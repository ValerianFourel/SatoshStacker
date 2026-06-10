"""LLM satoshi-stacker: an active BTC<->USDC trader that maximizes ENDING BTC.

Starts with $2000 USDC at a past point in time. Every 4h it computes momentum metrics
and fetches point-in-time news, and the LLM decides a TARGET BTC allocation (0 = all USDC,
1 = all BTC) to **stack the most satoshis** — i.e. hold USDC before drops (to rebuy lower)
and hold BTC at/after bottoms. It rebalances toward the target with realistic taker fees +
slippage. Reports ending BTC vs buy-and-hold, DCA, and a no-LLM momentum rule.

⚠️ RESEARCH sim. This trader BUYS AND SELLS — unlike the production SatoshiStacker agent
(accumulate-only, veto/shrink LLM). Timing with costs is hard; this measures whether
momentum+news actually stacks more sats than just holding.

No look-ahead: prices sliced to open_time <= T; news GDELT-bounded to seendate <= T.

    python3 backtest/sim_sats.py 2026-05-20 2026-06-10            # LLM run, 4h decisions
    python3 backtest/sim_sats.py 2026-05-20 2026-06-10 --no-llm   # free momentum-rule baseline
    python3 backtest/sim_sats.py 2022-05-05 2022-05-25            # the LUNA collapse
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.sim_news import fetch_pit_news  # noqa: E402  (point-in-time GDELT news)

DATA_4H = ROOT / "backtest" / "data" / "BTCUSDT_4h.parquet"
TAKER = 0.001       # Binance taker fee per trade
SLIP = 0.0005       # assumed slippage per market trade (round-trip cost ~0.3%)
REBAL_BAND = 0.10   # don't trade unless target differs from current by > this (anti-churn)


def _load_technicals() -> tuple[int, str]:
    """Read the weekly self-tuner's output (agent/technicals.json): the tuned RSI
    period and the context note that the LLM trader injects into its prompt."""
    try:
        d = json.loads((ROOT / "agent" / "technicals.json").read_text())
        return int(d.get("suggested", {}).get("rsi_period", 14)), d.get("context_note", "")
    except Exception:  # noqa: BLE001
        return 14, ""


RSI_PERIOD, WEEKLY_CONTEXT = _load_technicals()

TACTICAL_SYSTEM = (
    "You are a tactical crypto trader whose SOLE objective is to maximize the ENDING amount "
    "of BITCOIN (satoshis), starting from USDC. You swap between USDC and BTC. You gain "
    "satoshis by holding USDC BEFORE price falls (then rebuying more BTC lower) and holding "
    "BTC at/after bottoms and uptrends. You are given momentum metrics (RSI, fast/slow moving-"
    "average trend, 24h & 72h returns, volatility, drawdown-from-high), your current USDC and "
    "BTC, and RECENT NEWS HEADLINES known ONLY up to now. Decide a TARGET fraction of total "
    "portfolio value to hold in BTC right now: 0.0 = all USDC (expect lower), 1.0 = all BTC "
    "(expect flat/up). Round-trip trading costs ~0.3%, so only move decisively when momentum/"
    "news warrant it. Respond STRICT JSON only:\n"
    '{"target_btc_fraction":<float 0..1>,"stance":"long_btc|long_cash|hold","note":"<short>",'
    '"as_of_bar":"<ISO8601 now>"}'
)


# ───────────────────────── momentum metrics (point-in-time) ───────────────────
def momentum(closes: np.ndarray, highs: np.ndarray, price: float, bars_per_day: int = 6) -> dict:
    def rsi(c, n=14):
        if len(c) < n + 1:
            return 50.0
        d = np.diff(c[-(n + 1):])
        up, dn = d[d > 0].sum(), -d[d < 0].sum()
        return 100.0 if dn == 0 else float(100 - 100 / (1 + (up / n) / (dn / n)))

    def sma(c, n):
        return float(np.mean(c[-n:])) if len(c) >= n else float(np.mean(c))

    _rsi_n = RSI_PERIOD  # weekly-tuned RSI period (default 14)
    fast, slow = sma(closes, bars_per_day * 3), sma(closes, bars_per_day * 10)
    r24 = float(price / closes[-bars_per_day] - 1) * 100 if len(closes) > bars_per_day else 0.0
    r72 = float(price / closes[-bars_per_day * 3] - 1) * 100 if len(closes) > bars_per_day * 3 else 0.0
    rets = np.diff(closes[-30:]) / closes[-30:-1] if len(closes) > 2 else np.array([0.0])
    wh = float(highs.max()) if len(highs) else price
    return {
        "rsi14": round(rsi(closes, _rsi_n), 1),
        "trend_pct": round((fast / slow - 1) * 100, 2) if slow else 0.0,   # fast vs slow MA
        "ret_24h_pct": round(r24, 2),
        "ret_72h_pct": round(r72, 2),
        "vol_bar_pct": round(float(np.std(rets) * 100), 2),
        "drawdown_from_high_pct": round((price - wh) / wh * 100, 1) if wh else 0.0,
    }


# ───────────────────────── portfolio + execution ─────────────────────────────
@dataclass
class Book:
    usdc: float
    btc: float = 0.0
    trades: int = 0

    def value(self, price: float) -> float:
        return self.usdc + self.btc * price

    def btc_frac(self, price: float) -> float:
        v = self.value(price)
        return (self.btc * price) / v if v > 0 else 0.0


def rebalance(bk: Book, price: float, target: float) -> None:
    """Move toward `target` BTC fraction of portfolio value, with taker fee + slippage."""
    target = max(0.0, min(1.0, target))
    tv = bk.value(price)
    if tv <= 0:
        return
    cur = bk.btc_frac(price)
    if abs(target - cur) < REBAL_BAND:
        return
    delta_val = target * tv - bk.btc * price
    if delta_val > 0:  # buy BTC with USDC
        spend = min(delta_val, bk.usdc)
        if spend < 10:
            return
        bk.btc += spend * (1 - TAKER) / (price * (1 + SLIP))
        bk.usdc -= spend
        bk.trades += 1
    else:              # sell BTC for USDC
        sell_val = min(-delta_val, bk.btc * price)
        if sell_val < 10:
            return
        btc_sold = sell_val / price
        bk.usdc += btc_sold * price * (1 - SLIP) * (1 - TAKER)
        bk.btc -= btc_sold
        bk.trades += 1


def momentum_rule(m: dict) -> float:
    """Deterministic no-LLM baseline: buy dips, lighten into strength (mean-reversion)."""
    rsi = m["rsi14"]
    if rsi < 35:
        return 1.0
    if rsi > 68:
        return 0.2
    return 0.65 if m["trend_pct"] > 0 else 0.4


# ───────────────────────── LLM decision ──────────────────────────────────────
def llm_target(client, model, feats: dict, as_of: str, fallback: float) -> tuple[float, str, str]:
    """Ask the LLM for a target BTC fraction. Fail-safe: keep current allocation."""
    try:
        resp = client.chat.completions.create(
            model=model, temperature=0.0, timeout=18, max_tokens=400,
            messages=[{"role": "system", "content": TACTICAL_SYSTEM + (
                          f"\n\nTHIS WEEK'S TUNED GUIDANCE (from the weekly backtest+meta-LLM): "
                          f"{WEEKLY_CONTEXT}" if WEEKLY_CONTEXT else "")},
                      {"role": "user", "content": json.dumps({**feats, "as_of_bar": as_of})}])
        txt = resp.choices[0].message.content or ""
        i, j = txt.find("{"), txt.rfind("}")          # robustly extract the JSON object
        d = json.loads(txt[i:j + 1]) if 0 <= i < j else {}
        tgt = float(d.get("target_btc_fraction", fallback))
        return max(0.0, min(1.0, tgt)), str(d.get("stance", "?")), str(d.get("note", ""))[:70]
    except Exception as e:  # noqa: BLE001 - keep current allocation on any error
        return fallback, "error", f"{type(e).__name__}: hold"


# ───────────────────────── simulator ─────────────────────────────────────────
def simulate(bars, *, step_bars, budget, mode, client=None, model="qwen/qwen3.6-plus",
             use_news=True, verbose=True) -> Book:
    bk = Book(usdc=budget)
    closes = bars["close"].to_numpy()
    highs = bars["high"].to_numpy()
    bpd = max(1, 24 // 4)
    n = len(bars)
    for i in range(n):
        bar = bars.iloc[i]
        if i % step_bars != 0:
            continue  # decisions only at the cadence
        t = bar["dt"].to_pydatetime().astimezone(timezone.utc)
        price = float(bar["open"])
        m = momentum(closes[: i + 1], highs[: i + 1], price, bpd)
        cur_frac = bk.btc_frac(price)
        if mode == "rule":
            target, stance, note = momentum_rule(m), "rule", ""
        else:
            news = fetch_pit_news(t) if use_news else []
            feats = {"price": round(price, 2), "usdc": round(bk.usdc, 2),
                     "btc": round(bk.btc, 6), "btc_fraction_now": round(cur_frac, 2),
                     **m, "recent_news": news}
            target, stance, note = llm_target(
                client, model, feats, t.strftime("%Y-%m-%dT%H:%M:%SZ"), cur_frac)
        rebalance(bk, price, target)
        if verbose:
            print(f"  {t:%m-%d %H:%M} ${price:>7,.0f} rsi{m['rsi14']:>3.0f} "
                  f"r24{m['ret_24h_pct']:>+5.1f}% dd{m['drawdown_from_high_pct']:>5.0f}% | "
                  f"{stance:9} tgt{target:.2f}->{bk.btc_frac(price):.2f} | "
                  f"btc={bk.btc:.5f} usdc=${bk.usdc:>5,.0f}"
                  + (f" | {note}" if note else ""))
    return bk


def simulate_parallel(bars, *, step_bars, budget, client, model, use_news=True,
                      workers=16, verbose=True) -> Book:
    """Same as simulate(mode='llm') but FIRES ALL LLM CALLS CONCURRENTLY.

    Valid because the LLM's target is a *market view* from point-in-time momentum + news
    (portfolio-independent); execution (rebalancing) is then done sequentially and instantly.
    Cuts a ~30-min sequential run to ~2-3 min."""
    from concurrent.futures import ThreadPoolExecutor
    closes = bars["close"].to_numpy()
    highs = bars["high"].to_numpy()
    bpd = max(1, 24 // 4)
    dec = list(range(0, len(bars), step_bars))

    # phase 1: metrics (instant) + point-in-time news (sequential — GDELT is rate-limited,
    # but day-cached so only ~1 fetch per calendar day)
    inputs = []
    for i in dec:
        bar = bars.iloc[i]
        t = bar["dt"].to_pydatetime().astimezone(timezone.utc)
        price = float(bar["open"])
        m = momentum(closes[: i + 1], highs[: i + 1], price, bpd)
        news = fetch_pit_news(t) if use_news else []
        inputs.append((t, price, m, news))

    # phase 2: ALL LLM market-view calls in parallel
    def call(inp):
        t, price, m, news = inp
        feats = {"price": round(price, 2), **m, "recent_news": news}
        return llm_target(client, model, feats, t.strftime("%Y-%m-%dT%H:%M:%SZ"), 0.5)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        decisions = list(ex.map(call, inputs))

    # phase 3: sequential rebalance (instant) using the pre-computed targets
    bk = Book(usdc=budget)
    for (t, price, m, news), (target, stance, note) in zip(inputs, decisions):
        rebalance(bk, price, target)
        if verbose:
            print(f"  {t:%m-%d %H:%M} ${price:>7,.0f} rsi{m['rsi14']:>3.0f} "
                  f"r24{m['ret_24h_pct']:>+5.1f}% dd{m['drawdown_from_high_pct']:>5.0f}% | "
                  f"{stance:9} tgt{target:.2f}->{bk.btc_frac(price):.2f} | "
                  f"btc={bk.btc:.5f} usdc=${bk.usdc:>5,.0f}" + (f" | {note}" if note else ""))
    return bk


def buy_and_hold(bars, budget):
    bk = Book(usdc=budget)
    rebalance(bk, float(bars.iloc[0]["open"]), 1.0)
    return bk


def dca(bars, budget, step_bars):
    bk = Book(usdc=budget)
    idx = list(range(0, len(bars), step_bars))
    each = budget / len(idx)
    for i in idx:
        p = float(bars.iloc[i]["open"])
        bk.btc += each * (1 - TAKER) / p
        bk.usdc -= each
        bk.trades += 1
    return bk


def main():
    ap = argparse.ArgumentParser("sim_sats")
    ap.add_argument("start", nargs="?", default=None)
    ap.add_argument("end", nargs="?", default=None)
    ap.add_argument("--step-hours", type=int, default=4)
    ap.add_argument("--budget", type=float, default=2000.0)
    ap.add_argument("--no-news", action="store_true")
    ap.add_argument("--no-llm", action="store_true", help="momentum-rule baseline only (free)")
    ap.add_argument("--model", default=None, help="override LLM model (e.g. a faster one)")
    ap.add_argument("--parallel", action="store_true", help="fire LLM calls concurrently (fast)")
    ap.add_argument("--workers", type=int, default=16, help="parallel LLM workers")
    args = ap.parse_args()

    df = pd.read_parquet(DATA_4H)
    if args.start:
        s = pd.Timestamp(args.start, tz="UTC")
        e = pd.Timestamp(args.end, tz="UTC") if args.end else df["dt"].iloc[-1]
    else:
        e = df["dt"].iloc[-1]; s = e - pd.Timedelta(days=21)
    bars = df[(df["dt"] >= s) & (df["dt"] <= e)].reset_index(drop=True)
    if len(bars) < 6:
        print("window too short"); return
    step_bars = max(1, args.step_hours // 4)
    p0, pend = float(bars["open"].iloc[0]), float(bars["close"].iloc[-1])
    lo = float(bars["low"].min())
    n_dec = len(range(0, len(bars), step_bars))
    print(f"\n=== Satoshi-stacker sim  {bars['dt'].iloc[0]:%Y-%m-%d}..{bars['dt'].iloc[-1]:%Y-%m-%d}"
          f"  ${args.budget:,.0f} USDC, decide every {args.step_hours}h ({n_dec} decisions) ===")
    print(f"    BTC: open=${p0:,.0f}  low=${lo:,.0f}  close=${pend:,.0f}  "
          f"({(pend/p0-1)*100:+.1f}% over window)")

    client = None
    model = args.model or "qwen/qwen3.6-plus"
    if not args.no_llm:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=True)
        import os

        from agent.secrets import clean_secret
        key = clean_secret(os.getenv("LLM_API_KEY"))
        if key:
            from openai import OpenAI
            client = OpenAI(base_url=os.getenv("LLM_BASE_URL"), api_key=key)
            print(f"    LLM: ON  model={model}  news: "
                  f"{'ON' if not args.no_news else 'OFF'}  (~{n_dec} calls)")
        else:
            print("    no LLM key — momentum-rule only"); args.no_llm = True

    results = {}
    if client is not None:
        if args.parallel:
            print(f"\n  --- LLM trader (PARALLEL x{args.workers}, momentum + news) ---")
            results["LLM trader"] = simulate_parallel(
                bars.copy(), step_bars=step_bars, budget=args.budget, client=client,
                model=model, use_news=not args.no_news, workers=args.workers)
        else:
            print("\n  --- LLM trader (momentum + news) ---")
            results["LLM trader"] = simulate(bars.copy(), step_bars=step_bars,
                                             budget=args.budget, mode="llm", client=client,
                                             model=model, use_news=not args.no_news)
    print("\n  --- momentum-rule (no LLM) ---" if args.no_llm else "")
    results["momentum-rule"] = simulate(bars.copy(), step_bars=step_bars, budget=args.budget,
                                        mode="rule", verbose=args.no_llm)
    results["buy & hold"] = buy_and_hold(bars, args.budget)
    results["DCA"] = dca(bars, args.budget, step_bars)

    print("\n" + "=" * 82)
    print(f"  GOAL = most BTC.  (terminal value in BTC = how many sats you could hold at close)")
    print("=" * 82)
    print(f"{'strategy':<18}{'BTC held':>12}{'+ USDC':>9}{'TOTAL in BTC':>15}{'vs B&H':>9}{'trades':>8}")
    print("-" * 82)
    bh = results["buy & hold"].btc + results["buy & hold"].usdc / pend
    for name, bk in results.items():
        tot = bk.btc + bk.usdc / pend     # terminal value denominated in BTC = "max sats"
        print(f"{name:<18}{bk.btc:>12.6f}{f'${bk.usdc:,.0f}':>9}{tot:>15.6f}"
              f"{f'{(tot/bh-1)*100:+.1f}%':>9}{bk.trades:>8}")
    print("\n  'TOTAL in BTC' is the satoshi-maximization score (BTC held + USDC converted at the")
    print("  close). Beating 'buy & hold' means the momentum/news timing actually stacked more sats.")


if __name__ == "__main__":
    main()
