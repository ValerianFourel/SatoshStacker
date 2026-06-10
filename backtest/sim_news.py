"""Point-in-time, news-aware LLM paper-trading simulation.

Replays a historical window bar-by-bar. At each simulated decision time T the agent
sees ONLY information available at T:
  * price history with open_time <= T (sliced from the real Binance series), and
  * BTC news with GDELT `seendate` <= T (query bounded `enddatetime = T`).
It computes quick metrics, fetches the point-in-time news, asks the Qwen advisor
(veto/shrink ONLY), clamps the verdict (shrink-only safety), and paper-trades the
re-anchoring ladder — tracking USD + BTC held. It reports accumulation vs the no-LLM
deterministic spine and vs DCA, so we can see whether news-aware timing helped.

NO-LEAKAGE GUARANTEES (why this is honest):
  * prices  — only bars up to T are visible; resting limits fill on the *current* bar's low.
  * news    — GDELT is queried with enddatetime=T and startdatetime=T-lookback; articles are
              dated by `seendate` (first observation), so nothing published after T appears.
  * the LLM is the PRODUCTION advisor: it may only defer/halve the next tranche on bad
    news/metrics — never buy more, buy above the ladder, or sell. The ladder governs.

Usage:
    python3 backtest/sim_news.py                       # default recent ~14-day window
    python3 backtest/sim_news.py 2022-05-05 2022-05-20 # the LUNA/UST collapse
    python3 backtest/sim_news.py 2022-05-05 2022-05-20 --step-hours 12 --no-news
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.analysis import _parse  # noqa: E402
from agent.config import LadderConfig  # noqa: E402
from agent.ladder import build_ladder, clamp_multiplier, filter_min_notional  # noqa: E402

DATA_4H = ROOT / "backtest" / "data" / "BTCUSDT_4h.parquet"
DATA_1D = ROOT / "backtest" / "data" / "BTCUSDT_1d.parquet"

MAKER_FEE = 0.001
TAKER_FEE = 0.001
GDELT = "https://api.gdeltproject.org/api/v2/doc/doc"

SIM_SYSTEM = (
    "You are a cautious risk advisor for a deterministic BTC accumulation ladder that buys "
    "MORE as price falls toward a floor. You may ONLY make the NEXT tranche more cautious — "
    "shrink or defer it — you can NEVER increase it, buy above the ladder, or sell. You are "
    "given the current portfolio (USD left, BTC held, average cost), computed metrics (RSI, "
    "volatility, drawdown, % above the target floor), and RECENT NEWS HEADLINES known only up "
    "to now. If news/metrics point to imminent further downside, capitulation, or stablecoin/"
    "exchange risk, DEFER or SHRINK so capital is preserved for lower prices; otherwise PROCEED "
    "at full size. Respond with STRICT JSON only:\n"
    '{"stance":"proceed|shrink|defer","size_multiplier":<float 0..1>,"note":"<short>",'
    '"as_of_bar":"<ISO8601 of now>"}'
)


# ───────────────────────── point-in-time news (GDELT) ─────────────────────────
# GDELT free tier rate-limits to ~1 request / 5s, so we throttle + cache PER DAY
# (one fetch per calendar day, reused for that day's intraday decision steps — still
# strictly point-in-time, just not refreshed intra-day).
_news_cache: dict[str, list[str]] = {}
_last_call: list[float] = [0.0]


_CRYPTO_KW = ("bitcoin", "btc", "crypto", "ether", "eth", "stablecoin", "usdt", "usdc",
              "binance", "coinbase", "etf", "sec ", "tether", "luna", "ftx", "halving")


def fetch_pit_news(as_of: datetime, *, lookback_hours: int = 72, max_items: int = 6,
                   query: str = "bitcoin sourcelang:english") -> list[str]:
    """Headlines with GDELT seendate in [as_of - lookback, as_of]. Never future. Fail-safe [].
    Filters to English, crypto-relevant titles."""
    import time
    key = as_of.strftime("%Y%m%d")  # cache per calendar day
    if key in _news_cache:
        return _news_cache[key]
    start = (as_of - timedelta(hours=lookback_hours)).strftime("%Y%m%d%H%M%S")
    end = as_of.strftime("%Y%m%d%H%M%S")
    params = {"query": query, "mode": "artlist", "maxrecords": 50, "format": "json",
              "startdatetime": start, "enddatetime": end, "sort": "datedesc"}
    out: list[str] = []
    for _attempt in range(3):
        gap = time.monotonic() - _last_call[0]
        if gap < 6.0:
            time.sleep(6.0 - gap)  # respect GDELT's ~1-req/5s limit (6s margin)
        try:
            r = requests.get(GDELT, params=params, timeout=20)
            _last_call[0] = time.monotonic()
            if r.status_code == 429:
                time.sleep(6.0)
                continue
            r.raise_for_status()
            seen = set()
            for a in (r.json().get("articles") or []):
                sd, title = a.get("seendate", ""), (a.get("title") or "").strip()
                if not title or (sd and sd > end):  # hard guard: never after `as_of`
                    continue
                tl = title.lower()
                if not any(k in tl for k in _CRYPTO_KW):
                    continue
                if tl in seen:
                    continue
                seen.add(tl)
                out.append(f"{sd[:8]}: {title}")
                if len(out) >= max_items:
                    break
            break
        except Exception:  # noqa: BLE001 - news is advisory only; never break the sim
            _last_call[0] = time.monotonic()
            break
    _news_cache[key] = out
    return out


# ───────────────────────── quick metrics (PIT) ───────────────────────────────
def quick_metrics(closes: np.ndarray, highs: np.ndarray, price: float, floor: float) -> dict:
    """Compute RSI(14), recent volatility, drawdown-from-window-high, distance to floor.
    Uses only the passed (already PIT-sliced) arrays."""
    def rsi(c: np.ndarray, n: int = 14) -> float:
        if len(c) < n + 1:
            return 50.0
        d = np.diff(c[-(n + 1):])
        up, dn = d[d > 0].sum(), -d[d < 0].sum()
        if dn == 0:
            return 100.0
        rs = (up / n) / (dn / n)
        return float(100 - 100 / (1 + rs))

    rets = np.diff(closes[-30:]) / closes[-30:-1] if len(closes) > 2 else np.array([0.0])
    vol = float(np.std(rets) * 100)  # % per bar
    wh = float(highs.max()) if len(highs) else price
    dd = float((price - wh) / wh * 100) if wh else 0.0
    return {
        "rsi14": round(rsi(closes), 1),
        "bar_vol_pct": round(vol, 2),
        "drawdown_from_high_pct": round(dd, 1),
        "pct_above_target_floor": round((price - floor) / max(price, 1) * 100, 1),
    }


# ───────────────────────── fills ─────────────────────────────────────────────
@dataclass
class Rung:
    price: float
    usdc: float
    filled: bool = False


@dataclass
class Portfolio:
    budget: float
    usd: float
    btc: float = 0.0
    deployed: float = 0.0
    fills: int = 0

    @property
    def avg_cost(self) -> float:
        return self.deployed / self.btc if self.btc > 0 else float("nan")


def _fill(pf: Portfolio, usdc: float, price: float, fee: float) -> None:
    btc = (usdc / price) * (1 - fee)
    pf.btc += btc
    pf.usd -= usdc
    pf.deployed += usdc
    pf.fills += 1


# ───────────────────────── the simulator ─────────────────────────────────────
@dataclass
class SimConfig:
    budget: float = 2000.0
    floor_price: float = 35000.0
    floor_frac: float = 0.55
    gen_fraction: float = 0.5
    n_tranches: int = 8
    max_generations: int = 6
    cap_per_day: float = 400.0
    min_notional: float = 10.0
    deadline_frac: float = 0.70
    use_llm: bool = True
    use_news: bool = True
    min_mult: float = 0.5


def _advise(client, model: str, features: dict, as_of: str, max_tokens: int = 512):
    """One real advisor call. Returns a Verdict; fail-safe to proceed/1.0 on any error."""
    import json

    from agent.analysis import Verdict
    try:
        resp = client.chat.completions.create(
            model=model, temperature=0.0, timeout=30, max_tokens=max_tokens,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": SIM_SYSTEM},
                      {"role": "user", "content": json.dumps({**features, "as_of_bar": as_of})}],
        )
        return _parse(resp.choices[0].message.content or "", as_of)
    except Exception as e:  # noqa: BLE001
        return Verdict.safe(as_of, f"advisor error ({type(e).__name__}): spine size")


def simulate(bars: pd.DataFrame, cfg: SimConfig, *, step_bars: int, deploy_by: datetime,
             end: datetime, client=None, model="qwen/qwen3.6-plus", verbose=True) -> Portfolio:
    """Run one paper-trading pass. `bars` is 4h OHLC (PIT order). Decisions every
    `step_bars` bars; fills processed every bar."""
    pf = Portfolio(budget=cfg.budget, usd=cfg.budget)
    rungs: list[Rung] = []
    gen, anchor, gen_floor, gen_budget = 0, 0.0, 0.0, 0.0
    committed_day, committed_usd = None, 0.0
    phase = "ladder"
    dca_left, dca_bars_left = 0.0, 0
    closes_all = bars["close"].to_numpy()
    highs_all = bars["high"].to_numpy()

    def open_generation(g, anc):
        nonlocal gen, anchor, gen_floor, gen_budget, rungs
        remaining = cfg.budget - pf.deployed - sum(r.usdc for r in rungs if not r.filled)
        gen, anchor = g, anc
        gen_floor = anc * cfg.floor_frac
        gen_budget = max(0.0, remaining) * cfg.gen_fraction
        lc = LadderConfig(budget_usdc=gen_budget, floor_price=gen_floor, n_tranches=cfg.n_tranches,
                          weighting="geometric", floor_frac=cfg.floor_frac)
        built = build_ladder(anc, lc, budget_override=gen_budget, floor_override=gen_floor)
        placeable, _ = filter_min_notional(built, cfg.min_notional)
        rungs = [Rung(price=r.price, usdc=r.usdc) for r in placeable]

    n = len(bars)
    for i in range(n):
        bar = bars.iloc[i]
        t = bar["dt"].to_pydatetime().astimezone(timezone.utc)
        op, low, close = float(bar["open"]), float(bar["low"]), float(bar["close"])
        day = t.strftime("%Y-%m-%d")
        if day != committed_day:
            committed_day, committed_usd = day, 0.0

        # ── deploy-by-deadline: cancel rungs, force-DCA the leftover ──
        if phase == "ladder" and t >= deploy_by:
            phase = "deadline"
            leftover = cfg.budget - pf.deployed
            dca_left = leftover if leftover >= cfg.min_notional else 0.0
            remaining_bars = max(1, n - i)
            dca_bars_left = max(1, remaining_bars // max(1, step_bars))
            if verbose:
                print(f"  {t:%Y-%m-%d %H:%M}  ⏰ deadline — force-DCA ${dca_left:,.0f}")

        if phase == "deadline":
            if dca_left > 0 and dca_bars_left > 0 and i % step_bars == 0:
                tr = dca_left / dca_bars_left
                _fill(pf, min(tr, pf.usd), op, TAKER_FEE)
                dca_left -= tr
                dca_bars_left -= 1
            continue

        # ── fills (every bar): resting rung fills if the bar low reaches it ──
        for r in rungs:
            if not r.filled and low <= r.price:
                _fill(pf, r.usdc, min(r.price, op), MAKER_FEE)
                r.filled = True

        # ── re-anchor if price broke below the generation floor ──
        if gen and close < gen_floor and gen < cfg.max_generations:
            remaining = cfg.budget - pf.deployed - sum(r.usdc for r in rungs if not r.filled)
            if remaining >= cfg.min_notional:
                open_generation(gen + 1, close)

        # ── decision step (every step_bars): metrics + PIT news + LLM ──
        if i % step_bars == 0:
            price = op
            if gen == 0:
                open_generation(1, price)
            metrics = quick_metrics(closes_all[: i + 1], highs_all[: i + 1], price, cfg.floor_price)
            news = fetch_pit_news(t) if (cfg.use_llm and cfg.use_news) else []
            mult, stance, note = 1.0, "spine", "(no-llm baseline)"
            if cfg.use_llm and client is not None:
                resting = [round(r.price) for r in rungs if not r.filled][:6]
                feats = {
                    "price": round(price, 2), "target_floor": cfg.floor_price,
                    "generation_floor": round(gen_floor), "resting_levels": resting,
                    "usd_remaining": round(pf.usd, 2), "btc_held": round(pf.btc, 6),
                    "avg_cost": (round(pf.avg_cost) if pf.btc > 0 else None),
                    "pct_deployed": round(100 * pf.deployed / cfg.budget, 1),
                    "days_left": round((end - t).total_seconds() / 86400, 1),
                    **metrics, "recent_news": news,
                }
                v = _advise(client, model, feats, t.strftime("%Y-%m-%dT%H:%M:%SZ"))
                mult = clamp_multiplier(v.size_multiplier, cfg.min_mult)
                stance, note = v.stance, v.note

            # place this gen's not-yet-placed rungs, sized * mult, under the daily cap
            placed_this_step = 0.0
            for r in rungs:
                if r.filled or getattr(r, "_placed", False):
                    continue
                sized = r.usdc * mult
                if sized < cfg.min_notional:
                    continue
                if committed_usd + sized > cfg.cap_per_day + 1e-9:
                    continue
                r.usdc = sized
                r._placed = True  # type: ignore[attr-defined]
                committed_usd += sized
                placed_this_step += sized

            if verbose:
                head = (news[0][9:60] + "…") if news else "—"
                print(f"  {t:%m-%d %H:%M} ${price:>7,.0f} rsi{metrics['rsi14']:>4.0f} "
                      f"dd{metrics['drawdown_from_high_pct']:>5.0f}% | {stance:7} x{mult:.2f} "
                      f"| btc={pf.btc:.5f} usd=${pf.usd:>5,.0f} | news: {head}")
    return pf


# ───────────────────────── baselines + runner ────────────────────────────────
def run_dca(bars: pd.DataFrame, budget: float, step_bars: int) -> Portfolio:
    pf = Portfolio(budget=budget, usd=budget)
    idx = list(range(0, len(bars), step_bars))
    each = budget / len(idx)
    for i in idx:
        _fill(pf, each, float(bars.iloc[i]["open"]), TAKER_FEE)
    return pf


def main() -> None:
    ap = argparse.ArgumentParser("sim_news")
    ap.add_argument("start", nargs="?", default=None, help="window start YYYY-MM-DD")
    ap.add_argument("end", nargs="?", default=None, help="window end YYYY-MM-DD")
    ap.add_argument("--step-hours", type=int, default=12, help="decision cadence")
    ap.add_argument("--budget", type=float, default=2000.0)
    ap.add_argument("--floor", type=float, default=35000.0)
    ap.add_argument("--no-news", action="store_true")
    ap.add_argument("--no-llm", action="store_true", help="only run the deterministic baselines")
    args = ap.parse_args()

    df = pd.read_parquet(DATA_4H)
    if args.start:
        s = pd.Timestamp(args.start, tz="UTC")
        e = pd.Timestamp(args.end, tz="UTC") if args.end else df["dt"].iloc[-1]
    else:  # default: last ~14 days available
        e = df["dt"].iloc[-1]
        s = e - pd.Timedelta(days=14)
    bars = df[(df["dt"] >= s) & (df["dt"] <= e)].reset_index(drop=True)
    if len(bars) < 6:
        print("window too short"); return
    step_bars = max(1, args.step_hours // 4)
    end_dt = bars["dt"].iloc[-1].to_pydatetime().astimezone(timezone.utc)
    start_dt = bars["dt"].iloc[0].to_pydatetime().astimezone(timezone.utc)
    deploy_by = start_dt + (end_dt - start_dt) * 0.70

    cfg = SimConfig(budget=args.budget, floor_price=args.floor)
    print(f"\n=== PIT news-aware sim  {start_dt:%Y-%m-%d} .. {end_dt:%Y-%m-%d}  "
          f"(every {args.step_hours}h, {len(bars)} 4h-bars)  budget=${args.budget:,.0f} "
          f"floor=${args.floor:,.0f} ===")
    print(f"    open=${bars['open'].iloc[0]:,.0f}  low=${bars['low'].min():,.0f}  "
          f"close=${bars['close'].iloc[-1]:,.0f}")

    client = None
    if not args.no_llm:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=True)
        import os

        from agent.secrets import clean_secret
        key = clean_secret(os.getenv("LLM_API_KEY"))
        if key:
            from openai import OpenAI
            client = OpenAI(base_url=os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
                            api_key=key)
            print("    LLM: ON (OpenRouter), news:", "ON" if not args.no_news else "OFF")
        else:
            print("    LLM: no key found — running baselines only")

    # 1) news-aware LLM run
    res_llm = None
    if client is not None:
        c = SimConfig(budget=args.budget, floor_price=args.floor,
                      use_llm=True, use_news=not args.no_news)
        print("\n  --- news-aware LLM run ---")
        res_llm = simulate(bars.copy(), c, step_bars=step_bars, deploy_by=deploy_by,
                           end=end_dt, client=client)
    # 2) deterministic spine (no LLM) + 3) DCA
    spine = simulate(bars.copy(), SimConfig(budget=args.budget, floor_price=args.floor,
                     use_llm=False), step_bars=step_bars, deploy_by=deploy_by, end=end_dt,
                     verbose=False)
    dca = run_dca(bars, args.budget, step_bars)

    final = float(bars["close"].iloc[-1])
    print("\n" + "=" * 78)
    print(f"{'strategy':<22}{'BTC':>12}{'avg cost':>12}{'deployed':>11}{'port. $':>11}")
    print("-" * 78)

    def row(name, pf):
        ac = "n/a" if pf.btc <= 0 else f"${pf.avg_cost:,.0f}"
        print(f"{name:<22}{pf.btc:>12.6f}{ac:>12}{f'{100*pf.deployed/args.budget:.0f}%':>11}"
              f"{f'${pf.btc*final + pf.usd:,.0f}':>11}")

    if res_llm:
        row("news-aware LLM", res_llm)
    row("deterministic spine", spine)
    row("DCA", dca)
    if res_llm:
        d = (res_llm.btc / spine.btc - 1) * 100 if spine.btc else 0
        print(f"\n  news-aware LLM vs spine: {d:+.2f}% BTC  "
              f"({'more cautious — preserved USD' if res_llm.deployed < spine.deployed else 'similar'})")
    print("  NOTE: the LLM is veto/shrink-only, so it can only match or trail the spine on raw")
    print("  BTC — its job is to AVOID buying into further downside, trading a little stack for")
    print("  lower average cost / preserved dry powder. Judge it on avg cost + drawdown, not just BTC.")


if __name__ == "__main__":
    main()
