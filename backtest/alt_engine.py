"""Alt-stacker historical backtest + gate (SOL/ETH/HYPE/GOLD on real Kraken daily data).

Replays cached Kraken daily candles through the EXACT production sizing spine
(`agent.alts.clamp_targets` + `plan_trades` + `Position`) with a pluggable per-asset
allocator, and scores the bot against equal-weight DCA / HODL / all-cash portfolios.

    python3 backtest/alt_fetch.py                       # one-time: pull Kraken history
    python3 backtest/alt_engine.py                      # deterministic gate (SOL/ETH/PAXG, 2y)
    python3 backtest/alt_engine.py --assets SOL,ETH,HYPE,PAXG --window-days 130
    python3 backtest/alt_engine.py --allocator llm --assets SOL,ETH,HYPE,PAXG \
            --cadence 7 --window-days 120                # LLM-in-the-loop (bounded; costs tokens)

The GATE (writes backtest/results/ALT_GATE_PASSED) is certified on the DETERMINISTIC
reference allocator — reproducible, no hindsight. The `--allocator llm` run is illustrative
("how does the bot do when buying with the LLM") and is NOT the certified gate: it is a single
path, bounded in length, and the LLM can recognize an asset's price history (hindsight). See
the honesty caveats printed at the end.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.alts import (  # noqa: E402 - need sys.path first
    AssetSpec, Position, clamp_targets, plan_trades,
)
from agent.trader import momentum as _live_momentum  # noqa: E402 - exact live momentum

BARS_PER_DAY = {"5m": 288, "15m": 96, "30m": 48, "1h": 24, "4h": 6, "8h": 3, "12h": 2, "1d": 1}

DATA = Path(__file__).resolve().parent / "data" / "alt"
RESULTS = Path(__file__).resolve().parent / "results"
FEE = 0.0026   # Kraken taker fee (worst-case retail tier) — conservative
GATE_MARKER = RESULTS / "ALT_GATE_PASSED"

# default specs (objective + per-asset max fraction), mirrors the live default:
# XMR = MAKE USD (accumulate_quote: grow the quote, never net-hold); gold = ounces hedge.
# (SOL/ETH/HYPE kept for ad-hoc comparison runs but dropped from the live default.)
DEFAULT_SPECS = {
    "ZEC": AssetSpec("ZEC", "accumulate_quote", 0.6),
    "TRX": AssetSpec("TRX", "accumulate_quote", 0.6),
    "BNB": AssetSpec("BNB", "accumulate_quote", 0.6),
    "XMR": AssetSpec("XMR", "accumulate_quote", 0.6),
    "SOL": AssetSpec("SOL", "accumulate_quote", 0.6),
    "ETH": AssetSpec("ETH", "accumulate_quote", 0.6),
    "HYPE": AssetSpec("HYPE", "accumulate_quote", 0.6),
    "PAXG": AssetSpec("PAXG", "store_of_value", 0.5),
}


class _Rails:
    rebal_band = 0.08
    min_trade_usdc = 10.0
    max_cycle_turnover_usdc = 400.0
    max_daily_turnover_usdc = 1000.0
    max_total_allocation = 1.0


# ───────────────────────── data ────────────────────────────────────────────
def _load_csv(base: str, tf: str = "1d") -> dict[str, tuple[float, float]]:
    p = DATA / f"{base}_USD_{tf}.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} — run: python3 backtest/alt_fetch.py --tf {tf} {base}")
    out: dict[str, tuple[float, float]] = {}
    with p.open() as f:
        for row in csv.DictReader(f):
            out[row["date"]] = (float(row["high"]), float(row["close"]))
    return out


def load_series(bases: list[str], tf: str = "1d"):
    """Return (dates, closes{base:np}, highs{base:np}) aligned on the common timestamp set."""
    series = {b: _load_csv(b, tf) for b in bases}
    common = sorted(set.intersection(*[set(s) for s in series.values()]))
    closes = {b: np.array([series[b][d][1] for d in common], float) for b in bases}
    highs = {b: np.array([series[b][d][0] for d in common], float) for b in bases}
    return common, closes, highs


def load_fng() -> dict[str, int]:
    """Point-in-time crypto Fear&Greed by date (alternative.me dated history). LEAK-FREE:
    each value is the index AS OF that date, so feeding it into a past decision is honest."""
    import datetime as dt
    import requests
    try:
        data = requests.get("https://api.alternative.me/fng/", params={"limit": 0},
                            timeout=20).json()["data"]
    except Exception:  # noqa: BLE001 - sentiment is optional
        return {}
    out = {}
    for x in data:
        d = dt.datetime.fromtimestamp(int(x["timestamp"]), dt.timezone.utc).strftime("%Y-%m-%d")
        out[d] = int(x["value"])
    return out


def momentum_at(closes: np.ndarray, highs: np.ndarray, i: int, tf: str,
                rsi_n: int = 14) -> dict:
    """Point-in-time momentum at bar i. For 4h use the EXACT live `agent.trader.momentum`
    (bpd=6 bars/day); for 1d use the daily variant. Same feature keys either way."""
    if tf != "1d":
        bpd = BARS_PER_DAY.get(tf, 6)
        return _live_momentum(closes[: i + 1], highs[: i + 1], float(closes[i]), rsi_n, bpd=bpd)
    return daily_momentum(closes, highs, i, rsi_n)


def daily_momentum(closes: np.ndarray, highs: np.ndarray, i: int, rsi_n: int = 14) -> dict:
    """Point-in-time daily momentum at index i (uses ONLY data up to and incl. day i).
    Same feature keys as agent.trader.momentum so the LLM prompt is consistent."""
    c = closes[: i + 1]
    h = highs[: i + 1]

    def rsi(x, n):
        if len(x) < n + 1:
            return 50.0
        d = np.diff(x[-(n + 1):])
        up, dn = d[d > 0].sum(), -d[d < 0].sum()
        return 100.0 if dn == 0 else float(100 - 100 / (1 + (up / n) / (dn / n)))

    def sma(x, n):
        return float(np.mean(x[-n:])) if len(x) >= n else float(np.mean(x))

    px = float(c[-1])
    fast, slow = sma(c, 10), sma(c, 30)
    r1 = float(px / c[-2] - 1) * 100 if len(c) > 1 else 0.0
    r3 = float(px / c[-4] - 1) * 100 if len(c) > 3 else 0.0
    wh = float(h[-90:].max()) if len(h) else px
    return {"rsi": round(rsi(c, rsi_n), 1), "rsi_period": rsi_n,
            "trend_pct": round((fast / slow - 1) * 100, 2) if slow else 0.0,
            "ret_24h_pct": round(r1, 2), "ret_72h_pct": round(r3, 2),
            "drawdown_from_high_pct": round((px - wh) / wh * 100, 1) if wh else 0.0}


# ───────────────────────── allocators ──────────────────────────────────────
def momentum_target(objective: str, feats: dict, max_frac: float) -> float:
    """Deterministic reference: buy dips / hold for accumulate assets; range-trade the
    quote asset (high allocation when oversold/low, zero when overbought/high)."""
    rsi = feats["rsi"]
    dd = feats["drawdown_from_high_pct"]  # <= 0
    if objective == "accumulate_quote":
        if rsi <= 40:
            s = 1.0
        elif rsi >= 60:
            s = 0.0
        else:
            s = (60 - rsi) / 20
        return round(max_frac * s, 4)
    # accumulate_base / store_of_value: lean in on weakness, trim on strength
    if rsi <= 35:
        s = 0.95
    elif rsi >= 70:
        s = 0.25
    else:
        s = 0.25 + (70 - rsi) / 35 * 0.70
    boost = min(0.20, max(0.0, -dd) / 100.0 * 0.6)   # deeper drawdown -> accumulate more
    return round(min(1.0, s + boost) * max_frac, 4)


def momentum_trend_target(objective: str, feats: dict, max_frac: float) -> float:
    """TREND-FOLLOWING make-USD: the study found POSITIVE IC (momentum persists) on TRX/BNB at
    daily, so go LONG into confirmed strength and CASH on weakness — the opposite of the
    mean-reversion `momentum_target`. store_of_value (gold) accumulates steadily."""
    if objective == "store_of_value":
        return max_frac
    rsi = feats["rsi"]
    trend = feats["trend_pct"]          # fast EMA vs slow EMA (>0 = uptrend)
    r3 = feats["ret_72h_pct"]
    if trend > 0 and rsi >= 52 and r3 > 0:          # confirmed uptrend -> ride it
        s = 0.6 + min(0.4, max(0.0, trend) / 15.0)  # stronger trend -> larger (cap 1.0)
        return round(min(1.0, s) * max_frac, 4)
    if trend < 0 or rsi < 48:                        # downtrend/weak -> cash
        return 0.0
    return round(0.25 * max_frac, 4)                 # neutral -> small toehold


def make_llm_allocator(model: str | None, specs: dict):
    """LLM allocator: reuse AltStacker.decide_asset (the EXACT live prompt + fail-safe).
    News/Fear&Greed are OFF (they leak hindsight in a historical sim)."""
    import os

    from agent.alts import AltConfig, AltStacker, AltStore
    from agent.multi_exchange import MultiPaperExchange
    from agent.notify import Notifier
    from agent.secrets import clean_secret

    key = clean_secret(os.getenv("LLM_API_KEY"))
    if not key:
        raise SystemExit("LLM allocator needs LLM_API_KEY in the environment/.env")
    from openai import OpenAI
    client = OpenAI(base_url=os.getenv("LLM_BASE_URL") or None, api_key=key)
    fng = load_fng()   # point-in-time crypto sentiment (leak-free); {} if unavailable
    print(f"[llm] point-in-time Fear&Greed loaded for {len(fng)} dates "
          f"(headlines omitted — Yahoo isn't point-in-time; GDELT is a future add)")
    cfg = AltConfig(mode="dry_run", venue="kraken", quote="USD",
                    assets=tuple(specs.values()),   # so decide_asset knows each objective
                    model=model or os.getenv("ALT_MODEL", "qwen/qwen3.5-plus-20260420"),
                    news_enabled=False, self_tune=False)
    ex = MultiPaperExchange(list(specs), quote="USD", cash_usdc=0.0, taker_fee=FEE,
                            store_path="/tmp/_alt_bt_book.json", price_source=lambda b: 1.0)
    st = AltStacker(cfg=cfg, exchange=ex, store=AltStore("/tmp/_alt_bt.db"),
                    notifier=Notifier(), llm_client=client, ohlcv_source=lambda s: None)

    _prog = {"n": 0}

    def alloc(base, spec, feats, price, pos, pot, cash, cur, date):
        _prog["n"] += 1
        print(f"[llm-progress] call {_prog['n']} ({base} {date})", file=sys.stderr, flush=True)
        fg = fng.get(date)
        f = {"base": base, "objective": spec.objective, "max_fraction": spec.max_fraction,
             "pot_usd": round(pot, 2), "cash_usdc": round(cash, 2),
             "cash_fraction": round(cash / pot, 3) if pot > 0 else 1.0,
             "fear_greed": ({"value": fg} if fg is not None else None),
             "rsi_period": feats["rsi_period"],
             "price": round(price, 4), "units": round(pos.units, 6),
             "value_usd": round(pos.value(price), 2),
             "avg_entry": round(pos.avg, 4) if pos.units > 0 else None,
             "fraction_now": round(cur, 3), **feats, "headlines": []}
        tgt, _stance, _note = st.decide_asset(base, f, cur)
        return tgt
    return alloc


# ───────────────────────── the simulation ──────────────────────────────────
def run_backtest(bases, specs, allocator, *, stack, cadence, start_idx=0, fee=FEE, tf="1d",
                 rebal_band=None, end_trim=0):
    dates, closes, highs = load_series(bases, tf)
    T = len(dates)
    last = T - end_trim     # effective end (exclusive) — lets callers test disjoint windows
    rails = _Rails()
    if rebal_band is not None:   # smaller band -> more frequent trades (true scalping)
        rails.rebal_band = rebal_band
    decision_days = set(range(start_idx, last, cadence))
    n_decisions = len(decision_days)
    slice_ = stack / len(bases)
    max_by_base = {b: specs[b].max_fraction for b in bases}

    cash = stack
    pos = {b: Position() for b in bases}
    hodl = {b: slice_ * (1 - fee) / closes[b][start_idx] for b in bases}   # lump at start
    dca_units = {b: 0.0 for b in bases}
    dca_cash = {b: slice_ for b in bases}
    dca_slice = {b: slice_ / max(1, n_decisions) for b in bases}

    for t in range(start_idx, last):
        prices = {b: float(closes[b][t]) for b in bases}
        if t not in decision_days:
            continue
        # equal-weight DCA benchmark deploys its slice on each decision day
        for b in bases:
            spend = min(dca_slice[b], dca_cash[b])
            if spend > 0:
                dca_units[b] += spend * (1 - fee) / prices[b]
                dca_cash[b] -= spend
        # bot decides per asset, clamps via the cage, plans via the shared spine
        pot = cash + sum(pos[b].value(prices[b]) for b in bases)
        targets = {}
        for b in bases:
            feats = momentum_at(closes[b], highs[b], t, tf)
            cur = pos[b].value(prices[b]) / pot if pot > 0 else 0.0
            targets[b] = allocator(b, specs[b], feats, prices[b], pos[b], pot, cash,
                                   cur, dates[t])
        ctargets = clamp_targets(targets, max_by_base, rails.max_total_allocation)
        values = {b: pos[b].value(prices[b]) for b in bases}
        plan = plan_trades(prices, values, cash, pot, ctargets,
                           rebal_band=rails.rebal_band, min_trade=rails.min_trade_usdc,
                           cycle_cap=rails.max_cycle_turnover_usdc,
                           daily_left=rails.max_daily_turnover_usdc, fee=fee)
        for base, side, notional in plan:
            px = prices[base]
            if side == "sell":
                units = min(notional / px, pos[base].units)
                proceeds = units * px * (1 - fee)
                pos[base].realized_quote += proceeds - pos[base].avg * units
                pos[base].cost = max(0.0, pos[base].cost - pos[base].avg * units)
                pos[base].units -= units
                cash += proceeds
            else:
                spend = min(notional, cash)
                if spend < rails.min_trade_usdc:
                    continue
                pos[base].units += spend * (1 - fee) / px
                pos[base].cost += spend
                cash -= spend

    endpx = {b: float(closes[b][last - 1]) for b in bases}
    return {
        "dates": (dates[start_idx], dates[last - 1]), "n_bars": last - start_idx, "tf": tf,
        "fee": fee, "n_decisions": n_decisions, "bases": bases, "stack": stack,
        "bot_value": cash + sum(pos[b].value(endpx[b]) for b in bases),
        "bot_cash": cash,
        "hodl_value": sum(hodl[b] * endpx[b] for b in bases),
        "dca_value": sum(dca_units[b] * endpx[b] + dca_cash[b] for b in bases),
        "cash_value": stack,
        "per_asset": {b: {
            "objective": specs[b].objective,
            "start_px": float(closes[b][start_idx]), "end_px": endpx[b],
            "px_change_pct": round((endpx[b] / closes[b][start_idx] - 1) * 100, 1),
            "bot_units": pos[b].units, "dca_units": dca_units[b], "hodl_units": hodl[b],
            "bot_realized_usd": pos[b].realized_quote,
        } for b in bases},
    }


# ───────────────────────── reporting + gate ────────────────────────────────
def _print(res, allocator_name):
    d0, d1 = res["dates"]
    print(f"\n=== Alt-stacker backtest [{allocator_name}] {','.join(res['bases'])} ===")
    print(f"window {d0} .. {d1}  ({res['n_bars']} {res['tf']} bars, "
          f"{res['n_decisions']} decisions)  stack ${res['stack']:,.0f}  "
          f"fee {res['fee']*100:.2f}%/side")
    bot, hodl, dca, csh = (res["bot_value"], res["hodl_value"],
                           res["dca_value"], res["cash_value"])
    def pct(x): return f"{(x/res['stack']-1)*100:+.1f}%"
    print(f"\n  PORTFOLIO ENDING VALUE (USD):")
    print(f"    BOT        ${bot:,.2f}  ({pct(bot)})   cash left ${res['bot_cash']:,.2f}")
    print(f"    DCA  (eq)  ${dca:,.2f}  ({pct(dca)})")
    print(f"    HODL (eq)  ${hodl:,.2f}  ({pct(hodl)})")
    print(f"    all-USD    ${csh:,.2f}  (+0.0%)")
    print(f"    => bot vs DCA {bot-dca:+,.2f} ({(bot/dca-1)*100:+.1f}%)  "
          f"vs HODL {bot-hodl:+,.2f} ({(bot/hodl-1)*100:+.1f}%)")
    print(f"\n  PER-ASSET (units accumulated; the operator's real objective):")
    for b, a in res["per_asset"].items():
        if a["objective"] == "accumulate_quote":
            print(f"    {b:5} [{a['objective']}] px {a['px_change_pct']:+.1f}%  "
                  f"realized ${a['bot_realized_usd']:+,.2f} USD  (range-harvest)")
        else:
            tag = "✅" if a["bot_units"] >= max(a["dca_units"], a["hodl_units"]) else "⚠️"
            print(f"    {b:5} [{a['objective']}] px {a['px_change_pct']:+.1f}%  "
                  f"BOT {a['bot_units']:.4f} u  DCA {a['dca_units']:.4f}  "
                  f"HODL {a['hodl_units']:.4f}  {tag}")


def write_gate(res):
    """Make-USD gate. The alt-stacker's job (SOL/ETH/HYPE) is to GROW USD, not accumulate
    crypto, so the bar is dollars:
      MAKE-USD assets (`accumulate_quote`): the bot must END with MORE USD than (a) holding
        cash all along AND (b) HODLing the assets. (a) proves the trading actually nets USD;
        (b) proves active beat passive. Spot is long-only, so in a pure downtrend the honest
        bar 'beat holding cash' is HARD — a fail means: don't deploy / hold USD instead.
      `accumulate_base` assets (if any): still gated on UNITS vs DCA & HODL (satoshi-style).
      `store_of_value` (gold): reported on ounces, NOT gated (holding is the strategy).
    Run the gate on make-USD-only assets so gold's value doesn't muddy the dollar metric."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    pa = res["per_asset"]
    musd = [b for b in pa if pa[b]["objective"] == "accumulate_quote"]
    base = [b for b in pa if pa[b]["objective"] == "accumulate_base"]
    sov = [b for b in pa if pa[b]["objective"] == "store_of_value"]
    if not musd and not base:
        print("\n  GATE: (no make-USD / accumulate_base assets in this run — nothing to gate)")
        return False
    checks, passed = [], True
    if musd:
        if sov:
            checks.append("⚠ run includes a store_of_value asset; its value muddies the USD "
                          "metric — run make-USD assets alone for a clean gate")
        # MAKE-USD bar: end with more USD than holding cash (you made dollars) AND beat naive
        # DCA (active timing added value). NOT required to beat HODL — HODL ends holding the
        # crypto, which is the opposite of the goal (end in USD).
        v_ok = res["bot_value"] > res["cash_value"] and res["bot_value"] >= res["dca_value"]
        checks.append(f"make-USD value ${res['bot_value']:,.2f} > hold-cash "
                      f"${res['cash_value']:,.2f} AND >= DCA ${res['dca_value']:,.2f} "
                      f"(HODL ${res['hodl_value']:,.2f}, FYI) -> {v_ok}")
        passed &= v_ok
    if base:
        u_ok = all(pa[b]["bot_units"] >= pa[b]["dca_units"]
                   and pa[b]["bot_units"] >= pa[b]["hodl_units"] for b in base)
        checks.append(f"accumulate_base units beat DCA & HODL [{','.join(base)}] -> {u_ok}")
        passed &= u_ok
    if not passed:
        if GATE_MARKER.exists():
            GATE_MARKER.unlink()
        print("\n  GATE: ✗ NOT PASSED")
        for c in checks:
            print(f"      - {c}")
        return False
    d0, d1 = res["dates"]
    GATE_MARKER.write_text(
        f"ALT gate PASSED (deterministic reference allocator)\n"
        f"window {d0}..{d1} assets {','.join(res['bases'])} (Kraken daily, fee {FEE*100:.2f}%)\n"
        + "\n".join(f"- {c}" for c in checks) + "\n")
    print("\n  GATE: ✅ PASSED")
    for c in checks:
        print(f"      - {c}")
    print(f"      wrote {GATE_MARKER}")
    return True


def main(argv=None) -> int:
    p = argparse.ArgumentParser("alt-backtest")
    p.add_argument("--assets", default="XMR")   # make USD with Monero
    p.add_argument("--allocator", choices=["momentum", "llm"], default="momentum")
    p.add_argument("--strategy", choices=["trend", "meanrev"], default="trend",
                   help="deterministic rule: trend-following (study winner) or mean-reversion")
    p.add_argument("--timeframe",
                   choices=["5m", "15m", "30m", "1h", "4h", "8h", "12h", "1d"], default="1d")
    p.add_argument("--fee", type=float, default=FEE, help="per-side fee (0.0026 taker / 0.0016 maker)")
    p.add_argument("--rebal-band", type=float, default=None,
                   help="trade only when target moves > this frac of pot (smaller = more scalping)")
    p.add_argument("--stack", type=float, default=1000.0)
    p.add_argument("--cadence", type=int, default=1, help="decide every N bars (1d or 4h bars)")
    p.add_argument("--window-days", type=int, default=0, help="last N days only (0=all)")
    p.add_argument("--skip-recent-days", type=int, default=0,
                   help="trim the most recent N days (for disjoint out-of-sample windows)")
    p.add_argument("--model", default=None)
    p.add_argument("--write-gate", action="store_true")
    a = p.parse_args(argv)

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:  # noqa: BLE001
        pass

    bases = [b.strip().upper() for b in a.assets.split(",") if b.strip()]
    specs = {b: DEFAULT_SPECS.get(b, AssetSpec(b, "accumulate_base", 0.4)) for b in bases}
    _det = momentum_trend_target if a.strategy == "trend" else momentum_target
    allocator = (make_llm_allocator(a.model, specs) if a.allocator == "llm"
                 else (lambda b, s, f, px, pos, pot, cash, cur, date:
                       _det(s.objective, f, s.max_fraction)))

    dates, _, _ = load_series(bases, a.timeframe)
    bpd = BARS_PER_DAY[a.timeframe]
    skip = a.skip_recent_days * bpd
    effective_len = len(dates) - skip
    start_idx = max(0, effective_len - a.window_days * bpd) if a.window_days else 0
    if a.allocator == "llm":
        n = (effective_len - start_idx) // max(1, a.cadence)
        print(f"[llm] ~{n * len(bases)} LLM calls (assets×decisions, {a.timeframe} bars). "
              f"Ctrl-C to abort.")

    res = run_backtest(bases, specs, allocator, stack=a.stack, cadence=a.cadence,
                       start_idx=start_idx, tf=a.timeframe, fee=a.fee,
                       rebal_band=a.rebal_band, end_trim=skip)
    _print(res, a.allocator)
    if a.write_gate:
        if a.allocator != "momentum":
            print("\n  (refusing to write the gate from a non-deterministic LLM run — "
                  "the certified gate uses --allocator momentum)")
        else:
            write_gate(res)
    print("\n  CAVEATS: single historical path; equal-weight DCA/HODL baselines; Kraken "
          "taker fee; HYPE history is short (Kraken since 2026-01). An --allocator llm run "
          "is illustrative only — the LLM can recognize an asset's price history (hindsight) "
          "and news/Fear&Greed are disabled to avoid leakage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
