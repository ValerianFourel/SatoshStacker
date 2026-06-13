"""ZEC intraday-scalp study: can we MAKE USD scalping ZEC, and at which candle / fee?

The make-USD edge on spot is dominated by FEES at high frequency, so this sweeps timeframe ×
strategy × fee (taker 0.26% vs maker 0.16%) over a common recent window and asks: does any
combo end with MORE USD than holding cash (and beat DCA)? Reuses the exact production spine
(clamp_targets + plan_trades + Position) and the IC engine (agent.tune).

    python3 backtest/study_fetch.py --tfs 15m,30m,1h,4h,1d ZEC   # one-time
    python3 backtest/study_scalp.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import alt_engine as E  # noqa: E402
from agent import tune  # noqa: E402
from agent.alts import AssetSpec  # noqa: E402

ASSET = "ZEC"
TFS = ["15m", "30m", "1h", "4h", "1d"]
COMMON_DAYS = 180
STACK = 1000.0
TAKER, MAKER = 0.0026, 0.0016


def _run(spec, tf, start, strat_fn, fee):
    res = E.run_backtest([ASSET], spec, strat_fn, stack=STACK, cadence=1,
                         start_idx=start, tf=tf, fee=fee)
    return res["bot_value"] / STACK - 1, res


def main() -> int:
    spec = {ASSET: AssetSpec(ASSET, "accumulate_quote", 0.6)}
    trend = lambda b, s, f, px, p, pot, c, cur, d: E.momentum_trend_target(s.objective, f, s.max_fraction)  # noqa: E731
    mrev = lambda b, s, f, px, p, pot, c, cur, d: E.momentum_target(s.objective, f, s.max_fraction)  # noqa: E731

    print(f"{'='*92}\n{ASSET} intraday-scalp — make USD over last ~{COMMON_DAYS}d "
          f"(bot % vs holding cash; DCA/HODL FYI)\n{'='*92}")
    print(f"{'tf':>4} {'bars':>6} | {'trend@taker':>11} {'trend@maker':>11} "
          f"{'mrev@taker':>10} {'mrev@maker':>10} | {'dca%':>7} {'hodl%':>7} | top IC")
    winners = []
    for tf in TFS:
        try:
            dates, closes, highs = E.load_series([ASSET], tf)
        except FileNotFoundError:
            print(f"{tf:>4}  (no data — run study_fetch.py --tfs {tf} ZEC)"); continue
        bpd = E.BARS_PER_DAY[tf]
        start = max(0, len(dates) - COMMON_DAYS * bpd)
        tT, res = _run(spec, tf, start, trend, TAKER)
        tM, _ = _run(spec, tf, start, trend, MAKER)
        mT, _ = _run(spec, tf, start, mrev, TAKER)
        mM, _ = _run(spec, tf, start, mrev, MAKER)
        dca = res["dca_value"] / STACK - 1
        hodl = res["hodl_value"] / STACK - 1
        ic = tune.rank(closes[ASSET][start:])[:2]
        tech = ", ".join(f"{r['indicator']}({r['lag']},{r['ic']:+.2f})" for r in ic)
        print(f"{tf:>4} {len(dates)-start:>6} | {tT*100:>+10.1f}% {tM*100:>+10.1f}% "
              f"{mT*100:>+9.1f}% {mM*100:>+9.1f}% | {dca*100:>+6.1f}% {hodl*100:>+6.1f}% | {tech}")
        for name, val, fee in (("trend@taker", tT, TAKER), ("trend@maker", tM, MAKER),
                               ("mrev@taker", mT, TAKER), ("mrev@maker", mM, MAKER)):
            if val > 0 and (1 + val) * STACK >= res["dca_value"]:
                winners.append((tf, name, val))
    print()
    if winners:
        print("  ✅ MAKES USD (beats cash + DCA): " +
              ", ".join(f"{tf}/{n} {v*100:+.1f}%" for tf, n, v in winners))
    else:
        print("  ❌ NO timeframe/strategy/fee combo made USD (beat holding cash + DCA) over "
              "this window. Scalping ZEC on spot did not extract USD net of fees.")
    print(f"\n  Fee reality: taker {TAKER*100:.2f}% vs maker {MAKER*100:.2f}% per side. "
          "Each round-trip pays 2× the per-side fee — at high frequency this dominates P&L.")
    print("  CAVEAT: single path; deterministic rules; Binance public prices (live=Kraken). "
          "Real scalping also needs spread + slippage modeling this omits (so results are OPTIMISTIC).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
