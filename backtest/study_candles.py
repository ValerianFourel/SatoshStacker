"""Candle + technicals study for the make-USD alts (TRX, BNB).

Answers two questions the operator asked:
  1) WHICH CANDLE TIMEFRAME best extracts USD — runs the make-USD deterministic backtest per
     timeframe over a common recent window and ranks by USD made (vs holding cash / DCA / HODL).
  2) WHICH TECHNICALS / MOMENTUM METRICS make sense — ranks indicators by Information Coefficient
     (corr of the indicator with forward returns; |IC| = predictive power) per timeframe, reusing
     agent.tune (the same IC engine the BTC trader self-tunes with).

    python3 backtest/study_fetch.py            # one-time: deep multi-tf history
    python3 backtest/study_candles.py          # the study (TRX, BNB)

Deterministic (no LLM). The make-USD edge is regime-dependent (chop favors the swing bot; strong
trends favor hold/DCA), so read these as "what worked over this window," not a guarantee.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import alt_engine as E  # noqa: E402
from agent import tune  # noqa: E402
from agent.alts import AssetSpec  # noqa: E402

ASSETS = ["TRX", "BNB"]
TFS = ["1d", "12h", "8h", "4h", "1h"]
COMMON_DAYS = 365      # compare every timeframe over the SAME last ~365 days (apples-to-apples)
STACK = 1000.0


def _alloc_meanrev(b, s, f, px, pos, pot, cash, cur, date):
    return E.momentum_target(s.objective, f, s.max_fraction)        # buy oversold (swing)


def _alloc_trend(b, s, f, px, pos, pot, cash, cur, date):
    return E.momentum_trend_target(s.objective, f, s.max_fraction)  # ride strength (trend)


def study_asset(asset: str) -> None:
    spec = {asset: AssetSpec(asset, "accumulate_quote", 0.6)}
    print(f"\n{'='*82}\n{asset} — make USD (last ~{COMMON_DAYS}d, common window per timeframe)\n{'='*82}")
    print(f"{'tf':>4} {'bars':>6} {'meanrev%':>9} {'trend%':>8} {'dca%':>7} {'hodl%':>7}  "
          f"top technicals by |IC| (indicator lag IC)")
    rows = []
    for tf in TFS:
        try:
            dates, closes, highs = E.load_series([asset], tf)
        except FileNotFoundError:
            print(f"{tf:>4}  (no data — run study_fetch.py)")
            continue
        bpd = E.BARS_PER_DAY[tf]
        start = max(0, len(dates) - COMMON_DAYS * bpd)
        mr = E.run_backtest([asset], spec, _alloc_meanrev, stack=STACK, cadence=1,
                            start_idx=start, tf=tf)["bot_value"] / STACK - 1
        tr = E.run_backtest([asset], spec, _alloc_trend, stack=STACK, cadence=1,
                            start_idx=start, tf=tf)
        trend = tr["bot_value"] / STACK - 1
        dca = tr["dca_value"] / STACK - 1
        hodl = tr["hodl_value"] / STACK - 1
        ranked = tune.rank(closes[asset][start:])[:3]
        tech = ", ".join(f"{r['indicator']}({r['lag']},{r['ic']:+.2f})" for r in ranked)
        print(f"{tf:>4} {len(dates)-start:>6} {mr*100:>+8.1f}% {trend*100:>+7.1f}% "
              f"{dca*100:>+6.1f}% {hodl*100:>+6.1f}%  {tech}")
        rows.append((tf, mr, trend, dca, hodl, ranked))
    if rows:
        # best (strategy, tf) that makes USD = beats holding cash (return > 0) and beats DCA
        best_tf_trend = max(rows, key=lambda r: r[2])
        winners = [(r[0], "trend", r[2]) for r in rows if r[2] > 0 and r[2] >= r[3]] + \
                  [(r[0], "meanrev", r[1]) for r in rows if r[1] > 0 and r[1] >= r[3]]
        print(f"  → best for {asset}: trend@{best_tf_trend[0]} ({best_tf_trend[2]*100:+.1f}% USD)")
        print("  → makes-USD (beats cash+DCA): " +
              (", ".join(f"{tf}/{st} {v*100:+.1f}%" for tf, st, v in winners) if winners
               else "NONE over this window"))
        from collections import Counter
        c = Counter(r["indicator"] for row in rows for r in row[5])
        print("  → technicals recurring in top-3 across timeframes: " +
              ", ".join(f"{n}×{k}" for n, k in c.most_common(4)))


def main() -> int:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:  # noqa: BLE001
        pass
    for a in ASSETS:
        study_asset(a)
    print("\nCAVEAT: single historical path; deterministic swing rule; Binance public prices "
          "(live = Kraken). IC = linear predictive power, not a trading guarantee.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
