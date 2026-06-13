"""Make-USD sweep across a basket — the ROBUST gate (not the cherry-picked recent window).

For each asset: daily trend-following (the study's best candle+rule), at TAKER fee (the honest
assumption), over multiple lookback windows. An asset "makes USD" in a window only if it ends
with MORE USD than holding cash AND beats naive DCA (DCA is the dominant simple baseline). We
rank by how many windows it clears — a real edge should hold across most, not just the latest.

    python3 backtest/study_fetch.py --tfs 1d TAO SUI FET DOGE RENDER TON   # one-time
    python3 backtest/study_sweep.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import alt_engine as E  # noqa: E402
from agent.alts import AssetSpec  # noqa: E402

ASSETS = ["TAO", "SUI", "FET", "DOGE", "RENDER", "TON"]
WINDOWS = [180, 365, 730, 0]          # 0 = full history
WLABEL = {180: "180d", 365: "365d", 730: "730d", 0: "full"}
TAKER = 0.0026
STACK = 1000.0


def _trend(b, s, f, px, p, pot, c, cur, d):
    return E.momentum_trend_target(s.objective, f, s.max_fraction)


def sweep_asset(asset: str):
    spec = {asset: AssetSpec(asset, "accumulate_quote", 0.6)}
    try:
        dates, _, _ = E.load_series([asset], "1d")
    except FileNotFoundError:
        print(f"{asset:7}: no 1d data"); return None
    n = len(dates)
    cells, passes, valid = [], 0, 0
    for w in WINDOWS:
        start = max(0, n - w) if w else 0
        if w and w > n:                       # window longer than history: skip (dup of full)
            cells.append((WLABEL[w], None)); continue
        res = E.run_backtest([asset], spec, _trend, stack=STACK, cadence=1,
                             start_idx=start, tf="1d", fee=TAKER)
        bot = res["bot_value"] / STACK - 1
        dca = res["dca_value"] / STACK - 1
        ok = res["bot_value"] > STACK and res["bot_value"] >= res["dca_value"]  # cash AND DCA
        cells.append((WLABEL[w], (bot, dca, ok)))
        valid += 1
        passes += 1 if ok else 0
    return {"asset": asset, "n": n, "cells": cells, "passes": passes, "valid": valid}


def main() -> int:
    print(f"{'='*94}\nMAKE-USD SWEEP — daily trend-following, taker {TAKER*100:.2f}%/side, "
          f"gate = beat CASH and DCA\n{'='*94}")
    print(f"{'asset':>7} {'days':>5} | " +
          " | ".join(f"{WLABEL[w]:>14}" for w in WINDOWS) + " | gate")
    print(f"{'':>7} {'':>5} | " + " | ".join(f"{'bot% (vsDCA)':>14}" for _ in WINDOWS) + " |")
    rows = []
    for a in ASSETS:
        r = sweep_asset(a)
        if not r:
            continue
        rows.append(r)
        line = f"{a:>7} {r['n']:>5} | "
        parts = []
        for _lbl, cell in r["cells"]:
            if cell is None:
                parts.append(f"{'—':>14}")
            else:
                bot, dca, ok = cell
                parts.append(f"{bot*100:>+6.0f}%({bot*100-dca*100:>+4.0f}){'✓' if ok else '✗'}")
        line += " | ".join(parts) + f" | {r['passes']}/{r['valid']}"
        print(line)
    rows.sort(key=lambda r: (-r["passes"] / max(1, r["valid"]), -r["passes"]))
    print(f"\nRANKED by robustness (windows passing beat-cash+DCA / valid windows):")
    for r in rows:
        verdict = ("ROBUST edge" if r["passes"] == r["valid"] and r["valid"] >= 3
                   else "mixed/recent-only" if r["passes"] else "NO edge")
        print(f"  {r['asset']:>7}: {r['passes']}/{r['valid']}  → {verdict}")
    any_robust = any(r["passes"] == r["valid"] and r["valid"] >= 3 for r in rows)
    print("\n" + ("→ At least one asset shows a cross-window edge — verify it adversarially "
                  "before trusting." if any_robust else
                  "→ NONE shows a robust cross-window edge. Like ZEC, daily-trend beats cash only "
                  "via upward drift (long-only) and does not reliably beat DCA. Honest answer: DCA "
                  "(or hold cash) dominates; don't deploy the make-USD bot expecting an edge."))
    print("\nCAVEAT: single path/asset; taker fee; no spread/slippage (so OPTIMISTIC); daily-trend "
          "+ trend rule (intraday/scalp already shown to lose). Beating cash alone = upward drift, "
          "not skill — the real bar is beating DCA across windows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
