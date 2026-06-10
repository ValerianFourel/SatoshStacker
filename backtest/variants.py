"""Design search: can an HONEST deterministic ladder beat naive DCA?

Milestone-1 found the top-anchored ladder loses to DCA in deep bears because
its floor sits above the eventual bottom, so it fully deploys too high and
misses sub-floor coins. This module tests deterministic fixes that keep dry
powder for *below* the floor, against a STRONGER weekly-DCA baseline (the
honest baseline should be hard, not sandbagged).

All variants reuse the exact fill model & fee model from engine.py (no
look-ahead: only current-bar open/low/close are used).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.config import LadderConfig  # noqa: E402
from agent.ladder import build_ladder  # noqa: E402
from backtest.engine import (  # noqa: E402
    BUDGET, DEADLINE_FRAC, GEOM_RATIO, MAKER_FEE, N_TRANCHES, TAKER_FEE,
    WEIGHTING, WINDOWS, Result, _buy, slice_window,
)

DATA = Path(__file__).resolve().parent / "data" / "BTCUSDT_1d.parquet"


# ---------------- baselines ----------------
def run_lump(bars):
    r = Result("lump_sum")
    b = _buy(BUDGET, float(bars.iloc[0]["open"]), TAKER_FEE)
    r.btc, r.deployed, r.n_fills = b.btc, b.usdc, 1
    return r


def run_dca_weekly(bars, every: int = 7):
    """Stronger, realistic DCA: a market buy every `every` bars (~weekly)."""
    r = Result("dca_weekly")
    idx = list(range(0, len(bars), every))
    each = BUDGET / len(idx)
    for i in idx:
        b = _buy(each, float(bars.iloc[i]["open"]), TAKER_FEE)
        r.btc += b.btc
        r.deployed += b.usdc
        r.n_fills += 1
    return r


def _deadline_index(n: int) -> int:
    return int(round(DEADLINE_FRAC * (n - 1)))


def _force_dca(r: Result, bars, start_i: int, amount: float):
    """DCA `amount` equally over bars [start_i, end] at each open (taker)."""
    n = len(bars)
    rem = n - start_i
    if rem <= 0 or amount <= 0:
        return
    each = amount / rem
    for j in range(start_i, n):
        b = _buy(each, float(bars.iloc[j]["open"]), TAKER_FEE)
        r.btc += b.btc
        r.deployed += b.usdc
        r.n_fills += 1


# ---------------- variant strategies ----------------
def run_static_floor(bars, frac: float):
    """The current spine (single ladder anchor->floor) + deadline. Baseline ladder."""
    r = Result("ladder_static")
    anchor = float(bars.iloc[0]["open"])
    floor = anchor * frac
    cfg = LadderConfig(budget_usdc=BUDGET, floor_price=floor, n_tranches=N_TRANCHES,
                       weighting=WEIGHTING, geometric_ratio=GEOM_RATIO)
    rungs = [{"price": rg.price, "usdc": rg.usdc, "filled": False}
             for rg in build_ladder(anchor, cfg)]
    n = len(bars)
    di = _deadline_index(n)
    for i in range(n):
        if i >= di:
            leftover = sum(x["usdc"] for x in rungs if not x["filled"])
            for x in rungs:
                x["filled"] = True
            _force_dca(r, bars, i, leftover)
            break
        low, op = float(bars.iloc[i]["low"]), float(bars.iloc[i]["open"])
        for rg in sorted(rungs, key=lambda x: -x["price"]):
            if not rg["filled"] and low <= rg["price"]:
                b = _buy(rg["usdc"], min(rg["price"], op), MAKER_FEE)
                r.btc += b.btc
                r.deployed += b.usdc
                r.n_fills += 1
                rg["filled"] = True
    return r


def run_reanchor(bars, frac: float, gen_fraction: float = 0.5, max_gen: int = 6):
    """Trailing ladder: each 'generation' ladders only `gen_fraction` of the
    REMAINING budget from the current price down to current*frac. When price
    closes below the generation's floor, re-anchor a fresh, lower ladder with
    the still-unspent budget. Reserves powder for sub-floor capitulation.
    Deadline mops up anything left."""
    r = Result("ladder_reanchor")
    n = len(bars)
    di = _deadline_index(n)
    budget_left = BUDGET
    gens = 0

    def new_gen(anchor: float, bl: float):
        floor = anchor * frac
        cfg = LadderConfig(budget_usdc=bl * gen_fraction, floor_price=floor,
                           n_tranches=N_TRANCHES, weighting=WEIGHTING,
                           geometric_ratio=GEOM_RATIO)
        rungs = [{"price": rg.price, "usdc": rg.usdc, "filled": False}
                 for rg in build_ladder(anchor, cfg)]
        return rungs, floor

    anchor = float(bars.iloc[0]["open"])
    rungs, cur_floor = new_gen(anchor, budget_left)
    r.forced_usdc = 0.0  # capital deployed via the TAKER deadline force-DCA

    for i in range(n):
        if i >= di:
            d0 = r.deployed
            _force_dca(r, bars, i, budget_left)  # deploy ALL remaining
            r.forced_usdc += r.deployed - d0
            budget_left = 0.0
            break
        low = float(bars.iloc[i]["low"])
        op = float(bars.iloc[i]["open"])
        close = float(bars.iloc[i]["close"])
        for rg in sorted(rungs, key=lambda x: -x["price"]):
            if not rg["filled"] and low <= rg["price"]:
                b = _buy(rg["usdc"], min(rg["price"], op), MAKER_FEE)
                r.btc += b.btc
                r.deployed += b.usdc
                r.n_fills += 1
                rg["filled"] = True
                budget_left -= rg["usdc"]
        if close < cur_floor and budget_left > 1.0 and gens < max_gen:
            anchor = close
            rungs, cur_floor = new_gen(anchor, budget_left)
            gens += 1
    if budget_left > 1.0:  # safety: never strand
        d0 = r.deployed
        _force_dca(r, bars, n - 1, budget_left)
        r.forced_usdc += r.deployed - d0
    return r


def run_hybrid(bars, frac: float, ladder_share: float = 0.5):
    """Half-and-half: `ladder_share` of budget as a static anchor->floor ladder
    (+deadline), the rest as weekly DCA across the whole window. The DCA sleeve
    captures sub-floor coins; the ladder captures the descent to the floor."""
    r = Result("ladder_hybrid")
    n = len(bars)
    di = _deadline_index(n)
    ladder_budget = BUDGET * ladder_share
    dca_budget = BUDGET - ladder_budget

    # ladder sleeve
    anchor = float(bars.iloc[0]["open"])
    floor = anchor * frac
    cfg = LadderConfig(budget_usdc=ladder_budget, floor_price=floor,
                       n_tranches=N_TRANCHES, weighting=WEIGHTING,
                       geometric_ratio=GEOM_RATIO)
    rungs = [{"price": rg.price, "usdc": rg.usdc, "filled": False}
             for rg in build_ladder(anchor, cfg)]
    # weekly DCA sleeve schedule
    dca_idx = list(range(0, n, 7))
    dca_each = dca_budget / len(dca_idx)
    dca_set = set(dca_idx)
    ladder_done = False

    for i in range(n):
        op = float(bars.iloc[i]["open"])
        low = float(bars.iloc[i]["low"])
        # ladder fills (until deadline)
        if not ladder_done:
            if i >= di:
                leftover = sum(x["usdc"] for x in rungs if not x["filled"])
                for x in rungs:
                    x["filled"] = True
                _force_dca(r, bars, i, leftover)
                ladder_done = True
            else:
                for rg in sorted(rungs, key=lambda x: -x["price"]):
                    if not rg["filled"] and low <= rg["price"]:
                        b = _buy(rg["usdc"], min(rg["price"], op), MAKER_FEE)
                        r.btc += b.btc
                        r.deployed += b.usdc
                        r.n_fills += 1
                        rg["filled"] = True
        # weekly DCA sleeve
        if i in dca_set:
            b = _buy(dca_each, op, TAKER_FEE)
            r.btc += b.btc
            r.deployed += b.usdc
            r.n_fills += 1
    return r


# ---------------- runner ----------------
def run_all() -> pd.DataFrame:
    df = pd.read_parquet(DATA)
    rows = []
    for label, start, end, kind in WINDOWS:
        bars = slice_window(df, start, end)
        if len(bars) < 10:
            continue
        anchor = float(bars.iloc[0]["open"])
        final = float(bars.iloc[-1]["close"])
        wlow = float(bars["low"].min())
        for frac in (0.55, 0.65):
            floor = anchor * frac
            results = {
                "lump_sum": run_lump(bars),
                "dca_weekly": run_dca_weekly(bars),
                "ladder_static": run_static_floor(bars, frac),
                "ladder_reanchor": run_reanchor(bars, frac),
                "ladder_hybrid": run_hybrid(bars, frac),
            }
            for name, res in results.items():
                rows.append({
                    "window": label, "kind": kind, "floor_frac": frac,
                    "anchor": anchor, "final": final, "window_low": wlow,
                    "floor_hit": wlow <= floor, "strategy": name,
                    "btc": res.btc, "deployed": res.deployed,
                    "avg_cost": (res.deployed / res.btc) if res.btc else np.nan,
                    "terminal_usd": res.btc * final + (BUDGET - res.deployed),
                    "forced_usdc": getattr(res, "forced_usdc", np.nan),
                })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> None:
    print("\n" + "=" * 96)
    print("DESIGN SEARCH — BTC accumulated vs the STRONGER weekly-DCA baseline")
    print("=" * 96)
    piv = df.pivot_table(index=["window", "kind", "floor_frac"], columns="strategy", values="btc")
    order = ["lump_sum", "dca_weekly", "ladder_static", "ladder_reanchor", "ladder_hybrid"]
    piv = piv[order]
    for idx, row in piv.iterrows():
        win, kind, frac = idx
        best = row.idxmax()
        print(f"\n{win} [{kind}] floor={frac:.0%}")
        for s in order:
            r = row[s] / row["dca_weekly"]
            mark = " <-- best" if s == best else ""
            flag = "" if s in ("lump_sum", "dca_weekly") else f"  vsDCA={r:5.3f}"
            print(f"   {s:<16} btc={row[s]:.6f}{flag}{mark}")

    print("\n" + "=" * 96)
    print("VERDICT vs weekly DCA (the hard baseline)")
    print("=" * 96)
    dd = piv.reset_index()
    drawdown = dd[dd["kind"].isin(["drawdown", "v_crash", "crash_recover"])]
    allw = dd
    for name, sub in [("ALL windows", allw), ("drawdown/crash only", drawdown)]:
        print(f"\n  {name} ({len(sub)} cells):")
        for s in ["ladder_static", "ladder_reanchor", "ladder_hybrid"]:
            ratio = sub[s] / sub["dca_weekly"]
            wins = int((sub[s] > sub["dca_weekly"] * 1.0000001).sum())
            print(f"    {s:<16} beats weekly-DCA in {wins:2d}/{len(sub):2d}  "
                  f"| median ratio={ratio.median():.4f}  mean={ratio.mean():.4f}")

    # ---- audit-flagged honesty check: reanchor vs static HEAD-TO-HEAD ----
    print("\n" + "=" * 96)
    print("HONESTY CHECK — re-anchor vs static ladder, head-to-head (audit correction)")
    print("=" * 96)
    dsub = piv.reset_index()
    dsub = dsub[dsub["kind"].isin(["drawdown", "v_crash", "crash_recover"])]
    r_over_s = dsub["ladder_reanchor"] / dsub["ladder_static"]
    rea_wins = int((dsub["ladder_reanchor"] > dsub["ladder_static"] * 1.0000001).sum())
    sta_wins = int((dsub["ladder_static"] > dsub["ladder_reanchor"] * 1.0000001).sum())
    print(f"  per-cell ({len(dsub)} drawdown cells): reanchor wins {rea_wins}, "
          f"static wins {sta_wins}, median rea/static={r_over_s.median():.4f}")
    print(f"  AGGREGATE total BTC: reanchor={dsub['ladder_reanchor'].sum():.4f}  "
          f"static={dsub['ladder_static'].sum():.4f}  "
          f"(reanchor leads on total stack, mean ratio={r_over_s.mean():.4f})")
    print("  => 'beats static ladder' is true on TOTAL STACK / protracted bears, "
          "NOT per-cell (static wins sharp-V cells).")

    # ---- where does reanchor's capital actually go? maker ladder vs forced DCA ----
    print("\n  reanchor capital split (maker-ladder vs TAKER deadline-DCA), drawdown cells:")
    rr = df[(df["strategy"] == "ladder_reanchor")
            & df["kind"].isin(["drawdown", "v_crash", "crash_recover"])]
    for _, row in rr.iterrows():
        forced = row["forced_usdc"]
        maker = row["deployed"] - forced
        print(f"    {row['window']:<20} f={row['floor_frac']:.2f}  "
              f"maker={100*maker/BUDGET:5.1f}%  forced-DCA={100*forced/BUDGET:5.1f}%")


if __name__ == "__main__":
    d = run_all()
    out = Path(__file__).resolve().parent / "results" / "variants.csv"
    d.to_csv(out, index=False)
    summarize(d)
    print(f"\nsaved -> {out}")
