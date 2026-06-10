"""Accumulation backtest: deterministic adaptive ladder vs the honest baselines.

Scoreboard per (window, floor): total BTC accumulated, average cost basis,
% of budget deployed, and terminal portfolio value — for four strategies:

    1. lump_sum   — deploy the whole budget at the window open (taker).
    2. dca        — equal-interval market buys across the window (taker).
    3. static_ladder — resting weighted maker limit ladder, NO deadline;
                       leftover stays in USDC (the "no software" baseline).
    4. adaptive   — the agent's spine: same weighted ladder + per-day deploy
                    cap + deploy-by-deadline DCA of any leftover (maker fills,
                    taker for the forced deadline tranches).

Honesty guarantees:
  * Fill model uses ONLY the current bar's open & low (no look-ahead).
  * A resting limit buy at level L fills on bar b iff low_b <= L, at price
    min(L, open_b) — i.e. you get the limit or a better gap-down open.
  * The floor for each window is set by a forward rule (anchor * frac), never
    by hindsight knowledge of the actual bottom.
  * Both drawdown AND rising windows are included so the agent isn't only
    judged where laddering trivially wins.

The LLM advisor is deliberately ABSENT here: per the spec it may only make
buying *more cautious* (veto/shrink, never buy more/faster), so on a pure
"most BTC accumulated" score it can never beat the deterministic spine — it
can only match it or trade stack for tail-risk reduction. Milestone 1 therefore
tests the spine, which is the only thing that *can* beat the baselines.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.config import LadderConfig  # noqa: E402
from agent.ladder import build_ladder, deadline_dca_tranches  # noqa: E402

DATA = Path(__file__).resolve().parent / "data" / "BTCUSDT_1d.parquet"

MAKER_FEE = 0.001
TAKER_FEE = 0.001
BUDGET = 2_000.0
N_TRANCHES = 8
WEIGHTING = "geometric"
GEOM_RATIO = 1.5
DEADLINE_FRAC = 0.70        # deploy_by at 70% through the window
MAX_DEPLOY_PER_DAY = np.inf  # headline run: cap disabled (pure accumulation)


# ---------------------------------------------------------------------------
# Windows: real historical regimes. (label, start, end, kind)
# "kind" is only used to summarize behaviour, never fed to any strategy.
# ---------------------------------------------------------------------------
WINDOWS: list[tuple[str, str, str, str]] = [
    ("2018_bear",            "2018-01-15", "2018-12-15", "drawdown"),
    ("covid_crash",          "2020-02-01", "2020-08-01", "v_crash"),
    ("2021_midcycle",        "2021-04-12", "2021-10-01", "crash_recover"),
    ("2022_bear_full",       "2021-11-10", "2022-12-31", "drawdown"),
    ("2022_luna",            "2022-04-01", "2022-08-01", "drawdown"),
    ("2022_ftx",             "2022-09-01", "2023-01-15", "drawdown"),
    ("2024_summer",          "2024-03-13", "2024-09-15", "crash_recover"),
    ("2025_2026_drawdown",   "2025-10-01", "2026-06-10", "drawdown"),
    # adversarial / honesty windows where laddering should NOT trivially win:
    ("bull_2020H2",          "2020-08-01", "2021-01-15", "rising"),
    ("bull_2023H1",          "2023-01-01", "2023-07-01", "rising"),
]

FLOOR_FRACS = [0.55, 0.65]  # floor = anchor_open * frac (forward rule)


@dataclass
class Result:
    strategy: str
    btc: float = 0.0
    deployed: float = 0.0
    n_fills: int = 0

    @property
    def leftover(self) -> float:
        return BUDGET - self.deployed

    @property
    def avg_cost(self) -> float:
        return self.deployed / self.btc if self.btc > 0 else float("nan")

    @property
    def pct_deployed(self) -> float:
        return 100.0 * self.deployed / BUDGET


@dataclass
class Buy:
    btc: float
    usdc: float


def _buy(usdc: float, price: float, fee: float) -> Buy:
    """Convert USDC -> BTC at `price`, fee charged in received BTC."""
    if usdc <= 0 or price <= 0:
        return Buy(0.0, 0.0)
    btc = (usdc / price) * (1.0 - fee)
    return Buy(btc, usdc)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------
def run_lump_sum(bars: pd.DataFrame) -> Result:
    r = Result("lump_sum")
    px = float(bars.iloc[0]["open"])
    b = _buy(BUDGET, px, TAKER_FEE)
    r.btc, r.deployed, r.n_fills = b.btc, b.usdc, 1
    return r


def run_dca(bars: pd.DataFrame, n_buys: int = N_TRANCHES) -> Result:
    r = Result("dca")
    n = min(n_buys, len(bars))
    idx = np.linspace(0, len(bars) - 1, n).round().astype(int)
    each = BUDGET / n
    for i in idx:
        px = float(bars.iloc[int(i)]["open"])
        b = _buy(each, px, TAKER_FEE)
        r.btc += b.btc
        r.deployed += b.usdc
        r.n_fills += 1
    return r


def _run_ladder(bars: pd.DataFrame, floor: float, *, deadline: bool,
                cap: float = MAX_DEPLOY_PER_DAY) -> Result:
    """Shared engine for static_ladder (deadline=False) and adaptive (True)."""
    anchor = float(bars.iloc[0]["open"])
    cfg = LadderConfig(budget_usdc=BUDGET, floor_price=floor, n_tranches=N_TRANCHES,
                       weighting=WEIGHTING, geometric_ratio=GEOM_RATIO)
    rungs = build_ladder(anchor, cfg)
    # mutable rung state
    open_rungs = [{"price": rg.price, "usdc": rg.usdc, "filled": False} for rg in rungs]

    r = Result("adaptive" if deadline else "static_ladder")
    n = len(bars)
    deadline_i = int(round(DEADLINE_FRAC * (n - 1))) if deadline else None
    fired_deadline = False

    for i in range(n):
        bar = bars.iloc[i]
        low = float(bar["low"])
        op = float(bar["open"])

        # --- deploy-by-deadline: cancel remaining rungs, DCA the leftover ---
        if deadline and not fired_deadline and i >= deadline_i:
            fired_deadline = True
            leftover = sum(rg["usdc"] for rg in open_rungs if not rg["filled"])
            remaining_bars = n - i
            tranches = deadline_dca_tranches(leftover, remaining_bars)
            for rg in open_rungs:
                rg["filled"] = True  # cancelled / accounted for via forced DCA
            # schedule forced DCA across remaining bars (taker, at each open)
            r._dca_plan = tranches  # type: ignore[attr-defined]
            r._dca_start = i        # type: ignore[attr-defined]

        # --- forced deadline DCA tranche for this bar ---
        if deadline and fired_deadline:
            plan = getattr(r, "_dca_plan", [])
            start = getattr(r, "_dca_start", i)
            k = i - start
            if 0 <= k < len(plan) and plan[k] > 0:
                b = _buy(plan[k], op, TAKER_FEE)
                r.btc += b.btc
                r.deployed += b.usdc
                r.n_fills += 1
            continue  # once in deadline mode, no more resting-rung fills

        # --- normal resting-ladder fills (maker), highest price first ---
        deployed_today = 0.0
        for rg in sorted(open_rungs, key=lambda x: -x["price"]):
            if rg["filled"]:
                continue
            if low <= rg["price"]:
                if deployed_today + rg["usdc"] > cap:
                    continue  # per-day cap: leave resting for a later bar
                fill_price = min(rg["price"], op)  # limit, or better gap-down open
                b = _buy(rg["usdc"], fill_price, MAKER_FEE)
                r.btc += b.btc
                r.deployed += b.usdc
                r.n_fills += 1
                rg["filled"] = True
                deployed_today += rg["usdc"]
    return r


def run_static_ladder(bars: pd.DataFrame, floor: float) -> Result:
    return _run_ladder(bars, floor, deadline=False)


def run_adaptive(bars: pd.DataFrame, floor: float, cap: float = MAX_DEPLOY_PER_DAY) -> Result:
    return _run_ladder(bars, floor, deadline=True, cap=cap)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def slice_window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end, tz="UTC")
    w = df[(df["dt"] >= s) & (df["dt"] <= e)].reset_index(drop=True)
    return w


def run_all() -> pd.DataFrame:
    df = pd.read_parquet(DATA)
    rows = []
    for label, start, end, kind in WINDOWS:
        bars = slice_window(df, start, end)
        if len(bars) < 10:
            print(f"!! window {label} has too few bars ({len(bars)}), skipping")
            continue
        anchor = float(bars.iloc[0]["open"])
        final = float(bars.iloc[-1]["close"])
        wlow = float(bars["low"].min())
        for frac in FLOOR_FRACS:
            floor = anchor * frac
            results = {
                "lump_sum": run_lump_sum(bars),
                "dca": run_dca(bars),
                "static_ladder": run_static_ladder(bars, floor),
                "adaptive": run_adaptive(bars, floor),
            }
            for name, res in results.items():
                rows.append({
                    "window": label, "kind": kind, "floor_frac": frac,
                    "anchor": anchor, "final": final, "window_low": wlow,
                    "floor": floor, "floor_hit": wlow <= floor,
                    "strategy": name, "btc": res.btc, "avg_cost": res.avg_cost,
                    "deployed": res.deployed, "pct_deployed": res.pct_deployed,
                    "n_fills": res.n_fills,
                    "terminal_usd": res.btc * final + res.leftover,
                })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.float_format", lambda x: f"{x:,.6f}" if abs(x) < 1 else f"{x:,.2f}")

    print("\n" + "=" * 100)
    print("PER-WINDOW SCOREBOARD  (BTC accumulated from $2,000; higher = better)")
    print("=" * 100)
    for (win, frac), g in df.groupby(["window", "floor_frac"], sort=False):
        meta = g.iloc[0]
        hit = "FLOOR HIT" if meta["floor_hit"] else "floor NOT reached"
        print(f"\n{win}  | floor={meta['floor']:,.0f} ({frac:.0%} of "
              f"${meta['anchor']:,.0f})  | low=${meta['window_low']:,.0f} "
              f"final=${meta['final']:,.0f}  [{meta['kind']}, {hit}]")
        gg = g.set_index("strategy")[["btc", "avg_cost", "pct_deployed", "terminal_usd"]]
        gg = gg.reindex(["lump_sum", "dca", "static_ladder", "adaptive"])
        best = gg["btc"].idxmax()
        for strat, row in gg.iterrows():
            mark = " <-- most BTC" if strat == best else ""
            ac = "   n/a   " if not np.isfinite(row["avg_cost"]) else f"${row['avg_cost']:>9,.0f}"
            print(f"   {strat:<14} btc={row['btc']:.6f}  avg_cost={ac}  "
                  f"deployed={row['pct_deployed']:5.1f}%  term=${row['terminal_usd']:,.0f}{mark}")

    # ---- head-to-head verdict ----
    print("\n" + "=" * 100)
    print("VERDICT: adaptive spine vs each baseline (per window+floor cell)")
    print("=" * 100)
    piv = df.pivot_table(index=["window", "floor_frac"], columns="strategy", values="btc")
    cells = len(piv)
    for base in ["static_ladder", "dca", "lump_sum"]:
        wins = int((piv["adaptive"] > piv[base] * 1.0000001).sum())
        ties = int((np.abs(piv["adaptive"] - piv[base]) <= piv[base] * 1e-6).sum())
        losses = cells - wins - ties
        ratio = (piv["adaptive"] / piv[base]).replace([np.inf, -np.inf], np.nan)
        print(f"  adaptive vs {base:<14}: win {wins:2d} / tie {ties:2d} / lose {losses:2d}"
              f"  | median BTC ratio={ratio.median():.4f}  mean={ratio.mean():.4f}")

    # drawdown-only verdict (the regime the agent is actually built for)
    dd = df[df["kind"].isin(["drawdown", "v_crash", "crash_recover"])]
    pivd = dd.pivot_table(index=["window", "floor_frac"], columns="strategy", values="btc")
    print("\n  --- drawdown/crash windows only (the agent's design regime) ---")
    for base in ["static_ladder", "dca", "lump_sum"]:
        ratio = (pivd["adaptive"] / pivd[base]).replace([np.inf, -np.inf], np.nan)
        wins = int((pivd["adaptive"] > pivd[base] * 1.0000001).sum())
        print(f"  adaptive vs {base:<14}: win {wins}/{len(pivd)}  median ratio={ratio.median():.4f}")

    print("\n" + "=" * 100)
    print("RISING windows (honesty check — laddering is EXPECTED to lag lump_sum here)")
    print("=" * 100)
    rise = df[df["kind"] == "rising"]
    pr = rise.pivot_table(index=["window", "floor_frac"], columns="strategy", values="btc")
    for base in ["lump_sum", "dca"]:
        ratio = (pr["adaptive"] / pr[base]).replace([np.inf, -np.inf], np.nan)
        print(f"  adaptive vs {base:<14}: median BTC ratio={ratio.median():.4f} "
              f"(<1.0 expected — the deadline guard limits the damage)")


def main() -> None:
    df = run_all()
    out = Path(__file__).resolve().parent / "results" / "scoreboard.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    summarize(df)
    print(f"\nsaved raw results -> {out}")


if __name__ == "__main__":
    main()
