"""Replay the LIVE agent loop over a historical price path (integration demo).

Unlike engine.py/variants.py (which test the pure strategy math), this drives the
*actual production code* — Agent.run_cycle, OrderManager, Store (SQLite), RiskGate
— bar by bar against PaperExchange, proving the live agent re-anchors, fills,
fires the deadline, and never strands USDC on a real drawdown.

    python3 backtest/replay.py 2022_bear_full
    python3 backtest/replay.py 2025_2026_drawdown

Note: PaperExchange fills a resting bid at the touched price, so absolute BTC here
is not meant to reproduce engine.py's conservative min(L,open) fills — the rigorous
accumulation comparison lives in variants.py. This is a mechanics/behaviour demo.
"""
from __future__ import annotations

import sys
import tempfile
from datetime import timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.analysis import MockAdvisor  # noqa: E402
from agent.config import (  # noqa: E402
    AgentConfig, AnalysisConfig, DeployScheduleConfig, LadderConfig, RiskConfig,
)
from agent.exchange import PaperExchange  # noqa: E402
from agent.loop import Agent  # noqa: E402
from agent.notify import Notifier  # noqa: E402
from agent.orders import OrderManager  # noqa: E402
from agent.risk import RiskGate  # noqa: E402
from agent.state import Store  # noqa: E402
from backtest.engine import WINDOWS, slice_window  # noqa: E402

DATA = ROOT / "backtest" / "data" / "BTCUSDT_1d.parquet"


class _Px:
    def __init__(self, p): self.p = p
    def __call__(self): return self.p


def replay(window: str) -> None:
    win = next((w for w in WINDOWS if w[0] == window), None)
    if not win:
        print(f"unknown window {window!r}; choose from "
              f"{[w[0] for w in WINDOWS]}")
        return
    _, start, end, kind = win
    bars = slice_window(pd.read_parquet(DATA), start, end).reset_index(drop=True)
    anchor0 = float(bars.iloc[0]["open"])
    t0 = bars.iloc[0]["dt"].to_pydatetime().astimezone(timezone.utc)
    tend = bars.iloc[-1]["dt"].to_pydatetime().astimezone(timezone.utc)
    deploy_by = bars.iloc[int(0.70 * (len(bars) - 1))]["dt"].to_pydatetime().astimezone(timezone.utc)

    tmp = Path(tempfile.mkdtemp())
    cfg = AgentConfig(
        mode="dry_run",
        ladder=LadderConfig(budget_usdc=2_000.0, floor_price=35_000.0, n_tranches=8,
                            strategy="reanchor", gen_fraction=0.5, floor_frac=0.55),
        risk=RiskConfig(max_deploy_per_day_usdc=1e9, min_notional_usdc=10.0),
        schedule=DeployScheduleConfig(start=t0, end=tend, deploy_by=deploy_by,
                                      dca_every_hours=24),
        analysis=AnalysisConfig(enabled=True),
        db_path=str(tmp / "replay.db"))
    pricer = _Px(anchor0)
    ex = PaperExchange("BTC/USDC", budget_usdc=2_000.0, maker_fee=0.001,
                       taker_fee=0.001, store_path=str(tmp / "book.json"),
                       price_source=pricer)
    store = Store(cfg.db_path)
    rg = RiskGate(cfg.risk, store)
    quiet = Notifier()
    quiet.send = lambda *_a, **_k: None  # type: ignore[assignment]  # silence replay trace
    agent = Agent(cfg=cfg, store=store, ex=ex, risk=rg, advisor=MockAdvisor(),
                  om=OrderManager(cfg, ex, store, rg), notifier=quiet)

    print(f"\n=== REPLAY {window} [{kind}]  {start}..{end}  "
          f"anchor=${anchor0:,.0f} low=${bars['low'].min():,.0f} "
          f"final=${bars.iloc[-1]['close']:,.0f}  deploy_by={deploy_by.date()} ===")
    last_gen, deadline_announced = 0, False
    for i in range(len(bars)):
        bar = bars.iloc[i]
        now = bar["dt"].to_pydatetime().astimezone(timezone.utc)
        pricer.p = float(bar["low"])      # fills: resting bids the day's low touched
        agent.run_cycle(now=now)
        gen = int(store.get_meta("generation", 0))
        if gen != last_gen:
            print(f"  {now.date()}  gen {gen} opened @ anchor=${pricer.p:,.0f} "
                  f"floor=${float(store.get_meta('gen_floor')):,.0f}  "
                  f"deployed=${store.total_deployed():,.0f}")
            last_gen = gen
        if store.get_meta("mode_phase") == "deadline" and not deadline_announced:
            print(f"  {now.date()}  DEADLINE fired  "
                  f"deployed-so-far=${store.total_deployed():,.0f}")
            deadline_announced = True

    dep, btc = store.total_deployed(), store.total_btc()
    final = float(bars.iloc[-1]["close"])
    print(f"  RESULT: BTC={btc:.6f}  avg_cost="
          f"${(dep/btc) if btc else 0:,.0f}  deployed=${dep:,.0f} "
          f"({100*dep/2000:.1f}%)  generations={last_gen}  "
          f"portfolio=${btc*final + (2000-dep):,.0f}")
    assert dep > 2000 * 0.98, "INVARIANT VIOLATED: stranded USDC!"
    print("  invariant OK: budget fully deployed (no stranded USDC)")


if __name__ == "__main__":
    targets = sys.argv[1:] or ["2022_bear_full", "covid_crash", "2025_2026_drawdown"]
    for w in targets:
        replay(w)
