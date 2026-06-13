"""Tests for the multi-asset alt-stacker: shared-pot rebalance, avg-cost, and the
deterministic CAGE around the full-authority LLM (per-asset max fraction, total
allocation, per-cycle turnover, de-peg halt, bad-tick guard, fail-safe hold).

Deterministic — no real LLM, no network (prices + candles injected)."""
import json
import types
from datetime import datetime, timezone

import numpy as np

from agent.alts import AltConfig, AltStacker, AltStore, AssetSpec, Rails, _pos
from agent.multi_exchange import MultiPaperExchange
from agent.notify import Notifier

UTC = timezone.utc
T0 = datetime(2026, 6, 1, tzinfo=UTC)


def _client(holder):
    """Fake OpenAI-compatible client. Sequential per-asset: reads the asset under
    decision from the user message and returns holder['a'][base] as target_fraction."""
    def create(**kw):
        feats = json.loads(kw["messages"][-1]["content"])
        base = feats.get("base")
        frac = holder["a"].get(base, feats.get("fraction_now", 0.0))
        c = json.dumps({"target_fraction": frac, "stance": "x", "note": "t"})
        return types.SimpleNamespace(choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content=c))])
    return types.SimpleNamespace(chat=types.SimpleNamespace(
        completions=types.SimpleNamespace(create=create)))


def _raising_client():
    def create(**kw):
        raise RuntimeError("llm down")
    return types.SimpleNamespace(chat=types.SimpleNamespace(
        completions=types.SimpleNamespace(create=create)))


def _ohlcv_box(box, skew=None):
    """Candles whose last close = current price (sanity passes) unless skewed."""
    def src(symbol):
        base = symbol.split("/")[0]
        close = (skew or {}).get(base, box[base])
        return np.array([[box[base], box[base], box[base], close]] * 60, dtype=float)
    return src


def _make(tmp_path, box, holder, *, rails=None, cash=1000.0, stack=1000.0,
          assets=None, client=None, skew=None):
    assets = assets or (
        AssetSpec("SOL", "accumulate_base", 0.45),
        AssetSpec("ETH", "accumulate_base", 0.45),
        AssetSpec("HYPE", "accumulate_quote", 0.35),
    )
    cfg = AltConfig(mode="dry_run", quote="USDC", stack_usdc=stack, taker_fee=0.001,
                    cycle_hours=4, assets=assets,
                    rails=rails or Rails(max_cycle_turnover_usdc=5000.0,
                                         max_daily_turnover_usdc=50000.0),
                    news_enabled=False, decision_pings=False, self_tune=False,
                    db_path=str(tmp_path / "a.db"))
    ex = MultiPaperExchange([a.base for a in assets], quote="USDC", cash_usdc=cash,
                            taker_fee=0.001, store_path=str(tmp_path / "book.json"),
                            price_source=lambda b: box[b])
    store = AltStore(str(tmp_path / "a.db"))
    st = AltStacker(cfg=cfg, exchange=ex, store=store, notifier=Notifier(),
                    llm_client=client or _client(holder),
                    ohlcv_source=_ohlcv_box(box, skew))
    return st


def _val(st, base, box):
    return _pos(st.store.get(f"pos:{base}")).value(box[base])


def test_buy_allocates_shared_pot(tmp_path):
    box = {"SOL": 100.0, "ETH": 2000.0, "HYPE": 20.0}
    st = _make(tmp_path, box, {"a": {"SOL": 0.4, "ETH": 0.4, "HYPE": 0.0}})
    st.run_cycle(now=T0)
    assert abs(_val(st, "SOL", box) - 400) < 10      # ~40% of $1,000 pot
    assert abs(_val(st, "ETH", box) - 400) < 10
    assert _val(st, "HYPE", box) < 1                  # nothing into HYPE
    assert 180 < st.ex.cash() < 220                   # ~$200 left in USDC


def test_per_asset_max_fraction_clamp(tmp_path):
    box = {"SOL": 100.0, "ETH": 2000.0, "HYPE": 20.0}
    st = _make(tmp_path, box, {"a": {"SOL": 0.0, "ETH": 0.0, "HYPE": 0.9}})
    st.run_cycle(now=T0)
    # LLM asked for 90% HYPE; the cage clamps to its 0.35 max_fraction
    assert _val(st, "HYPE", box) <= 0.36 * 1000
    assert _val(st, "HYPE", box) > 0.30 * 1000


def test_total_allocation_clamp(tmp_path):
    box = {"SOL": 100.0, "ETH": 2000.0, "HYPE": 20.0}
    # 0.45+0.45+0.35 = 1.25 > max_total 1.0 -> scaled down, cash ~ fully deployed
    st = _make(tmp_path, box, {"a": {"SOL": 0.45, "ETH": 0.45, "HYPE": 0.35}})
    st.run_cycle(now=T0)
    deployed = sum(_val(st, b, box) for b in ("SOL", "ETH", "HYPE"))
    assert deployed > 950                            # ~all $1,000 deployed
    assert st.ex.cash() < 50


def test_cycle_turnover_cap(tmp_path):
    box = {"SOL": 100.0, "ETH": 2000.0, "HYPE": 20.0}
    rails = Rails(max_cycle_turnover_usdc=200.0, max_daily_turnover_usdc=50000.0)
    st = _make(tmp_path, box, {"a": {"SOL": 0.45, "ETH": 0.0, "HYPE": 0.0}}, rails=rails)
    st.run_cycle(now=T0)
    # target SOL = $450 but only $200 may trade this cycle
    assert abs(_val(st, "SOL", box) - 200) < 5
    assert st.ex.cash() > 790


def test_daily_turnover_cap_across_cycles(tmp_path):
    box = {"SOL": 100.0, "ETH": 2000.0, "HYPE": 20.0}
    rails = Rails(max_cycle_turnover_usdc=5000.0, max_daily_turnover_usdc=150.0)
    st = _make(tmp_path, box, {"a": {"SOL": 0.45, "ETH": 0.45, "HYPE": 0.0}}, rails=rails)
    st.run_cycle(now=T0)
    st.run_cycle(now=T0.replace(hour=4))             # same UTC day
    traded = sum(_val(st, b, box) for b in ("SOL", "ETH"))
    assert traded <= 155                             # day cap holds across cycles


def test_hype_round_trip_realizes_usdc(tmp_path):
    box = {"SOL": 100.0, "ETH": 2000.0, "HYPE": 20.0}
    holder = {"a": {"SOL": 0.0, "ETH": 0.0, "HYPE": 0.3}}
    st = _make(tmp_path, box, holder)
    st.run_cycle(now=T0)                             # buy HYPE ~ $300 @ 20
    assert _pos(st.store.get("pos:HYPE")).units > 10
    box["HYPE"] = 24.0                               # range high
    holder["a"] = {"SOL": 0.0, "ETH": 0.0, "HYPE": 0.0}
    st.run_cycle(now=T0.replace(hour=4))            # sell HYPE high
    pos = _pos(st.store.get("pos:HYPE"))
    assert pos.units < 0.01                          # flat again
    assert pos.realized_quote > 40                   # harvested USDC (bought ~20 sold ~24)


def test_depeg_halts(tmp_path):
    box = {"SOL": 100.0, "ETH": 2000.0, "HYPE": 20.0}
    st = _make(tmp_path, box, {"a": {"SOL": 0.4, "ETH": 0.0, "HYPE": 0.0}})
    st.ex.usd_quote_estimate = lambda: 0.90          # simulate USDC de-peg
    rep = st.run_cycle(now=T0)
    assert rep.get("halted") == "depeg"
    assert st.store.is_halted()
    assert _val(st, "SOL", box) == 0                 # nothing bought during de-peg


def test_bad_tick_skips_asset(tmp_path):
    box = {"SOL": 100.0, "ETH": 2000.0, "HYPE": 20.0}
    # SOL's last close (200) is 2x its tick (100) -> >35% jump -> skip SOL, trade ETH
    st = _make(tmp_path, box, {"a": {"SOL": 0.4, "ETH": 0.4, "HYPE": 0.0}},
               skew={"SOL": 200.0})
    rep = st.run_cycle(now=T0)
    assert "SOL" in rep["bad_ticks"]
    assert _val(st, "SOL", box) == 0                 # SOL never traded on a bad tick
    assert _val(st, "ETH", box) > 300                # ETH still rebalanced


def test_llm_error_holds(tmp_path):
    box = {"SOL": 100.0, "ETH": 2000.0, "HYPE": 20.0}
    st = _make(tmp_path, box, {"a": {}}, client=_raising_client())
    rep = st.run_cycle(now=T0)
    assert all(d["stance"] == "error" for d in rep["decisions"].values())
    assert rep["actions"] == []                      # fail-safe: no churn
    assert _val(st, "SOL", box) == 0


def test_idempotent_market_order(tmp_path):
    box = {"SOL": 100.0}
    ex = MultiPaperExchange(["SOL"], quote="USDC", cash_usdc=1000.0, taker_fee=0.001,
                            store_path=str(tmp_path / "b.json"),
                            price_source=lambda b: box[b])
    ex.market_buy_quote("SOL", 100.0, "dup-coid")
    ex.market_buy_quote("SOL", 100.0, "dup-coid")    # same coid -> no-op
    assert abs(ex.cash() - 900.0) < 1e-6
    assert abs(ex.balances()["SOL"] - 100.0 * 0.999 / 100.0) < 1e-9


def test_kill_switch_blocks_trading(tmp_path):
    box = {"SOL": 100.0, "ETH": 2000.0, "HYPE": 20.0}
    st = _make(tmp_path, box, {"a": {"SOL": 0.4, "ETH": 0.0, "HYPE": 0.0}})
    st.store.halt("manual")
    rep = st.run_cycle(now=T0)
    assert rep.get("halted") == "manual"
    assert _val(st, "SOL", box) == 0


def test_reports_do_not_raise(tmp_path):
    box = {"SOL": 100.0, "ETH": 2000.0, "HYPE": 20.0}
    st = _make(tmp_path, box, {"a": {"SOL": 0.3, "ETH": 0.3, "HYPE": 0.2}})
    st.run_cycle(now=T0)
    st.daily_report()


# ───────────── audit-fix regressions ─────────────
def _const_client(payload_json):
    """Client returning a fixed raw JSON string for every asset (to test parsing)."""
    def create(**kw):
        return types.SimpleNamespace(choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content=payload_json))])
    return types.SimpleNamespace(chat=types.SimpleNamespace(
        completions=types.SimpleNamespace(create=create)))


def test_malformed_target_holds_not_max_buy(tmp_path):
    # Audit CRITICAL-class #2/#10: a NaN/"nan"/bool target must HOLD, never clamp to 1.0.
    box = {"SOL": 100.0, "ETH": 2000.0, "HYPE": 20.0}
    for payload in ('{"target_fraction": NaN, "stance":"hold"}',
                    '{"target_fraction": "nan", "stance":"hold"}',
                    '{"target_fraction": "inf"}',
                    '{"target_fraction": true}'):
        st = _make(tmp_path, box, {"a": {}}, client=_const_client(payload))
        st.store.set("_x", payload)  # unique-ish; tmp_path reused so reset positions
        rep = st.run_cycle(now=T0)
        assert all(d["stance"] == "error" for d in rep["decisions"].values()), payload
        assert rep["actions"] == [], payload
        assert _val(st, "SOL", box) == 0, payload
        # clean up for next iteration
        for b in ("SOL", "ETH", "HYPE"):
            st.store.set(f"pos:{b}", {"units": 0.0, "cost": 0.0, "realized_quote": 0.0})


def test_reconcile_prevents_double_buy_after_lost_write(tmp_path):
    # Audit CRITICAL #1: simulate a crash that filled on the exchange but lost the
    # local position write. Next cycle must reconcile from the exchange and NOT re-buy.
    box = {"SOL": 100.0, "ETH": 2000.0, "HYPE": 20.0}
    st = _make(tmp_path, box, {"a": {"SOL": 0.4, "ETH": 0.0, "HYPE": 0.0}})
    st.run_cycle(now=T0)                          # buys ~ $400 SOL
    units_after = _pos(st.store.get("pos:SOL")).units
    cash_after = st.ex.cash()
    assert units_after > 0
    # simulate the torn write: the exchange holds the units but the local pos was lost
    st.store.set("pos:SOL", {"units": 0.0, "cost": 0.0, "realized_quote": 0.0})
    st.run_cycle(now=T0.replace(hour=4))         # same target 0.4
    # reconcile re-adopts the exchange units; target already met -> NO second buy
    assert abs(st.ex.cash() - cash_after) < 1.0
    assert abs(_pos(st.store.get("pos:SOL")).units - units_after) < 1e-6


def test_daily_turnover_is_crash_atomic_from_trades(tmp_path):
    # Audit #13/#15: the per-day cap is derived from the trades log, not a counter,
    # so wiping the meta counter cannot re-open the cap.
    box = {"SOL": 100.0, "ETH": 2000.0, "HYPE": 20.0}
    rails = Rails(max_cycle_turnover_usdc=5000.0, max_daily_turnover_usdc=150.0)
    st = _make(tmp_path, box, {"a": {"SOL": 0.45, "ETH": 0.45, "HYPE": 0.0}}, rails=rails)
    st.run_cycle(now=T0)
    assert st.store.turnover_today(T0.timestamp()) <= 155
    # even after blowing away any meta state, the cap is recomputed from trades
    st.store.set("consec_bad_cycles", 0)
    assert st.store.turnover_today(T0.timestamp()) <= 155


def test_depeg_estimate_failure_fails_closed(tmp_path):
    # Audit #17: if the peg estimate raises, skip trading (don't trade blind).
    box = {"SOL": 100.0, "ETH": 2000.0, "HYPE": 20.0}
    st = _make(tmp_path, box, {"a": {"SOL": 0.4, "ETH": 0.0, "HYPE": 0.0}})
    def boom():
        raise RuntimeError("peg feed down")
    st.ex.usd_quote_estimate = boom
    rep = st.run_cycle(now=T0)
    assert rep.get("skipped") == "depeg_estimate_failed"
    assert _val(st, "SOL", box) == 0


def test_usd_quote_skips_depeg(tmp_path):
    # Kraken/USD: a fiat quote has no peg risk, so the de-peg path is skipped entirely
    # and trading proceeds normally.
    box = {"SOL": 100.0, "ETH": 2000.0, "HYPE": 20.0}
    assets = (AssetSpec("SOL", "accumulate_base", 0.5),)
    st = _make(tmp_path, {"SOL": 100.0}, {"a": {"SOL": 0.4}}, assets=assets)
    object.__setattr__(st.cfg, "quote", "USD")   # frozen dataclass; force fiat quote
    def boom():
        raise RuntimeError("should not be called for USD")
    st.ex.usd_quote_estimate = boom
    rep = st.run_cycle(now=T0)
    assert "skipped" not in rep and "halted" not in rep
    assert _val(st, "SOL", {"SOL": 100.0}) > 0


def test_clamp_targets_pure():
    from agent.alts import clamp_targets
    # per-asset max clamp (0.9 -> 0.5 each), total 1.0 within cap
    out = clamp_targets({"A": 0.9, "B": 0.9}, {"A": 0.5, "B": 0.5}, 1.0)
    assert out == {"A": 0.5, "B": 0.5}
    # total clamp: 0.5+0.5=1.0 > 0.8 -> scaled to 0.4 each
    out2 = clamp_targets({"A": 0.5, "B": 0.5}, {"A": 0.6, "B": 0.6}, 0.8)
    assert abs(out2["A"] - 0.4) < 1e-9 and abs(out2["B"] - 0.4) < 1e-9


def test_plan_trades_sells_first_then_buys():
    from agent.alts import plan_trades
    prices = {"A": 100.0, "B": 50.0}
    values = {"A": 600.0, "B": 0.0}          # overweight A, nothing in B
    plan = plan_trades(prices, values, cash=400.0, pot=1000.0,
                       targets={"A": 0.2, "B": 0.5}, rebal_band=0.05, min_trade=10.0,
                       cycle_cap=5000.0, daily_left=5000.0, fee=0.0)
    # A: 0.2*1000-600 = -400 -> SELL 400 (first); B: 0.5*1000-0 = +500 -> BUY 500
    assert plan[0] == ("A", "sell", 400.0)
    assert plan[1][0] == "B" and plan[1][1] == "buy" and abs(plan[1][2] - 500.0) < 1e-6


def test_plan_trades_respects_cycle_cap_and_band():
    from agent.alts import plan_trades
    prices = {"A": 100.0}
    plan = plan_trades(prices, {"A": 0.0}, cash=1000.0, pot=1000.0, targets={"A": 0.9},
                       rebal_band=0.08, min_trade=10.0, cycle_cap=200.0,
                       daily_left=5000.0, fee=0.0)
    assert plan == [("A", "buy", 200.0)]      # capped to the per-cycle turnover
    # below-band move -> no trade
    assert plan_trades({"A": 100.0}, {"A": 500.0}, 500.0, 1000.0, {"A": 0.52},
                       rebal_band=0.08, min_trade=10.0, cycle_cap=5000.0,
                       daily_left=5000.0, fee=0.0) == []


def test_gold_store_of_value(tmp_path):
    # Gold (PAXG) uses the store_of_value objective: safe-haven prompt + units-based
    # accumulation & benchmarks (like accumulate_base, NOT the HYPE quote-harvest path).
    from agent.alts import BASE_OBJECTIVES, _system_for
    sysmsg = _system_for("PAXG", "store_of_value")
    assert "SAFE-HAVEN" in sysmsg and "PAXG" in sysmsg
    assert "store_of_value" in BASE_OBJECTIVES
    box = {"PAXG": 4000.0}
    assets = (AssetSpec("PAXG", "store_of_value", 0.6),)
    st = _make(tmp_path, box, {"a": {"PAXG": 0.5}}, assets=assets)
    st.run_cycle(now=T0)
    assert _val(st, "PAXG", box) > 400                 # accumulated ~50% of the $1,000 pot
    assert st.store.get("hodl:PAXG") is not None        # units-based benchmark seeded
    assert st.store.get("dca:PAXG") is not None
    st.daily_report()                                   # report path handles gold


def test_pot_includes_held_skipped_asset(tmp_path):
    # Audit #11: an asset that is held but skipped (bad tick) this cycle must still
    # count toward the pot, so the surviving asset isn't force-sized against a shrunk pot.
    box = {"SOL": 100.0, "ETH": 2000.0, "HYPE": 20.0}
    st = _make(tmp_path, box, {"a": {"SOL": 0.4, "ETH": 0.4, "HYPE": 0.0}})
    st.run_cycle(now=T0)                          # establish SOL+ETH positions
    eth_units = _pos(st.store.get("pos:ETH")).units
    # next cycle: ETH bad-ticks (skipped); SOL target unchanged -> SOL must not be
    # force-sold to fund a target computed on a pot that dropped ETH's value.
    st2 = _make(tmp_path, box, {"a": {"SOL": 0.4, "ETH": 0.4, "HYPE": 0.0}},
                skew={"ETH": 5000.0})
    rep = st2.run_cycle(now=T0.replace(hour=4))
    assert "ETH" in rep["bad_ticks"]
    assert _pos(st2.store.get("pos:ETH")).units == eth_units  # ETH untouched (skipped)
