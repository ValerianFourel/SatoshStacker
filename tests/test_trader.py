"""Tests for the active satoshi-stacker: pot-proportion rebalancing, avg-entry
accounting, benchmark seeding — deterministic (no real orders, no real LLM)."""
import types
from datetime import datetime, timezone

import agent.trader as T
from agent.exchange import PaperExchange
from agent.notify import Notifier
from agent.trader import Lot, SatoshiTrader, TraderStore

UTC = timezone.utc


def test_lot_buy_lower_drops_avg():
    lot = Lot(usdc=1000.0)
    lot.buy(500, 60000)
    a1 = lot.avg
    lot.buy(500, 40000)               # buying lower lowers the average
    assert lot.avg < a1 and 40000 < lot.avg < 60000


def _client(holder):
    def create(**kw):
        c = '{"target_btc_fraction": %s, "stance":"x", "note":"t"}' % holder["t"]
        return types.SimpleNamespace(choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content=c))])
    return types.SimpleNamespace(chat=types.SimpleNamespace(
        completions=types.SimpleNamespace(create=create)))


def _trader(tmp_path, price, holder):
    box = {"p": float(price)}
    ex = PaperExchange("BTC/USDT", budget_usdc=2000.0, maker_fee=0.001, taker_fee=0.001,
                       store_path=str(tmp_path / "book.json"), price_source=lambda: box["p"])
    store = TraderStore(str(tmp_path / "t.db"))
    tr = SatoshiTrader(exchange=ex, store=store, notifier=Notifier(), llm_client=_client(holder),
                       model="x", symbol="BTC/USDT", dca_days=30, cycle_hours=4,
                       news_enabled=False)
    return tr, box


def test_first_cycle_snapshots_and_buys_to_target(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "public_ohlcv", lambda *a, **k: [[60000, 61000, 59000, 60000]] * 60)
    holder = {"t": 1.0}                               # LLM says go all-BTC
    tr, _ = _trader(tmp_path, 60000, holder)
    tr.run_cycle(now=datetime(2026, 6, 1, tzinfo=UTC))
    usdc, btc = tr.balances()
    assert btc > 0 and usdc < 50                      # bot deployed ~the whole pot
    # benchmarks seeded from the SAME starting pot ($2000)
    assert Lot(**tr.store.get("hodl")).btc > 0
    assert tr.store.get("dca") is not None
    assert abs(tr.store.get("start")["pot"] - 2000) < 1  # snapshotted the live pot


def test_avg_entry_from_cost_basis(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "public_ohlcv", lambda *a, **k: [[60000, 61000, 59000, 60000]] * 60)
    tr, _ = _trader(tmp_path, 60000, {"t": 1.0})
    tr.run_cycle(now=datetime(2026, 6, 1, tzinfo=UTC))
    usdc, btc = tr.balances()
    avg = float(tr.store.get("cost_basis")) / btc
    assert 59900 < avg < 60200                        # ~price incl. fee


def test_rebalance_sells_when_target_drops(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "public_ohlcv", lambda *a, **k: [[60000, 61000, 59000, 60000]] * 60)
    holder = {"t": 1.0}
    tr, _ = _trader(tmp_path, 60000, holder)
    tr.run_cycle(now=datetime(2026, 6, 1, tzinfo=UTC))   # buy to ~100% BTC
    _, btc1 = tr.balances()
    holder["t"] = 0.2                                    # now de-risk to 20% BTC
    tr.run_cycle(now=datetime(2026, 6, 1, 4, tzinfo=UTC))
    usdc2, btc2 = tr.balances()
    assert btc2 < btc1 and usdc2 > 50                    # sold BTC -> raised cash


def test_reports_do_not_raise(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "public_ohlcv", lambda *a, **k: [[60000, 61000, 59000, 60000]] * 60)
    tr, _ = _trader(tmp_path, 60000, {"t": 0.6})
    tr.run_cycle(now=datetime(2026, 6, 1, tzinfo=UTC))
    tr.daily_report(60000)
    tr.weekly_report(60000)
