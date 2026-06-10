"""Tests for the active satoshi-stacker: a logical STACK (e.g. $2k) traded in pot
proportions, avg-entry accounting, benchmark seeding — deterministic (no real LLM)."""
import types
from datetime import datetime, timezone

import agent.trader as T
from agent.exchange import PaperExchange
from agent.notify import Notifier
from agent.trader import Lot, SatoshiTrader, TraderStore, _lot

UTC = timezone.utc


def test_lot_buy_lower_drops_avg():
    lot = Lot(usdc=1000.0)
    lot.buy(500, 60000)
    a1 = lot.avg
    lot.buy(500, 40000)               # buying lower lowers the average
    assert lot.avg < a1 and 40000 < lot.avg < 60000


def test_lot_sell_keeps_avg_then_rebuy_lower_drops():
    lot = Lot(usdc=1000.0)
    lot.buy(1000, 60000)
    a = lot.avg
    lot.sell(lot.btc * 60000 * 0.5, 60000)   # sell half at 60k -> remainder avg unchanged
    assert abs(lot.avg - a) < a * 0.01
    lot.buy(lot.usdc, 30000)                  # redeploy freed cash lower -> avg drops
    assert lot.avg < a


def _client(holder):
    def create(**kw):
        c = '{"target_btc_fraction": %s, "stance":"x", "note":"t"}' % holder["t"]
        return types.SimpleNamespace(choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content=c))])
    return types.SimpleNamespace(chat=types.SimpleNamespace(
        completions=types.SimpleNamespace(create=create)))


def _trader(tmp_path, price, holder, stack=2000.0):
    box = {"p": float(price)}
    # seed the paper exchange with a BIG balance (mimics the pre-seeded testnet account)
    ex = PaperExchange("BTC/USDT", budget_usdc=50000.0, maker_fee=0.001, taker_fee=0.001,
                       store_path=str(tmp_path / "book.json"), price_source=lambda: box["p"])
    store = TraderStore(str(tmp_path / "t.db"))
    tr = SatoshiTrader(exchange=ex, store=store, notifier=Notifier(), llm_client=_client(holder),
                       model="x", symbol="BTC/USDT", stack_usdc=stack, dca_days=30,
                       cycle_hours=4, news_enabled=False)
    return tr, box


def test_bot_manages_only_the_stack_not_the_full_balance(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "public_ohlcv", lambda *a, **k: [[60000, 61000, 59000, 60000]] * 60)
    tr, _ = _trader(tmp_path, 60000, {"t": 1.0}, stack=2000.0)   # exchange has $50k, stack is $2k
    tr.run_cycle(now=datetime(2026, 6, 1, tzinfo=UTC))
    bot = _lot(tr.store.get("bot"))
    assert abs(bot.value(60000) - 2000) < 5      # manages ~$2,000, NOT the $50k account
    assert bot.btc > 0 and bot.usdc < 50         # target 1.0 deployed the stack
    assert _lot(tr.store.get("hodl")).btc > 0    # benchmarks seeded from the $2k stack too


def test_avg_entry_from_cost_basis(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "public_ohlcv", lambda *a, **k: [[60000, 61000, 59000, 60000]] * 60)
    tr, _ = _trader(tmp_path, 60000, {"t": 1.0})
    tr.run_cycle(now=datetime(2026, 6, 1, tzinfo=UTC))
    assert 59900 < _lot(tr.store.get("bot")).avg < 60200    # ~price incl. fee


def test_rebalance_sells_when_target_drops(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "public_ohlcv", lambda *a, **k: [[60000, 61000, 59000, 60000]] * 60)
    holder = {"t": 1.0}
    tr, _ = _trader(tmp_path, 60000, holder)
    tr.run_cycle(now=datetime(2026, 6, 1, tzinfo=UTC))           # buy to ~100% BTC
    btc1 = _lot(tr.store.get("bot")).btc
    holder["t"] = 0.2                                            # de-risk to 20% BTC
    tr.run_cycle(now=datetime(2026, 6, 1, 4, tzinfo=UTC))
    bot = _lot(tr.store.get("bot"))
    assert bot.btc < btc1 and bot.usdc > 50                     # sold BTC -> raised cash


def test_reports_do_not_raise(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "public_ohlcv", lambda *a, **k: [[60000, 61000, 59000, 60000]] * 60)
    tr, _ = _trader(tmp_path, 60000, {"t": 0.6})
    tr.run_cycle(now=datetime(2026, 6, 1, tzinfo=UTC))
    tr.daily_report(60000)
    tr.weekly_report(60000)
