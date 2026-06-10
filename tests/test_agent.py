"""End-to-end agent tests: fills, re-anchoring, the deadline guard, de-peg halt,
reconcile-on-restart, idempotency, per-day cap, and the LLM shrink-only invariant.

All tests run against PaperExchange with an INJECTED price (no network) and an
explicit clock, so the safety-critical paths are deterministic.
"""
from __future__ import annotations

import dataclasses as dc
from datetime import datetime, timedelta, timezone

import pytest

from agent.analysis import MockAdvisor, Verdict, gate_multiplier
from agent.config import (
    AgentConfig, AnalysisConfig, DeployScheduleConfig, LadderConfig, RiskConfig,
)
from agent.exchange import PaperExchange
from agent.loop import Agent
from agent.notify import Notifier
from agent.orders import OrderManager
from agent.risk import RiskGate
from agent.state import Store

UTC = timezone.utc
T0 = datetime(2026, 6, 10, 0, 0, tzinfo=UTC)


class Price:
    """Mutable injected price source."""
    def __init__(self, p: float) -> None:
        self.p = p

    def __call__(self) -> float:
        return self.p


def make_agent(tmp_path, price: float, *, advisor=None, usdc_usd: float = 1.0,
               **overrides):
    """Build an Agent on PaperExchange with an injected price + clock window."""
    budget = overrides.pop("budget", 2_000.0)
    ladder = LadderConfig(budget_usdc=budget, floor_price=35_000.0, n_tranches=8,
                          strategy=overrides.pop("strategy", "reanchor"),
                          gen_fraction=overrides.pop("gen_fraction", 0.5),
                          floor_frac=overrides.pop("floor_frac", 0.55))
    risk = RiskConfig(max_deploy_per_day_usdc=overrides.pop("cap", 1e9),
                      min_notional_usdc=overrides.pop("min_notional", 10.0))
    sched = DeployScheduleConfig(
        start=T0, end=overrides.pop("end", T0 + timedelta(days=90)),
        deploy_by=overrides.pop("deploy_by", T0 + timedelta(days=60)),
        dca_every_hours=overrides.pop("dca_every_hours", 24))
    cfg = AgentConfig(mode="dry_run", ladder=ladder, risk=risk, schedule=sched,
                      analysis=AnalysisConfig(enabled=True),
                      db_path=str(tmp_path / "s.db"))
    pricer = Price(price)
    ex = PaperExchange("BTC/USDC", budget_usdc=budget, maker_fee=0.001,
                       taker_fee=0.001, store_path=str(tmp_path / "book.json"),
                       price_source=pricer)
    if usdc_usd != 1.0:
        ex.usdc_usd_estimate = lambda: usdc_usd  # type: ignore[method-assign]
    store = Store(cfg.db_path)
    rg = RiskGate(cfg.risk, store)
    om = OrderManager(cfg, ex, store, rg)
    agent = Agent(cfg=cfg, store=store, ex=ex, risk=rg,
                  advisor=advisor or MockAdvisor(), om=om, notifier=Notifier())
    return agent, pricer


# ── basic placement / idempotency ───────────────────────────────────────────
def test_places_first_generation(tmp_path):
    agent, _ = make_agent(tmp_path, 60_000)
    rep = agent.run_cycle(now=T0)
    assert any("placed_rungs" in a for a in rep["actions"])
    resting = agent.store.rungs_by_status("resting")
    assert len(resting) == 8
    # gen-1 commits exactly gen_fraction of budget
    assert agent.store.resting_usdc() == pytest.approx(1_000.0, rel=1e-6)


def test_idempotent_second_cycle_places_nothing(tmp_path):
    agent, _ = make_agent(tmp_path, 60_000)
    agent.run_cycle(now=T0)
    rep = agent.run_cycle(now=T0 + timedelta(minutes=15))
    assert all("placed_rungs" not in a for a in rep["actions"])


# ── fills ────────────────────────────────────────────────────────────────────
def test_limit_fills_when_price_drops(tmp_path):
    agent, pricer = make_agent(tmp_path, 60_000)
    agent.run_cycle(now=T0)
    assert agent.store.total_deployed() == 0.0
    pricer.p = 45_000  # drop through several rungs
    agent.run_cycle(now=T0 + timedelta(days=1))
    assert agent.store.total_deployed() > 0
    assert agent.store.total_btc() > 0
    # only rungs at/above 45k should have filled
    for r in agent.store.fills():
        assert r["price"] >= 45_000 - 1


# ── re-anchoring ───────────────────────────────────────────────────────────────
def test_reanchor_opens_lower_generation_without_overcommitting(tmp_path):
    agent, pricer = make_agent(tmp_path, 60_000)
    agent.run_cycle(now=T0)
    gen1_floor = float(agent.store.get_meta("gen_floor"))
    pricer.p = gen1_floor - 500  # break below gen-1 floor
    rep = agent.run_cycle(now=T0 + timedelta(days=2))
    assert any("reanchored_to_generation" in a for a in rep["actions"])
    assert int(agent.store.get_meta("generation")) == 2
    # never commit more than the budget across deployed + resting
    committed = agent.store.total_deployed() + agent.store.resting_usdc()
    assert committed <= 2_000.0 + 1e-6


# ── deploy-by-deadline guard (the safety net) ──────────────────────────────────
def test_deadline_forces_full_deployment_never_strands(tmp_path):
    # price stays HIGH (bearish call wrong) so the ladder barely fills
    agent, pricer = make_agent(tmp_path, 60_000,
                               deploy_by=T0 + timedelta(days=1),
                               end=T0 + timedelta(days=11), dca_every_hours=24)
    agent.run_cycle(now=T0)
    assert agent.store.total_deployed() < 50  # almost nothing filled at 60k
    # advance past the deadline and run daily cycles to execute the forced DCA
    now = T0 + timedelta(days=1)
    for d in range(12):
        agent.run_cycle(now=now + timedelta(days=d))
    assert agent.store.get_meta("mode_phase") == "deadline"
    # essentially the entire budget is deployed — USDC was NOT stranded
    assert agent.store.total_deployed() == pytest.approx(2_000.0, rel=1e-3)


def test_deadline_beats_a_silent_ladder_even_if_floor_never_hit(tmp_path):
    agent, _ = make_agent(tmp_path, 100_000, floor_frac=0.55,
                          deploy_by=T0 + timedelta(days=1),
                          end=T0 + timedelta(days=6))
    for d in range(8):
        agent.run_cycle(now=T0 + timedelta(days=d))
    assert agent.store.total_btc() > 0
    assert agent.store.total_deployed() == pytest.approx(2_000.0, rel=1e-3)


# ── LLM shrink-only invariant ──────────────────────────────────────────────────
@pytest.mark.parametrize("mult,expected", [(1.5, 1.0), (0.7, 0.7), (0.2, 0.5)])
def test_llm_can_only_shrink_new_rungs(tmp_path, mult, expected):
    agent, _ = make_agent(tmp_path, 60_000,
                          advisor=MockAdvisor(stance="shrink", multiplier=mult))
    agent.run_cycle(now=T0)
    # each resting rung is sized at base * clamped(mult); compare to a no-advice run
    total = agent.store.resting_usdc()
    assert total == pytest.approx(1_000.0 * expected, rel=1e-6)


def test_stale_verdict_fails_safe_to_spine_size():
    stale = Verdict(stance="shrink", size_multiplier=0.5,
                    as_of_bar="2020-01-01T00:00:00Z")
    assert gate_multiplier(stale, current_bar="2026-06-10T00:00:00Z",
                           min_fraction=0.5) == 1.0


def test_missing_verdict_fails_safe():
    assert gate_multiplier(None, current_bar="2026-06-10T00:00:00Z",
                           min_fraction=0.5) == 1.0


# ── de-peg halt ────────────────────────────────────────────────────────────────
def test_depeg_halts_and_blocks_orders(tmp_path):
    agent, _ = make_agent(tmp_path, 60_000, usdc_usd=0.95)  # USDC de-pegged
    rep = agent.run_cycle(now=T0)
    assert agent.store.is_halted()
    assert "depeg" in (rep.get("halted") or "")
    assert len(agent.store.rungs_by_status("resting")) == 0  # nothing placed


def test_halt_requires_manual_clear(tmp_path):
    agent, pricer = make_agent(tmp_path, 60_000, usdc_usd=0.95)
    agent.run_cycle(now=T0)
    assert agent.store.is_halted()
    # even after the peg returns, we stay halted until cleared
    agent.ex.usdc_usd_estimate = lambda: 1.0  # type: ignore[method-assign]
    agent.run_cycle(now=T0 + timedelta(minutes=15))
    assert agent.store.is_halted()
    agent.store.clear_halt()
    agent.run_cycle(now=T0 + timedelta(minutes=30))
    assert not agent.store.is_halted()
    assert len(agent.store.rungs_by_status("resting")) == 8


# ── per-day cap ────────────────────────────────────────────────────────────────
def test_per_day_cap_limits_placement(tmp_path):
    # cap below the largest rung -> the big floor rungs get skipped this day
    agent, _ = make_agent(tmp_path, 60_000, cap=100.0)
    agent.run_cycle(now=T0)
    assert agent.store.resting_usdc() <= 100.0 + 1e-6


# ── reconcile-on-restart (exchange = source of truth) ──────────────────────────
def test_reconcile_on_restart_adopts_fills(tmp_path):
    agent, pricer = make_agent(tmp_path, 60_000)
    agent.run_cycle(now=T0)
    # simulate fills happening on the EXCHANGE while the agent is down
    pricer.p = 40_000
    agent.ex.settle()  # paper book fills resting bids >= 40k
    deployed_before_restart = agent.store.total_deployed()
    assert deployed_before_restart == 0.0  # agent state hasn't seen the fills yet
    agent.store.close()

    # restart: brand-new Store + Agent on the SAME db + paper book
    cfg = agent.cfg
    store2 = Store(cfg.db_path)
    from agent.exchange import PaperExchange
    ex2 = PaperExchange("BTC/USDC", budget_usdc=2_000.0, maker_fee=0.001,
                        taker_fee=0.001, store_path=str(tmp_path / "book.json"),
                        price_source=Price(40_000))
    rg2 = RiskGate(cfg.risk, store2)
    om2 = OrderManager(cfg, ex2, store2, rg2)
    agent2 = Agent(cfg=cfg, store=store2, ex=ex2, risk=rg2,
                   advisor=MockAdvisor(), om=om2, notifier=Notifier())
    agent2.run_cycle(now=T0 + timedelta(days=1))
    # the restarted agent reconciled the fills from the exchange
    assert agent2.store.total_deployed() > 0
    assert agent2.store.total_btc() > 0


# ── regression: audit fixes (M2 safety audit, 2026-06-10) ──────────────────────
def _journal(store):
    return store.conn.execute("SELECT kind, payload FROM journal").fetchall()


def test_torn_write_crash_recovers_no_double_buy_no_lost_fill(tmp_path):
    """Crash AFTER placing the exchange order but BEFORE persisting 'resting'.
    On restart, reconcile must adopt the orphan order (record its fill if filled)
    and never re-place it. (Audit: STATE-INTEGRITY HIGH.)"""
    agent, pricer = make_agent(tmp_path, 60_000)

    # crash the FIRST 'resting' transition (models power loss right after placement)
    orig = agent.store.set_rung_status
    state = {"crashed": False}

    def crashy(coid, status, exchange_order_id=None):
        if status == "resting" and not state["crashed"]:
            state["crashed"] = True
            raise RuntimeError("power loss between create_limit_buy and persist")
        return orig(coid, status, exchange_order_id)

    agent.store.set_rung_status = crashy  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        agent.run_cycle(now=T0)
    # ss-g1-r1: order exists on the exchange book but is still 'pending' locally
    assert agent.ex.fetch_order_by_client_id("ss-g1-r1") is not None
    assert agent.store.get_rung("ss-g1-r1")["status"] == "pending"
    # the orphan fills while we are "down" (r1 level ≈ $56,625 for a $60k anchor)
    pricer.p = 50_000
    agent.ex.settle()
    agent.store.close()

    # restart on the SAME db + paper book
    cfg = agent.cfg
    store2 = Store(cfg.db_path)
    ex2 = PaperExchange("BTC/USDC", budget_usdc=2_000.0, maker_fee=0.001,
                        taker_fee=0.001, store_path=str(tmp_path / "book.json"),
                        price_source=Price(50_000))
    rg2 = RiskGate(cfg.risk, store2)
    agent2 = Agent(cfg=cfg, store=store2, ex=ex2, risk=rg2, advisor=MockAdvisor(),
                   om=OrderManager(cfg, ex2, store2, rg2), notifier=Notifier())
    agent2.run_cycle(now=T0 + timedelta(hours=1))

    # exactly ONE exchange order for r1 (no double-buy) and its fill IS counted
    n_r1_orders = sum(1 for o in ex2._book["orders"].values()
                      if o["clientOrderId"] == "ss-g1-r1")
    assert n_r1_orders == 1, "double-buy: r1 was re-placed"
    assert store2.fill_exists("ss-g1-r1"), "lost fill: orphan not reconciled"
    assert store2.total_deployed() > 0


def test_deadline_subm_min_leftover_surfaced_as_dust_not_looped(tmp_path):
    """Leftover below MIN_NOTIONAL is un-deployable on a real exchange; it must be
    surfaced as dust, not silently retried forever. (Audit: DEADLINE-WINS MEDIUM.)"""
    agent, _ = make_agent(tmp_path, 60_000, budget=8.0, min_notional=10.0,
                          deploy_by=T0 + timedelta(days=1), end=T0 + timedelta(days=6))
    for d in range(8):
        agent.run_cycle(now=T0 + timedelta(days=d))
    kinds = [r["kind"] for r in _journal(agent.store)]
    assert "deadline_dust" in kinds          # dust surfaced
    assert "dca_skipped" not in kinds        # NOT an infinite reject loop
    assert agent.store.pending_schedule_usdc() == 0.0  # no doomed pending tranche


def test_deadline_above_min_still_fully_deploys(tmp_path):
    """Leftover >= MIN_NOTIONAL must still deploy 100% (every tranche clears min)."""
    agent, _ = make_agent(tmp_path, 60_000, budget=45.0, min_notional=10.0,
                          deploy_by=T0 + timedelta(days=1), end=T0 + timedelta(days=6))
    for d in range(8):
        agent.run_cycle(now=T0 + timedelta(days=d))
    assert agent.store.total_deployed() == pytest.approx(45.0, rel=1e-3)


def test_clamp_floor_misconfig_cannot_exceed_spine():
    """An operator setting min_fraction > 1 must NOT lift a verdict above spine."""
    v = Verdict("shrink", 0.3, as_of_bar="2026-06-10T00:00:00Z")
    assert gate_multiplier(v, current_bar="2026-06-10T00:00:00Z", min_fraction=2.0) <= 1.0


def test_advisor_exception_fails_safe_to_spine(tmp_path):
    class Boom(MockAdvisor):
        def advise(self, features, *, as_of_bar):
            raise RuntimeError("LLM 500")

    agent, _ = make_agent(tmp_path, 60_000, advisor=Boom())
    rep = agent.run_cycle(now=T0)  # must NOT raise
    assert rep.get("size_multiplier") == 1.0          # degraded to spine size
    assert agent.store.resting_usdc() == pytest.approx(1_000.0, rel=1e-6)


def test_depeg_estimate_exception_fails_closed(tmp_path):
    agent, _ = make_agent(tmp_path, 60_000)
    agent.ex.usdc_usd_estimate = lambda: (_ for _ in ()).throw(RuntimeError("feed down"))  # type: ignore
    rep = agent.run_cycle(now=T0)  # must NOT raise; must place nothing
    assert "error" in rep
    assert len(agent.store.rungs_by_status("resting")) == 0
    assert int(agent.store.get_meta("consec_api_failures", 0)) >= 1  # escalates


def test_exception_strings_are_redacted_in_journal(tmp_path, monkeypatch):
    monkeypatch.setenv("BINANCE_API_KEY", "MyBinanceSecret_9f8e7d6c5b4a")
    agent, _ = make_agent(tmp_path, 60_000)

    def leaky(*_a, **_k):
        raise RuntimeError("InvalidSignature url=/api/v3/order?"
                           "apiKey=MyBinanceSecret_9f8e7d6c5b4a&signature=deadbeef")

    agent.ex.create_limit_buy = leaky  # type: ignore[method-assign]
    agent.run_cycle(now=T0)
    blob = " ".join(str(r["payload"]) for r in _journal(agent.store))
    assert "MyBinanceSecret_9f8e7d6c5b4a" not in blob
    assert "signature=deadbeef" not in blob
    assert "REDACTED" in blob


# ── regression: re-audit fixes (M2 safety re-audit, 2026-06-10) ────────────────
def test_price_fetch_exception_redacted_in_report(tmp_path, monkeypatch):
    """The price-fetch error path must not leak creds into report['error']/stdout."""
    monkeypatch.setenv("BINANCE_API_KEY", "MyBinanceSecret_9f8e7d6c5b4a")
    agent, _ = make_agent(tmp_path, 60_000)

    def boom():
        raise RuntimeError("AuthError url=/sapi/v1/account?"
                           "apiKey=MyBinanceSecret_9f8e7d6c5b4a&signature=deadbeefcafe")

    agent.ex.get_price = boom  # type: ignore[method-assign]
    rep = agent.run_cycle(now=T0)  # must not raise
    assert "MyBinanceSecret_9f8e7d6c5b4a" not in str(rep)
    assert "signature=deadbeefcafe" not in str(rep)


def test_deadline_dust_guard_matches_gate_at_float_boundary(tmp_path):
    """A leftover just below MIN_NOTIONAL must be dust, never a doomed looping schedule.
    (Re-audit: the dust guard's -1e-6 slack re-opened the strand.)"""
    agent, _ = make_agent(tmp_path, 60_000, budget=9.9999999, min_notional=10.0,
                          deploy_by=T0 + timedelta(days=1), end=T0 + timedelta(days=6))
    for d in range(8):
        agent.run_cycle(now=T0 + timedelta(days=d))
    kinds = [r["kind"] for r in _journal(agent.store)]
    assert "deadline_dust" in kinds
    assert "dca_skipped" not in kinds            # no infinite reject loop
    assert agent.store.pending_schedule_usdc() == 0.0


def test_per_day_cap_crash_atomic_not_breached(tmp_path):
    """committed_today is derived from rung rows, so a 'pending' rung orphaned by a
    crash still counts against the cap — it cannot be evaded. (Re-audit MEDIUM.)"""
    agent, _ = make_agent(tmp_path, 60_000, cap=300.0)
    day = int(T0.timestamp() - (T0.timestamp() % 86_400))
    # inject an orphaned pending rung (models a crash before the counter bump)
    agent.store.upsert_rung(client_order_id="ss-crash", generation=1, idx=99,
                            price=50_000, usdc=250.0, amount_btc=250.0 / 50_000,
                            exchange_order_id=None, status="pending", placed_day=day)
    # the derived counter already reflects the orphan
    assert agent.store.committed_today(T0.timestamp()) >= 250.0
    agent.run_cycle(now=T0)
    # new placements respect the cap GIVEN the orphan — no breach
    assert agent.store.committed_today(T0.timestamp()) <= 300.0 + 1e-6


def test_telegram_url_token_is_redacted():
    from agent.secrets import redact
    url = "ConnectionError https://api.telegram.org/bot987654321:AAH_directToken_xyz/sendMessage"
    out = redact(url)
    assert "AAH_directToken_xyz" not in out
    assert "REDACTED" in out


@pytest.mark.parametrize("text,secret", [
    ('{"apiKey":"Sup3rSecretValue_abc123"}', "Sup3rSecretValue_abc123"),
    ("sent header X-MBX-APIKEY Sup3rSecretValue_abc123 to host", "Sup3rSecretValue_abc123"),
    ("GET url?apiKey=Sup3rSecretValue_abc123&signature=deadbeef", "Sup3rSecretValue_abc123"),
])
def test_redact_no_delimiter_shapes_without_env(text, secret):
    """redact() masks quoted-JSON and space-delimited header creds even when the
    value is NOT registered in the env (regex-only defense). (Confirm-audit LOW.)"""
    from agent.secrets import redact
    out = redact(text)
    assert secret not in out


@pytest.mark.parametrize("raw,expected", [
    ("# «ADD» your key here", ""),   # leaked python-dotenv inline comment -> empty
    ("  # comment", ""),
    ("", ""),
    (None, ""),
    ("  sk-realkey123  ", "sk-realkey123"),  # trimmed, kept
    ("ExampleKey789", "ExampleKey789"),
])
def test_clean_secret_drops_leaked_comments(raw, expected):
    """A secret env value that is actually a leaked '# comment' must be treated as
    empty so the agent never runs with a garbage credential."""
    from agent.secrets import clean_secret
    assert clean_secret(raw) == expected
