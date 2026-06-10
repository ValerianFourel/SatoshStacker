"""The agent loop: observe -> reconcile -> risk gate -> plan -> place -> persist.

State machine (re-anchoring strategy):

    laddering  --(now >= deploy_by)-->  deadline (force-DCA leftover)

While *laddering* we keep a generation's weighted maker ladder resting; when
price breaks the generation floor we re-anchor a fresh, lower generation with the
remaining uncommitted budget (the Milestone-1 trailing-ladder edge). The LLM may
shrink newly-placed rungs (clamped, fail-safe). When the deploy-by-deadline fires,
all resting bids are cancelled and the leftover is force-DCA'd over the remaining
window — the spine and the deadline always win over the advisor.
"""
from __future__ import annotations

import logging
import signal
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from .analysis import Advisor, Verdict, build_features, gate_multiplier
from .config import AgentConfig
from .exchange import Exchange
from .ladder import (
    build_ladder, deadline_dca_tranches, filter_min_notional, next_generation,
)
from .notify import Notifier
from .orders import OrderManager
from .risk import RiskGate
from .secrets import redact
from .state import Store

log = logging.getLogger("satoshistacker.loop")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _bar_iso(now: datetime, cadence_hours: int) -> str:
    """ISO8601 of the current advisory bar (floored to the cadence)."""
    h = (now.hour // cadence_hours) * cadence_hours
    return now.replace(hour=h, minute=0, second=0, microsecond=0).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


@dataclass
class Agent:
    cfg: AgentConfig
    store: Store
    ex: Exchange
    risk: RiskGate
    advisor: Advisor
    om: OrderManager
    notifier: Notifier

    # ---- generation helpers ----
    def _current_generation_rungs(self):
        anchor = float(self.store.get_meta("anchor"))
        gen_budget = float(self.store.get_meta("gen_budget"))
        gen_floor = float(self.store.get_meta("gen_floor"))
        rungs = build_ladder(anchor, self.cfg.ladder, budget_override=gen_budget,
                             floor_override=gen_floor)
        placeable, _ = filter_min_notional(rungs, self.cfg.risk.min_notional_usdc)
        return placeable

    def _open_new_generation(self, generation: int, anchor: float) -> None:
        budget = self.cfg.ladder.budget_usdc
        remaining_uncommitted = budget - self.store.total_deployed() - self.store.resting_usdc()
        rungs, gen_floor = next_generation(anchor, remaining_uncommitted, self.cfg.ladder)
        gen_budget = remaining_uncommitted * self.cfg.ladder.gen_fraction
        self.store.set_meta("generation", generation)
        self.store.set_meta("anchor", anchor)
        self.store.set_meta("gen_budget", gen_budget)
        self.store.set_meta("gen_floor", gen_floor)
        self.store.journal("generation_opened",
                           {"generation": generation, "anchor": anchor,
                            "gen_floor": gen_floor, "gen_budget": round(gen_budget, 2)})

    # ---- advisory ----
    def _get_verdict(self, now: datetime, price: float) -> Verdict | None:
        bar = _bar_iso(now, self.cfg.analysis.cadence_hours)
        if self.store.get_meta("last_advice_bar") == bar:
            d = self.store.get_meta("last_verdict")
            return Verdict(**d) if d else None
        if not self.cfg.analysis.enabled:
            return None
        anchor = float(self.store.get_meta("anchor", price))
        gen_floor = float(self.store.get_meta("gen_floor", self.cfg.ladder.floor_price))
        days_left = max(0.0, (self.cfg.schedule.end - now).total_seconds() / 86_400)
        resting = [r["price"] for r in self.store.rungs_by_status("resting")]
        features = build_features(
            price=price, anchor=anchor, gen_floor=gen_floor,
            floor_price=self.cfg.ladder.floor_price,
            deployed=self.store.total_deployed(), budget=self.cfg.ladder.budget_usdc,
            days_left=days_left, resting_levels=resting)
        try:
            verdict = self.advisor.advise(features, as_of_bar=bar)
        except Exception:  # noqa: BLE001 - a broken advisor degrades to spine size
            verdict = Verdict.safe(bar, "advisor raised: spine size")
        self.store.set_meta("last_advice_bar", bar)
        self.store.set_meta("last_verdict", verdict.__dict__)
        self.store.journal("verdict", verdict.__dict__)
        return verdict

    # ---- deadline ----
    def _transition_to_deadline(self, now: datetime) -> None:
        cancelled = self.om.cancel_all_resting()
        leftover = self.cfg.ladder.budget_usdc - self.store.total_deployed()
        min_notional = self.cfg.risk.min_notional_usdc
        self.store.set_meta("mode_phase", "deadline")

        # Below the exchange MIN_NOTIONAL a market buy is physically rejected, so a
        # sub-min remainder is un-deployable DUST, not a tradeable strand. Surface
        # it (never silently loop dca_skipped) and stop — do not emit a doomed order.
        # NOTE: this predicate MUST match RiskGate.check_order's strict
        # `usdc < min_notional` rejection exactly, or a leftover in the gap between
        # them would be scheduled yet rejected forever (re-audit finding).
        if leftover < min_notional:
            self.store.journal("deadline_dust",
                               {"dust_usdc": round(leftover, 2),
                                "cancelled_rungs": cancelled})
            if leftover > 0.01:
                self.notifier.send(
                    f"⏰ Deadline reached. ${leftover:,.2f} leftover is below "
                    f"MIN_NOTIONAL (${min_notional:,.0f}) — un-deployable dust, "
                    f"nothing tradeable stranded.")
            return

        end = self.cfg.schedule.end
        cadence_h = self.cfg.schedule.dca_every_hours
        hours = max(cadence_h, (end - now).total_seconds() / 3600)
        n_desired = max(1, int(hours // cadence_h))
        # size n so EVERY forced tranche clears MIN_NOTIONAL (leftover >= min here,
        # and n = floor(leftover/min) => each tranche = leftover/n >= min_notional)
        n = max(1, min(n_desired, int(leftover // min_notional)))
        tranches = deadline_dca_tranches(leftover, n)
        rows = [(i, int(now.timestamp() + i * cadence_h * 3600), usdc)
                for i, usdc in enumerate(tranches)]
        self.store.set_schedule(rows)
        self.store.journal("deadline_fired",
                           {"leftover": round(leftover, 2), "tranches": n,
                            "cancelled_rungs": cancelled})
        self.notifier.send(
            f"⏰ Deploy-by-deadline reached. Cancelled {cancelled} resting bids; "
            f"force-DCAing ${leftover:,.0f} over {n} tranches.")

    def _run_due_tranches(self, now: datetime) -> int:
        n = 0
        for row in self.store.due_tranches(int(now.timestamp())):
            if self.om.execute_deadline_tranche(row["idx"], row["usdc"]):
                n += 1
        return n

    # ---- summary ----
    def _maybe_summary(self, now: datetime, price: float) -> None:
        today = now.strftime("%Y-%m-%d")
        if self.store.get_meta("last_summary_date") == today:
            return
        if now.hour < self.cfg.summary_hour_utc:
            return
        deployed = self.store.total_deployed()
        btc = self.store.total_btc()
        budget = self.cfg.ladder.budget_usdc
        self.notifier.daily_summary(
            mode=self.cfg.mode, deployed=deployed, budget=budget, btc=btc,
            avg_cost=(deployed / btc if btc > 0 else 0.0), price=price,
            remaining=budget - deployed)
        self.store.set_meta("last_summary_date", today)

    # ---- one cycle ----
    def run_cycle(self, now: datetime | None = None) -> dict:
        now = now or _utcnow()
        report: dict = {"ts": now.isoformat(), "actions": []}

        # 1) exchange = source of truth: settle paper fills, then reconcile
        try:
            self.ex.settle()
            changes = self.store.reconcile(self.ex)
            self.risk.note_api_result(True)
            if changes:
                report["actions"].append({"reconciled": changes})
        except Exception as e:  # noqa: BLE001
            self.risk.note_api_result(False)
            self.store.journal("reconcile_error", {"error": redact(e)})

        if self.store.is_halted():
            report["halted"] = self.store.get_meta("halt_reason")
            return report

        # 2) price + de-peg + anomaly band
        try:
            price = self.ex.get_price()
            self.risk.note_api_result(True)
        except Exception as e:  # noqa: BLE001
            self.risk.note_api_result(False)
            report["error"] = f"price fetch failed: {redact(e)}"  # redact: may carry creds
            return report
        report["price"] = price

        try:
            usdc_usd = self.ex.usdc_usd_estimate()
            self.risk.note_api_result(True)
        except Exception as e:  # noqa: BLE001 - cannot verify peg -> fail closed, skip
            self.risk.note_api_result(False)
            self.store.journal("depeg_estimate_error", {"error": redact(e)})
            report["error"] = "depeg estimate failed (skipping cycle)"
            return report
        depeg = self.risk.check_depeg(usdc_usd)
        if not depeg.ok:
            self.notifier.send(f"🛑 HALT: {self.store.get_meta('halt_reason')}")
            report["halted"] = depeg.reason
            return report

        budget = self.cfg.ladder.budget_usdc

        # 3) deadline transition + execution (always wins over the advisor)
        phase = self.store.get_meta("mode_phase", "laddering")
        if now >= self.cfg.schedule.deploy_by and phase != "deadline":
            self._transition_to_deadline(now)
            phase = "deadline"
        if phase == "deadline":
            done = self._run_due_tranches(now)
            if done:
                report["actions"].append({"deadline_tranches": done})
            self._maybe_summary(now, price)
            return report

        # 4) laddering — initialize first generation if needed
        if not self.store.get_meta("initialized", False):
            self._open_new_generation(1, price)
            self.store.set_meta("initialized", True)
            self.store.set_meta("started_at", now.isoformat())
            self.notifier.send(
                f"🚀 SatoshiStacker started [{self.cfg.mode}] "
                f"anchor=${price:,.0f} floor=${self.cfg.ladder.floor_price:,.0f} "
                f"budget=${budget:,.0f} ({self.cfg.ladder.strategy})")

        # 5) re-anchor if price broke the current generation floor
        generation = int(self.store.get_meta("generation"))
        gen_floor = float(self.store.get_meta("gen_floor"))
        remaining_uncommitted = budget - self.store.total_deployed() - self.store.resting_usdc()
        if (self.cfg.ladder.strategy == "reanchor" and price < gen_floor
                and remaining_uncommitted >= self.cfg.risk.min_notional_usdc
                and generation < self.cfg.ladder.max_generations):
            generation += 1
            self._open_new_generation(generation, price)
            report["actions"].append({"reanchored_to_generation": generation,
                                      "anchor": price})

        # 6) advisory shrink (clamped, fail-safe) applied to NEW placements
        verdict = self._get_verdict(now, price)
        bar = _bar_iso(now, self.cfg.analysis.cadence_hours)
        mult = gate_multiplier(verdict, current_bar=bar,
                               min_fraction=self.cfg.analysis.min_size_multiplier)
        report["size_multiplier"] = mult

        # 7) ensure this generation's rungs are resting (idempotent)
        placed = self.om.ensure_resting(self._current_generation_rungs(), generation,
                                        size_multiplier=mult, now_ts=now.timestamp())
        if placed:
            report["actions"].append({"placed_rungs": placed})

        # 8) housekeeping
        self._maybe_summary(now, price)
        self.store.set_meta("last_cycle", now.isoformat())
        return report

    # ---- long-running loop with graceful shutdown ----
    def run(self, *, max_cycles: int | None = None) -> None:
        stop = {"flag": False}

        def _graceful(signum, _frame):
            log.info("signal %s — graceful shutdown (resting bids LEFT in place)", signum)
            stop["flag"] = True

        for s in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(s, _graceful)
            except ValueError:
                pass  # not in main thread (e.g. tests) — skip

        i = 0
        while not stop["flag"]:
            try:
                rep = self.run_cycle()
                log.info("cycle %d: %s", i, rep)
            except Exception as e:  # noqa: BLE001 - never crash the loop; back off
                log.error("cycle error: %s", redact(e))
                self.store.journal("cycle_exception", {"error": redact(e)})
            i += 1
            if max_cycles is not None and i >= max_cycles:
                break
            # interruptible sleep (no spin-loop)
            for _ in range(self.cfg.poll_interval_s):
                if stop["flag"]:
                    break
                time.sleep(1)
        log.info("loop stopped after %d cycles (resting bids preserved)", i)
