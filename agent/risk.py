"""Deterministic risk guardrails. Fail closed.

Every order passes `RiskGate.check_order` first. Anomalies (USDC de-peg, an
order that deviates absurdly from the ladder, repeated API failures) trip a
persisted halt that requires a manual `--clear-halt` before trading resumes.
The LLM advisor has no access here; these checks are pure and deterministic.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from .config import RiskConfig
from .state import Store


@dataclass
class RiskResult:
    ok: bool
    reason: str = ""


def _start_of_utc_day(now: float | None = None) -> int:
    now = now if now is not None else time.time()
    return int(now - (now % 86_400))


class RiskGate:
    def __init__(self, cfg: RiskConfig, store: Store) -> None:
        self.cfg = cfg
        self.store = store

    # ---- anomaly halts ----
    def check_depeg(self, usdc_usd: float) -> RiskResult:
        """Halt if the USDC/USD estimate leaves the safe band (de-peg)."""
        if not (self.cfg.depeg_low <= usdc_usd <= self.cfg.depeg_high):
            self.store.halt(f"USDC de-peg: USDC/USD={usdc_usd:.4f} outside "
                            f"[{self.cfg.depeg_low},{self.cfg.depeg_high}]")
            return RiskResult(False, "usdc_depeg")
        return RiskResult(True)

    def note_api_result(self, ok: bool) -> RiskResult:
        """Track consecutive API failures; halt past the threshold."""
        n = int(self.store.get_meta("consec_api_failures", 0))
        n = 0 if ok else n + 1
        self.store.set_meta("consec_api_failures", n)
        if n >= self.cfg.max_consecutive_api_failures:
            self.store.halt(f"{n} consecutive API failures")
            return RiskResult(False, "api_failures")
        return RiskResult(True)

    # ---- per-order checks ----
    def check_order(self, *, price: float, usdc: float, expected_level: float,
                    now: float | None = None, enforce_daily_cap: bool = True) -> RiskResult:
        """Validate a single buy before placement.

        - halt state blocks everything (incl. the deadline DCA — a halt means
          something is wrong, e.g. USDC de-peg, so we do NOT buy);
        - per-UTC-day deploy cap (skipped for the deploy-by-deadline guard, which
          must always win — pass ``enforce_daily_cap=False``);
        - order price must not deviate absurdly from its intended ladder level;
        - notional must clear MIN_NOTIONAL.
        """
        if self.store.is_halted():
            return RiskResult(False, "halted")
        if usdc < self.cfg.min_notional_usdc:
            return RiskResult(False, f"below min_notional ({usdc:.2f}<"
                                     f"{self.cfg.min_notional_usdc})")
        # per-day deploy cap. With resting GTC orders we cannot bound *fills*
        # (the exchange fills them whenever price hits), so we bound the new
        # resting/forced notional we COMMIT per UTC day — this is what stops the
        # agent dumping the whole budget into one candle.
        if enforce_daily_cap:
            now_ts = now if now is not None else time.time()
            committed_today = self.store.committed_today(now_ts)
            if committed_today + usdc > self.cfg.max_deploy_per_day_usdc + 1e-9:
                return RiskResult(False, f"daily deploy cap "
                                         f"({committed_today:.0f}+{usdc:.0f}>"
                                         f"{self.cfg.max_deploy_per_day_usdc:.0f})")
        # price sanity vs intended level
        if expected_level > 0:
            dev = abs(price - expected_level) / expected_level
            if dev > self.cfg.max_order_deviation_pct:
                self.store.halt(f"order price {price:,.0f} deviates {dev:.1%} from "
                                f"level {expected_level:,.0f}")
                return RiskResult(False, "price_deviation")
        return RiskResult(True)
