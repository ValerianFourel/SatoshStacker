"""Central configuration for SatoshiStacker.

Pure dataclasses + env loading. No exchange or LLM imports here so this
module (and `ladder.py`) stay importable by the backtest and tests without
network or credentials.

Bracketed defaults below mirror the spec's placeholders; override via env
or by constructing the dataclasses directly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Literal

Mode = Literal["dry_run", "testnet", "live"]
Weighting = Literal["linear", "geometric"]


def _f(env: str, default: float) -> float:
    v = os.getenv(env)
    return float(v) if v not in (None, "") else default


def _i(env: str, default: int) -> int:
    v = os.getenv(env)
    return int(v) if v not in (None, "") else default


def _b(env: str, default: bool) -> bool:
    v = os.getenv(env)
    if v in (None, ""):
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _dt(env: str, default: datetime) -> datetime:
    v = os.getenv(env)
    if v in (None, ""):
        return default
    return datetime.fromisoformat(v.strip().replace("Z", "+00:00"))


Strategy = Literal["reanchor", "single_anchor"]


@dataclass(frozen=True)
class LadderConfig:
    """Shape of the deterministic accumulation ladder.

    `budget_usdc` is split across `n_tranches` price levels from just below
    the anchor price down to `floor_price`, weighted to allocate *more*
    capital to lower rungs (the bearish tilt).
    """
    budget_usdc: float = 2_000.0
    floor_price: float = 35_000.0
    n_tranches: int = 8
    weighting: Weighting = "geometric"
    geometric_ratio: float = 1.5  # each lower rung gets ratio x the rung above
    # Fraction of a rung the LLM advisor may *defer* but never below this floor.
    min_tranche_fraction: float = 0.5
    # Production strategy (Milestone-1 backtest winner = reanchor).
    strategy: Strategy = "reanchor"
    # Re-anchoring (trailing) ladder: each generation ladders this fraction of the
    # remaining budget; re-anchors a fresh lower ladder when price breaks its floor.
    gen_fraction: float = 0.5
    max_generations: int = 6
    floor_frac: float = 0.55  # each generation's floor = anchor * floor_frac


@dataclass(frozen=True)
class RiskConfig:
    """Hard, deterministic guardrails. Fail closed."""
    max_deploy_per_day_usdc: float = 400.0
    depeg_low: float = 0.985   # halt if USDC/USD (or stablecoin px) below this
    depeg_high: float = 1.015  # ...or above this
    max_order_deviation_pct: float = 0.25  # order price vs ladder level sanity
    max_consecutive_api_failures: int = 6
    min_notional_usdc: float = 10.0  # Binance MIN_NOTIONAL; aggregate/defer below this


@dataclass(frozen=True)
class DeployScheduleConfig:
    """Deploy-by-deadline safety net.

    Any USDC still undeployed by `deploy_by` is DCA'd over the remaining
    window so a wrong floor call can't strand capital in USDC. The spine and
    this deadline ALWAYS win over the LLM advisor.
    """
    # window the accumulation runs over
    start: datetime = datetime(2026, 6, 10, tzinfo=timezone.utc)
    end: datetime = datetime(2026, 9, 10, tzinfo=timezone.utc)
    deploy_by: datetime = datetime(2026, 8, 20, tzinfo=timezone.utc)
    dca_every_hours: int = 24  # cadence of forced deadline DCA tranches


@dataclass(frozen=True)
class AnalysisConfig:
    """Qwen3.6-Plus advisory (veto/shrink ONLY). OpenAI-compatible endpoint."""
    enabled: bool = True
    base_url: str = field(default_factory=lambda: os.getenv(
        "LLM_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"))
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "qwen3.6-plus"))
    api_key_env: str = "LLM_API_KEY"
    cadence_hours: int = 4
    request_timeout_s: int = 30
    max_tokens: int = 512  # advisor emits a tiny JSON object; cap output (also avoids
                           # OpenRouter's 402 that reserves the model's full max output)
    web_search: bool = False          # off by default; sends queries off-box
    web_search_max_queries: int = 3
    web_search_timeout_s: int = 12
    min_size_multiplier: float = 0.5  # advisor can never shrink below this


@dataclass(frozen=True)
class AgentConfig:
    mode: Mode = "dry_run"
    symbol: str = "BTC/USDC"
    enable_trim: bool = False  # selling stays OFF — accumulate only
    maker_fee: float = 0.001   # Binance spot default; override per VIP/BNB tier
    taker_fee: float = 0.001
    ladder: LadderConfig = field(default_factory=LadderConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    schedule: DeployScheduleConfig = field(default_factory=DeployScheduleConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    db_path: str = "satoshistacker.db"
    poll_interval_s: int = 900   # 15m loop cadence (ladder is per-bar/4h; poll faster)
    summary_hour_utc: int = 0    # daily Telegram summary at this UTC hour
    # live trading requires this marker file (written by the backtest gate) to exist:
    backtest_gate_marker: str = "backtest/results/GATE_PASSED"

    @staticmethod
    def from_env() -> "AgentConfig":
        """Build the full config from environment (.env) with spec defaults."""
        _def = DeployScheduleConfig()
        ladder = LadderConfig(
            budget_usdc=_f("BUDGET_USDC", 2_000.0),
            floor_price=_f("FLOOR_PRICE", 35_000.0),
            n_tranches=_i("N_TRANCHES", 8),
            weighting=os.getenv("WEIGHTING", "geometric"),  # type: ignore[arg-type]
            strategy=os.getenv("STRATEGY", "reanchor"),      # type: ignore[arg-type]
            gen_fraction=_f("GEN_FRACTION", 0.5),
            floor_frac=_f("FLOOR_FRAC", 0.55),
        )
        risk = RiskConfig(
            max_deploy_per_day_usdc=_f("MAX_DEPLOY_PER_DAY_USDC", 400.0),
            min_notional_usdc=_f("MIN_NOTIONAL_USDC", 10.0),
        )
        schedule = DeployScheduleConfig(
            start=_dt("WINDOW_START", _def.start),
            end=_dt("WINDOW_END", _def.end),
            deploy_by=_dt("DEPLOY_BY", _def.deploy_by),
            dca_every_hours=_i("DCA_EVERY_HOURS", _def.dca_every_hours),
        )
        analysis = AnalysisConfig(
            enabled=_b("ANALYSIS_ENABLED", True),
            web_search=_b("ANALYSIS_WEB_SEARCH", False),
            cadence_hours=_i("ANALYSIS_CADENCE_HOURS", 4),
        )
        return AgentConfig(
            mode=os.getenv("MODE", "dry_run"),  # type: ignore[arg-type]
            symbol=os.getenv("SYMBOL", "BTC/USDC"),
            maker_fee=_f("MAKER_FEE", 0.001),
            taker_fee=_f("TAKER_FEE", 0.001),
            ladder=ladder, risk=risk, schedule=schedule, analysis=analysis,
            db_path=os.getenv("DB_PATH", "satoshistacker.db"),
            poll_interval_s=_i("POLL_INTERVAL_S", 900),
        )
