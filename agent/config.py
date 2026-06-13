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
class WatchConfig:
    """BTC market-watch service (`agent.btcwatch`) — monitor + LLM analyst + Telegram Q&A.

    Completely separate from the trading agent: public Binance data only, NO keys,
    NO orders. The LLM is read-only analysis; it may write to a sandboxed scratch
    dir but never to the repo or the exchange.
    """
    symbol: str = "BTC/USDT"              # USDT for the deepest public order book
    scan_interval_s: int = 15             # how often to refresh metrics + check anomalies
    kline_tf: str = "1m"                  # fast candles for spikes/volume/vol
    kline_limit: int = 240                # ~4h of 1m candles
    trend_tf: str = "1h"                  # slower candles for trend/RSI/24h range
    trend_limit: int = 200
    book_limit: int = 100                 # order-book depth levels to pull
    depth_bands_pct: tuple = (0.5, 1.0, 2.0)  # bid/ask depth windows around mid
    # anomaly thresholds (an event = "out of the norm" -> auto LLM read)
    rsi_overbought: float = 72.0          # peaking
    rsi_oversold: float = 28.0            # bottoming
    vol_z_threshold: float = 3.0          # volume z-score spike (std devs over recent mean)
    ret_z_threshold: float = 3.5          # short-window return z-score (price spike)
    near_extreme_pct: float = 0.25        # within this % of 24h high/low = peak/bottom test
    imbalance_threshold: float = 0.60     # |book imbalance| in [-1,1] that's notable
    alert_cooldown_s: int = 1800          # min seconds between alerts of the SAME signal
    analyst_enabled: bool = True          # call the LLM on events (else numeric-only alert)
    analyst_max_tokens: int = 450
    poll_telegram: bool = True            # run the inbound Q&A listener
    news_enabled: bool = True             # attach BTC headlines + Fear&Greed (keyless)
    news_ttl_s: int = 600                 # cache news this long (don't refetch every scan)
    web_search_enabled: bool = True       # let the analyst look things up online
    search_max_results: int = 5           # SEARXNG_URL / TAVILY_API_KEY / SERPER_API_KEY / DuckDuckGo
    fetch_articles: int = 2               # open & READ the top N results (0 = snippets only)
    article_max_chars: int = 2500         # cap extracted article text fed to the LLM
    scratch_dir: str = "scratch"          # sandbox the analyst may write to (temp files)
    snapshot_path: str = "state/btc_snapshot.json"
    state_path: str = "state/btc_watch_state.json"  # detector cooldowns survive restarts
    onboarded_marker: str = "state/btc_watch_onboarded"  # one-time onboarding tip guard

    @staticmethod
    def from_env() -> "WatchConfig":
        return WatchConfig(
            symbol=os.getenv("WATCH_SYMBOL", "BTC/USDT"),
            scan_interval_s=_i("WATCH_SCAN_INTERVAL_S", 15),
            kline_tf=os.getenv("WATCH_KLINE_TF", "1m"),
            trend_tf=os.getenv("WATCH_TREND_TF", "1h"),
            rsi_overbought=_f("WATCH_RSI_OVERBOUGHT", 72.0),
            rsi_oversold=_f("WATCH_RSI_OVERSOLD", 28.0),
            vol_z_threshold=_f("WATCH_VOL_Z", 3.0),
            ret_z_threshold=_f("WATCH_RET_Z", 3.5),
            near_extreme_pct=_f("WATCH_NEAR_EXTREME_PCT", 0.25),
            imbalance_threshold=_f("WATCH_IMBALANCE_THRESHOLD", 0.60),
            alert_cooldown_s=_i("WATCH_ALERT_COOLDOWN_S", 1800),
            analyst_enabled=_b("WATCH_ANALYST_ENABLED", True),
            poll_telegram=_b("WATCH_POLL_TELEGRAM", True),
            scratch_dir=os.getenv("WATCH_SCRATCH_DIR", "scratch"),
            news_enabled=_b("WATCH_NEWS_ENABLED", True),
            news_ttl_s=_i("WATCH_NEWS_TTL_S", 600),
            web_search_enabled=_b("WATCH_WEB_SEARCH", True),
            fetch_articles=_i("WATCH_FETCH_ARTICLES", 2),
            article_max_chars=_i("WATCH_ARTICLE_MAX_CHARS", 2500),
        )


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
