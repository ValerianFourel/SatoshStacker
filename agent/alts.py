"""Alt-stacker — a multi-asset LLM portfolio trader (SOL / ETH / HYPE), testnet first.

Sibling of the BTC ``agent/trader.py`` but for a *portfolio* sharing ONE USDC cash
pot. Each cycle the LLM is given per-asset momentum + the live pot and decides a
TARGET ALLOCATION per asset (fraction of total pot value to hold in that base);
the bot rebalances the shared pot toward those targets — buying *and* selling.

Two objectives live side by side (per ``AssetSpec.objective``):

* ``accumulate_base`` (SOL, ETH) — operator is BEARISH vs USD: the goal is to END
  with the most BASE units by buying the grind down cheap (hold cash before drops,
  rebuy lower). Benchmarked against per-asset DCA and HODL on *units accumulated*.
* ``accumulate_quote`` (HYPE) — range-stable: the goal is to GROW USDC by buying the
  low end of HYPE's range and selling the high end (do NOT net-accumulate HYPE).
  Tracked as realized USDC, benchmarked against just holding HYPE.

THE LLM HAS FULL DECISION AUTHORITY over the target allocation — but only INSIDE a
deterministic cage (``Rails``): per-asset max fraction, per-cycle & per-day turnover
caps, USDC de-peg halt, price-sanity bad-tick guard, min-notional, and a kill switch.
A broken/malformed LLM response is fail-safe: HOLD the current allocation (no churn).
Spot only — never margin/leverage. Lean: numpy + requests only.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests

from .multi_exchange import STABLE_QUOTES, MultiExchange
from .notify import Notifier
from .secrets import redact
from .trader import Lot, momentum  # reuse avg-cost lot + momentum metrics
from .tune import run_tune          # weekly technical backtest + Qwen tune (per asset)

ROOT = Path(__file__).resolve().parent.parent

# One SEQUENTIAL call per asset (so each decision + rationale is observable). The
# system prompt is tailored to the asset's objective.
_SYSTEM_BASE = (
    "You are a tactical crypto trader deciding ONE asset, {BASE}, inside a shared USDC pot. "
    "The operator is BEARISH on {BASE} vs USD and wants to END the window holding the MOST "
    "UNITS of {BASE}. You gain units by holding USDC BEFORE a fall then rebuying more units "
    "lower, and by holding {BASE} near/after bottoms and in uptrends. You are given {BASE}'s "
    "momentum (incl. a weekly-tuned RSI period + this week's tuned guidance), recent {BASE} "
    "NEWS HEADLINES, a market FEAR & GREED index (0=extreme fear often marks bottoms → favor "
    "accumulating; 100=greed often precedes pullbacks → favor cash — let momentum confirm), "
    "your current {BASE} holding and the shared cash. Decide the TARGET fraction of TOTAL POT "
    "VALUE to hold in {BASE} right now (0.0=none, rest stays USDC). Round-trip costs ~0.2% so "
    "only move decisively. Respond STRICT JSON only:\n"
    '{"target_fraction":<0..1>,"stance":"long_'  '{base}|long_cash|hold","note":"<short>"}'
)
_SYSTEM_QUOTE = (
    "You are a tactical trader deciding ONE asset, {BASE}, inside a shared USD pot. Your SOLE "
    "objective is to GROW USD — to END with more dollars than you started, NOT to accumulate "
    "{BASE}. You are spot-only (you CANNOT short). You make USD two ways: (1) be LONG {BASE} "
    "during a clear UPTREND and BANK profits into strength / when momentum rolls over; (2) buy "
    "oversold bounces likely to recover, then sell the rip. SIT IN CASH during downtrends or "
    "chop you can't read. NEVER ride a position all the way back down (no bag-holding) and never "
    "catch a falling knife. You are given {BASE}'s momentum (weekly-tuned RSI + guidance), recent "
    "{BASE} NEWS, a market crypto FEAR & GREED index (0=extreme fear, often near bottoms; "
    "100=greed, often near tops), your current {BASE} inventory and the shared cash. Decide the "
    "TARGET fraction of TOTAL POT VALUE to hold in {BASE} now (high to ride a confirmed uptrend, "
    "near 0 in a downtrend/unclear chop). Round-trip costs ~0.2-0.5% — move decisively. Respond "
    "STRICT JSON only:\n"
    '{"target_fraction":<0..1>,"stance":"ride_up|take_profit|hold_cash","note":"<short>"}'
)


_SYSTEM_GOLD = (
    "You are a tactical trader deciding ONE asset, {BASE} (TOKENIZED GOLD, ~1 token = 1oz), "
    "inside a shared USD pot. {BASE} is a SAFE-HAVEN STORE OF VALUE — the operator is "
    "LONG-TERM CONSTRUCTIVE on it as a hedge while crypto is in a bear market, and wants to "
    "END holding the MOST UNITS (ounces) of {BASE}. Accumulate on dips/weakness and hold; do "
    "not chase strength. Gold is far LESS volatile than crypto, so move gradually and keep a "
    "meaningful baseline allocation rather than flipping to all-cash. You are given {BASE}'s "
    "momentum (incl. a weekly-tuned RSI period + tuned guidance) and recent {BASE} NEWS. A "
    "market crypto FEAR & GREED index is given but is only LOOSELY relevant to gold (gold often "
    "rises when crypto fear is extreme — a flight to safety). Decide the TARGET fraction of "
    "TOTAL POT VALUE to hold in {BASE} now (rest stays USD). Respond STRICT JSON only:\n"
    '{"target_fraction":<0..1>,"stance":"accumulate|hold|trim","note":"<short>"}'
)

# objectives that hold the BASE asset for its units (vs accumulate_quote which round-trips
# the base to grow cash). Used for prompt routing and unit-based DCA/HODL benchmarks.
BASE_OBJECTIVES = ("accumulate_base", "store_of_value")


def _system_for(base: str, objective: str) -> str:
    if objective == "accumulate_quote":
        tmpl = _SYSTEM_QUOTE
    elif objective == "store_of_value":
        tmpl = _SYSTEM_GOLD
    else:
        tmpl = _SYSTEM_BASE
    return tmpl.replace("{BASE}", base).replace("{base}", base.lower())


def _safe(s: str, n: int = 24) -> str:
    """Sanitize untrusted LLM/news text before it enters a Markdown Telegram message:
    strip Markdown/URL/control chars and cap length (alert-integrity, #21)."""
    s = str(s or "")
    for ch in "`*_[]()~>#|\n\r\t":
        s = s.replace(ch, " ")
    s = " ".join(s.split())
    return s[:n]


# ───────────────────── pure sizing spine (shared by live + backtest) ─────────
# These are deterministic and import-clean (no exchange/LLM/network) so the
# backtest exercises the EXACT production sizing, not a reimplementation.
def clamp_targets(targets: dict[str, float], max_fraction_by_base: dict[str, float],
                  max_total: float) -> dict[str, float]:
    """The deterministic cage on LLM targets: clamp each to its per-asset max, then
    scale all down proportionally if their sum exceeds the total-allocation cap."""
    out = {b: max(0.0, min(float(targets.get(b, 0.0)), mx))
           for b, mx in max_fraction_by_base.items()}
    total = sum(out.values())
    if total > max_total and total > 0:
        s = max_total / total
        out = {b: v * s for b, v in out.items()}
    return out


def plan_trades(prices: dict[str, float], values: dict[str, float], cash: float,
                pot: float, targets: dict[str, float], *, rebal_band: float,
                min_trade: float, cycle_cap: float, daily_left: float,
                fee: float) -> list[tuple[str, str, float]]:
    """Decide the ordered list of (base, side, notional_usd) trades to move the pot
    toward `targets` — SELLS first (free cash), then BUYS — respecting the rebalance
    band, per-cycle and per-day turnover caps, and available cash. Pure: no I/O. Both
    the live `_rebalance` and the backtest engine call this so sizing is identical."""
    band = rebal_band * pot
    cycle_left, dleft, avail = cycle_cap, daily_left, cash
    trades: list[tuple[str, str, float]] = []
    deltas = {b: targets.get(b, 0.0) * pot - values[b] for b in prices}
    for base in sorted(prices, key=lambda b: deltas[b]):  # most-negative (sells) first
        if cycle_left < min_trade or dleft < min_trade:
            break
        delta = targets.get(base, 0.0) * pot - values[base]
        if delta < -band:  # SELL
            sell_usd = min(-delta, values[base], cycle_left, dleft)
            if sell_usd < min_trade:
                continue
            trades.append((base, "sell", sell_usd))
            avail += sell_usd * (1 - fee)
            cycle_left -= sell_usd
            dleft -= sell_usd
        elif delta > band:  # BUY
            spend = min(delta, avail, cycle_left, dleft)
            if spend < min_trade:
                continue
            trades.append((base, "buy", spend))
            avail -= spend
            cycle_left -= spend
            dleft -= spend
    return trades


# ───────────────────────────── config ──────────────────────────────────────
@dataclass(frozen=True)
class AssetSpec:
    base: str
    objective: str = "accumulate_base"   # accumulate_base | accumulate_quote
    max_fraction: float = 0.5            # rail: max fraction of pot held in this base
    symbol: str | None = None            # venue/symbol override (default base/quote)


@dataclass(frozen=True)
class Rails:
    """Hard deterministic guardrails around the LLM's decisions. Fail closed."""
    rebal_band: float = 0.08             # ignore target moves smaller than this fraction of pot
    min_trade_usdc: float = 10.0         # MIN_NOTIONAL: never emit a smaller order
    max_cycle_turnover_usdc: float = 400.0   # cap notional traded per cycle
    max_daily_turnover_usdc: float = 1000.0  # cap notional traded per UTC day
    max_total_allocation: float = 1.0    # sum of target fractions across assets
    depeg_low: float = 0.985
    depeg_high: float = 1.015
    max_price_jump_pct: float = 0.35     # reject a tick deviating this far from last close
    max_consecutive_api_failures: int = 6


@dataclass(frozen=True)
class AltConfig:
    mode: str = "dry_run"
    venue: str = "binance"               # binance | kraken (kraken lists HYPE, but vs USD only)
    quote: str = "USDC"
    stack_usdc: float = 1_000.0
    taker_fee: float = 0.001
    cycle_hours: int = 4
    assets: tuple[AssetSpec, ...] = ()
    rails: Rails = field(default_factory=Rails)
    base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", ""))
    model: str = field(default_factory=lambda: os.getenv("ALT_MODEL",
                       "qwen/qwen3.5-plus-20260420"))
    news_enabled: bool = True
    news_every_hours: int = 8            # refresh sentiment/news every N h (decisions stay 4h)
    decision_pings: bool = True
    self_tune: bool = True               # weekly per-asset technical backtest + Qwen tune
    tune_model: str = field(default_factory=lambda: os.getenv(
        "ALT_TUNE_MODEL", os.getenv("TUNE_MODEL", "qwen/qwen3.5-plus-20260420")))
    db_path: str = "alts.db"
    # live gating: an alt backtest/dry-run review gate marker, mirroring the BTC bot
    gate_marker: str = "backtest/results/ALT_GATE_PASSED"

    @property
    def bases(self) -> list[str]:
        return [a.base for a in self.assets]

    @property
    def symbol_overrides(self) -> dict[str, str]:
        return {a.base: a.symbol for a in self.assets if a.symbol}

    def spec(self, base: str) -> AssetSpec:
        for a in self.assets:
            if a.base == base:
                return a
        raise KeyError(base)

    @staticmethod
    def _parse_assets(raw: str) -> tuple[AssetSpec, ...]:
        """'SOL:accumulate_base:0.45,ETH:accumulate_base:0.45,HYPE:accumulate_quote:0.35'."""
        specs: list[AssetSpec] = []
        for tok in (raw or "").split(","):
            tok = tok.strip()
            if not tok:
                continue
            parts = [p.strip() for p in tok.split(":")]
            base = parts[0].upper()
            obj = parts[1] if len(parts) > 1 else "accumulate_base"
            mx = float(parts[2]) if len(parts) > 2 else 0.5
            sym = parts[3] if len(parts) > 3 and parts[3] else None
            specs.append(AssetSpec(base=base, objective=obj, max_fraction=mx, symbol=sym))
        return tuple(specs)

    @staticmethod
    def from_env() -> "AltConfig":
        def _f(env, d):
            v = os.getenv(env)
            return float(v) if v not in (None, "") else d

        def _b(env, d):
            v = os.getenv(env)
            return d if v in (None, "") else v.strip().lower() in ("1", "true", "yes", "on")

        assets = AltConfig._parse_assets(os.getenv(
            "ALT_ASSETS",
            # MAKE USD by intraday-scalping ZEC (accumulate_quote = grow the quote=USD, never
            # net-hold); gold = accumulate ounces (store_of_value hedge). ZEC/USD on Kraken (EU-ok).
            # History: SOL/ETH/HYPE (can't make USD on fallers), XMR & TRX/BNB dropped.
            "ZEC:accumulate_quote:0.6,PAXG:store_of_value:0.5"))
        venue = os.getenv("ALT_VENUE", "binance").strip().lower()
        # Kraken lists HYPE only vs USD; default the shared cash leg to USD there.
        quote = os.getenv("ALT_QUOTE") or ("USD" if venue == "kraken" else "USDC")
        rails = Rails(
            rebal_band=_f("ALT_REBAL_BAND", 0.08),
            min_trade_usdc=_f("ALT_MIN_TRADE_USDC", 10.0),
            max_cycle_turnover_usdc=_f("ALT_MAX_CYCLE_TURNOVER_USDC", 400.0),
            max_daily_turnover_usdc=_f("ALT_MAX_DAILY_TURNOVER_USDC", 1000.0),
            max_total_allocation=_f("ALT_MAX_TOTAL_ALLOCATION", 1.0),
            max_price_jump_pct=_f("ALT_MAX_PRICE_JUMP_PCT", 0.35),
        )
        return AltConfig(
            # independent of the BTC bots' MODE (alts live on a different venue with a
            # different lifecycle — e.g. Kraken has no testnet), so it never inherits
            # MODE=testnet from the shared .env. CLI --mode still overrides.
            mode=os.getenv("ALT_MODE", "dry_run"),
            venue=venue, quote=quote,
            stack_usdc=_f("ALT_STACK_USDC", 1_000.0),
            taker_fee=_f("TAKER_FEE", 0.001),
            cycle_hours=int(_f("ALT_CYCLE_HOURS", 4)),
            assets=assets, rails=rails,
            news_enabled=_b("NEWS_ENABLED", True),
            news_every_hours=int(_f("ALT_NEWS_EVERY_HOURS", 8)),
            decision_pings=_b("DECISION_PINGS", True),
            self_tune=_b("ALT_SELF_TUNE", True),
            db_path=os.getenv("ALT_DB_PATH", "alts.db"),
        )


# ───────────────────────────── state ───────────────────────────────────────
@dataclass
class Position:
    """A base holding with average-cost basis, sharing the agent's external cash."""
    units: float = 0.0
    cost: float = 0.0            # USDC cost basis of the units currently held
    realized_quote: float = 0.0  # cumulative realized USDC PnL (for accumulate_quote)

    @property
    def avg(self) -> float:
        return self.cost / self.units if self.units > 1e-12 else 0.0

    def value(self, px: float) -> float:
        return self.units * px

    def asdict(self) -> dict:
        return {"units": self.units, "cost": self.cost,
                "realized_quote": self.realized_quote}


def _pos(d) -> Position:
    return Position(**d) if d else Position()


class AltStore:
    """Tiny SQLite state: meta key/value + a trades log + a persisted halt."""

    def __init__(self, path: str) -> None:
        self.conn = sqlite3.connect(path)
        self.conn.execute("CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS trades(ts INT, asset TEXT, side TEXT,"
                          " usd REAL, units REAL, price REAL, coid TEXT)")
        self.conn.commit()

    def get(self, k, default=None):
        r = self.conn.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
        return json.loads(r[0]) if r else default

    def set(self, k, v):
        self.conn.execute("INSERT INTO meta(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE "
                          "SET v=excluded.v", (k, json.dumps(v)))
        self.conn.commit()

    def trade(self, asset, side, usd, units, price, coid):
        self.conn.execute("INSERT INTO trades VALUES(?,?,?,?,?,?,?)",
                          (int(time.time()), asset, side, usd, units, price, coid))
        self.conn.commit()

    def recent_trades(self, n=10):
        return self.conn.execute(
            "SELECT ts,asset,side,usd,price FROM trades ORDER BY ts DESC LIMIT ?",
            (n,)).fetchall()

    def turnover_today(self, now_ts: float) -> float:
        """Notional traded this UTC day, DERIVED from the trades log (not an
        incremental counter) so the per-day cap is crash-atomic with the order: a
        trade row is written in the same step as the fill, so a torn write can't
        evade the cap (mirrors the BTC bot's committed_today)."""
        day = int(now_ts - (now_ts % 86_400))
        r = self.conn.execute(
            "SELECT COALESCE(SUM(ABS(usd)),0) FROM trades WHERE ts>=?", (day,)).fetchone()
        return float(r[0] or 0.0)

    # ---- halt / kill switch ----
    def halt(self, reason: str) -> None:
        self.set("halted", True)
        self.set("halt_reason", reason)

    def is_halted(self) -> bool:
        return bool(self.get("halted", False))

    def clear_halt(self) -> None:
        self.set("halted", False)
        self.set("halt_reason", None)


# ─────────────────────────── helpers ───────────────────────────────────────
def market_news(bases: list[str], max_items: int = 4) -> dict:
    """Market-wide Fear&Greed + a few per-asset headlines (Yahoo RSS). Fail-safe."""
    import xml.etree.ElementTree as ET
    out: dict = {"fear_greed": None, "headlines": {}}
    try:
        d = requests.get("https://api.alternative.me/fng/", params={"limit": 1},
                         timeout=8).json()["data"][0]
        out["fear_greed"] = {"value": int(d["value"]), "label": d["value_classification"]}
    except Exception:  # noqa: BLE001 - sentiment is advisory only
        pass
    for base in bases:
        try:
            r = requests.get("https://feeds.finance.yahoo.com/rss/2.0/headline",
                             params={"s": f"{base}-USD", "region": "US", "lang": "en-US"},
                             headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
            root = ET.fromstring(r.text)
            out["headlines"][base] = [(it.findtext("title") or "").strip()
                                      for it in root.findall(".//item")[:max_items]
                                      if it.findtext("title")]
        except Exception:  # noqa: BLE001
            out["headlines"][base] = []
    return out


def _ohlcv(symbol: str) -> np.ndarray:
    from .exchange import public_ohlcv
    return np.array(public_ohlcv(symbol, "4h", 200))


def _tech_path(base: str) -> Path:
    """Per-asset tuned-technicals file, e.g. agent/technicals_SOL.json."""
    return ROOT / "agent" / f"technicals_{base}.json"


def load_tuned(base: str) -> tuple[int, str]:
    """(rsi_period, context_note) from the per-asset tune file; (14, '') if absent."""
    try:
        d = json.loads(_tech_path(base).read_text())
        return int(d.get("suggested", {}).get("rsi_period", 14)), d.get("context_note", "")
    except Exception:  # noqa: BLE001
        return 14, ""


# ─────────────────────────── the agent ─────────────────────────────────────
class AltStacker:
    def __init__(self, *, cfg: AltConfig, exchange: MultiExchange, store: AltStore,
                 notifier: Notifier, llm_client, ohlcv_source=None) -> None:
        self.cfg = cfg
        self.ex = exchange
        self.store = store
        self.notify = notifier
        self.client = llm_client
        self._ohlcv = ohlcv_source or _ohlcv
        # weekly-tuned (rsi_period, context_note) per asset; refreshed by _maybe_tune
        self.tuned: dict[str, tuple[int, str]] = {b: load_tuned(b) for b in cfg.bases}

    # ---- shared cash ----
    def _cash(self) -> float:
        return self.ex.cash()

    # ---- sentiment cache: decisions run every 4h, but news refreshes every news_every_hours ----
    def _cached_news(self, now: datetime) -> dict:
        if not self.cfg.news_enabled:
            return {}
        ts = float(self.store.get("news_cache_ts", 0) or 0)
        if (now.timestamp() - ts) >= self.cfg.news_every_hours * 3600:
            try:
                news = market_news(self.cfg.bases)
            except Exception as e:  # noqa: BLE001 - news is advisory; reuse last on failure
                self.store.set("last_error", redact(e))
                return self.store.get("news_cache", {}) or {}
            self.store.set("news_cache", news)
            self.store.set("news_cache_ts", now.timestamp())
            return news
        return self.store.get("news_cache", {}) or {}

    def _symbol(self, base: str) -> str:
        spec = self.cfg.spec(base)
        return spec.symbol or f"{base}/{self.cfg.quote}"

    # ---- LLM decision: ONE SEQUENTIAL CALL PER ASSET ----
    def decide_asset(self, base: str, feats: dict,
                     current: float) -> tuple[float, str, str]:
        """One LLM call for a single asset. Returns (target_fraction, stance, note).
        Fail-safe: HOLD the current fraction on any error/missing client."""
        if self.client is None:
            return current, "no-llm", "hold (no LLM client)"
        spec = self.cfg.spec(base)
        sysmsg = _system_for(base, spec.objective)
        _, ctx = self.tuned.get(base, (14, ""))
        if ctx:
            sysmsg += f"\n\nTHIS WEEK'S TUNED GUIDANCE for {base}: {ctx}"
        try:
            r = self.client.chat.completions.create(
                model=self.cfg.model, temperature=0.0, timeout=30, max_tokens=400,
                messages=[{"role": "system", "content": sysmsg},
                          {"role": "user", "content": json.dumps(feats)}])
            t = r.choices[0].message.content or ""
            i, j = t.find("{"), t.rfind("}")
            d = json.loads(t[i:j + 1]) if 0 <= i < j else {}
            raw = d.get("target_fraction", current)
            # FAIL-SAFE: a bool / non-numeric / NaN / inf target must HOLD (return
            # current), NOT silently clamp to 1.0 (min(1.0, nan) -> 1.0 = full buy).
            if isinstance(raw, bool):
                return current, "error", "non-numeric target (bool): hold"
            try:
                v = float(raw)
            except (TypeError, ValueError):
                return current, "error", "non-numeric target: hold"
            import math
            if not math.isfinite(v):
                return current, "error", "non-finite target: hold"
            tgt = max(0.0, min(1.0, v))
            return tgt, str(d.get("stance", "?"))[:40], str(d.get("note", ""))[:80]
        except Exception as e:  # noqa: BLE001 - keep this asset's allocation on any error
            return current, "error", f"{type(e).__name__}: hold"

    # ---- the deterministic cage applied to LLM targets ----
    def _clamp_targets(self, targets: dict[str, float]) -> dict[str, float]:
        """Per-asset max-fraction clamp + total-allocation clamp (proportional)."""
        max_by_base = {b: self.cfg.spec(b).max_fraction for b in self.cfg.bases}
        return clamp_targets(targets, max_by_base, self.cfg.rails.max_total_allocation)

    def _daily_turnover_used(self, now: datetime) -> float:
        # derived from the trades log (crash-atomic), not an evadable counter
        return self.store.turnover_today(now.timestamp())

    # ---- reconcile: EXCHANGE = SOURCE OF TRUTH for held units ----
    def _reconcile_positions(self, prices: dict[str, float]) -> list[str]:
        """Sync each local pos:{b}.units to the exchange's ACTUAL base balance before
        any decision/trade. This is the crash-safety backstop: if the process died
        between an irreversible market order and persisting the position, the next
        cycle re-reads the truth from the exchange, so it can never double-buy/double-
        sell an already-executed allocation. Cost basis (our private accounting the
        exchange doesn't know) is reconstructed best-effort. Assumes a DEDICATED
        account/sub-account for the alt-stacker (its base balances are its positions)."""
        changes: list[str] = []
        try:
            bal = self.ex.balances()
        except Exception as e:  # noqa: BLE001 - transient: keep local view this cycle
            self.store.set("last_error", redact(e))
            return changes
        for b in self.cfg.bases:
            pos = _pos(self.store.get(f"pos:{b}"))
            actual = float(bal.get(b, 0.0) or 0.0)
            tol = max(1e-9, 1e-6 * max(actual, pos.units))
            if abs(actual - pos.units) <= tol:
                continue
            px = prices.get(b) or float(self.store.get(f"last_price:{b}", 0.0) or 0.0)
            if actual < pos.units:                 # a sell we didn't record: keep avg
                pos.cost = pos.avg * actual
            elif px:                               # a buy we didn't record: add at px
                pos.cost = pos.cost + (actual - pos.units) * px
            pos.units = max(0.0, actual)
            self.store.set(f"pos:{b}", pos.asdict())
            changes.append(f"{b}: pos {pos.units:.6f} (reconciled to exchange)")
        if changes:
            self.store.set("last_reconcile", changes)
        return changes

    # ---- one cycle ----
    def run_cycle(self, now: datetime | None = None) -> dict:
        now = now or datetime.now(timezone.utc)
        cfg, rails = self.cfg, self.cfg.rails

        if self.store.is_halted():
            return {"halted": self.store.get("halt_reason")}

        degraded = False  # any price/order/peg failure this cycle -> bad-cycle counter

        # 1) market data per asset (price + momentum); price-sanity bad-tick guard
        prices: dict[str, float] = {}
        moms: dict[str, dict] = {}
        bad: list[str] = []
        for base in cfg.bases:
            try:
                px = self.ex.price(base)
                ohlc = self._ohlcv(self._symbol(base))
                closes, highs = ohlc[:, 3], ohlc[:, 1]
                last_close = float(closes[-1])
                if last_close > 0 and abs(px / last_close - 1) > rails.max_price_jump_pct:
                    bad.append(base)
                    continue
                prices[base] = px
                self.store.set(f"last_price:{base}", px)
                rsi_n, _ = self.tuned.get(base, (14, ""))
                moms[base] = momentum(closes, highs, px, rsi_n)
            except Exception as e:  # noqa: BLE001 - skip this asset this cycle
                degraded = True
                self.store.set("last_error", redact(e))
                continue
        if bad:                                  # surface bad ticks to the operator (#19)
            warn = "bad ticks skipped: " + ",".join(bad)
            if self.store.get("last_bad_warn") != warn:
                self.notify.send(f"⚠️ alt: {warn} (price deviates >"
                                 f"{rails.max_price_jump_pct:.0%} from last 4h close)")
                self.store.set("last_bad_warn", warn)
        if not prices:
            self._note_cycle(not degraded)
            return {"error": "no usable prices this cycle", "bad_ticks": bad}

        # 2) de-peg halt — only for a STABLE quote (fiat USD on Kraken carries no peg risk).
        #    FAIL CLOSED: if the peg can't be verified, skip trading this cycle (#17).
        if cfg.quote in STABLE_QUOTES:
            try:
                peg = self.ex.usd_quote_estimate()
            except Exception as e:  # noqa: BLE001 - cannot verify -> do NOT trade blind
                degraded = True
                self.store.set("last_error", redact(e))
                self._note_cycle(False)
                return {"skipped": "depeg_estimate_failed", "bad_ticks": bad}
            if not (rails.depeg_low <= peg <= rails.depeg_high):
                self.store.halt(f"{cfg.quote} de-peg: {cfg.quote}/USD={peg:.4f}")
                self.notify.send(f"🛑 HALT (alt): {cfg.quote} de-peg {peg:.4f}")
                return {"halted": "depeg", "peg": peg}

        # 3) reconcile (EXCHANGE = source of truth), weekly tune, seed
        self._reconcile_positions(prices)
        self._maybe_tune(now)
        self._maybe_seed(now, prices)

        # 4) read state + market-wide sentiment. Pot includes the value of assets HELD but
        #    skipped this cycle (bad tick / fetch fail) at their last price, so a skip can't
        #    shrink the other assets' dollar targets and force-sell them (#11).
        positions = {b: _pos(self.store.get(f"pos:{b}")) for b in cfg.bases}
        cash = self._cash()
        held_skipped = sum(
            positions[b].units * float(self.store.get(f"last_price:{b}", 0.0) or 0.0)
            for b in cfg.bases if b not in prices)
        pot = cash + sum(positions[b].value(prices[b]) for b in prices) + held_skipped
        cur_frac = {b: (positions[b].value(prices[b]) / pot if pot > 0 else 0.0)
                    for b in prices}
        news = self._cached_news(now)        # refreshed every news_every_hours (not every 4h cycle)
        fg = (news or {}).get("fear_greed")
        headlines = (news or {}).get("headlines", {}) or {}

        # 5) ONE SEQUENTIAL LLM CALL PER ASSET (each decision observable) -> cage clamps
        decisions: dict[str, dict] = {}
        for base in prices:
            spec = cfg.spec(base)
            rsi_n, _ = self.tuned.get(base, (14, ""))
            feats_b = {
                "base": base, "objective": spec.objective,
                "max_fraction": spec.max_fraction,
                "pot_usd": round(pot, 2), "cash_usdc": round(cash, 2),
                "cash_fraction": round(cash / pot, 3) if pot > 0 else 1.0,
                "fear_greed": fg, "rsi_period": rsi_n,
                "price": round(prices[base], 4),
                "units": round(positions[base].units, 6),
                "value_usd": round(positions[base].value(prices[base]), 2),
                "avg_entry": (round(positions[base].avg, 4)
                              if positions[base].units > 0 else None),
                "fraction_now": round(cur_frac[base], 3),
                **moms[base],
                "headlines": headlines.get(base, []),
            }
            tgt, stance, note = self.decide_asset(base, feats_b, cur_frac[base])
            decisions[base] = {"raw_target": tgt, "stance": stance, "note": note}

        targets = self._clamp_targets({b: decisions[b]["raw_target"] for b in prices})
        for b in prices:
            decisions[b]["target"] = round(targets[b], 3)

        # 6) rebalance the shared pot toward targets (SELLS first to free cash, then BUYS),
        #    capped by per-cycle and per-day turnover.
        actions, order_failures = self._rebalance(now, prices, positions, pot, targets)
        degraded = degraded or order_failures > 0

        # 7) per-cycle health (one increment/alert per cycle), benchmarks, reports, ping
        self._note_cycle(not degraded)
        self._step_benchmarks(prices)
        self._maybe_reports(now, prices)
        next_at = now + timedelta(hours=cfg.cycle_hours)
        if cfg.decision_pings:
            tg = " ".join(f"{b}:{_safe(decisions[b]['stance'])}→{int(round(targets[b]*100))}%"
                          for b in prices)
            self.notify.send(f"🤖 {now:%H:%M}Z alt · [{tg}] · "
                             f"{', '.join(actions) or 'hold'} · next in {cfg.cycle_hours}h")
        return {"ts": now.isoformat(), "pot": round(pot),
                "decisions": decisions, "actions": actions, "bad_ticks": bad,
                "next_decision_at": next_at.strftime("%Y-%m-%dT%H:%M:%SZ")}

    # ---- rebalance core ----
    def _coid(self, now, base: str, side: str) -> str:
        """Deterministic clientOrderId per (asset, side, cycle-bucket) so a retry within
        the same cycle reuses the exact id (idempotent on the exchange); cross-cycle
        double-trades are prevented by _reconcile_positions reading the exchange first."""
        bucket = int(now.timestamp() // max(1, self.cfg.cycle_hours * 3600))
        return f"alt-{base}-{side}-{bucket}"

    def _rebalance(self, now, prices, positions, pot, targets) -> tuple[list[str], int]:
        rails = self.cfg.rails
        fee = self.cfg.taker_fee
        values = {b: positions[b].value(prices[b]) for b in prices}
        daily_left = max(0.0, rails.max_daily_turnover_usdc - self._daily_turnover_used(now))
        # SHARED SPINE: decide what to trade (sells-first, caps, band) deterministically.
        plan = plan_trades(prices, values, self._cash(), pot, targets,
                           rebal_band=rails.rebal_band, min_trade=rails.min_trade_usdc,
                           cycle_cap=rails.max_cycle_turnover_usdc, daily_left=daily_left,
                           fee=fee)
        actions: list[str] = []
        failures = 0
        for base, side, notional in plan:
            px, pos = prices[base], positions[base]
            if side == "sell":
                want_units = min(notional / px, pos.units)
                order = self._execute(now, base, "sell", want_units, px)
                if order is None:
                    failures += 1
                    continue
                # use the ACTUAL filled amount/proceeds, not the requested size (#4)
                units = float(order.get("amount") or order.get("filled") or want_units)
                proceeds = float(order.get("cost") or units * px * (1 - fee))
                traded = units * px
                cost_removed = pos.avg * units
                pos.realized_quote += proceeds - cost_removed
                pos.cost = max(0.0, pos.cost - cost_removed)
                pos.units = max(0.0, pos.units - units)
                self.store.set(f"pos:{base}", pos.asdict())
                self.store.trade(base, "sell", traded, units, px,
                                 self._coid(now, base, "sell"))
                actions.append(f"SELL {base} ${traded:,.0f}")
            else:  # buy — re-cap by live cash so a fill drift can't overspend
                spend = min(notional, self._cash())
                if spend < rails.min_trade_usdc:
                    continue
                order = self._execute(now, base, "buy", spend, px)
                if order is None:
                    failures += 1
                    continue
                units = float(order.get("amount") or order.get("filled")
                              or spend * (1 - fee) / px)
                cost = float(order.get("cost") or spend)
                pos.units += units
                pos.cost += cost
                self.store.set(f"pos:{base}", pos.asdict())
                self.store.trade(base, "buy", cost, units, px, self._coid(now, base, "buy"))
                actions.append(f"BUY {base} ${cost:,.0f}")
        return actions, failures

    def _execute(self, now, base, side, amount, px) -> dict | None:
        """Place a real spot market order. Idempotent by deterministic coid (a same-bucket
        retry returns the prior order). Returns the order dict, or None on failure."""
        coid = self._coid(now, base, side)
        try:
            if side == "buy":
                return self.ex.market_buy_quote(base, amount, coid)   # amount = USDC to spend
            return self.ex.market_sell(base, amount, coid)            # amount = base units
        except Exception as e:  # noqa: BLE001 - leave state unchanged; retry next cycle
            self.store.set("last_error", redact(e))
            return None

    # ---- per-cycle health: one increment + one alert per degraded cycle (#9,#16,#18) ----
    def _note_cycle(self, clean: bool) -> None:
        n = 0 if clean else int(self.store.get("consec_bad_cycles", 0)) + 1
        self.store.set("consec_bad_cycles", n)
        if not clean and n == self.cfg.rails.max_consecutive_api_failures:
            self.store.halt(f"{n} consecutive degraded cycles (price/order/peg failures)")
            self.notify.send(f"🛑 HALT (alt): {n} consecutive degraded cycles — manual "
                             f"--clear-halt required")

    # ---- weekly per-asset technical backtest + tune ----
    def _maybe_tune(self, now) -> None:
        """Backtest which technicals best predict each asset's returns and let the
        smart Qwen model pick RSI period / regime — at startup and once a week, per
        asset. Writes agent/technicals_<BASE>.json and reloads it. Fail-safe: any
        error leaves that asset on its previous (or default RSI-14) technicals."""
        if not self.cfg.self_tune:
            return
        wk = now.strftime("%Y-W%U")
        if self.store.get("last_tune") == wk:
            return
        summary = []
        for base in self.cfg.bases:
            res = run_tune(self._symbol(base), self.cfg.tune_model,
                           out_path=_tech_path(base))
            if res.get("error"):
                self.store.set(f"tune_error:{base}", res["error"])
                continue
            self.tuned[base] = load_tuned(base)               # reload freshly-written config
            sug, ld = res.get("suggested", {}), res.get("leader", {})
            summary.append(f"{base}: RSI{sug.get('rsi_period', '?')}/"
                           f"{sug.get('regime', '?')} (lead {ld.get('indicator', '?')} "
                           f"IC {ld.get('ic', 0):+.2f})")
        self.store.set("last_tune", wk)
        if summary:
            self.notify.send("🔧 alt self-tune (weekly technicals):\n  " +
                             "\n  ".join(summary))

    # ---- seeding + benchmarks ----
    def _maybe_seed(self, now, prices) -> None:
        if self.store.get("seeded"):
            return
        n = max(1, len(self.cfg.bases))
        slice_ = self.cfg.stack_usdc / n
        cycles_per_window = max(1, int(30 * 24 / self.cfg.cycle_hours))  # ~30-day DCA
        for b in self.cfg.bases:
            self.store.set(f"pos:{b}", Position().asdict())
            if b in prices:
                hodl = Lot(usdc=slice_); hodl.buy(slice_, prices[b])
                self.store.set(f"hodl:{b}", hodl.asdict())
                self.store.set(f"dca:{b}", Lot(usdc=slice_).asdict())
        self.store.set("dca_slice", slice_ / cycles_per_window)
        self.store.set("start", {"ts": now.isoformat(),
                                 "prices": {b: prices[b] for b in prices}})
        self.store.set("seeded", True)
        self.notify.send(
            f"🚀 alt-stacker started [{self.cfg.mode}] — ${self.cfg.stack_usdc:,.0f} pot "
            f"across {', '.join(self.cfg.bases)} (quote {self.cfg.quote})")

    def _step_benchmarks(self, prices) -> None:
        slice_dca = float(self.store.get("dca_slice", 0) or 0)
        for b in self.cfg.bases:
            if b not in prices:
                continue
            dca = Lot(**self.store.get(f"dca:{b}", Lot().asdict()))
            if dca.usdc > 0 and slice_dca > 0:
                dca.buy(min(slice_dca, dca.usdc), prices[b])
                self.store.set(f"dca:{b}", dca.asdict())

    # ---- reports ----
    def _snapshot(self, prices) -> dict:
        positions = {b: _pos(self.store.get(f"pos:{b}")) for b in self.cfg.bases}
        cash = self._cash()
        pot = cash + sum(positions[b].value(prices[b]) for b in prices)
        return {"positions": positions, "cash": cash, "pot": pot}

    def _report(self, tag: str, prices) -> None:
        snap = self._snapshot(prices)
        lines = [f"{tag} *alt-stacker* — pot ${snap['pot']:,.0f} "
                 f"(cash ${snap['cash']:,.0f})"]
        for b in self.cfg.bases:
            if b not in prices:
                continue
            spec = self.cfg.spec(b)
            pos = snap["positions"][b]
            px = prices[b]
            if spec.objective != "accumulate_quote":   # base + store_of_value: units matter
                dca = Lot(**self.store.get(f"dca:{b}", Lot().asdict()))
                hodl = Lot(**self.store.get(f"hodl:{b}", Lot().asdict()))
                beat = pos.units >= dca.btc and pos.units >= hodl.btc
                ae = f"${pos.avg:,.2f}" if pos.units > 0 else "n/a"
                lines.append(
                    f"  {b}: {pos.units:.4f} u (avg {ae})  "
                    f"DCA {dca.btc:.4f}  HODL {hodl.btc:.4f}  "
                    f"{'✅' if beat else '⚠️'}")
            else:  # accumulate_quote (HYPE): realized USDC vs holding HYPE
                hodl = Lot(**self.store.get(f"hodl:{b}", Lot().asdict()))
                start_px = (self.store.get("start", {}).get("prices", {}) or {}).get(b, px)
                hodl_pnl = hodl.btc * (px - start_px)
                lines.append(
                    f"  {b}: realized ${pos.realized_quote:+,.2f} USDC  "
                    f"(hold-HYPE PnL ${hodl_pnl:+,.2f})  inv {pos.units:.4f} u")
        self.notify.send("\n".join(lines))

    def _maybe_reports(self, now, prices) -> None:
        today = now.strftime("%Y-%m-%d")
        if self.store.get("last_daily") != today:
            self._report("📅 Daily", prices); self.store.set("last_daily", today)
        wk = now.strftime("%Y-W%U")
        if self.store.get("last_weekly") != wk:
            self._report("🗓️ Weekly", prices); self.store.set("last_weekly", wk)

    def daily_report(self) -> None:
        prices = {b: self.ex.price(b) for b in self.cfg.bases}
        self._report("📅 Daily", prices)
