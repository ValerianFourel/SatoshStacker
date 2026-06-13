"""BTC market monitor — continuously computes microstructure + technical metrics
from PUBLIC Binance data, writes an atomic snapshot, and fires an LLM read on
out-of-the-norm events (peaks / bottoms / volume & volatility spikes).

NO keys, NO orders, NO trading — this process only observes and talks. It is run
by ``agent.btcwatch`` as a separate service from the accumulation trader.

Design:
  * ``compute_metrics`` is a PURE function (inject price/klines/book) so the metric
    math and the detector are unit-testable with zero network.
  * ``AnomalyDetector`` is stateful but deterministic: per-signal cooldown + a
    re-arm (hysteresis) flag so one episode = one alert, surviving restarts.
  * ``MarketMonitor`` does the I/O: fetch -> compute -> snapshot -> detect ->
    (on fire) analyst read -> Telegram. Analyst + notifier are injected.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .config import WatchConfig

log = logging.getLogger("satoshistacker.monitor")


# ───────────────────────── pure metric math (no I/O) ─────────────────────────

def _rsi(closes: np.ndarray, n: int = 14) -> float:
    if len(closes) < n + 1:
        return 50.0
    d = np.diff(closes[-(n + 1):])
    up, dn = d[d > 0].sum(), -d[d < 0].sum()
    return 100.0 if dn == 0 else float(100 - 100 / (1 + (up / n) / (dn / n)))


def _ema(values: np.ndarray, n: int) -> float:
    if len(values) == 0:
        return 0.0
    if len(values) < n:
        return float(values.mean())
    k = 2.0 / (n + 1)
    e = float(values[0])
    for v in values[1:]:
        e = float(v) * k + e * (1 - k)
    return e


def _atr_pct(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
             n: int = 14) -> float:
    if len(closes) < 2:
        return 0.0
    m = min(len(highs), len(lows), len(closes))
    h, l, c = highs[-m:], lows[-m:], closes[-m:]
    prev = c[:-1]
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - prev),
                                              np.abs(l[1:] - prev)))
    atr = float(tr[-n:].mean()) if len(tr) else 0.0
    px = float(c[-1]) or 1.0
    return round(atr / px * 100, 4)


def realized_vol_pct(closes: np.ndarray, n: int = 60) -> float:
    """Stdev of per-candle log returns over the last n candles, in %."""
    if len(closes) < 3:
        return 0.0
    c = closes[-(n + 1):]
    r = np.diff(np.log(c))
    return round(float(np.std(r)) * 100, 4)


def _zscore(series: np.ndarray, value: float) -> float:
    if len(series) < 5:
        return 0.0
    mu, sd = float(series.mean()), float(series.std())
    return 0.0 if sd == 0 else round((value - mu) / sd, 2)


def book_metrics(order_book: dict, *, bands_pct=(0.5, 1.0, 2.0)) -> dict:
    bids = order_book.get("bids") or []
    asks = order_book.get("asks") or []
    if not bids or not asks:
        return {"ok": False}
    best_bid, best_ask = float(bids[0][0]), float(asks[0][0])
    mid = (best_bid + best_ask) / 2
    spread_bps = round((best_ask - best_bid) / mid * 1e4, 2) if mid else 0.0
    out = {"ok": True, "mid": round(mid, 2), "best_bid": best_bid,
           "best_ask": best_ask, "spread_bps": spread_bps, "bands": {}}
    for band in bands_pct:
        lo, hi = mid * (1 - band / 100), mid * (1 + band / 100)
        bid_vol = sum(float(q) for p, q in bids if float(p) >= lo)
        ask_vol = sum(float(q) for p, q in asks if float(p) <= hi)
        tot = bid_vol + ask_vol
        imb = round((bid_vol - ask_vol) / tot, 3) if tot else 0.0
        out["bands"][str(band)] = {"bid": round(bid_vol, 3),
                                   "ask": round(ask_vol, 3), "imbalance": imb}
    # biggest single resting wall on each side (size, price)
    tb = max(bids, key=lambda x: float(x[1]))
    ta = max(asks, key=lambda x: float(x[1]))
    out["top_bid_wall"] = [float(tb[0]), round(float(tb[1]), 3)]
    out["top_ask_wall"] = [float(ta[0]), round(float(ta[1]), 3)]
    return out


def _tuned_values(klines_trend, tuned: dict) -> dict:
    """Current reading of the tuned best top/bottom oscillators (from signal_tuner)."""
    from .signal_tuner import compute_one
    a = np.array(klines_trend, dtype=float)
    o, h, l, c, vv = a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5]
    out = {}
    for side in ("best_top", "best_bottom"):
        sig = tuned.get(side)
        if not sig or sig.get("threshold") is None:
            continue
        series = compute_one(sig["name"], o, h, l, c, vv)
        val = series[~np.isnan(series)]
        if len(val):
            out["top" if side == "best_top" else "bottom"] = {
                "name": sig["name"], "value": round(float(val[-1]), 2),
                "threshold": sig["threshold"], "auc": sig.get("auc")}
    return out


def compute_metrics(*, price: float, klines_fast: list[list[float]],
                    klines_trend: list[list[float]], order_book: dict,
                    cfg: WatchConfig, now_ts: float, tuned: dict | None = None,
                    ticker24: dict | None = None, funding: dict | None = None) -> dict:
    """Build the full metric snapshot. klines rows are
    [open_time_ms, open, high, low, close, volume]. Pure — no network.
    ``ticker24`` (Binance official 24h stats) gives the accurate, time-correct 24h
    high/low/change; ``funding`` adds perp funding-rate + open-interest. ``tuned``
    adds the backtest-chosen top/bottom oscillators' current values."""
    f = np.array(klines_fast, dtype=float) if klines_fast else np.empty((0, 6))
    t = np.array(klines_trend, dtype=float) if klines_trend else np.empty((0, 6))
    fc = f[:, 4] if len(f) else np.array([price])
    fv = f[:, 5] if len(f) else np.array([0.0])
    tc = t[:, 4] if len(t) else fc
    th, tl = (t[:, 2], t[:, 3]) if len(t) else (tc, tc)

    last_vol = float(fv[-1]) if len(fv) else 0.0
    vol_window = fv[-61:-1] if len(fv) > 5 else fv  # exclude the live (partial) candle
    vol_z = _zscore(vol_window, last_vol)
    vol_mean = float(vol_window.mean()) if len(vol_window) else 0.0
    surge_x = round(last_vol / vol_mean, 2) if vol_mean else 0.0

    # short-window return + its z-score (price spike detection)
    rets = np.diff(fc) / fc[:-1] * 100 if len(fc) > 1 else np.array([0.0])
    ret_last = float(rets[-1]) if len(rets) else 0.0
    ret_z = _zscore(rets[-120:-1] if len(rets) > 5 else rets, ret_last)
    ret_5m = round(float(price / fc[-5] - 1) * 100, 3) if len(fc) >= 5 else 0.0
    ret_1h = round(float(price / fc[-60] - 1) * 100, 3) if len(fc) >= 60 else 0.0

    # TIME-ACCURATE ranges: prefer Binance's official rolling-24h ticker; else select
    # candles by their actual timestamp (NOT the whole 200-bar trend window).
    now_ms = now_ts * 1000

    def _wlh(hours):
        if len(t):
            mask = t[:, 0] >= now_ms - hours * 3600 * 1000
            if mask.any():
                return float(t[mask, 3].min()), float(t[mask, 2].max())
        return (float(tl.min()) if len(tl) else price,
                float(th.max()) if len(th) else price)

    if ticker24:
        low_24h, high_24h = ticker24["low"], ticker24["high"]
        change_24h = ticker24["change_pct"]
    else:
        low_24h, high_24h = _wlh(24)
        change_24h = round((price / tc[-24] - 1) * 100, 2) if len(tc) >= 24 else 0.0
    low_7d, high_7d = _wlh(168)
    rng = max(high_24h - low_24h, 1e-9)
    rsi = round(_rsi(tc, 14), 1)
    ema_fast, ema_slow = _ema(tc, 9), _ema(tc, 21)
    last_candle_ms = int(t[-1, 0]) if len(t) else int(now_ms)

    m = {
        "ts": now_ts,
        "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now_ts)),
        "symbol": cfg.symbol,
        "price": round(price, 2),
        "technicals": {
            "rsi_14": rsi,
            "ema_fast_9": round(ema_fast, 2),
            "ema_slow_21": round(ema_slow, 2),
            "ema_trend_pct": round((ema_fast / ema_slow - 1) * 100, 3) if ema_slow else 0.0,
            "atr_pct": _atr_pct(th, tl, tc, 14),
            "realized_vol_pct": realized_vol_pct(fc, 60),
            "ret_1m_pct": round(ret_last, 3),
            "ret_5m_pct": ret_5m,
            "ret_1h_pct": ret_1h,
            "ret_z": ret_z,
            "change_24h_pct": round(change_24h, 2),
            "high_24h": round(high_24h, 2),
            "low_24h": round(low_24h, 2),
            "pct_from_high_24h": round((price - high_24h) / high_24h * 100, 3),
            "pct_from_low_24h": round((price - low_24h) / low_24h * 100, 3),
            "range_position_pct": round((price - low_24h) / rng * 100, 1),
            "high_7d": round(high_7d, 2),
            "low_7d": round(low_7d, 2),
            "pct_from_high_7d": round((price - high_7d) / high_7d * 100, 2),
        },
        "volume": {
            "last_1m": round(last_vol, 3), "mean_1m": round(vol_mean, 3),
            "z": vol_z, "surge_x": surge_x,
        },
        "order_book": book_metrics(order_book, bands_pct=cfg.depth_bands_pct),
        "time": {
            "scan_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now_ts)),
            "last_candle_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                             time.gmtime(last_candle_ms / 1000)),
            "candle_age_s": int(now_ts - last_candle_ms / 1000),
            "trend_tf": cfg.trend_tf,
        },
        "futures": _futures_block(funding),
    }
    if tuned:
        tv = _tuned_values(klines_trend, tuned)
        if tv:
            m["tuned"] = tv
    return m


def _futures_block(funding: dict | None) -> dict:
    """Perp funding rate (per 8h + annualized) + open interest, from public_funding_oi."""
    if not funding:
        return {}
    fr = funding.get("funding_rate")
    return {
        "funding_rate_pct": round(fr * 100, 4) if fr is not None else None,
        "funding_annualized_pct": round(fr * 100 * 3 * 365, 1) if fr is not None else None,
        "open_interest": funding.get("open_interest"),
        "oi_change_24h_pct": funding.get("oi_change_24h_pct"),
        "long_short_ratio": funding.get("long_short_ratio"),
        "mark": funding.get("mark"),
    }


# ───────────────────────── anomaly detection (stateful) ──────────────────────

@dataclass
class Signal:
    name: str
    kind: str          # peak | bottom | spike | microstructure
    detail: str


@dataclass
class AnomalyDetector:
    cfg: WatchConfig
    level: "str | None" = None     # None -> raw cfg thresholds (tests); else a preset name
    muted: bool = False            # True -> no proactive alerts at all
    overrides: dict = field(default_factory=dict)  # manual per-bar overrides (preset key -> val)
    disabled: tuple = ()           # Signal.name values the user turned off (e.g. book_imbalance)
    # per-signal state: {"last_fired": ts, "armed": bool, "clear_since": ts|None}
    _state: dict = field(default_factory=dict)

    def load_state(self, d: dict) -> None:
        if isinstance(d, dict):
            self._state = d

    def dump_state(self) -> dict:
        return self._state

    def _thr(self) -> dict:
        """Active threshold set: a sensitivity preset when ``level`` is set, else the raw
        cfg fields (keeps unit tests deterministic) — with manual overrides merged on top."""
        if self.level:
            from .sensitivity import resolve
            base = dict(resolve(self.level))
        else:
            c = self.cfg
            base = {"rsi_ob": c.rsi_overbought, "rsi_os": c.rsi_oversold, "vol_z": c.vol_z_threshold,
                    "ret_z": c.ret_z_threshold, "near": c.near_extreme_pct,
                    "imb": c.imbalance_threshold, "fund": c.funding_extreme_pct, "oi": c.oi_spike_pct,
                    "ls_long": 2.0, "ls_short": 0.6,
                    "cooldown": c.alert_cooldown_s, "rearm": c.rearm_clear_s}
        for k, v in (self.overrides or {}).items():
            if k in base:
                base[k] = v
        return base

    def _conditions(self, m: dict) -> list[Signal]:
        tech, vol, book = m["technicals"], m["volume"], m.get("order_book", {})
        c, T = self.cfg, self._thr()
        out: list[Signal] = []
        near_high = tech["pct_from_high_24h"] >= -T["near"]
        near_low = tech["pct_from_low_24h"] <= T["near"]
        from .signal_tuner import _pretty, _score
        tuned = m.get("tuned", {})
        tp, bt = tuned.get("top"), tuned.get("bottom")
        # TOP: tuned backtest-winner if present, else the default RSI+near-high rule
        if tp:
            if tp["value"] >= tp["threshold"]:
                out.append(Signal("peak", "peak",
                    f"{_pretty(tp['name'])} at {tp['value']} (fires ≥{tp['threshold']}) "
                    f"— tuned top-caller, score {_score(tp.get('auc'))}"))
        elif near_high and tech["rsi_14"] >= T["rsi_ob"]:
            out.append(Signal("peak", "peak",
                f"at 24h-high, RSI {tech['rsi_14']} overbought "
                f"({tech['pct_from_high_24h']:+.2f}% from high)"))
        # BOTTOM: tuned winner if present, else default RSI+near-low rule
        if bt:
            if bt["value"] <= bt["threshold"]:
                out.append(Signal("bottom", "bottom",
                    f"{_pretty(bt['name'])} at {bt['value']} (fires ≤{bt['threshold']}) "
                    f"— tuned bottom-caller, score {_score(bt.get('auc'))}"))
        elif near_low and tech["rsi_14"] <= T["rsi_os"]:
            out.append(Signal("bottom", "bottom",
                f"at 24h-low, RSI {tech['rsi_14']} oversold "
                f"({tech['pct_from_low_24h']:+.2f}% from low)"))
        if vol["z"] >= T["vol_z"]:
            out.append(Signal("volume_spike", "spike",
                f"volume {vol['surge_x']}x normal (z={vol['z']})"))
        if tech["ret_z"] >= T["ret_z"]:
            out.append(Signal("price_spike_up", "spike",
                f"fast up-move {tech['ret_1m_pct']:+.2f}%/1m (z={tech['ret_z']})"))
        if tech["ret_z"] <= -T["ret_z"]:
            out.append(Signal("price_spike_down", "spike",
                f"fast down-move {tech['ret_1m_pct']:+.2f}%/1m (z={tech['ret_z']})"))
        if book.get("ok"):
            band = book["bands"].get(str(c.depth_bands_pct[1]), {})
            imb = band.get("imbalance", 0.0)
            if abs(imb) >= T["imb"]:
                side = "bid-heavy (support)" if imb > 0 else "ask-heavy (resistance)"
                out.append(Signal("book_imbalance", "microstructure",
                    f"order book {side}, imbalance {imb:+.2f}"))
        fut = m.get("futures", {})
        fr = fut.get("funding_rate_pct")
        if fr is not None and abs(fr) >= T["fund"]:
            who = "crowded longs paying" if fr > 0 else "crowded shorts paying"
            out.append(Signal("funding_extreme", "spike",
                f"funding {fr:+.4f}%/8h ({fut.get('funding_annualized_pct')}%/yr) — {who}"))
        oic = fut.get("oi_change_24h_pct")
        if oic is not None and abs(oic) >= T["oi"]:
            d = ("leverage building" if oic > 0 else "deleveraging / liquidations")
            out.append(Signal("oi_spike", "spike",
                f"open interest {oic:+.1f}% in 24h — {d}"))
        lsr = fut.get("long_short_ratio")
        if lsr is not None and (lsr >= T["ls_long"] or lsr <= T["ls_short"]):
            crowd = "crowded LONG (retail)" if lsr >= T["ls_long"] else "crowded SHORT (retail)"
            out.append(Signal("long_short_extreme", "spike",
                f"long/short ratio {lsr:.2f} — {crowd}"))
        if self.disabled:                       # user turned specific signals off
            out = [s for s in out if s.name not in self.disabled]
        return out

    def evaluate(self, m: dict, *, now_ts: float) -> list[Signal]:
        """Signals that should fire NOW. A signal fires only if it is armed AND its
        cooldown has elapsed; firing disarms it. It re-arms ONLY after its reading has
        been clear of the bar for ``rearm`` seconds straight — so a value hovering at the
        threshold (one flickering scan off) does NOT re-fire. One episode = one alert."""
        if self.muted:
            return []
        T = self._thr()
        cooldown, rearm = T["cooldown"], T["rearm"]
        active = {s.name: s for s in self._conditions(m)}
        fired: list[Signal] = []
        # hysteresis re-arm: require a *sustained* clear streak, not a single off-scan
        for name, st in self._state.items():
            if name in active:
                st["clear_since"] = None                       # back on the bar -> reset streak
            else:
                if st.get("clear_since") is None:
                    st["clear_since"] = now_ts                 # streak starts
                elif now_ts - st["clear_since"] >= rearm:
                    st["armed"] = True                         # clear long enough -> re-arm
        for name, sig in active.items():
            st = self._state.setdefault(name,
                                        {"last_fired": None, "armed": True, "clear_since": None})
            lf = st.get("last_fired")
            cooled = lf is None or (now_ts - lf) >= cooldown
            if st.get("armed", True) and cooled:
                fired.append(sig)
                st["last_fired"] = now_ts
                st["armed"] = False
                st["clear_since"] = None
        return fired


# ─────────────────────────────── the monitor loop ───────────────────────────

def _atomic_write_json(path: str, obj: dict) -> None:
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, separators=(",", ":"))
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class MarketMonitor:
    """Fetch -> compute -> snapshot -> detect -> (on event) analyst read -> notify.

    Fetchers/analyst/notifier are injected so the whole pipeline is testable with
    no network. Each defaults to the real public-Binance helper.
    """

    def __init__(self, cfg: WatchConfig, *, notifier, analyst=None,
                 price_fn: Callable | None = None, klines_fn: Callable | None = None,
                 book_fn: Callable | None = None, ticker_fn: Callable | None = None,
                 funding_fn: Callable | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        from . import exchange  # lazy: keeps tests import-light
        self.cfg = cfg
        self.notifier = notifier
        self.analyst = analyst
        self.clock = clock
        self.price_fn = price_fn or (lambda: exchange.public_price(cfg.symbol))
        self.klines_fn = klines_fn or (
            lambda tf, lim: exchange.public_klines(cfg.symbol, tf, lim))
        self.book_fn = book_fn or (
            lambda: exchange.public_order_book(cfg.symbol, cfg.book_limit))
        self.ticker_fn = ticker_fn or (lambda: exchange.public_ticker_24h(cfg.symbol))
        self.funding_fn = funding_fn or (lambda: exchange.public_funding_oi(cfg.symbol))
        self.detector = AnomalyDetector(cfg)
        self._prefs_mtime: float = -2.0
        self._refresh_prefs(force=True)        # sets detector.level / .muted from prefs file
        self._mtf_cache: dict = {}
        self._mtf_ts: float = 0.0
        self._oc_cache: dict = {}
        self._oc_ts: float = 0.0
        from .alerts import AlertStore
        self.alerts = AlertStore(cfg.user_alerts_path)
        self._load_state()

    def _refresh_prefs(self, *, force: bool = False) -> None:
        """Pick up live sensitivity / mute changes from the prefs file (mtime-gated, cheap)."""
        from .sensitivity import read_prefs
        path = self.cfg.prefs_path
        try:
            mt = os.path.getmtime(path)
        except OSError:
            mt = -1.0
        if not force and mt == self._prefs_mtime:
            return
        self._prefs_mtime = mt
        p = read_prefs(path, default_level=self.cfg.sensitivity)
        self.detector.level = p["sensitivity"]
        self.detector.muted = p["muted"]
        self.detector.overrides = p["overrides"]
        self.detector.disabled = tuple(p["disabled"])

    def _check_user_alerts(self, m: dict) -> None:
        """Evaluate user-defined trigger rules against the live snapshot; ping on fire."""
        from .alerts import evaluate
        rules = self.alerts.load()
        if not rules:
            return
        before = [r.get("armed", True) for r in rules]
        fired = evaluate(rules, m)
        if before != [r.get("armed", True) for r in rules]:
            self.alerts.save(rules)        # persist arm/disarm only when it changed
        for rule, val in fired:
            self.notifier.send(
                f"🔔 *Trigger fired* — `{rule['metric']} {rule['op']} {rule['value']}`"
                f"  →  now `{val:g}`   (#{rule['id']})")

    def _multi_tf(self, now: float) -> dict:
        """RSI(14) + EMA trend per 5m/1h/4h/1d so the LLM sees each timeframe by name.
        Cached ~3 min (these change slowly; keeps the per-scan loop light)."""
        if self._mtf_cache and now - self._mtf_ts < 180:
            return self._mtf_cache
        out = {}
        for tf in ("5m", "1h", "4h", "1d"):
            try:
                c = np.array(self.klines_fn(tf, 200), dtype=float)[:, 4]
                slow = _ema(c, 21)
                out[tf] = {"rsi_14": round(_rsi(c, 14), 1),
                           "ema_trend_pct": round((_ema(c, 9) / slow - 1) * 100, 2) if slow else 0.0}
            except Exception:  # noqa: BLE001 - skip a timeframe that fails
                continue
        self._mtf_cache, self._mtf_ts = out, now
        return out

    def _onchain(self, now: float) -> dict:
        """CryptoQuant on-chain metrics, cached ~4h (daily data; needs a plan that grants
        the endpoints — empty until then)."""
        if self._oc_cache and now - self._oc_ts < 14400:
            return self._oc_cache
        try:
            from .onchain import fetch_onchain
            self._oc_cache = fetch_onchain()
        except Exception:  # noqa: BLE001
            self._oc_cache = {}
        self._oc_ts = now
        return self._oc_cache

    def _load_state(self) -> None:
        self._sched: dict = {}        # {tz: last-fired YYYY-MM-DD} for the daily digest
        try:
            with open(self.cfg.state_path) as f:
                d = json.load(f)
            self.detector.load_state(d.get("detector", {}))
            self._sched = d.get("schedule", {}) or {}
        except Exception:  # noqa: BLE001 - first run / corrupt -> fresh state
            pass

    def _save_state(self) -> None:
        try:
            _atomic_write_json(self.cfg.state_path,
                               {"detector": self.detector.dump_state(),
                                "schedule": getattr(self, "_sched", {})})
        except Exception as e:  # noqa: BLE001
            log.warning("could not persist state: %s", e)

    def _maybe_daily_update(self, now_ts: float, m: dict) -> None:
        """Fire a daily briefing at ``daily_update_hour`` local time in each configured
        zone, once per day per zone (survives restarts via the schedule state)."""
        import datetime
        try:
            from zoneinfo import ZoneInfo
        except Exception:  # noqa: BLE001 - no zoneinfo -> skip scheduling
            return
        for tzname in self.cfg.daily_update_tzs:
            try:
                local = datetime.datetime.fromtimestamp(now_ts, ZoneInfo(tzname))
            except Exception:  # noqa: BLE001 - bad tz / missing tzdata -> skip
                continue
            today = local.date().isoformat()
            if local.hour == self.cfg.daily_update_hour and self._sched.get(tzname) != today:
                self._sched[tzname] = today
                self._save_state()
                self._send_daily_update(m, tzname, local)

    def _send_daily_update(self, m: dict, tzname: str, local) -> None:
        from .analyst import numeric_summary
        city = tzname.split("/")[-1].replace("_", " ")
        head = f"🗓️ *Daily BTC briefing* — 9:00 {city} ({local:%a %d %b})"
        read = ""
        if self.cfg.analyst_enabled and self.analyst is not None:
            try:
                read = self.analyst.answer(
                    "Daily BTC briefing: trend & key levels (note timeframes), funding/OI, "
                    "day/week/month sentiment, and anything notable since yesterday.", m)
            except Exception as e:  # noqa: BLE001
                log.warning("daily briefing LLM failed: %s", e)
        body = f"{head}\n{numeric_summary(m)}"
        if read:
            body += f"\n\n🧠 {read}"
        self.notifier.send(body)
        self._maybe_chart(m)          # attach a chart (LLM picks set by the answer above)

    def latest_snapshot(self) -> dict | None:
        try:
            with open(self.cfg.snapshot_path) as f:
                return json.load(f)
        except Exception:  # noqa: BLE001
            return None

    def run_once(self) -> dict:
        now = self.clock()
        self._refresh_prefs()        # apply any live /sensitivity or /mute change
        price = self.price_fn()
        fast = self.klines_fn(self.cfg.kline_tf, self.cfg.kline_limit)
        trend = self.klines_fn(self.cfg.trend_tf, self.cfg.trend_limit)
        book = self.book_fn()
        try:
            ticker24 = self.ticker_fn()
        except Exception:  # noqa: BLE001 - accurate-but-optional; fall back to candles
            ticker24 = None
        try:
            funding = self.funding_fn()
        except Exception:  # noqa: BLE001 - futures may be geo-blocked; flags just skip
            funding = None
        from .signal_tuner import load_tuned
        tuned = load_tuned(self.cfg.tuned_signals_path)
        m = compute_metrics(price=price, klines_fast=fast, klines_trend=trend,
                            order_book=book, cfg=self.cfg, now_ts=now, tuned=tuned,
                            ticker24=ticker24, funding=funding)
        m["multi_tf"] = self._multi_tf(now)        # RSI/trend per 5m/1h/4h/1d (cached)
        m["onchain"] = self._onchain(now)          # MVRV/NUPL/SOPR/netflow (cached ~4h)
        fired = self.detector.evaluate(m, now_ts=now)
        m["events"] = [s.name for s in fired]
        _atomic_write_json(self.cfg.snapshot_path, m)
        if fired:
            self._handle_events(m, fired)
            self._save_state()
        self._check_user_alerts(m)
        self._maybe_daily_update(now, m)
        return m

    _ICON = {"peak": "📈", "bottom": "📉", "spike": "⚡", "microstructure": "📖"}
    _TITLE = {"peak": "possible TOP forming", "bottom": "possible BOTTOM forming",
              "spike": "unusual move", "microstructure": "order-book shift"}

    def _handle_events(self, m: dict, fired: list[Signal]) -> None:
        log.info("anomaly fired: %s", ", ".join(s.name for s in fired))
        head = (f"🚨 *BTC alert* — {self._TITLE.get(fired[0].kind, 'unusual behaviour')}"
                f"  ·  `${m.get('price', 0):,.0f}`")
        triggers = "\n".join(f"{self._ICON.get(s.kind, '•')} {s.detail}" for s in fired)
        body = f"{head}\n{triggers}"
        if self.cfg.analyst_enabled and self.analyst is not None:
            try:
                read = self.analyst.event_read(m, fired)
                self.notifier.send(f"{body}\n\n🧠 {read}")
                self._maybe_chart(m)
                return
            except Exception as e:  # noqa: BLE001 - never let analysis break the loop
                log.warning("analyst event_read failed: %s", e)
        self.notifier.send(body)  # numeric-only fallback
        self._maybe_chart(m)

    def _maybe_chart(self, m: dict) -> None:
        """Attach a price + leading-indicator chart to the alert (if enabled)."""
        if not getattr(self.cfg, "alert_charts", False):
            return
        try:
            from .plotter import build_btc_chart
            from .signal_tuner import load_tuned
            groups = getattr(self.analyst, "last_plot", None) or [None]  # LLM's patchwork
            tuned = load_tuned(self.cfg.tuned_signals_path)
            for g in groups:
                png, cap = build_btc_chart(self.cfg, tuned, snapshot=m, indicators=g)
                if png:
                    self.notifier.send_photo(png, cap)
        except Exception as e:  # noqa: BLE001 - a chart must never break the alert
            log.warning("alert chart failed: %s", e)

    def run(self, stop) -> None:
        """Loop until ``stop`` (a threading.Event-like with .is_set/.wait) is set."""
        log.info("market monitor started: %s every %ss", self.cfg.symbol,
                 self.cfg.scan_interval_s)
        while not stop.is_set():
            try:
                self.run_once()
            except Exception as e:  # noqa: BLE001 - transient fetch errors are normal
                log.warning("monitor cycle error: %s", e)
            stop.wait(self.cfg.scan_interval_s)
