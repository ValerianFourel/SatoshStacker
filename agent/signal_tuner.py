"""Backtest a BATTERY of momentum/oscillator metrics over the past weeks to find which
best mark local TOPS and BOTTOMS, and write the winners to agent/watch_signals.json so
the live monitor uses them onward.

Read-only research — this never trades; it only tunes the watch service's anomaly
detector (which itself only describes the market). Pure numpy; data is injected so the
scoring is unit-testable with no network.

Method
------
* Compute a battery of oscillators on the candles (each normalized so HIGHER = more
  overbought / topside).
* Label ground-truth swing tops/bottoms: i is a TOP if its high is the max in ±swing_w,
  a BOTTOM if its low is the min in ±swing_w.
* Score each oscillator as a TOP detector (does a high reading coincide, within ±tol
  bars, with a real top?) and as a BOTTOM detector (low reading near a real bottom),
  using rank-AUC (threshold-free) plus the best-F1 threshold.
* Rank; the top-AUC and bottom-AUC winners (name, period, threshold) are written out and
  picked up live by the monitor.
"""
from __future__ import annotations

import json
import logging
import os
import re
import tempfile

import numpy as np

log = logging.getLogger("satoshistacker.tuner")
SIGNALS_PATH = "agent/watch_signals.json"


# ───────────────────────────── oscillator battery ───────────────────────────
# Each fn takes (o,h,l,c,v) arrays and returns an array aligned to c, where a HIGHER
# value means more overbought (topside). Warmup positions are NaN.

def _rsi(c, n):
    out = np.full(len(c), np.nan)
    d = np.diff(c)
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    for i in range(n, len(c)):
        au, ad = up[i - n:i].mean(), dn[i - n:i].mean()
        out[i] = 100.0 if ad == 0 else 100 - 100 / (1 + au / ad)
    return out


def _stoch_k(c, h, l, n):
    out = np.full(len(c), np.nan)
    for i in range(n, len(c)):
        hh, ll = h[i - n + 1:i + 1].max(), l[i - n + 1:i + 1].min()
        out[i] = 50.0 if hh == ll else 100 * (c[i] - ll) / (hh - ll)
    return out


def _williams(c, h, l, n):
    k = _stoch_k(c, h, l, n)        # %R = %K - 100, same orientation after shift
    return k                        # already 0..100, higher = overbought


def _cci(c, h, l, n):
    out = np.full(len(c), np.nan)
    tp = (h + l + c) / 3
    for i in range(n, len(c)):
        w = tp[i - n + 1:i + 1]
        md = np.abs(w - w.mean()).mean()
        out[i] = 0.0 if md == 0 else (tp[i] - w.mean()) / (0.015 * md)
    return out


def _roc(c, n):
    out = np.full(len(c), np.nan)
    out[n:] = (c[n:] / c[:-n] - 1) * 100
    return out


def _mfi(c, h, l, v, n):
    out = np.full(len(c), np.nan)
    tp = (h + l + c) / 3
    rmf = tp * v
    for i in range(n, len(c)):
        pos = rmf[i - n + 1:i + 1][np.diff(tp[i - n:i + 1]) > 0].sum()
        neg = rmf[i - n + 1:i + 1][np.diff(tp[i - n:i + 1]) < 0].sum()
        out[i] = 100.0 if neg == 0 else 100 - 100 / (1 + pos / neg)
    return out


def _ema(c, n):
    k = 2 / (n + 1)
    e = np.full(len(c), np.nan)
    e[0] = c[0]
    for i in range(1, len(c)):
        e[i] = c[i] * k + e[i - 1] * (1 - k)
    return e


def _sma(x, n):
    out = np.full(len(x), np.nan)
    cs = np.cumsum(np.insert(np.nan_to_num(x), 0, 0.0))
    out[n - 1:] = (cs[n:] - cs[:-n]) / n
    return out


def _true_range(h, l, c):
    tr = np.empty(len(c))
    tr[0] = h[0] - l[0]
    tr[1:] = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]),
                                                  np.abs(l[1:] - c[:-1])))
    return tr


def _atr(h, l, c, n):
    return _sma(_true_range(h, l, c), n)


def _ema_cross(c, a, b):              # momentum/trend: higher = fast above slow
    return (_ema(c, a) - _ema(c, b)) / _ema(c, b) * 100


def _stoch_rsi(c, n):                 # stochastic of RSI — twitchier, crypto-popular
    r = _rsi(c, n)
    out = np.full(len(c), np.nan)
    for i in range(2 * n, len(c)):
        w = r[i - n + 1:i + 1]
        lo, hi = np.nanmin(w), np.nanmax(w)
        out[i] = 50.0 if hi == lo else 100 * (r[i] - lo) / (hi - lo)
    return out


def _macd_hist(c):                    # MACD histogram (12/26/9); higher = bullish thrust
    macd = _ema(c, 12) - _ema(c, 26)
    return macd - _ema(macd, 9)


def _bb_pctb(c, n):                   # Bollinger %B: >1 above upper band (stretched up)
    mid, out = _sma(c, n), np.full(len(c), np.nan)
    for i in range(n - 1, len(c)):
        sd = c[i - n + 1:i + 1].std()
        out[i] = 0.5 if sd == 0 else (c[i] - (mid[i] - 2 * sd)) / (4 * sd)
    return out * 100


def _bb_width(c, n):                  # Bollinger band width (volatility squeeze gauge)
    mid, out = _sma(c, n), np.full(len(c), np.nan)
    for i in range(n - 1, len(c)):
        sd = c[i - n + 1:i + 1].std()
        out[i] = 0.0 if mid[i] == 0 else 4 * sd / mid[i] * 100
    return out


def _keltner_pos(h, l, c, n):         # position vs Keltner channel (EMA ± ATR)
    e, a = _ema(c, n), _atr(h, l, c, n)
    return np.where(a == 0, 0.0, (c - e) / a)


def _atr_pct(h, l, c, n):
    a = _atr(h, l, c, n)
    return np.where(c == 0, 0.0, a / c * 100)


def _sma_dist(c, n):                  # % distance above/below the SMA (e.g. 200)
    s = _sma(c, n)
    return np.where(s == 0, np.nan, (c - s) / s * 100)


def _supertrend_pos(h, l, c, n=10, mult=3.0):
    a = _atr(h, l, c, n)
    return np.where(a == 0, 0.0, (c - _sma(c, n)) / a)   # ATR-normalized trend position


def _obv_slope(c, v, n):              # On-Balance Volume momentum (accumulation)
    obv = np.zeros(len(c))
    for i in range(1, len(c)):
        obv[i] = obv[i - 1] + (v[i] if c[i] > c[i - 1] else -v[i] if c[i] < c[i - 1] else 0)
    out = np.full(len(c), np.nan)
    out[n:] = (obv[n:] - obv[:-n]) / (np.abs(obv[:-n]) + 1e-9)
    return out


def _vwap_dist(h, l, c, v, n):        # % distance from rolling VWAP
    tp = (h + l + c) / 3
    pv, out = tp * v, np.full(len(c), np.nan)
    for i in range(n - 1, len(c)):
        vol = v[i - n + 1:i + 1].sum()
        if vol > 0:
            vwap = pv[i - n + 1:i + 1].sum() / vol
            out[i] = (c[i] - vwap) / vwap * 100
    return out


def _cvd_slope(o, c, v, n):           # cumulative volume delta proxy (buy vs sell pressure)
    delta = np.where(c > o, v, np.where(c < o, -v, 0.0))
    cvd = np.cumsum(delta)
    out = np.full(len(c), np.nan)
    out[n:] = (cvd[n:] - cvd[:-n]) / (np.abs(cvd[:-n]) + 1e-9)
    return out


# Full battery, grouped by family (combine ACROSS families, per the operator's note).
def battery(o, h, l, c, v) -> dict:
    return {
        # momentum / oscillators
        "rsi_14": _rsi(c, 14), "rsi_21": _rsi(c, 21),
        "stoch_rsi_14": _stoch_rsi(c, 14), "stoch_14": _stoch_k(c, h, l, 14),
        "macd_hist": _macd_hist(c), "cci_20": _cci(c, h, l, 20),
        "williams_14": _williams(c, h, l, 14), "roc_10": _roc(c, 10), "roc_20": _roc(c, 20),
        # volatility
        "bb_pctb_20": _bb_pctb(c, 20), "bb_width_20": _bb_width(c, 20),
        "keltner_pos_20": _keltner_pos(h, l, c, 20), "atr_pct_14": _atr_pct(h, l, c, 14),
        # trend
        "ema_cross_21_50": _ema_cross(c, 21, 50), "ema_cross_50_200": _ema_cross(c, 50, 200),
        "sma_dist_200": _sma_dist(c, 200), "supertrend_10": _supertrend_pos(h, l, c, 10),
        # volume
        "mfi_14": _mfi(c, h, l, v, 14), "obv_slope_14": _obv_slope(c, v, 14),
        "vwap_dist_20": _vwap_dist(h, l, c, v, 20), "cvd_slope_14": _cvd_slope(o, c, v, 14),
    }


FAMILY = {
    "rsi_14": "momentum", "rsi_21": "momentum", "stoch_rsi_14": "momentum",
    "stoch_14": "momentum", "macd_hist": "momentum", "cci_20": "momentum",
    "williams_14": "momentum", "roc_10": "momentum", "roc_20": "momentum",
    "bb_pctb_20": "volatility", "bb_width_20": "volatility",
    "keltner_pos_20": "volatility", "atr_pct_14": "volatility",
    "ema_cross_21_50": "trend", "ema_cross_50_200": "trend",
    "sma_dist_200": "trend", "supertrend_10": "trend",
    "mfi_14": "volume", "obv_slope_14": "volume", "vwap_dist_20": "volume",
    "cvd_slope_14": "volume",
}
PERIODS = {"rsi_14": 14, "rsi_21": 21, "stoch_rsi_14": 14, "stoch_14": 14,
           "macd_hist": 26, "cci_20": 20, "williams_14": 14, "roc_10": 10, "roc_20": 20,
           "bb_pctb_20": 20, "bb_width_20": 20, "keltner_pos_20": 20, "atr_pct_14": 14,
           "ema_cross_21_50": 50, "ema_cross_50_200": 200, "sma_dist_200": 200,
           "supertrend_10": 10, "mfi_14": 14, "obv_slope_14": 14, "vwap_dist_20": 20,
           "cvd_slope_14": 14}


# ── parameter SWEEP: try several periods per family so tuning can pick e.g. RSI(15) > RSI(14) ──
# {family: (periods, fn(o,h,l,c,v,n) -> series, family_group)}. Each default period is included
# so a retune never regresses below the current live signal.
TUNE_GRID = {
    # momentum oscillators
    "rsi":        ([7, 9, 11, 14, 15, 21, 28], lambda o, h, l, c, v, n: _rsi(c, n), "momentum"),
    "stoch_rsi":  ([14, 21],                    lambda o, h, l, c, v, n: _stoch_rsi(c, n), "momentum"),
    "stoch":      ([9, 14, 21],                 lambda o, h, l, c, v, n: _stoch_k(c, h, l, n), "momentum"),
    "williams":   ([9, 14, 21],                 lambda o, h, l, c, v, n: _williams(c, h, l, n), "momentum"),
    "cci":        ([14, 20, 30],                lambda o, h, l, c, v, n: _cci(c, h, l, n), "momentum"),
    "roc":        ([10, 14, 20],                lambda o, h, l, c, v, n: _roc(c, n), "momentum"),
    # volatility
    "bb_pctb":    ([14, 20, 26],                lambda o, h, l, c, v, n: _bb_pctb(c, n), "volatility"),
    "keltner_pos": ([14, 20],                   lambda o, h, l, c, v, n: _keltner_pos(h, l, c, n), "volatility"),
    "atr_pct":    ([14, 20],                     lambda o, h, l, c, v, n: _atr_pct(h, l, c, n), "volatility"),
    # trend
    "sma_dist":   ([100, 200],                  lambda o, h, l, c, v, n: _sma_dist(c, n), "trend"),
    "supertrend": ([10, 14],                    lambda o, h, l, c, v, n: _supertrend_pos(h, l, c, n), "trend"),
    # volume
    "mfi":        ([9, 14, 21],                 lambda o, h, l, c, v, n: _mfi(c, h, l, v, n), "volume"),
    "obv_slope":  ([14, 21],                    lambda o, h, l, c, v, n: _obv_slope(c, v, n), "volume"),
    "vwap_dist":  ([14, 20],                    lambda o, h, l, c, v, n: _vwap_dist(h, l, c, v, n), "volume"),
    "cvd_slope":  ([14, 21],                    lambda o, h, l, c, v, n: _cvd_slope(o, c, v, n), "volume"),
}
_GRID_FAMS = sorted(TUNE_GRID, key=len, reverse=True)   # longest first: "stoch_rsi" before "stoch"


def _sweep(o, h, l, c, v) -> dict:
    """{name: (series, family_group, period)} — the battery WITH period variants, for tuning."""
    out = {}
    for fam, (periods, fn, grp) in TUNE_GRID.items():
        for n in periods:
            out[f"{fam}_{n}"] = (fn(o, h, l, c, v, n), grp, n)
    out["macd_hist"] = (_macd_hist(c), "momentum", 26)            # fixed-param extras
    out["ema_cross_21_50"] = (_ema_cross(c, 21, 50), "trend", 50)
    out["ema_cross_50_200"] = (_ema_cross(c, 50, 200), "trend", 200)
    return out


def compute_one(name: str, o, h, l, c, v) -> np.ndarray:
    """Compute a single named indicator (used live to evaluate the tuned winners). Handles both
    the fixed battery names AND swept names like 'rsi_15' (family_period)."""
    bat = battery(o, h, l, c, v)
    if name in bat:
        return bat[name]
    for fam in _GRID_FAMS:
        m = re.fullmatch(re.escape(fam) + r"_(\d+)", name)
        if m:
            return TUNE_GRID[fam][1](o, h, l, c, v, int(m.group(1)))
    return np.full(len(c), np.nan)


def latest_values(klines: list) -> dict:
    """Current (last finite) value of every battery indicator, for the live snapshot — so
    alarms/questions can reference RSI, MFI, Boll %B, Stoch, CCI, Williams %R, MACD, etc.
    Returns {} on too-little data; per-indicator skipped when all-NaN (warmup)."""
    try:
        a = np.asarray(klines, dtype=float)
    except Exception:  # noqa: BLE001
        return {}
    if a.ndim != 2 or a.shape[0] < 3 or a.shape[1] < 6:
        return {}
    o, h, l, c, v = a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5]
    out = {}
    for name, series in battery(o, h, l, c, v).items():
        fin = series[~np.isnan(series)]
        if len(fin):
            out[name] = round(float(fin[-1]), 4)
    return out


# ───────────────────────────── labelling + scoring ──────────────────────────

def label_extrema(h, l, w: int):
    tops, bottoms = set(), set()
    n = len(h)
    for i in range(w, n - w):
        if h[i] == h[i - w:i + w + 1].max():
            tops.add(i)
        if l[i] == l[i - w:i + w + 1].min():
            bottoms.add(i)
    return tops, bottoms


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based ROC-AUC (probability a positive outranks a negative)."""
    pos, neg = labels.sum(), len(labels) - labels.sum()
    if pos == 0 or neg == 0:
        return 0.5
    order = scores.argsort()
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def _near(idxs: set, n: int, tol: int) -> np.ndarray:
    """Boolean array: is bar i within ±tol of any index in idxs?"""
    lab = np.zeros(n, dtype=int)
    for j in idxs:
        lab[max(0, j - tol):min(n, j + tol + 1)] = 1
    return lab


def score_signal(values: np.ndarray, extrema: set, tol: int, *, side: str) -> dict:
    """side='top' rewards HIGH readings near extrema; 'bottom' rewards LOW readings."""
    n = len(values)
    lab = _near(extrema, n, tol)
    m = ~np.isnan(values)
    if m.sum() < 10 or lab[m].sum() == 0:
        return {"auc": 0.5, "threshold": None, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    s = values[m] if side == "top" else -values[m]   # bottoms: low value = signal
    y = lab[m]
    auc = _auc(s, y)
    # best-F1 threshold over candidate quantiles
    best = {"auc": round(auc, 3), "threshold": None, "f1": 0.0,
            "precision": 0.0, "recall": 0.0}
    for q in np.quantile(s, [0.7, 0.8, 0.85, 0.9, 0.95]):
        pred = s >= q
        tp = int((pred & (y == 1)).sum())
        if tp == 0:
            continue
        prec = tp / pred.sum()
        rec = tp / (y == 1).sum()
        f1 = 2 * prec * rec / (prec + rec)
        if f1 > best["f1"]:
            raw = q if side == "top" else -q       # convert back to oscillator units
            best.update(threshold=round(float(raw), 2), f1=round(f1, 3),
                        precision=round(prec, 3), recall=round(rec, 3))
    return best


def tune_signals(klines: list, *, swing_w: int = 12, tol: int = 6) -> dict:
    """Rank the battery for top- and bottom-calling. klines rows:
    [open_time_ms, open, high, low, close, volume]."""
    a = np.asarray(klines, dtype=float)
    o, h, l, c, v = a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5]
    tops, bottoms = label_extrema(h, l, swing_w)
    sweep = _sweep(o, h, l, c, v)            # battery WITH period variants (rsi_7..rsi_28, etc.)
    top_rank, bot_rank = [], []
    for name, (series, grp, period) in sweep.items():
        st = score_signal(series, tops, tol, side="top")
        sb = score_signal(series, bottoms, tol, side="bottom")
        top_rank.append({"name": name, "family": grp, "period": period, **st})
        bot_rank.append({"name": name, "family": grp, "period": period, **sb})
    top_rank.sort(key=lambda d: d["auc"], reverse=True)
    bot_rank.sort(key=lambda d: d["auc"], reverse=True)

    def best_per_family(rank):
        seen, out = set(), []
        for d in rank:                       # rank already sorted by AUC desc
            if d["family"] not in seen and d["threshold"] is not None:
                seen.add(d["family"]); out.append(d)
        return out

    return {
        "bars": len(c), "swing_w": swing_w, "tol": tol,
        "n_tops": len(tops), "n_bottoms": len(bottoms),
        "top_leaderboard": top_rank, "bottom_leaderboard": bot_rank,
        "top_by_family": best_per_family(top_rank),     # the "one from each family" combo
        "bottom_by_family": best_per_family(bot_rank),
        "best_top": next((d for d in top_rank if d["threshold"] is not None), None),
        "best_bottom": next((d for d in bot_rank if d["threshold"] is not None), None),
    }


_TF_MIN = {"5m": 5, "15m": 15, "30m": 30, "1h": 60, "2h": 120, "4h": 240,
           "6h": 360, "12h": 720, "1d": 1440}


def run_tune(symbol: str, *, timeframes=("5m", "1h", "4h", "1d"), live_tf: str = "1h",
             weeks: float = 8.0, out_path: str = SIGNALS_PATH, stamp: str = "",
             swing_w: int = 12, tol: int = 6, lookback_days: int = 90) -> dict:
    """Backtest the battery across MULTIPLE timeframes (5m/1h/4h/1d), pick per-family
    and overall winners on each, and persist. NEVER pulls more than ``lookback_days``
    (~3 months) of candles per timeframe. The live detector uses the ``live_tf`` winners."""
    from .exchange import public_klines
    days = min(lookback_days, max(1, int(round(weeks * 7))))   # `weeks` actually sizes the window now
    per_tf = {}
    for tf in timeframes:
        try:
            bars = min(1000, max(60, days * 1440 // _TF_MIN.get(tf, 60)))   # still <= lookback_days
            kl = public_klines(symbol, tf, bars)
            per_tf[tf] = tune_signals(kl, swing_w=swing_w, tol=tol)
        except Exception as e:  # noqa: BLE001 - skip a TF that fails, keep the rest
            log.warning("tune %s failed: %s", tf, e)
    live = per_tf.get(live_tf) or (next(iter(per_tf.values())) if per_tf else None)
    keep = ("bars", "n_tops", "n_bottoms", "best_top", "best_bottom",
            "top_by_family", "bottom_by_family")
    res = {
        "symbol": symbol, "as_of": stamp, "weeks": weeks, "live_tf": live_tf,
        "timeframes": list(per_tf),
        "best_top": live["best_top"] if live else None,        # used live on live_tf
        "best_bottom": live["best_bottom"] if live else None,
        "per_timeframe": {tf: {k: r[k] for k in keep} for tf, r in per_tf.items()},
    }
    try:                                          # atomic write so the monitor never reads a half file
        d = os.path.dirname(out_path) or "."
        os.makedirs(d, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(res, f, indent=2)
        os.replace(tmp, out_path)
    except Exception as e:  # noqa: BLE001
        log.warning("could not write %s: %s", out_path, e)
    return res


def load_tuned(path: str = SIGNALS_PATH) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001 - none tuned yet -> detector uses RSI defaults
        return None


_PRETTY = {
    "rsi_14": "RSI(14)", "rsi_21": "RSI(21)", "stoch_rsi_14": "StochRSI",
    "stoch_14": "Stoch(14)", "macd_hist": "MACD hist", "cci_20": "CCI(20)",
    "williams_14": "Williams%R", "roc_10": "ROC(10)", "roc_20": "ROC(20)",
    "bb_pctb_20": "Boll %B", "bb_width_20": "Boll width", "keltner_pos_20": "Keltner",
    "atr_pct_14": "ATR%", "ema_cross_21_50": "EMA 21/50", "ema_cross_50_200": "EMA 50/200",
    "sma_dist_200": "SMA-200", "supertrend_10": "Supertrend", "mfi_14": "MFI(14)",
    "obv_slope_14": "OBV", "vwap_dist_20": "VWAP(20)", "cvd_slope_14": "CVD",
}


_PRETTY_BASE = {"rsi": "RSI", "stoch_rsi": "StochRSI", "stoch": "Stoch", "williams": "Williams%R",
                "cci": "CCI", "roc": "ROC", "bb_pctb": "Boll %B", "keltner_pos": "Keltner",
                "atr_pct": "ATR%", "sma_dist": "SMA", "supertrend": "Supertrend", "mfi": "MFI",
                "obv_slope": "OBV", "vwap_dist": "VWAP", "cvd_slope": "CVD"}


def _pretty(name):
    if name in _PRETTY:
        return _PRETTY[name]
    m = re.fullmatch(r"([a-z_]+?)_(\d+)", name or "")   # swept name -> e.g. RSI(15)
    if m and m.group(1) in _PRETTY_BASE:
        return f"{_PRETTY_BASE[m.group(1)]}({m.group(2)})"
    return name


def _score(auc):
    return int(round((auc or 0.5) * 100))


def _quality(auc):
    s = _score(auc)
    return ("🟢 strong" if s >= 75 else "🟢 good" if s >= 65
            else "🟡 fair" if s >= 58 else "🔴 weak")


def leaderboard_text(res: dict) -> str:
    tf = res.get("live_tf", "1h")
    bt, bb = res.get("best_top"), res.get("best_bottom")
    L = [f"🔬 *BTC top/bottom scan* — live candles `{tf}`"]

    # ── the headline: which indicator is best RIGHT NOW ──
    L.append("\n👑 *Best caller right now*")
    if bt:
        L.append(f"📈 Tops:  *{_pretty(bt['name'])}*  —  {_score(bt['auc'])}/100  {_quality(bt['auc'])}")
    if bb:
        L.append(f"📉 Bottoms:  *{_pretty(bb['name'])}*  —  {_score(bb['auc'])}/100  {_quality(bb['auc'])}")

    # ── leading indicators (best one per family), ranked, winner crowned ──
    def block(items):
        rows = []
        for i, d in enumerate(items[:4]):
            crown = "👑" if i == 0 else "  "
            rows.append(f"{crown} {_pretty(d['name']):<11} {d['family']:<10} {_score(d['auc']):>3}/100")
        return "```\n" + "\n".join(rows) + "\n```" if rows else ""

    live = res.get("per_timeframe", {}).get(tf, {})
    if live.get("top_by_family"):
        L.append(f"\n🏅 *Leading top-callers* (`{tf}`, best of each family)")
        L.append(block(live["top_by_family"]))
    if live.get("bottom_by_family"):
        L.append(f"🏅 *Leading bottom-callers* (`{tf}`)")
        L.append(block(live["bottom_by_family"]))

    # ── one glance across all timeframes ──
    rows = [f"{'TF':<4} {'TOP-caller':<16} BOTTOM-caller"]
    best_tf, best_s = tf, 0
    for t, r in res.get("per_timeframe", {}).items():
        tt, bbx = r.get("best_top"), r.get("best_bottom")
        ts = f"{_pretty(tt['name'])} {_score(tt['auc'])}" if tt else "—"
        bs = f"{_pretty(bbx['name'])} {_score(bbx['auc'])}" if bbx else "—"
        s = max(_score(tt['auc']) if tt else 0, _score(bbx['auc']) if bbx else 0)
        if s > best_s:
            best_s, best_tf = s, t
        rows.append(f"{t:<4} {ts:<16} {bs}{'  ← live' if t == tf else ''}")
    L.append("\n🕐 *Best across timeframes*")
    L.append("```\n" + "\n".join(rows) + "\n```")

    L.append(f"_score = hit-rate vs coin-flip (50). 65+ = usable, 75+ = strong. "
             f"Cleanest signals on `{best_tf}`. Live detector uses the `{tf}` winners._")
    return "\n".join(L)
