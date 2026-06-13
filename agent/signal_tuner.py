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


def _ema_cross(c, a, b):
    return (_ema(c, a) - _ema(c, b)) / _ema(c, b) * 100   # higher = fast above slow


# name -> (callable, period) ; period is metadata for the live monitor
def battery(o, h, l, c, v) -> dict:
    return {
        "rsi_7": _rsi(c, 7), "rsi_14": _rsi(c, 14), "rsi_21": _rsi(c, 21),
        "rsi_28": _rsi(c, 28),
        "stoch_14": _stoch_k(c, h, l, 14), "stoch_21": _stoch_k(c, h, l, 21),
        "williams_14": _williams(c, h, l, 14),
        "cci_20": _cci(c, h, l, 20),
        "roc_10": _roc(c, 10), "roc_20": _roc(c, 20),
        "mfi_14": _mfi(c, h, l, v, 14),
        "ema_cross_9_21": _ema_cross(c, 9, 21),
    }


PERIODS = {"rsi_7": 7, "rsi_14": 14, "rsi_21": 21, "rsi_28": 28, "stoch_14": 14,
           "stoch_21": 21, "williams_14": 14, "cci_20": 20, "roc_10": 10,
           "roc_20": 20, "mfi_14": 14, "ema_cross_9_21": 21}


def compute_one(name: str, o, h, l, c, v) -> np.ndarray:
    """Compute a single named oscillator (used live to evaluate the tuned winners)."""
    return battery(o, h, l, c, v).get(name, np.full(len(c), np.nan))


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
    bat = battery(o, h, l, c, v)
    top_rank, bot_rank = [], []
    for name, series in bat.items():
        st = score_signal(series, tops, tol, side="top")
        sb = score_signal(series, bottoms, tol, side="bottom")
        top_rank.append({"name": name, "period": PERIODS[name], **st})
        bot_rank.append({"name": name, "period": PERIODS[name], **sb})
    top_rank.sort(key=lambda d: d["auc"], reverse=True)
    bot_rank.sort(key=lambda d: d["auc"], reverse=True)
    return {
        "bars": len(c), "swing_w": swing_w, "tol": tol,
        "n_tops": len(tops), "n_bottoms": len(bottoms),
        "top_leaderboard": top_rank, "bottom_leaderboard": bot_rank,
        "best_top": top_rank[0] if top_rank else None,
        "best_bottom": bot_rank[0] if bot_rank else None,
    }


def run_tune(symbol: str, *, timeframe: str = "1h", weeks: float = 8.0,
             out_path: str = SIGNALS_PATH, stamp: str = "") -> dict:
    """Fetch ~`weeks` of candles, tune, and persist the winners. Returns the result."""
    from .exchange import public_klines
    per_week = {"15m": 672, "1h": 168, "4h": 42, "1d": 7}.get(timeframe, 168)
    limit = min(1000, int(per_week * weeks) + 50)
    klines = public_klines(symbol, timeframe, limit)
    res = tune_signals(klines)
    res.update(symbol=symbol, timeframe=timeframe, weeks=weeks, as_of=stamp)
    try:
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
    except Exception as e:  # noqa: BLE001
        log.warning("could not write %s: %s", out_path, e)
    return res


def load_tuned(path: str = SIGNALS_PATH) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001 - none tuned yet -> detector uses RSI defaults
        return None


def leaderboard_text(res: dict) -> str:
    def row(d):
        th = "" if d["threshold"] is None else f" thr={d['threshold']}"
        return f"  {d['name']:<14} AUC {d['auc']:.2f}  F1 {d['f1']:.2f}{th}"
    lt = "\n".join(row(d) for d in res["top_leaderboard"][:5])
    lb = "\n".join(row(d) for d in res["bottom_leaderboard"][:5])
    return (f"Tuned on {res.get('bars','?')} {res.get('timeframe','')} bars "
            f"({res.get('n_tops','?')} tops, {res.get('n_bottoms','?')} bottoms)\n"
            f"BEST TOP caller: {res['best_top']['name']} (AUC {res['best_top']['auc']:.2f})\n"
            f"{lt}\n"
            f"BEST BOTTOM caller: {res['best_bottom']['name']} (AUC {res['best_bottom']['auc']:.2f})\n"
            f"{lb}")
