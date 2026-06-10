"""Lean (numpy-only) self-tune the bot runs AT STARTUP and once a week.

Backtests which technicals best PREDICT BTC returns (Information Coefficient = corr of
the indicator with the forward return) from live public candles, then asks the smart
Qwen model to pick the leading timeframe + indicator + parameters (RSI period, lag,
momentum-vs-reversion) for the period, and writes agent/technicals.json — which the
trader reads for its RSI period + context note. No pandas, so it runs on the slim image.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .exchange import public_ohlcv
from .secrets import clean_secret, redact

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "agent" / "technicals.json"

META_SYSTEM = (
    "You are a quant research lead tuning a BTC trader. You are given, per candle "
    "timeframe, indicators ranked by Information Coefficient (predictive corr with forward "
    "returns; POSITIVE = momentum/trend persists, NEGATIVE = mean-reversion). Choose for the "
    "period: the leading timeframe, primary indicator + parameters, forward lag (bars), whether "
    "the regime is MOMENTUM or REVERSION, the RSI period to use, and 1-2 confirming indicators. "
    "Write a concise CONTEXT note (<=70 words) telling the 4h trader how to read these signals. "
    "Respond STRICT JSON only:\n"
    '{"timeframe":"4h","primary_indicator":"rsi_28","rsi_period":28,"lag":6,"regime":"momentum",'
    '"confirming":["ema_cross_20_50"],"context_note":"...","rationale":"..."}'
)


def _roll_mean(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        cs = np.cumsum(np.insert(x, 0, 0.0))
        out[n - 1:] = (cs[n:] - cs[:-n]) / n
    return out


def _rsi(c: np.ndarray, n: int) -> np.ndarray:
    d = np.diff(c)
    up, dn = np.where(d > 0, d, 0.0), np.where(d < 0, -d, 0.0)
    ru, rd = _roll_mean(up, n), _roll_mean(dn, n)
    rsi = 100 - 100 / (1 + ru / np.where(rd == 0, np.nan, rd))
    return np.insert(rsi, 0, np.nan)  # realign to closes (diff dropped one)


def _mom(c: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(c), np.nan)
    out[n:] = c[n:] / c[:-n] - 1
    return out


def _signals(c: np.ndarray) -> dict[str, np.ndarray]:
    f, s = _roll_mean(c, 8), _roll_mean(c, 21)
    f2, s2 = _roll_mean(c, 20), _roll_mean(c, 50)
    return {"rsi_7": _rsi(c, 7), "rsi_14": _rsi(c, 14), "rsi_21": _rsi(c, 21),
            "rsi_28": _rsi(c, 28), "momentum_6": _mom(c, 6), "momentum_12": _mom(c, 12),
            "momentum_24": _mom(c, 24), "ema_cross_8_21": (f - s) / s,
            "ema_cross_20_50": (f2 - s2) / s2}


def _ic(ind: np.ndarray, c: np.ndarray, lag: int) -> float:
    n = len(c)
    fwd = np.full(n, np.nan)
    fwd[:n - lag] = c[lag:] / c[:n - lag] - 1
    m = ~(np.isnan(ind) | np.isnan(fwd))
    if m.sum() < 30:
        return 0.0
    cc = np.corrcoef(ind[m], fwd[m])[0, 1]
    return 0.0 if np.isnan(cc) else float(cc)


def rank(closes: np.ndarray, lags=(1, 2, 3, 6)) -> list[dict]:
    rows = []
    for name, ind in _signals(closes).items():
        best = max(lags, key=lambda L: abs(_ic(ind, closes, L)))
        v = _ic(ind, closes, best)
        rows.append({"indicator": name, "lag": best, "ic": round(v, 4),
                     "direction": "momentum" if v > 0 else "reversion",
                     "abs_ic": round(abs(v), 4)})
    return sorted(rows, key=lambda r: -r["abs_ic"])


def _qwen_meta(payload: dict, model: str) -> dict:
    key = clean_secret(os.getenv("LLM_API_KEY"))
    if not key:
        return {}
    try:
        from openai import OpenAI
        c = OpenAI(base_url=os.getenv("LLM_BASE_URL"), api_key=key)
        r = c.chat.completions.create(
            model=model, temperature=0.2, timeout=60, max_tokens=600,
            messages=[{"role": "system", "content": META_SYSTEM},
                      {"role": "user", "content": json.dumps(payload)}])
        t = r.choices[0].message.content or ""
        i, j = t.find("{"), t.rfind("}")
        return json.loads(t[i:j + 1]) if 0 <= i < j else {}
    except Exception as e:  # noqa: BLE001
        return {"error": type(e).__name__}


def run_tune(symbol: str = "BTC/USDT", model: str | None = None) -> dict:
    """Backtest + (smart-Qwen) pick the leading technicals; write technicals.json.
    Returns a short summary. Fail-safe: on any error leaves the existing config."""
    model = model or os.getenv("TUNE_MODEL", "qwen/qwen3.5-plus-20260420")
    try:
        c4 = np.array(public_ohlcv(symbol, "4h", 400))[:, 3]
        c1 = np.array(public_ohlcv(symbol, "1h", 700))[:, 3]
    except Exception as e:  # noqa: BLE001
        return {"error": f"candles: {redact(e)}"}
    ranked = {"4h": rank(c4), "1h": rank(c1)}
    payload = {tf: [{k: r[k] for k in ("indicator", "lag", "ic", "direction")}
                    for r in ranked[tf][:6]] for tf in ranked}
    leader = sorted([{**ranked[tf][0], "timeframe": tf} for tf in ranked],
                    key=lambda r: -r["abs_ic"])[0]
    decision = _qwen_meta(payload, model)
    # if Qwen fails, fall back to the raw backtest leader (incl. its RSI period), not a hardcode
    leader_rsi = (int(leader["indicator"].split("_")[1])
                  if leader["indicator"].startswith("rsi") else 14)
    sug = {
        "timeframe": decision.get("timeframe", leader["timeframe"]),
        "primary_indicator": decision.get("primary_indicator", leader["indicator"]),
        "rsi_period": int(decision.get("rsi_period", leader_rsi)),
        "lag": int(decision.get("lag", leader["lag"])),
        "regime": decision.get("regime", leader["direction"]),
        "confirming": decision.get("confirming", []),
    }
    cfg = {"as_of": "startup/weekly", "backtest_leader": leader, "ranked": ranked,
           "qwen_decision": decision, "context_note": decision.get("context_note", ""),
           "suggested": sug}
    try:
        OUT.write_text(json.dumps(cfg, indent=2))
    except Exception:  # noqa: BLE001
        pass
    return {"leader": leader, "suggested": sug, "context_note": cfg["context_note"],
            "qwen_ok": "error" not in decision and bool(decision)}
