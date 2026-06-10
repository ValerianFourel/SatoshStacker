"""Multi-timeframe momentum-technical backtester — which indicators best PREDICT
forward BTC returns, per candle timeframe (5m / 1h / 4h) and per forward lag.

For each indicator+parameter it computes the **Information Coefficient** (IC =
Pearson corr of the indicator value at time t with the realized return over the next
`lag` bars). Indicators are ranked by |IC|; the SIGN reveals momentum (+, trend
persists) vs mean-reversion (−, extremes revert). The winners + their best params
(the RSI period and the `lag` that scored highest) are written to
`agent/technicals.json` for the weekly Qwen tuner and the tactical trader to consume.

    python3 backtest/indicators.py                 # default: last 60 days, 1h + 4h
    python3 backtest/indicators.py --days 30 --tfs 5m,1h,4h
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from backtest.fetch_data import fetch_klines  # noqa: E402  paginated Binance fetch

DATA = ROOT / "backtest" / "data"
OUT = ROOT / "agent" / "technicals.json"


def load_tf(tf: str, days: int) -> pd.DataFrame:
    """Load (or fetch) BTCUSDT candles for a timeframe, return the last `days`."""
    path = DATA / f"BTCUSDT_{tf}.parquet"
    need = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days + 5)
    df = pd.read_parquet(path) if path.exists() else None
    if df is None or df["dt"].iloc[0] > need or df["dt"].iloc[-1] < pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=2):
        start = int(need.timestamp() * 1000)
        end = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
        print(f"  fetching {tf} candles…")
        df = fetch_klines("BTCUSDT", tf, start, end)
        df.to_parquet(path)
    cut = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    return df[df["dt"] >= cut].reset_index(drop=True)


# ── indicator signal builders (higher value = expected more bullish) ──────────
def _rsi(c: pd.Series, n: int) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _ema(c: pd.Series, n: int) -> pd.Series:
    return c.ewm(span=n, adjust=False).mean()


def signals(close: pd.Series) -> dict[str, pd.Series]:
    macd = _ema(close, 12) - _ema(close, 26)
    macd_sig = macd - macd.ewm(span=9, adjust=False).mean()
    bb_n = 20
    ma = close.rolling(bb_n).mean()
    sd = close.rolling(bb_n).std()
    return {
        "rsi_7": _rsi(close, 7), "rsi_14": _rsi(close, 14),
        "rsi_21": _rsi(close, 21), "rsi_28": _rsi(close, 28),
        "ema_cross_8_21": (_ema(close, 8) - _ema(close, 21)) / _ema(close, 21),
        "ema_cross_20_50": (_ema(close, 20) - _ema(close, 50)) / _ema(close, 50),
        "momentum_6": close / close.shift(6) - 1,
        "momentum_12": close / close.shift(12) - 1,
        "momentum_24": close / close.shift(24) - 1,
        "macd_hist": macd_sig,
        "bb_pctb_20": (close - ma) / (2 * sd),
    }


def ic(sig: pd.Series, fwd: pd.Series) -> float:
    s = pd.concat([sig, fwd], axis=1).dropna()
    if len(s) < 30:
        return 0.0
    c = s.corr().iloc[0, 1]
    return 0.0 if np.isnan(c) else float(c)


def rank_tf(df: pd.DataFrame, lags: list[int]) -> list[dict]:
    close = df["close"]
    sigs = signals(close)
    rows = []
    for name, sig in sigs.items():
        best = max(lags, key=lambda L: abs(ic(sig, close.shift(-L) / close - 1)))
        v = ic(sig, close.shift(-best) / close - 1)
        rows.append({"indicator": name, "lag": best, "ic": round(v, 4),
                     "direction": "momentum" if v > 0 else "reversion",
                     "abs_ic": round(abs(v), 4)})
    return sorted(rows, key=lambda r: -r["abs_ic"])


def main() -> None:
    ap = argparse.ArgumentParser("indicators")
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--tfs", default="1h,4h")
    ap.add_argument("--lags", default="1,2,3,6")
    args = ap.parse_args()
    lags = [int(x) for x in args.lags.split(",")]
    tfs = args.tfs.split(",")

    print(f"\n=== Technical predictive-power backtest — last {args.days} days, "
          f"timeframes {tfs}, forward lags {lags} ===")
    print("    IC = corr(indicator_t, forward return). |IC|>~0.05 = some edge; "
          "sign: momentum(+) / reversion(-)\n")

    all_ranked: dict[str, list[dict]] = {}
    leaders = []
    for tf in tfs:
        df = load_tf(tf, args.days)
        ranked = rank_tf(df, lags)
        all_ranked[tf] = ranked
        print(f"  [{tf}]  ({len(df)} candles)")
        for r in ranked[:5]:
            print(f"      {r['indicator']:16} lag={r['lag']}  IC={r['ic']:+.4f}  ({r['direction']})")
        top = ranked[0]
        leaders.append({**top, "timeframe": tf})

    leaders.sort(key=lambda r: -r["abs_ic"])
    best = leaders[0]
    # extract the winning RSI period if an RSI indicator leads, else default 14
    rsi_period = int(best["indicator"].split("_")[1]) if best["indicator"].startswith("rsi") else 14

    config = {
        "as_of": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "lookback_days": args.days,
        "leader": best,                 # best (tf, indicator, lag, direction)
        "leaders_per_tf": leaders,
        "ranked": all_ranked,
        "suggested": {"timeframe": best["timeframe"], "primary_indicator": best["indicator"],
                      "lag": best["lag"], "rsi_period": rsi_period},
    }
    OUT.write_text(json.dumps(config, indent=2))
    print(f"\n  LEADER: {best['indicator']} on {best['timeframe']} (lag {best['lag']}, "
          f"IC {best['ic']:+.4f}, {best['direction']})")
    print(f"  -> wrote suggested technicals to {OUT.relative_to(ROOT)} "
          f"(primary={best['indicator']}, tf={best['timeframe']}, lag={best['lag']}, rsi_period={rsi_period})")


if __name__ == "__main__":
    main()
