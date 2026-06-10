"""Fetch real historical BTC klines from Binance public REST API.

No API key required (public market data). Paginates through Binance's
1000-candle-per-request limit. Saves tidy parquet files under data/.

We use BTCUSDT as the price proxy for the full history because BTCUSDC
liquidity/history is shorter; for *price* backtesting USDT≈USDC≈$1. The
USDC de-peg risk is a separate *live* concern handled in agent/risk.py,
not a price-series concern here.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://api.binance.com/api/v3/klines"
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_base", "taker_quote", "ignore",
]


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Paginated fetch of klines in [start_ms, end_ms]. Returns a typed DataFrame."""
    rows: list[list] = []
    cursor = start_ms
    session = requests.Session()
    while cursor < end_ms:
        params = {
            "symbol": symbol, "interval": interval,
            "startTime": cursor, "endTime": end_ms, "limit": 1000,
        }
        for attempt in range(5):
            try:
                r = session.get(BASE, params=params, timeout=15)
                r.raise_for_status()
                batch = r.json()
                break
            except Exception as e:  # noqa: BLE001 - simple retry/backoff for a CLI fetch
                wait = 2 ** attempt
                print(f"  retry {attempt+1} ({e}) sleeping {wait}s", file=sys.stderr)
                time.sleep(wait)
        else:
            raise RuntimeError(f"failed to fetch {symbol} {interval} at cursor {cursor}")
        if not batch:
            break
        rows.extend(batch)
        last_close = batch[-1][6]
        nxt = last_close + 1
        if nxt <= cursor:  # safety: no forward progress
            break
        cursor = nxt
        time.sleep(0.12)  # be polite to the public endpoint
        if len(batch) < 1000:
            break

    df = pd.DataFrame(rows, columns=_COLS)
    if df.empty:
        return df
    for c in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["dt"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    return df[["dt", "open_time", "open", "high", "low", "close", "volume", "quote_volume", "trades"]]


def main() -> None:
    # Full daily history from Binance BTCUSDT genesis (2017-08-17) to "now".
    start = int(pd.Timestamp("2017-08-01", tz="UTC").timestamp() * 1000)
    end = int(pd.Timestamp.utcnow().timestamp() * 1000)

    print("Fetching BTCUSDT 1d full history ...")
    daily = fetch_klines("BTCUSDT", "1d", start, end)
    out = DATA_DIR / "BTCUSDT_1d.parquet"
    daily.to_parquet(out)
    print(f"  saved {len(daily)} daily candles -> {out}")
    print(f"  range: {daily['dt'].iloc[0].date()} .. {daily['dt'].iloc[-1].date()}")
    print(f"  price range: ${daily['low'].min():,.0f} .. ${daily['high'].max():,.0f}")

    # Higher-resolution 4h history (used for fill-realism robustness checks).
    print("Fetching BTCUSDT 4h full history ...")
    h4 = fetch_klines("BTCUSDT", "4h", start, end)
    out4 = DATA_DIR / "BTCUSDT_4h.parquet"
    h4.to_parquet(out4)
    print(f"  saved {len(h4)} 4h candles -> {out4}")


if __name__ == "__main__":
    main()
