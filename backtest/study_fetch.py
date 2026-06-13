"""Deep, multi-timeframe history for the candle/technicals study (Binance PUBLIC klines).

Kraken's public OHLC only serves ~720 recent candles, so for a LONG study we pull Binance
public klines (no account — market data only; prices are ~identical across major venues, and
live trading stays on Kraken). Paginates back years. Writes the same CSV shape the alt engine
reads: backtest/data/alt/{BASE}_USD_{tf}.csv  (date is a full timestamp so sub-daily bars stay
distinct). The "USD" in the name is nominal — Binance quote is USDT (≈ USD).

    python3 backtest/study_fetch.py                 # TRX,BNB,PAXG across 1h/4h/8h/12h/1d
    python3 backtest/study_fetch.py TRX BNB         # subset
"""
from __future__ import annotations

import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

DATA = Path(__file__).resolve().parent / "data" / "alt"
KLINES = "https://api.binance.com/api/v3/klines"
DEFAULT_BASES = ["TRX", "BNB", "PAXG"]

_MS = {"5m": 300_000, "15m": 900_000, "30m": 1_800_000, "1h": 3_600_000,
       "4h": 14_400_000, "8h": 28_800_000, "12h": 43_200_000, "1d": 86_400_000}
# how far back to pull per timeframe (intraday capped so the request count stays sane)
_START = {"5m": "2026-03-01", "15m": "2025-09-01", "30m": "2025-01-01", "1h": "2024-06-01",
          "4h": "2021-01-01", "8h": "2021-01-01", "12h": "2019-01-01", "1d": "2019-01-01"}


def _ms(date: str) -> int:
    return int(datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def fetch(base: str, tf: str, *, quote: str = "USDT") -> list[list]:
    sym = f"{base}{quote}"
    start = _ms(_START[tf])
    step = _MS[tf]
    now = int(time.time() * 1000)
    rows: list[list] = []
    while start < now:
        r = requests.get(KLINES, params={"symbol": sym, "interval": tf,
                                         "startTime": start, "limit": 1000}, timeout=20)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < 1000:
            break
        start = batch[-1][0] + step
        time.sleep(0.15)
    # dedupe by openTime, sort
    seen, out = set(), []
    for k in sorted(rows, key=lambda x: x[0]):
        if k[0] in seen:
            continue
        seen.add(k[0])
        out.append(k)
    return out


def save(base: str, tf: str, klines: list[list]) -> Path:
    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / f"{base}_USD_{tf}.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "open", "high", "low", "close"])
        for k in klines:
            d = datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M")
            w.writerow([d, k[1], k[2], k[3], k[4]])
    return out


def main(argv=None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    tfs = ("5m", "15m", "30m", "1h", "4h", "8h", "12h", "1d")
    if "--tfs" in args:
        i = args.index("--tfs"); tfs = tuple(args[i + 1].split(",")); del args[i:i + 2]
    bases = args or DEFAULT_BASES
    for base in bases:
        for tf in tfs:
            try:
                kl = fetch(base, tf)
                if not kl:
                    print(f"  ✗ {base} {tf}: no data (symbol listed on Binance?)")
                    continue
                out = save(base, tf, kl)
                d0 = datetime.fromtimestamp(kl[0][0] / 1000, tz=timezone.utc).date()
                d1 = datetime.fromtimestamp(kl[-1][0] / 1000, tz=timezone.utc).date()
                print(f"  ✓ {base} {tf:>3}: {len(kl):>5} bars {d0}..{d1} -> {out.name}")
            except Exception as e:  # noqa: BLE001
                print(f"  ✗ {base} {tf}: {type(e).__name__}: {str(e)[:60]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
