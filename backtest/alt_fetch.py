"""One-time: pull real Kraken DAILY OHLCV for the alt-stacker assets and cache as CSV.

    python3 backtest/alt_fetch.py            # SOL/ETH/HYPE/PAXG vs USD
    python3 backtest/alt_fetch.py SOL ETH    # a subset

Daily candles keep the backtest (and any LLM-in-the-loop run) to a tractable number of
cycles. Kraken serves ~720 daily candles per request with no API key. HYPE only launched
2024-11, so the 4-asset *aligned* window is bounded by HYPE's history; SOL/ETH/PAXG have
longer single-asset history. Lean CSV (no pandas) so the engine stays import-light.
"""
from __future__ import annotations

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data" / "alt"
# XMR (make USD) + PAXG (gold hedge) are the live assets; SOL/ETH/HYPE kept for comparison.
DEFAULT_BASES = ["XMR", "PAXG", "SOL", "ETH", "HYPE"]
QUOTE = "USD"


def fetch(base: str, quote: str = QUOTE, timeframe: str = "1d", limit: int = 720) -> list[list]:
    import ccxt
    k = ccxt.kraken({"enableRateLimit": True})
    sym = f"{base}/{quote}"
    raw = k.fetch_ohlcv(sym, timeframe, limit=limit)  # [ts_ms,o,h,l,c,v]
    return raw


def save(base: str, rows: list[list], timeframe: str = "1d") -> Path:
    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / f"{base}_{QUOTE}_{timeframe}.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "open", "high", "low", "close"])
        for ts, o, h, l, c, *_ in rows:
            # full timestamp (date+hour) so sub-daily (e.g. 4h) bars stay distinct
            d = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M")
            w.writerow([d, o, h, l, c])
    return out


def main(argv=None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    tf = "1d"
    if "--tf" in args:
        i = args.index("--tf")
        tf = args[i + 1]
        del args[i:i + 2]
    bases = args or DEFAULT_BASES
    for base in bases:
        try:
            rows = fetch(base, timeframe=tf)
            out = save(base, rows, tf)
            d0 = datetime.fromtimestamp(rows[0][0] / 1000, tz=timezone.utc).date()
            d1 = datetime.fromtimestamp(rows[-1][0] / 1000, tz=timezone.utc).date()
            print(f"  ✓ {base}/{QUOTE}: {len(rows)} {tf} candles {d0}..{d1} -> {out}")
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ {base}/{QUOTE}: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
