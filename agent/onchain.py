"""CryptoQuant on-chain metrics — MVRV(-Z), NUPL, SOPR, exchange netflow (cycle-level).

Needs CRYPTOQUANT_API_KEY *and* a plan that grants these endpoints. The basic API tier
only allows price-ohlcv — the on-chain endpoints return 403 "you don't have an authority";
this fetcher is fail-safe (returns {} / 'unavailable') and will light up automatically once
the plan is upgraded. Cloudflare blocks default Python UAs, so a browser UA is sent.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("satoshistacker.onchain")

_BASE = "https://api.cryptoquant.com/v1"
_METRICS = {
    "mvrv": ("/btc/network-indicator/mvrv", {"window": "day", "limit": 1}),
    "nupl": ("/btc/network-indicator/nupl", {"window": "day", "limit": 1}),
    "sopr": ("/btc/network-indicator/sopr", {"window": "day", "limit": 1}),
    "exchange_netflow": ("/btc/exchange-flows/netflow",
                         {"exchange": "all_exchange", "window": "day", "limit": 1}),
}
_SKIP = {"date", "datetime", "start_time", "blockheight", "block_height"}


def _latest_value(payload: dict):
    try:
        data = payload.get("result", {}).get("data") or []
        if not data:
            return None
        for k, v in data[-1].items():
            if k not in _SKIP and isinstance(v, (int, float)):
                return round(float(v), 4)
    except Exception:  # noqa: BLE001
        return None
    return None


def fetch_onchain(*, timeout: float = 12.0) -> dict:
    """Latest MVRV/NUPL/SOPR/netflow, or {} if no key / no plan access (fail-safe)."""
    key = os.getenv("CRYPTOQUANT_API_KEY", "").strip()
    if not key:
        return {}
    import requests
    h = {"Authorization": "Bearer " + key, "User-Agent": "Mozilla/5.0"}
    out: dict = {}
    for name, (path, params) in _METRICS.items():
        try:
            r = requests.get(_BASE + path, headers=h, params=params, timeout=timeout)
            if r.status_code == 200:
                v = _latest_value(r.json())
                if v is not None:
                    out[name] = v
        except Exception as e:  # noqa: BLE001
            log.warning("onchain %s failed: %s", name, e)
    return out


def onchain_text(data: dict) -> str:
    if not data:
        return ("🔗 *On-chain:* unavailable — the CryptoQuant key works but your plan grants "
                "only price data; MVRV / NUPL / SOPR / netflow return 403. Upgrade the "
                "CryptoQuant API tier and they light up automatically.")
    label = {"mvrv": "MVRV", "nupl": "NUPL", "sopr": "SOPR", "exchange_netflow": "Exch netflow"}
    return "🔗 *On-chain* (cycle-level):\n" + "\n".join(
        f"  {label.get(k, k)}: `{v}`" for k, v in data.items())
