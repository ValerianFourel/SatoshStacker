"""Multi-asset spot exchange: paper (dry_run) and ccxt/Binance (testnet, live).

The alt agent (SOL/ETH/HYPE) trades several base assets against ONE shared USDC
cash pot. The single-symbol :class:`agent.exchange.Exchange` cannot model a shared
quote balance across pairs, so this module adds a small multi-symbol interface:

    price(base)                  -> last price of base/quote
    balances()                   -> {"USDC":.., "USDT":.., "SOL":.., ...}
    market_buy_quote(base, q, c) -> spend `q` USDC of cash on base (taker)
    market_sell(base, amt, c)    -> sell `amt` base units for USDC (taker)
    usd_quote_estimate()         -> USDC/USD for de-peg detection
    can_withdraw()               -> withdrawal permission (live MUST refuse if True)

Spot only. No margin/leverage/futures — `market_*` map to plain spot orders. ccxt
is imported lazily so the paper path and the test suite need no exchange SDK.

Per-asset venue note: v1 routes every pair through ONE venue (Binance) so the USDC
pot is genuinely shared. If a base (e.g. HYPE) is not listed there, point its
``symbol`` at a venue that lists it via config — but cross-venue USDC is NOT shared,
so that base must then run as its own instance. See DEPLOY notes.
"""
from __future__ import annotations

import abc
import json
import time
from pathlib import Path
from typing import Callable

from .exchange import public_price

# cash = any stable/fiat quote balance the venue reports (Binance: USDC/USDT;
# Kraken HYPE only trades vs USD, so the Kraken pot's cash leg is USD)
CASH_ASSETS = ("USDC", "USDT", "USD")
STABLE_QUOTES = ("USDC", "USDT")  # quotes that carry de-peg risk (fiat USD does not)


def _sym(base: str, quote: str) -> str:
    """'SOL','USDC' -> 'SOL/USDC' (ccxt unified symbol)."""
    return f"{base}/{quote}"


def public_feeds(venue: str):
    """Return (price_fn(symbol)->float, ohlcv_fn(symbol)->np.ndarray[[o,h,l,c],...]) for a
    venue, using PUBLIC market data (no API key). Binance uses the lightweight REST in
    exchange.py (keeps dry-run ccxt-free); any other venue (e.g. 'kraken') uses a
    lazily-built public ccxt client. Candles are normalized to an [open,high,low,close]
    float array (the shape AltStacker indexes as ohlc[:, 3] / ohlc[:, 1])."""
    import numpy as np
    if venue == "binance":
        from .exchange import public_ohlcv, public_price
        return (lambda sym: public_price(sym),
                lambda sym: np.array(public_ohlcv(sym, "4h", 200), dtype=float))
    import ccxt  # lazy: only when a non-Binance venue is used
    client = getattr(ccxt, venue)({"enableRateLimit": True})

    def price(sym: str) -> float:
        return float(client.fetch_ticker(sym)["last"])

    def ohlcv(sym: str):
        raw = client.fetch_ohlcv(sym, "4h", limit=200)  # [ts,o,h,l,c,v]
        return np.array([[r[1], r[2], r[3], r[4]] for r in raw], dtype=float)

    return price, ohlcv


class MultiExchange(abc.ABC):
    """Minimal multi-asset spot interface the alt agent depends on."""

    mode: str
    quote: str
    bases: list[str]

    @abc.abstractmethod
    def price(self, base: str) -> float: ...

    @abc.abstractmethod
    def balances(self) -> dict[str, float]: ...

    @abc.abstractmethod
    def market_buy_quote(self, base: str, quote_usdc: float,
                         client_order_id: str) -> dict: ...

    @abc.abstractmethod
    def market_sell(self, base: str, base_amount: float,
                    client_order_id: str) -> dict: ...

    @abc.abstractmethod
    def can_withdraw(self) -> bool:
        """True if the API key has withdrawal permission. Live MUST refuse this."""

    def cash(self) -> float:
        """Total spendable stable-quote balance (USDC + USDT)."""
        b = self.balances()
        return sum(float(b.get(a, 0) or 0) for a in CASH_ASSETS)

    def usd_quote_estimate(self) -> float:
        """Estimate USDC/USD for de-peg detection. Default 1.0 (paper/unknown)."""
        return 1.0

    def fetch_order_by_client_id(self, client_order_id: str) -> dict | None:
        return None


class MultiPaperExchange(MultiExchange):
    """In-process multi-asset spot simulator with a shared cash pot and real prices.

    One persisted book holds the shared cash plus a unit balance per base. Market
    orders fill immediately at the current price minus the taker fee and are
    idempotent by ``clientOrderId`` so a mid-cycle restart never double-trades.
    """

    def __init__(self, bases: list[str], *, quote: str, cash_usdc: float,
                 taker_fee: float, store_path: str,
                 price_source: Callable[[str], float] | None = None) -> None:
        self.mode = "dry_run"
        self.quote = quote
        self.bases = list(bases)
        self.taker_fee = taker_fee
        self._store = Path(store_path)
        self._price_source = price_source or (
            lambda base: public_price(_sym(base, self.quote)))
        bal = {quote: float(cash_usdc)}
        bal.update({b: 0.0 for b in self.bases})
        self._book: dict = {"orders": {}, "balances": bal, "seq": 0}
        if self._store.exists():
            self._book = json.loads(self._store.read_text())
            # tolerate a new base added to config after the book was created
            for b in self.bases:
                self._book["balances"].setdefault(b, 0.0)

    # ---- persistence ----
    def _save(self) -> None:
        self._store.parent.mkdir(parents=True, exist_ok=True)
        self._store.write_text(json.dumps(self._book, indent=2))

    def _next_id(self) -> str:
        self._book["seq"] += 1
        return f"paper-{self._book['seq']}"

    # ---- interface ----
    def price(self, base: str) -> float:
        return float(self._price_source(base))

    def balances(self) -> dict[str, float]:
        return dict(self._book["balances"])

    def market_buy_quote(self, base: str, quote_usdc: float,
                         client_order_id: str) -> dict:
        for o in self._book["orders"].values():
            if o["clientOrderId"] == client_order_id:
                return o  # idempotent
        px = self.price(base)
        bal = self._book["balances"]
        spend = min(quote_usdc, bal.get(self.quote, 0.0))
        units = (spend / px) * (1.0 - self.taker_fee)
        bal[self.quote] = bal.get(self.quote, 0.0) - spend
        bal[base] = bal.get(base, 0.0) + units
        order = self._record("buy", base, px, units, spend, client_order_id)
        return order

    def market_sell(self, base: str, base_amount: float,
                    client_order_id: str) -> dict:
        for o in self._book["orders"].values():
            if o["clientOrderId"] == client_order_id:
                return o  # idempotent
        px = self.price(base)
        bal = self._book["balances"]
        amt = min(base_amount, bal.get(base, 0.0))
        proceeds = amt * px * (1.0 - self.taker_fee)
        bal[base] = bal.get(base, 0.0) - amt
        bal[self.quote] = bal.get(self.quote, 0.0) + proceeds
        order = self._record("sell", base, px, amt, proceeds, client_order_id)
        return order

    def _record(self, side: str, base: str, px: float, amount: float,
                cost: float, coid: str) -> dict:
        oid = self._next_id()
        order = {
            "id": oid, "clientOrderId": coid, "symbol": _sym(base, self.quote),
            "side": side, "type": "market", "price": px, "amount": amount,
            "filled": amount, "status": "closed", "average": px, "cost": cost,
            "timestamp": int(time.time() * 1000),
        }
        self._book["orders"][oid] = order
        self._save()
        return order

    def fetch_order_by_client_id(self, client_order_id: str) -> dict | None:
        for o in self._book["orders"].values():
            if o["clientOrderId"] == client_order_id:
                return o
        return None

    def can_withdraw(self) -> bool:
        return False  # paper key cannot withdraw


class MultiCcxtExchange(MultiExchange):
    """Binance spot via a single ccxt client, operated per base symbol.

    The real account naturally shares one USDC/USDT balance across all pairs, so
    the shared-pot model maps directly. ccxt is imported lazily.
    """

    def __init__(self, bases: list[str], *, quote: str, mode: str,
                 api_key: str, api_secret: str, venue: str = "binance",
                 symbol_overrides: dict[str, str] | None = None) -> None:
        assert mode in ("testnet", "live")
        import ccxt  # lazy: only needed off the paper path
        self.mode = mode
        self.quote = quote
        self.venue = venue
        self.bases = list(bases)
        self._overrides = symbol_overrides or {}
        self._ccxt = ccxt
        opts = {"enableRateLimit": True, "options": {"defaultType": "spot"}}
        self.client = getattr(ccxt, venue)({"apiKey": api_key, "secret": api_secret, **opts})
        if mode == "testnet":
            # only Binance exposes a spot sandbox; guarded in build_multi_exchange
            self.client.set_sandbox_mode(True)

    def _symbol(self, base: str) -> str:
        return self._overrides.get(base, _sym(base, self.quote))

    def price(self, base: str) -> float:
        return float(self.client.fetch_ticker(self._symbol(base))["last"])

    def balances(self) -> dict[str, float]:
        b = self.client.fetch_balance()
        free = b.get("free", {})
        return {k: float(v) for k, v in free.items() if v}

    def market_buy_quote(self, base: str, quote_usdc: float,
                         client_order_id: str) -> dict:
        # 'clientOrderId' is the ccxt-unified param (mapped per venue); 'quoteOrderQty'
        # (spend N quote) is Binance-specific — Kraken markets buys are sized in base, so
        # convert via the current price there.
        params = {"clientOrderId": client_order_id}
        if self.venue == "binance":
            return self.client.create_order(
                self._symbol(base), "market", "buy", None, None,
                {"quoteOrderQty": quote_usdc, **params})
        amount = quote_usdc / self.price(base)
        return self.client.create_order(
            self._symbol(base), "market", "buy", amount, None, params)

    def market_sell(self, base: str, base_amount: float,
                    client_order_id: str) -> dict:
        return self.client.create_order(
            self._symbol(base), "market", "sell", base_amount, None,
            {"clientOrderId": client_order_id})

    def fetch_order_by_client_id(self, client_order_id: str) -> dict | None:
        for base in self.bases:
            try:
                return self.client.fetch_order(
                    None, self._symbol(base),
                    {"origClientOrderId": client_order_id})
            except Exception:  # noqa: BLE001 - try next symbol / give up
                continue
        return None

    def can_withdraw(self) -> bool:
        """Best-effort permission probe. Fail SAFE: if we cannot prove the key is
        withdrawal-disabled, treat it as withdrawal-enabled so live refuses to run."""
        try:
            info = self.client.sapi_get_account_apirestrictions()  # type: ignore[attr-defined]
            return bool(info.get("enableWithdrawals", True))
        except Exception:
            return True  # cannot verify -> assume unsafe

    def usd_quote_estimate(self) -> float:
        """USDC/USD via a liquid base's USDT/USDC ratio (de-peg detector)."""
        base = self.bases[0] if self.bases else "ETH"
        try:
            usdc = float(self.client.fetch_ticker(_sym(base, "USDC"))["last"])
            usdt = float(self.client.fetch_ticker(_sym(base, "USDT"))["last"])
            return usdt / usdc if usdc else 1.0
        except Exception:
            return 1.0  # cannot estimate -> assume pegged (do not false-halt)


def build_multi_exchange(cfg, *, store_path: str) -> MultiExchange:
    """Factory: returns the right MultiExchange for the configured mode + venue."""
    import os

    from .secrets import clean_secret
    venue = getattr(cfg, "venue", "binance")
    if cfg.mode == "dry_run":
        price_fn, _ = public_feeds(venue)          # price_fn expects a full SYMBOL
        overrides = cfg.symbol_overrides
        quote = cfg.quote

        def paper_price(base: str) -> float:       # MultiPaperExchange calls with a BASE
            return price_fn(overrides.get(base) or _sym(base, quote))

        return MultiPaperExchange(
            cfg.bases, quote=cfg.quote, cash_usdc=cfg.stack_usdc,
            taker_fee=cfg.taker_fee, store_path=store_path,
            price_source=paper_price)
    if cfg.mode == "testnet":
        if venue != "binance":
            raise ValueError(f"venue {venue!r} has no spot testnet — use dry_run, then "
                             "live with a small amount (Kraken has no spot sandbox).")
        from .exchange import _binance_secret
        return MultiCcxtExchange(
            cfg.bases, quote=cfg.quote, mode="testnet", venue=venue,
            api_key=clean_secret(os.getenv("BINANCE_TESTNET_API_KEY")),
            api_secret=_binance_secret("BINANCE_TESTNET_API_SECRET",
                                       "BINANCE_TESTNET_API_SECRET_FILE"),
            symbol_overrides=cfg.symbol_overrides)
    if cfg.mode == "live":
        from .exchange import _binance_secret
        key_env = "KRAKEN_API_KEY" if venue == "kraken" else "BINANCE_API_KEY"
        sec_env = "KRAKEN_API_SECRET" if venue == "kraken" else "BINANCE_API_SECRET"
        return MultiCcxtExchange(
            cfg.bases, quote=cfg.quote, mode="live", venue=venue,
            api_key=clean_secret(os.getenv(key_env)),
            api_secret=_binance_secret(sec_env, sec_env + "_FILE"),
            symbol_overrides=cfg.symbol_overrides)
    raise ValueError(f"unknown mode {cfg.mode!r}")
