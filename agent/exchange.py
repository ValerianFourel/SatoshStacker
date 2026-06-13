"""Exchange abstraction: paper (dry_run) and ccxt/Binance (testnet, live).

The agent talks ONLY to this interface. `dry_run` uses `PaperExchange` — a
self-contained simulator with its own persisted order book (an independent
"source of truth" so reconcile-on-restart is genuinely exercised) and real
public Binance prices. `testnet`/`live` use `CcxtExchange`, which imports ccxt
lazily so the paper path and the test suite need no exchange SDK.

Order dicts follow the ccxt unified shape (subset the agent uses):
    {id, clientOrderId, symbol, side, type, price, amount, filled,
     status: open|closed|canceled, average, cost, fee:{cost,currency}, timestamp}
"""
from __future__ import annotations

import abc
import json
import time
from pathlib import Path
from typing import Callable

import requests

_BINANCE_PUBLIC = "https://api.binance.com/api/v3/ticker/price"


def binance_symbol(symbol: str) -> str:
    """'BTC/USDC' -> 'BTCUSDC' (Binance REST symbol)."""
    return symbol.replace("/", "")


def public_price(symbol: str, *, timeout: float = 8.0) -> float:
    """Last trade price from Binance public API (no key). Falls back USDC->USDT."""
    bs = binance_symbol(symbol)
    for sym in (bs, bs.replace("USDC", "USDT")):
        try:
            r = requests.get(_BINANCE_PUBLIC, params={"symbol": sym}, timeout=timeout)
            r.raise_for_status()
            return float(r.json()["price"])
        except Exception:  # noqa: BLE001 - try the fallback symbol
            continue
    raise RuntimeError(f"could not fetch public price for {symbol}")


def public_ohlcv(symbol: str, timeframe: str = "4h", limit: int = 200,
                 *, timeout: float = 10.0) -> list[list[float]]:
    """Recent [open, high, low, close] candles from Binance public API (no key).
    Market data is identical for testnet/live, so the trader uses this for momentum."""
    bs = binance_symbol(symbol)
    for sym in (bs, bs.replace("USDC", "USDT")):
        try:
            r = requests.get("https://api.binance.com/api/v3/klines",
                             params={"symbol": sym, "interval": timeframe, "limit": limit},
                             timeout=timeout)
            r.raise_for_status()
            return [[float(k[1]), float(k[2]), float(k[3]), float(k[4])] for k in r.json()]
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError(f"could not fetch public ohlcv for {symbol}")


def public_klines(symbol: str, timeframe: str = "1m", limit: int = 200,
                  *, timeout: float = 10.0) -> list[list[float]]:
    """Recent [open_time_ms, open, high, low, close, volume] candles (no key).

    Like ``public_ohlcv`` but keeps the open-time and base-asset volume, which the
    market monitor needs for volume-surge and realized-volatility metrics."""
    bs = binance_symbol(symbol)
    for sym in (bs, bs.replace("USDC", "USDT")):
        try:
            r = requests.get("https://api.binance.com/api/v3/klines",
                             params={"symbol": sym, "interval": timeframe, "limit": limit},
                             timeout=timeout)
            r.raise_for_status()
            return [[float(k[0]), float(k[1]), float(k[2]), float(k[3]),
                     float(k[4]), float(k[5])] for k in r.json()]
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError(f"could not fetch public klines for {symbol}")


def public_order_book(symbol: str, limit: int = 100,
                      *, timeout: float = 8.0) -> dict[str, list[list[float]]]:
    """L2 order book {"bids":[[price,qty],...], "asks":[...]} from Binance (no key).
    Bids are descending, asks ascending (Binance order). Falls back USDC->USDT."""
    bs = binance_symbol(symbol)
    for sym in (bs, bs.replace("USDC", "USDT")):
        try:
            r = requests.get("https://api.binance.com/api/v3/depth",
                             params={"symbol": sym, "limit": limit}, timeout=timeout)
            r.raise_for_status()
            d = r.json()
            return {"bids": [[float(p), float(q)] for p, q in d.get("bids", [])],
                    "asks": [[float(p), float(q)] for p, q in d.get("asks", [])]}
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError(f"could not fetch public order book for {symbol}")


class Exchange(abc.ABC):
    """Minimal spot-buy interface the agent depends on."""

    mode: str
    symbol: str

    @abc.abstractmethod
    def get_price(self) -> float: ...

    @abc.abstractmethod
    def fetch_balance(self) -> dict[str, float]: ...

    @abc.abstractmethod
    def create_limit_buy(self, price: float, amount_btc: float,
                         client_order_id: str) -> dict: ...

    @abc.abstractmethod
    def create_market_buy_quote(self, quote_usdc: float,
                                client_order_id: str) -> dict: ...

    def create_market_sell(self, amount_btc: float, client_order_id: str) -> dict:
        """Market-sell `amount_btc` BTC for USDC (used by the active trader)."""
        raise NotImplementedError

    @abc.abstractmethod
    def cancel_order(self, order_id: str) -> None: ...

    @abc.abstractmethod
    def fetch_open_orders(self) -> list[dict]: ...

    @abc.abstractmethod
    def fetch_order(self, order_id: str) -> dict: ...

    @abc.abstractmethod
    def can_withdraw(self) -> bool:
        """True if the API key has withdrawal permission. Live MUST refuse this."""

    def settle(self) -> list[dict]:  # paper-only hook; real exchanges fill autonomously
        return []

    def usdc_usd_estimate(self) -> float:
        """Estimate USDC/USD for de-peg detection. Default 1.0 (paper/unknown)."""
        return 1.0

    def fetch_order_by_client_id(self, client_order_id: str) -> dict | None:
        """Look up an order by our clientOrderId (any status), or None if unknown.

        Used by reconcile to recover from a crash between placing an order and
        persisting it locally — so an orphaned exchange order is never missed.
        """
        return None


class PaperExchange(Exchange):
    """In-process spot simulator with persisted books and real public prices.

    Resting limit buys fill when the (real or injected) price touches them, at the
    maker fee; market buys fill immediately at the current price, taker fee. Its
    book persists to a JSON file so a restart can reconcile against it.
    """

    def __init__(self, symbol: str, *, budget_usdc: float, maker_fee: float,
                 taker_fee: float, store_path: str,
                 price_source: Callable[[], float] | None = None) -> None:
        self.mode = "dry_run"
        self.symbol = symbol
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self._store = Path(store_path)
        self._price_source = price_source or (lambda: public_price(symbol))
        self._book: dict = {"orders": {}, "balances": {"USDC": budget_usdc, "BTC": 0.0},
                            "seq": 0}
        if self._store.exists():
            self._book = json.loads(self._store.read_text())

    # ---- persistence ----
    def _save(self) -> None:
        self._store.parent.mkdir(parents=True, exist_ok=True)
        self._store.write_text(json.dumps(self._book, indent=2))

    def _next_id(self) -> str:
        self._book["seq"] += 1
        return f"paper-{self._book['seq']}"

    # ---- interface ----
    def get_price(self) -> float:
        return float(self._price_source())

    def fetch_balance(self) -> dict[str, float]:
        return dict(self._book["balances"])

    def create_limit_buy(self, price: float, amount_btc: float,
                         client_order_id: str) -> dict:
        # idempotency: if a live (open) order with this clientOrderId exists, return it
        for o in self._book["orders"].values():
            if o["clientOrderId"] == client_order_id and o["status"] == "open":
                return o
        oid = self._next_id()
        order = {
            "id": oid, "clientOrderId": client_order_id, "symbol": self.symbol,
            "side": "buy", "type": "limit", "price": float(price),
            "amount": float(amount_btc), "filled": 0.0, "status": "open",
            "average": None, "cost": 0.0, "fee": {"cost": 0.0, "currency": "BTC"},
            "timestamp": int(time.time() * 1000),
        }
        self._book["orders"][oid] = order
        self._save()
        return order

    def create_market_buy_quote(self, quote_usdc: float, client_order_id: str) -> dict:
        for o in self._book["orders"].values():
            if o["clientOrderId"] == client_order_id:
                return o  # already executed (idempotent)
        px = self.get_price()
        bal = self._book["balances"]
        spend = min(quote_usdc, bal["USDC"])
        btc = (spend / px) * (1.0 - self.taker_fee)
        bal["USDC"] -= spend
        bal["BTC"] += btc
        oid = self._next_id()
        order = {
            "id": oid, "clientOrderId": client_order_id, "symbol": self.symbol,
            "side": "buy", "type": "market", "price": px, "amount": btc,
            "filled": btc, "status": "closed", "average": px, "cost": spend,
            "fee": {"cost": btc * self.taker_fee / (1 - self.taker_fee), "currency": "BTC"},
            "timestamp": int(time.time() * 1000),
        }
        self._book["orders"][oid] = order
        self._save()
        return order

    def create_market_sell(self, amount_btc: float, client_order_id: str) -> dict:
        for o in self._book["orders"].values():
            if o["clientOrderId"] == client_order_id:
                return o  # idempotent
        px = self.get_price()
        bal = self._book["balances"]
        amt = min(amount_btc, bal["BTC"])
        proceeds = amt * px * (1.0 - self.taker_fee)
        bal["BTC"] -= amt
        bal["USDC"] += proceeds
        oid = self._next_id()
        order = {
            "id": oid, "clientOrderId": client_order_id, "symbol": self.symbol,
            "side": "sell", "type": "market", "price": px, "amount": amt,
            "filled": amt, "status": "closed", "average": px, "cost": proceeds,
            "fee": {"cost": amt * px * self.taker_fee, "currency": "USDC"},
            "timestamp": int(time.time() * 1000),
        }
        self._book["orders"][oid] = order
        self._save()
        return order

    def cancel_order(self, order_id: str) -> None:
        o = self._book["orders"].get(order_id)
        if o and o["status"] == "open":
            o["status"] = "canceled"
            self._save()

    def fetch_open_orders(self) -> list[dict]:
        return [o for o in self._book["orders"].values() if o["status"] == "open"]

    def fetch_order(self, order_id: str) -> dict:
        return self._book["orders"][order_id]

    def fetch_order_by_client_id(self, client_order_id: str) -> dict | None:
        for o in self._book["orders"].values():
            if o["clientOrderId"] == client_order_id:
                return o
        return None

    def can_withdraw(self) -> bool:
        return False  # paper key cannot withdraw

    def settle(self) -> list[dict]:
        """Fill any resting limit buy the current price has reached. Returns fills."""
        px = self.get_price()
        bal = self._book["balances"]
        filled: list[dict] = []
        for o in self._book["orders"].values():
            if o["status"] != "open" or o["type"] != "limit":
                continue
            if px <= o["price"]:
                fill_px = min(o["price"], px)
                cost = o["amount"] * fill_px
                if cost > bal["USDC"] + 1e-9:  # insufficient (shouldn't happen if sized right)
                    continue
                btc = o["amount"] * (1.0 - self.maker_fee)
                bal["USDC"] -= cost
                bal["BTC"] += btc
                o.update(status="closed", filled=o["amount"], average=fill_px,
                         cost=cost, fee={"cost": o["amount"] * self.maker_fee,
                                         "currency": "BTC"})
                filled.append(o)
        if filled:
            self._save()
        return filled


class CcxtExchange(Exchange):
    """Binance spot via ccxt (testnet or live). ccxt imported lazily."""

    def __init__(self, symbol: str, *, mode: str, api_key: str, api_secret: str) -> None:
        assert mode in ("testnet", "live")
        import ccxt  # lazy: only needed off the paper path
        self.mode = mode
        self.symbol = symbol
        self._ccxt = ccxt
        self.client = ccxt.binance({
            "apiKey": api_key, "secret": api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
        if mode == "testnet":
            self.client.set_sandbox_mode(True)

    def get_price(self) -> float:
        return float(self.client.fetch_ticker(self.symbol)["last"])

    def fetch_balance(self) -> dict[str, float]:
        b = self.client.fetch_balance()
        free = b.get("free", {})
        return {k: float(v) for k, v in free.items() if v}

    def create_limit_buy(self, price: float, amount_btc: float,
                         client_order_id: str) -> dict:
        return self.client.create_order(
            self.symbol, "limit", "buy", amount_btc, price,
            {"newClientOrderId": client_order_id})

    def create_market_buy_quote(self, quote_usdc: float, client_order_id: str) -> dict:
        # Binance spot market buy by quote amount (quoteOrderQty)
        return self.client.create_order(
            self.symbol, "market", "buy", None, None,
            {"quoteOrderQty": quote_usdc, "newClientOrderId": client_order_id})

    def create_market_sell(self, amount_btc: float, client_order_id: str) -> dict:
        return self.client.create_order(
            self.symbol, "market", "sell", amount_btc, None,
            {"newClientOrderId": client_order_id})

    def cancel_order(self, order_id: str) -> None:
        self.client.cancel_order(order_id, self.symbol)

    def fetch_open_orders(self) -> list[dict]:
        return self.client.fetch_open_orders(self.symbol)

    def fetch_order(self, order_id: str) -> dict:
        return self.client.fetch_order(order_id, self.symbol)

    def fetch_order_by_client_id(self, client_order_id: str) -> dict | None:
        # Binance supports querying by origClientOrderId; fall back to open-order scan.
        try:
            return self.client.fetch_order(None, self.symbol,
                                           {"origClientOrderId": client_order_id})
        except Exception:
            try:
                for o in self.client.fetch_open_orders(self.symbol):
                    if o.get("clientOrderId") == client_order_id:
                        return o
            except Exception:
                return None
            return None

    def can_withdraw(self) -> bool:
        """Best-effort permission probe. Fail SAFE: if we cannot prove the key is
        withdrawal-disabled, treat it as withdrawal-enabled (return True) so live
        startup refuses to run."""
        try:
            info = self.client.sapi_get_account_apirestrictions()  # type: ignore[attr-defined]
            return bool(info.get("enableWithdrawals", True))
        except Exception:
            return True  # cannot verify -> assume unsafe

    def usdc_usd_estimate(self) -> float:
        """USDC/USD via the BTC/USDT vs BTC/USDC ratio (de-peg detector)."""
        try:
            usdc = float(self.client.fetch_ticker(self.symbol)["last"])
            usdt = float(self.client.fetch_ticker(
                self.symbol.replace("USDC", "USDT"))["last"])
            return usdt / usdc if usdc else 1.0
        except Exception:
            return 1.0  # cannot estimate -> assume pegged (do not false-halt)


def _binance_secret(inline_env: str, file_env: str) -> str:
    """Resolve a Binance API secret. For HMAC keys it is the one-line secret in
    ``inline_env``. For Ed25519/RSA keys it is the PRIVATE KEY in PEM form, supplied
    either as a file path in ``file_env`` (recommended — PEMs are multi-line) or
    inline with escaped ``\\n`` newlines. ccxt detects the PEM and signs accordingly.
    """
    import os
    from pathlib import Path

    from .secrets import clean_secret
    path = clean_secret(os.getenv(file_env))
    if path:
        return Path(path).expanduser().read_text().strip()
    val = clean_secret(os.getenv(inline_env))
    return val.replace("\\n", "\n") if "BEGIN" in val else val


def build_exchange(cfg, *, store_path: str) -> Exchange:
    """Factory: returns the right Exchange for the configured mode."""
    import os

    from .secrets import clean_secret
    if cfg.mode == "dry_run":
        return PaperExchange(
            cfg.symbol, budget_usdc=cfg.ladder.budget_usdc,
            maker_fee=cfg.maker_fee, taker_fee=cfg.taker_fee, store_path=store_path)
    if cfg.mode == "testnet":
        return CcxtExchange(
            cfg.symbol, mode="testnet",
            api_key=clean_secret(os.getenv("BINANCE_TESTNET_API_KEY")),
            api_secret=_binance_secret("BINANCE_TESTNET_API_SECRET",
                                       "BINANCE_TESTNET_API_SECRET_FILE"))
    if cfg.mode == "live":
        return CcxtExchange(
            cfg.symbol, mode="live",
            api_key=clean_secret(os.getenv("BINANCE_API_KEY")),
            api_secret=_binance_secret("BINANCE_API_SECRET", "BINANCE_API_SECRET_FILE"))
    raise ValueError(f"unknown mode {cfg.mode!r}")
