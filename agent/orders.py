"""Order placement: idempotent maker limit rungs + deadline market DCA.

Idempotency & crash-safety: every order carries a stable ``clientOrderId``
(``ss-g{gen}-r{idx}`` for rungs, ``ss-dca-{idx}`` for deadline tranches).

**Write-ahead:** a rung is persisted as ``pending`` (with its clientOrderId)
BEFORE the exchange order is placed, then promoted to ``resting`` with the
exchange id after. If the process dies between the two steps, reconcile finds
the order on the exchange by clientOrderId and adopts it (or, if it was never
placed, the pending rung is re-attempted) — so a crash can neither double-buy
nor lose a fill. Exception text is redacted before it ever reaches the journal.
"""
from __future__ import annotations

from .config import AgentConfig
from .exchange import Exchange
from .ladder import Rung
from .risk import RiskGate
from .secrets import redact
from .state import Store


def rung_coid(generation: int, idx: int) -> str:
    return f"ss-g{generation}-r{idx}"


def dca_coid(idx: int) -> str:
    return f"ss-dca-{idx}"


class OrderManager:
    def __init__(self, cfg: AgentConfig, exchange: Exchange, store: Store,
                 risk: RiskGate) -> None:
        self.cfg = cfg
        self.ex = exchange
        self.store = store
        self.risk = risk

    def ensure_resting(self, rungs: list[Rung], generation: int, *,
                       size_multiplier: float = 1.0,
                       now_ts: float | None = None) -> list[str]:
        """Place any not-yet-resting rungs of this generation (write-ahead, idempotent).

        The LLM ``size_multiplier`` (already clamped to [floor,1]) shrinks each NEW
        rung; the deferred remainder is simply not committed (stays available for a
        lower generation or the deadline).
        """
        placed: list[str] = []
        for r in rungs:
            coid = rung_coid(generation, r.index)
            if self.store.fill_exists(coid):
                continue  # already executed
            row = self.store.get_rung(coid)
            if row is not None and row["status"] in ("resting", "filled", "canceled"):
                continue  # already handled — idempotent no-op

            if row is None:
                usdc = r.usdc * size_multiplier
                chk = self.risk.check_order(price=r.price, usdc=usdc,
                                            expected_level=r.price, now=now_ts)
                if not chk.ok:
                    self.store.journal("rung_skipped",
                                       {"coid": coid, "reason": chk.reason, "usdc": usdc})
                    continue
                amount_btc = usdc / r.price
                price, amount = r.price, amount_btc
                # WRITE-AHEAD: persist pending (with clientOrderId + placed_day) BEFORE
                # placing. placed_day makes the per-day cap counter crash-atomic — it is
                # derived from this row, so a crash before placement cannot evade the cap.
                placed_day = int(now_ts - (now_ts % 86_400)) if now_ts is not None else None
                self.store.upsert_rung(
                    client_order_id=coid, generation=generation, idx=r.index,
                    price=price, usdc=usdc, amount_btc=amount_btc,
                    exchange_order_id=None, status="pending", placed_day=placed_day)
            else:
                # an existing 'pending' rung from a prior crashed attempt: re-place
                # it (commit already counted). Re-validate halt/min-notional only.
                chk = self.risk.check_order(price=row["price"], usdc=row["usdc"],
                                            expected_level=row["price"], now=now_ts,
                                            enforce_daily_cap=False)
                if not chk.ok:
                    self.store.journal("rung_skipped",
                                       {"coid": coid, "reason": chk.reason})
                    continue
                price, amount = row["price"], row["amount_btc"]

            try:
                order = self.ex.create_limit_buy(price, amount, coid)
                self.risk.note_api_result(True)
            except Exception as e:  # noqa: BLE001 - stays 'pending', retried next cycle
                self.risk.note_api_result(False)
                self.store.journal("order_error", {"coid": coid, "error": redact(e)})
                continue
            self.store.set_rung_status(coid, "resting", exchange_order_id=str(order.get("id")))
            self.store.journal("rung_placed",
                               {"coid": coid, "price": price, "usdc": round(price * amount, 2),
                                "mult": size_multiplier})
            placed.append(coid)
        return placed

    def cancel_all_resting(self) -> int:
        """Cancel every resting/pending rung (used at the deadline transition)."""
        n = 0
        for row in self.store.rungs_by_status("resting", "pending"):
            oid = row["exchange_order_id"]
            try:
                if oid:
                    self.ex.cancel_order(oid)
                self.store.set_rung_status(row["client_order_id"], "canceled")
                n += 1
            except Exception as e:  # noqa: BLE001
                self.store.journal("cancel_error",
                                   {"coid": row["client_order_id"], "error": redact(e)})
        return n

    def _record_market_fill(self, coid: str, order: dict) -> None:
        fill_px = float(order.get("average") or order.get("price") or self.ex.get_price())
        btc = float(order.get("filled") or order.get("amount") or 0.0)
        fee_btc = float((order.get("fee") or {}).get("cost") or 0.0)
        cost = float(order.get("cost") or 0.0)
        self.store.record_fill(client_order_id=coid, price=fill_px, btc=btc - fee_btc,
                               usdc=cost, fee_btc=fee_btc, kind="taker")

    def execute_deadline_tranche(self, idx: int, usdc: float) -> bool:
        """Market-buy one deploy-by-deadline DCA tranche. Bypasses the per-day cap
        (the deadline always wins) but still respects halt / de-peg / min-notional.
        Crash-safe: recovers an already-placed tranche by clientOrderId."""
        coid = dca_coid(idx)
        if self.store.fill_exists(coid):
            self.store.mark_tranche_done(idx, coid)
            return True
        # recovery: a prior crashed attempt may have already placed/filled this
        try:
            existing = self.ex.fetch_order_by_client_id(coid)
        except Exception:  # noqa: BLE001
            existing = None
        if existing and existing.get("status") == "closed" and (existing.get("filled") or 0) > 0:
            self._record_market_fill(coid, existing)
            self.store.mark_tranche_done(idx, coid)
            return True

        chk = self.risk.check_order(price=self.ex.get_price(), usdc=usdc,
                                    expected_level=0.0, enforce_daily_cap=False)
        if not chk.ok:
            self.store.journal("dca_skipped", {"idx": idx, "reason": chk.reason})
            return False
        try:
            order = self.ex.create_market_buy_quote(usdc, coid)
            self.risk.note_api_result(True)
        except Exception as e:  # noqa: BLE001
            self.risk.note_api_result(False)
            self.store.journal("dca_error", {"idx": idx, "error": redact(e)})
            return False
        self._record_market_fill(coid, order)
        self.store.mark_tranche_done(idx, coid)
        self.store.journal("dca_filled", {"idx": idx, "usdc": round(usdc, 2)})
        return True
