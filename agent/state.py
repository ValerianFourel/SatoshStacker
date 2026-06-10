"""Crash-safe persistent state (SQLite) + reconcile-on-startup.

Invariant: the **exchange is the source of truth** for fills and open orders.
On startup (and every cycle) the agent reconciles its recorded rungs against the
exchange before sizing or placing anything. Deployed totals are *derived* from
the `fills` table so they can never drift from what actually executed.
"""
from __future__ import annotations

import json
import sqlite3
import time
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS rungs (
  client_order_id   TEXT PRIMARY KEY,
  generation        INTEGER NOT NULL,
  idx               INTEGER NOT NULL,
  price             REAL NOT NULL,
  usdc              REAL NOT NULL,
  amount_btc        REAL NOT NULL,
  exchange_order_id TEXT,
  status            TEXT NOT NULL,  -- pending|resting|filled|canceled
  placed_day        INTEGER         -- UTC day (epoch//86400*86400) the rung was committed
);
CREATE TABLE IF NOT EXISTS fills (
  client_order_id TEXT PRIMARY KEY,
  price   REAL NOT NULL,
  btc     REAL NOT NULL,
  usdc    REAL NOT NULL,
  fee_btc REAL NOT NULL,
  ts      INTEGER NOT NULL,
  kind    TEXT NOT NULL  -- maker|taker
);
CREATE TABLE IF NOT EXISTS schedule (
  idx     INTEGER PRIMARY KEY,
  due_ts  INTEGER NOT NULL,
  usdc    REAL NOT NULL,
  status  TEXT NOT NULL,  -- pending|done
  client_order_id TEXT
);
CREATE TABLE IF NOT EXISTS journal (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts INTEGER NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL);
"""


class Store:
    def __init__(self, path: str) -> None:
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ---- meta (typed key/value) ----
    def get_meta(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def set_meta(self, key: str, value: Any) -> None:
        self.conn.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value)))
        self.conn.commit()

    # ---- rungs ----
    def upsert_rung(self, *, client_order_id: str, generation: int, idx: int,
                    price: float, usdc: float, amount_btc: float,
                    exchange_order_id: str | None, status: str,
                    placed_day: int | None = None) -> None:
        # placed_day is set once (on the write-ahead insert) and preserved on
        # conflict, so the per-day cap counter can be derived from the rung rows.
        self.conn.execute(
            "INSERT INTO rungs(client_order_id,generation,idx,price,usdc,amount_btc,"
            "exchange_order_id,status,placed_day) VALUES(?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(client_order_id) DO UPDATE SET "
            "exchange_order_id=excluded.exchange_order_id, status=excluded.status",
            (client_order_id, generation, idx, price, usdc, amount_btc,
             exchange_order_id, status, placed_day))
        self.conn.commit()

    def set_rung_status(self, client_order_id: str, status: str,
                        exchange_order_id: str | None = None) -> None:
        if exchange_order_id is not None:
            self.conn.execute(
                "UPDATE rungs SET status=?, exchange_order_id=? WHERE client_order_id=?",
                (status, exchange_order_id, client_order_id))
        else:
            self.conn.execute("UPDATE rungs SET status=? WHERE client_order_id=?",
                              (status, client_order_id))
        self.conn.commit()

    def rungs_by_status(self, *statuses: str) -> list[sqlite3.Row]:
        q = ",".join("?" * len(statuses))
        return self.conn.execute(
            f"SELECT * FROM rungs WHERE status IN ({q}) ORDER BY price DESC",
            statuses).fetchall()

    def resting_usdc(self) -> float:
        """USDC committed to currently-resting (open, unfilled) limit rungs."""
        return float(self.conn.execute(
            "SELECT COALESCE(SUM(usdc),0) AS s FROM rungs WHERE status='resting'"
        ).fetchone()["s"])

    def has_client_order(self, client_order_id: str) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM rungs WHERE client_order_id=? "
            "UNION SELECT 1 FROM fills WHERE client_order_id=?",
            (client_order_id, client_order_id)).fetchone() is not None

    def get_rung(self, client_order_id: str):
        return self.conn.execute(
            "SELECT * FROM rungs WHERE client_order_id=?",
            (client_order_id,)).fetchone()

    def fill_exists(self, client_order_id: str) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM fills WHERE client_order_id=?",
            (client_order_id,)).fetchone() is not None

    # ---- fills (deployed totals derived from here) ----
    def record_fill(self, *, client_order_id: str, price: float, btc: float,
                    usdc: float, fee_btc: float, kind: str) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO fills(client_order_id,price,btc,usdc,fee_btc,ts,kind)"
            " VALUES(?,?,?,?,?,?,?)",
            (client_order_id, price, btc, usdc, fee_btc, int(time.time()), kind))
        self.conn.commit()

    def total_deployed(self) -> float:
        return float(self.conn.execute(
            "SELECT COALESCE(SUM(usdc),0) AS s FROM fills").fetchone()["s"])

    def total_btc(self) -> float:
        return float(self.conn.execute(
            "SELECT COALESCE(SUM(btc),0) AS s FROM fills").fetchone()["s"])

    def deployed_since(self, ts: int) -> float:
        return float(self.conn.execute(
            "SELECT COALESCE(SUM(usdc),0) AS s FROM fills WHERE ts>=?",
            (ts,)).fetchone()["s"])

    # ---- per-UTC-day committed-notional (per-day deploy cap) ----
    def committed_today(self, now_ts: float) -> float:
        """Resting/forced notional committed this UTC day, DERIVED from the rung
        rows (not an incremental counter) so it is crash-atomic with the
        write-ahead insert and cannot be breached by a torn write."""
        day = int(now_ts - (now_ts % 86_400))
        return float(self.conn.execute(
            "SELECT COALESCE(SUM(usdc),0) AS s FROM rungs "
            "WHERE placed_day=? AND status IN ('pending','resting','filled')",
            (day,)).fetchone()["s"])

    def fills(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM fills ORDER BY ts").fetchall()

    # ---- schedule (deploy-by-deadline DCA tranches) ----
    def set_schedule(self, tranches: list[tuple[int, int, float]]) -> None:
        """tranches: list of (idx, due_ts, usdc)."""
        self.conn.executemany(
            "INSERT OR REPLACE INTO schedule(idx,due_ts,usdc,status,client_order_id)"
            " VALUES(?,?,?, 'pending', NULL)", tranches)
        self.conn.commit()

    def due_tranches(self, now_ts: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM schedule WHERE status='pending' AND due_ts<=? ORDER BY idx",
            (now_ts,)).fetchall()

    def mark_tranche_done(self, idx: int, client_order_id: str) -> None:
        self.conn.execute(
            "UPDATE schedule SET status='done', client_order_id=? WHERE idx=?",
            (client_order_id, idx))
        self.conn.commit()

    def has_schedule(self) -> bool:
        return self.conn.execute("SELECT 1 FROM schedule LIMIT 1").fetchone() is not None

    def pending_schedule_usdc(self) -> float:
        return float(self.conn.execute(
            "SELECT COALESCE(SUM(usdc),0) AS s FROM schedule WHERE status='pending'"
        ).fetchone()["s"])

    # ---- journal / halt ----
    def journal(self, kind: str, payload: Any) -> None:
        self.conn.execute("INSERT INTO journal(ts,kind,payload) VALUES(?,?,?)",
                          (int(time.time()), kind, json.dumps(payload)))
        self.conn.commit()

    def halt(self, reason: str) -> None:
        self.set_meta("halted", True)
        self.set_meta("halt_reason", reason)
        self.journal("halt", {"reason": reason})

    def is_halted(self) -> bool:
        return bool(self.get_meta("halted", False))

    def clear_halt(self) -> None:
        self.set_meta("halted", False)
        self.set_meta("halt_reason", None)
        self.journal("halt_cleared", {})

    def _adopt_order(self, coid: str, row, o: dict, changes: list[str]) -> None:
        """Apply an exchange order's truth to a local rung row."""
        status = o.get("status")
        if status == "closed" and (o.get("filled") or 0) > 0:
            fill_px = o.get("average") or o.get("price") or row["price"]
            btc = float(o.get("filled") or row["amount_btc"])
            fee_btc = float((o.get("fee") or {}).get("cost") or 0.0)
            usdc = float(o.get("cost") or btc * fill_px)
            self.record_fill(client_order_id=coid, price=fill_px, btc=btc - fee_btc,
                             usdc=usdc, fee_btc=fee_btc, kind="maker")
            self.set_rung_status(coid, "filled", exchange_order_id=str(o.get("id")))
            changes.append(f"{coid}: filled {btc:.8f} BTC @ {fill_px:,.0f}")
        elif status == "open":
            if row["status"] == "pending":
                self.set_rung_status(coid, "resting", exchange_order_id=str(o.get("id")))
                changes.append(f"{coid}: adopted orphaned open order")
        elif status == "canceled":
            self.set_rung_status(coid, "canceled")
            changes.append(f"{coid}: canceled on exchange")

    # ---- reconcile (exchange = source of truth) ----
    def reconcile(self, exchange) -> list[str]:
        """Sync recorded rungs against the EXCHANGE (the source of truth).

        Crash-safe: a rung may be persisted as ``pending`` *before* its order is
        placed (write-ahead). Here we look each rung up on the exchange — by
        exchange order id if known, else by our clientOrderId — and adopt whatever
        actually happened (filled / still open / canceled). This recovers orders
        orphaned by a crash between placement and persistence, so a fill is never
        lost and a duplicate is never placed. Transient lookup errors are left
        untouched (never cancel on a transient failure).
        """
        changes: list[str] = []
        for row in self.rungs_by_status("resting", "pending"):
            coid = row["client_order_id"]
            oid = row["exchange_order_id"]
            try:
                o = exchange.fetch_order(oid) if oid \
                    else exchange.fetch_order_by_client_id(coid)
            except Exception:  # noqa: BLE001 - transient: leave as-is, retry next cycle
                continue
            if o is None:
                # not on the exchange. A 'resting' rung with a known id that has
                # vanished was canceled; a 'pending' rung was never placed — leave
                # it for ensure_resting to place (idempotent).
                if row["status"] == "resting":
                    self.set_rung_status(coid, "canceled")
                    changes.append(f"{coid}: missing on exchange -> canceled")
                continue
            self._adopt_order(coid, row, o, changes)
        return changes
