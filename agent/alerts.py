"""User-defined trigger alerts: "ping me when rsi > 70", "price < 60000", etc.

A rule is {metric, op, value}. Each scan the monitor resolves the metric from the live
snapshot and fires when the condition is met — once per crossing (re-arms when the
condition clears), so you get a ping each time it crosses, not every 15s while true.
Rules persist to a JSON file. Atomic + fail-safe.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import time

_OPS = {">": lambda a, b: a > b, "<": lambda a, b: a < b,
        ">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b,
        "=": lambda a, b: abs(a - b) < 1e-9, "==": lambda a, b: abs(a - b) < 1e-9}

# user-facing metric names -> where they live in the snapshot (see resolve_metric)
SUPPORTED = ("price, rsi (=rsi_14), rsi_5m/rsi_1h/rsi_4h/rsi_1d, funding, oi_change, "
             "ls (long/short), ret_1m/ret_5m/ret_1h, change_24h, range_pos, atr, trend, "
             "vol_z, vol_surge, high_24h, low_24h")

_CMD = re.compile(r"^\s*([a-z][\w]*(?:\([\w]+\))?)\s*(>=|<=|==|>|<|=)\s*(-?\d+(?:\.\d+)?)\s*$", re.I)


def parse_rule(text: str):
    """Parse 'rsi > 70' / 'price<60000' / 'funding >= 0.05' -> (metric, op, value) or None."""
    m = _CMD.match((text or "").strip())
    if not m:
        return None
    metric = m.group(1).lower().replace("(", "_").replace(")", "").strip("_")
    return metric, m.group(2), float(m.group(3))


def resolve_metric(snapshot: dict, name: str):
    """Current value of a named metric from the snapshot (or None if unknown/missing)."""
    t = snapshot.get("technicals", {})
    v = snapshot.get("volume", {})
    f = snapshot.get("futures", {})
    mt = snapshot.get("multi_tf", {})
    name = name.lower().strip()
    direct = {
        "price": snapshot.get("price"),
        "rsi": t.get("rsi_14"), "rsi_14": t.get("rsi_14"),
        "ret_1m": t.get("ret_1m_pct"), "ret_5m": t.get("ret_5m_pct"), "ret_1h": t.get("ret_1h_pct"),
        "change_24h": t.get("change_24h_pct"), "change": t.get("change_24h_pct"),
        "range_pos": t.get("range_position_pct"), "range": t.get("range_position_pct"),
        "atr": t.get("atr_pct"), "trend": t.get("ema_trend_pct"), "ema_trend": t.get("ema_trend_pct"),
        "high_24h": t.get("high_24h"), "low_24h": t.get("low_24h"),
        "vol_z": v.get("z"), "volume_z": v.get("z"), "vol_surge": v.get("surge_x"),
        "funding": f.get("funding_rate_pct"), "oi_change": f.get("oi_change_24h_pct"),
        "ls": f.get("long_short_ratio"), "long_short": f.get("long_short_ratio"),
        "longshort": f.get("long_short_ratio"),
    }
    if name in direct:
        return direct[name]
    mtf = re.match(r"rsi_(5m|15m|30m|1h|4h|1d)$", name)
    if mtf and mtf.group(1) in mt:
        return mt[mtf.group(1)].get("rsi_14")
    return None


def evaluate(rules: list, snapshot: dict) -> list:
    """Return [(rule, value)] for rules that fire NOW; mutates each rule's 'armed' flag
    (fire once per crossing, re-arm when the condition clears)."""
    fired = []
    for r in rules:
        val = resolve_metric(snapshot, r["metric"])
        if val is None:
            continue
        cond = _OPS.get(r["op"], lambda a, b: False)(val, r["value"])
        if cond and r.get("armed", True):
            r["armed"] = False
            fired.append((r, val))
        elif not cond:
            r["armed"] = True
    return fired


class AlertStore:
    def __init__(self, path: str) -> None:
        self.path = path

    def load(self) -> list:
        try:
            with open(self.path) as f:
                d = json.load(f)
            return d if isinstance(d, list) else []
        except Exception:  # noqa: BLE001
            return []

    def save(self, rules: list) -> None:
        try:
            d = os.path.dirname(self.path) or "."
            os.makedirs(d, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
            with os.fdopen(fd, "w") as f:
                json.dump(rules, f)
            os.replace(tmp, self.path)
        except Exception:  # noqa: BLE001
            pass

    def add(self, metric: str, op: str, value: float, *, chat: str = "", clock=time.time) -> dict:
        rules = self.load()
        rid = (max((r.get("id", 0) for r in rules), default=0) + 1)
        rule = {"id": rid, "metric": metric, "op": op, "value": value,
                "chat": chat, "armed": True, "created": clock()}
        rules.append(rule)
        self.save(rules)
        return rule

    def remove(self, rid: int) -> bool:
        rules = self.load()
        kept = [r for r in rules if r.get("id") != rid]
        self.save(kept)
        return len(kept) != len(rules)

    def clear(self) -> None:
        self.save([])
