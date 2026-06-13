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


# ── natural-language composite alarms ("tell me when those 3 metrics are in sell mode") ──
# Per-metric "mode" condition: what "in sell mode" (frothy/overbought) and "in buy mode"
# (washed-out/oversold) means for each metric. Aligned with the monitor's own bars.
MODE_SELL = {"rsi": (">=", 70.0), "funding": (">=", 0.05), "range_pos": (">=", 80.0),
             "ls": (">=", 2.0), "oi_change": (">=", 8.0), "change_24h": (">=", 3.0),
             "trend": (">=", 0.3), "vol_z": (">=", 3.0)}
MODE_BUY = {"rsi": ("<=", 30.0), "funding": ("<=", -0.03), "range_pos": ("<=", 20.0),
            "ls": ("<=", 0.6), "oi_change": ("<=", -8.0), "change_24h": ("<=", -3.0),
            "trend": ("<=", -0.3)}
CORE_METRICS = ("rsi", "funding", "range_pos")     # the default trio for an unnamed "N metrics"
_SELL_WORDS = ("sell mode", "sell-mode", "sell signal", "sell side", "selling", "overbought",
               "take profit", "take-profit", "distribut", "topping", "froth")
_BUY_WORDS = ("buy mode", "buy-mode", "buy signal", "buy side", "oversold", "accumulat",
              "capitulat", "bottoming", "washed out", "washed-out")
# longest phrases first so "long/short" wins over "short", "open interest" over "oi", etc.
_METRIC_PHRASES = (("long/short", "ls"), ("long short", "ls"), ("longshort", "ls"),
                   ("open interest", "oi_change"), ("range position", "range_pos"),
                   ("range pos", "range_pos"), ("funding", "funding"), ("rsi", "rsi"),
                   ("trend", "trend"), ("volume", "vol_z"), ("range", "range_pos"),
                   ("price", "price"), ("change", "change_24h"), ("oi", "oi_change"),
                   ("ls", "ls"))

_METRIC_ALIASES = {
    "price": "price", "rsi": "rsi", "rsi_14": "rsi", "ret_1m": "ret_1m", "ret_5m": "ret_5m",
    "ret_1h": "ret_1h", "change_24h": "change_24h", "change": "change_24h",
    "range_pos": "range_pos", "range": "range_pos", "range_position": "range_pos",
    "atr": "atr", "trend": "trend", "ema_trend": "trend", "high_24h": "high_24h",
    "low_24h": "low_24h", "vol_z": "vol_z", "volume_z": "vol_z", "vol_surge": "vol_surge",
    "funding": "funding", "oi_change": "oi_change", "oi": "oi_change",
    "ls": "ls", "long_short": "ls", "longshort": "ls",
}


def canon_metric(name: str) -> str:
    """Canonical metric name resolve_metric understands, or '' if unknown."""
    n = (name or "").lower().strip().replace("(", "_").replace(")", "").strip("_")
    if n in _METRIC_ALIASES:
        return _METRIC_ALIASES[n]
    return n if re.match(r"rsi_(5m|15m|30m|1h|4h|1d)$", n) else ""


def _word(phrase: str, low: str) -> bool:
    """Whole-token match (so 'oi' doesn't match 'point', 'range' not 'arrange', etc.)."""
    return re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", low) is not None


def nl_to_composite(text: str):
    """Best-effort 'alert me when rsi, funding & range are in sell mode' (or 'those 3 metrics
    in sell mode') -> {conditions, match, mode, label}, else None. If no metric is named, uses
    the core trio (rsi, funding, range_pos), trimmed to a stated count ('those 3'). All matching
    is on WORD BOUNDARIES so ordinary words ('exchange', 'signals', 'arrange') don't inject metrics."""
    low = (text or "").lower()
    if _word("neither", low) or "none of" in low:   # negation we don't model -> don't guess
        return None
    if any(w in low for w in _SELL_WORDS):
        mode, table = "sell", MODE_SELL
    elif any(w in low for w in _BUY_WORDS):
        mode, table = "buy", MODE_BUY
    else:
        return None
    named, seen, recognized = [], set(), False
    for phrase, canon in _METRIC_PHRASES:           # named metrics (longest-first, word-bounded, deduped)
        if _word(phrase, low):
            recognized = True                        # the user DID name a metric (even if not valid here)
            if canon in table and canon not in seen:
                named.append(canon)
                seen.add(canon)
    if recognized and not named:                     # named metric(s) but none valid for this mode
        return None                                  # -> don't silently substitute the default trio
    metrics = named or list(CORE_METRICS)
    if not named:                                    # count-trim ONLY the default trio, and only a real count
        nm = (re.search(r"(?:those|these|top|first|all|any(?:\s+of)?)\s+(\d+)\b", low)
              or re.search(r"\b(\d+)\s+metric", low))
        if nm:
            n = int(nm.group(1))
            if 1 <= n <= len(metrics):
                metrics = metrics[:n]
    conds = [{"metric": mt, "op": table[mt][0], "value": table[mt][1]}
             for mt in metrics if mt in table]
    if not conds:
        return None
    match = "any" if (_word("either", low) or "any of" in low) else "all"
    names = ", ".join(c["metric"] for c in conds)
    return {"conditions": conds, "match": match, "mode": mode,
            "label": f"{len(conds)} metric{'s' if len(conds) != 1 else ''} in {mode} mode ({names})"}


def validate_conditions(spec):
    """Sanitize a composite spec (dropping unknown metrics / bad ops) -> a clean spec or None."""
    if not isinstance(spec, dict):
        return None
    out = []
    for c in (spec.get("conditions") or [])[:6]:
        if not isinstance(c, dict):
            continue
        m = canon_metric(str(c.get("metric", "")))
        op = str(c.get("op", "")).strip()
        try:
            v = float(c.get("value"))
        except (TypeError, ValueError):
            continue
        if m and op in _OPS:
            out.append({"metric": m, "op": op, "value": v})
    if not out:
        return None
    match = spec.get("match", "all")
    return {"conditions": out, "match": match if match in ("all", "any") else "all",
            "mode": spec.get("mode"), "label": spec.get("label") or ""}


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
    """Return [(rule, info)] for rules that fire NOW; mutates each rule's 'armed' flag
    (fire once per crossing, re-arm when the condition clears).

    A rule is either SINGLE ({metric, op, value} -> info is the metric value) or COMPOSITE
    ({conditions:[...], match:'all'|'any'} -> info is [(cond, value, met)] per condition).
    A composite fires when all (match='all', the default) / any (match='any') conditions hold."""
    fired = []
    for r in rules:
        conds = r.get("conditions")
        if conds:                                          # composite (multi-metric) alarm
            rows = []
            for c in conds:
                v = resolve_metric(snapshot, c["metric"])
                met = v is not None and _OPS.get(c["op"], lambda a, b: False)(v, c["value"])
                rows.append((c, v, met))
            flags = [m for _c, _v, m in rows]
            avail = all(v is not None for _c, v, _m in rows)   # is the data even there?
            hit = (all(flags) if r.get("match", "all") != "any" else any(flags)) if flags else False
            if hit and r.get("armed", True):
                r["armed"] = False
                fired.append((r, rows))
            elif not hit and avail:                            # genuine clear -> re-arm
                r["armed"] = True
            # else: a data gap (some metric None) -> hold state, never re-arm on missing data
            #       (mirrors single-metric `continue`, so a futures hiccup can't cause re-fire spam)
            continue
        val = resolve_metric(snapshot, r["metric"])        # single-metric trigger
        if val is None:
            continue
        cond = _OPS.get(r["op"], lambda a, b: False)(val, r["value"])
        if cond and r.get("armed", True):
            r["armed"] = False
            fired.append((r, val))
        elif not cond:
            r["armed"] = True
    return fired


def fired_text(rule: dict, info) -> str:
    """Telegram message for a fired rule (single or composite)."""
    rid = rule.get("id")
    if rule.get("conditions"):
        lines = []
        for c, v, met in info:
            vs = "n/a" if v is None else f"{v:g}"
            lines.append(f"  {'✅' if met else '◻️'} `{c['metric']} {c['op']} {c['value']:g}`"
                         f"  (now {vs})")
        label = rule.get("label") or "composite alarm"
        return f"🔔 *Alarm fired* — {label}  (#{rid})\n" + "\n".join(lines)
    return (f"🔔 *Trigger fired* — `{rule['metric']} {rule['op']} {rule['value']}`  →  "
            f"now `{info:g}`   (#{rid})")


def describe_rule(rule: dict) -> str:
    """One-line rendering of a rule for /alerts."""
    rid = rule.get("id")
    if rule.get("conditions"):
        joiner = " AND " if rule.get("match", "all") != "any" else " OR "
        body = joiner.join(f"{c['metric']} {c['op']} {c['value']:g}" for c in rule["conditions"])
        lbl = rule.get("label")
        return f"#{rid}  {(lbl + ' — ') if lbl else ''}`{body}`"
    return f"#{rid}  `{rule['metric']} {rule['op']} {rule['value']}`"


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

    def add_composite(self, conditions: list, match: str = "all", label: str = "",
                      *, chat: str = "", clock=time.time) -> dict:
        rules = self.load()
        rid = (max((r.get("id", 0) for r in rules), default=0) + 1)
        rule = {"id": rid, "conditions": conditions, "match": match if match in ("all", "any") else "all",
                "label": label, "chat": chat, "armed": True, "created": clock()}
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
