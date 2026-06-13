"""Short-term conversation memory — a rolling 24h temp file of recent exchanges so
the analyst has context across messages (e.g. "what about the 4h?" after a prior Q).

Pruned by TTL **and** turn count, so it never grows unbounded. Atomic, fail-safe.
"""
from __future__ import annotations

import json
import os
import tempfile
import time


class Conversation:
    def __init__(self, path: str, *, ttl_s: int = 86_400, max_turns: int = 12,
                 clock=time.time) -> None:
        self.path = path
        self.ttl_s = ttl_s
        self.max_turns = max_turns
        self._clock = clock

    def _load(self) -> list:
        try:
            with open(self.path) as f:
                d = json.load(f)
            return d if isinstance(d, list) else []
        except Exception:  # noqa: BLE001 - first run / corrupt -> empty
            return []

    def _prune(self, items: list) -> list:
        now = self._clock()
        items = [x for x in items if now - float(x.get("ts", 0)) <= self.ttl_s]
        return items[-self.max_turns:]

    def recent(self) -> list:
        """Recent {role, text} turns within the TTL (oldest first)."""
        return [{"role": x["role"], "text": x["text"]} for x in self._prune(self._load())]

    def add(self, role: str, text: str) -> None:
        items = self._prune(self._load())
        items.append({"ts": self._clock(), "role": role, "text": str(text)[:1500]})
        items = items[-self.max_turns:]
        try:
            d = os.path.dirname(self.path) or "."
            os.makedirs(d, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
            with os.fdopen(fd, "w") as f:
                json.dump(items, f)
            os.replace(tmp, self.path)
        except Exception:  # noqa: BLE001 - memory is best-effort, never crash
            pass
