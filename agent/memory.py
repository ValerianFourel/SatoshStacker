"""Multi-day rolling memory — a JSONL transcript (like Claude Code's) the analyst keeps
across days so it remembers recent conversation AND the searches it ran.

Append-only: each chat turn / search is one JSON line. Auto-erased past ``ttl_s`` (default
~1 week) — pruned lazily on read (the file self-compacts), so nothing is hoarded. The
operator can wipe it with /clear (all) or /clear old (keep only today). Atomic, fail-safe:
memory is best-effort and must never crash the bot.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time

log = logging.getLogger("satoshistacker.memory")

WEEK_S = 7 * 86_400


class Memory:
    def __init__(self, path: str, *, ttl_s: int = WEEK_S, max_chat_turns: int = 24,
                 clock=time.time) -> None:
        self.path = path
        self.ttl_s = ttl_s
        self.max_chat_turns = max_chat_turns
        self._clock = clock

    # ── io ──
    def _load_raw(self) -> list[dict]:
        try:
            with open(self.path) as f:
                out = []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if isinstance(rec, dict):
                            out.append(rec)
                    except Exception:  # noqa: BLE001 - skip a corrupt line, keep the rest
                        continue
                return out
        except Exception:  # noqa: BLE001 - first run / missing
            return []

    def _live(self) -> list[dict]:
        """All records still within the TTL. Compacts the file if anything expired."""
        raw = self._load_raw()
        now = self._clock()
        live = [r for r in raw if now - float(r.get("ts", 0)) <= self.ttl_s]
        if len(live) < len(raw):          # something expired -> auto-erase (rewrite compacted)
            self._rewrite(live)
        return live

    def _rewrite(self, items: list[dict]) -> bool:
        """Atomically replace the file with ``items``. Returns True on success, False if the
        write/replace failed (so callers like clear() can report the truth, not assume it worked)."""
        try:
            d = os.path.dirname(self.path) or "."
            os.makedirs(d, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
            with os.fdopen(fd, "w") as f:
                for r in items:
                    f.write(json.dumps(r, separators=(",", ":")) + "\n")
            os.replace(tmp, self.path)
            return True
        except Exception as e:  # noqa: BLE001 - memory is best-effort
            log.warning("could not rewrite memory: %s", e)
            return False

    def _append(self, rec: dict) -> None:
        rec = {"ts": self._clock(),
               "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self._clock())), **rec}
        try:
            d = os.path.dirname(self.path) or "."
            os.makedirs(d, exist_ok=True)
            with open(self.path, "a") as f:
                f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        except Exception as e:  # noqa: BLE001 - never crash on a memory write
            log.warning("could not append memory: %s", e)

    # ── writes ──
    def add_chat(self, role: str, text: str) -> None:
        self._append({"kind": "chat", "role": role, "text": str(text)[:1500]})

    def add_search(self, query: str, *, summary: str = "", after=None, before=None) -> None:
        self._append({"kind": "search", "query": str(query)[:300],
                      "summary": str(summary)[:400], "after": after, "before": before})

    # ── reads ──
    def recent_chat(self, max_turns: int | None = None) -> list[dict]:
        """Recent {role, text} turns within the TTL (oldest first)."""
        turns = [{"role": r.get("role", "user"), "text": r.get("text", "")}
                 for r in self._live() if r.get("kind") == "chat"]
        return turns[-(max_turns or self.max_chat_turns):]

    def recent_searches(self, *, days: float | None = None, limit: int = 12) -> list[dict]:
        """Recent searches within ``days`` (default: full TTL window), newest last."""
        now = self._clock()
        win = days * 86_400 if days else self.ttl_s
        out = [{"iso": r.get("iso", ""), "query": r.get("query", ""),
                "summary": r.get("summary", ""), "after": r.get("after"),
                "before": r.get("before")}
               for r in self._live()
               if r.get("kind") == "search" and now - float(r.get("ts", 0)) <= win]
        return out[-limit:]

    def clear(self, scope: str = "all") -> int:
        """``all`` wipes everything; ``old`` keeps only today's records (UTC). Returns the
        number of records removed."""
        raw = self._load_raw()
        if scope == "old":
            today = time.strftime("%Y-%m-%d", time.gmtime(self._clock()))
            keep = [r for r in raw if str(r.get("iso", ""))[:10] == today]
        else:
            keep = []
        if not self._rewrite(keep):               # storage error -> report failure, don't lie
            return -1
        return len(raw) - len(keep)

    def stats(self) -> dict:
        live = self._live()
        chats = sum(1 for r in live if r.get("kind") == "chat")
        searches = sum(1 for r in live if r.get("kind") == "search")
        oldest = min((r.get("iso", "") for r in live), default="")
        return {"chats": chats, "searches": searches, "oldest": oldest,
                "ttl_days": round(self.ttl_s / 86_400, 1)}
