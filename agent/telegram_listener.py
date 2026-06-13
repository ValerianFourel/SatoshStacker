"""Inbound Telegram — lets the operator ask the bot questions; it replies with an
LLM read over the latest market snapshot. Read-only: it never trades.

Hardened: only messages from the configured operator chat id are answered (every
other chat is ignored), and `getUpdates` long-polling is resilient to transient
errors. Routing is in ``handle_text`` (pure given the injected analyst/snapshot),
so it is unit-testable without the network.
"""
from __future__ import annotations

import logging
import re

from .config import WatchConfig

log = logging.getLogger("satoshistacker.listener")

_HELP = (
    "🛰️ *SatoshiStacker BTC watch* — read-only. I never trade; I watch, analyze & explain.\n"
    "\n"
    "*On my own:* I monitor BTC 24/7 (order book, volume, volatility, RSI/EMA) and ping you "
    "when something's out of the norm — a likely *top/peak*, *bottom*, or *volume/price spike* "
    "— with an LLM read (plus news if it's relevant).\n"
    "\n"
    "*Commands*\n"
    "`/btc` or `/status` — my read of BTC right now\n"
    "`/raw` — just the numbers, no LLM\n"
    "`/chart` — price + the leading indicators, plotted\n"
    "`/news` — BTC headlines + Fear&Greed (day/week/month)\n"
    "`/search <query>` — search the web, read the top articles, summarize\n"
    "`/notes` — list scratch files · `/get <file>` — read one\n"
    "`/help` — this message\n"
    "\n"
    "*Or just ask in plain words:*\n"
    "• _is this a local top?_\n"
    "• _why did BTC just spike?_  (I'll search if I need to)\n"
    "• _what's the news on ETF flows this week?_\n"
    "• _save me a table of the last hour's metrics_  (I write it to a scratch file)\n"
    "\n"
    "*Search by date* — say it naturally, or pin it:\n"
    "• _what happened to BTC between Jan 1 and Feb 1 2026?_\n"
    "• _any regulation news since the halving?_\n"
    "• `/search etf flows between:2026-01-01..2026-02-01`\n"
    "• `/search halving news after:2024-04-01`  (or `before:YYYY-MM-DD`)"
)

# Sent ONCE, the first time the service ever launches (trimmed /help).
ONBOARDING = (
    "👋 *Welcome to SatoshiStacker BTC watch.*\n"
    "I'm read-only — I never trade. I watch BTC 24/7 and *ping you* when something's out of "
    "the norm (a likely top, bottom, or volume/price spike), with a quick read.\n"
    "\n"
    "You can also just *ask me in plain words*:\n"
    "• _is this a local top?_\n"
    "• _why did BTC just move?_\n"
    "• _BTC news between Jan 1 and Feb 1?_  — I can search the web & read articles, by date\n"
    "\n"
    "Type `/help` any time for the full guide."
)

_D = r"(\d{4}-\d{2}-\d{2})"
_BETWEEN = re.compile(rf"between:{_D}\.\.{_D}")
_AFTER = re.compile(rf"\bafter:{_D}")
_BEFORE = re.compile(rf"\bbefore:{_D}")


def _parse_search_dates(text: str):
    """Pull optional after:/before:/between: date tokens out of a /search query."""
    after = before = None
    mb = _BETWEEN.search(text)
    if mb:
        after, before = mb.group(1), mb.group(2)
    else:
        ma, mbf = _AFTER.search(text), _BEFORE.search(text)
        after = ma.group(1) if ma else None
        before = mbf.group(1) if mbf else None
    text = _BETWEEN.sub("", text)
    text = _AFTER.sub("", text)
    text = _BEFORE.sub("", text)
    return re.sub(r"\s+", " ", text).strip(), after, before


class TelegramListener:
    def __init__(self, cfg: WatchConfig, *, token: str, chat_id: str,
                 analyst, notifier, snapshot_fn, scratch=None, news_fn=None) -> None:
        from .secrets import clean_secret
        self.cfg = cfg
        self.token = clean_secret(token)
        # chat_id may be a comma-separated ALLOWLIST (operator + shared users)
        self.chat_ids = {c.strip() for c in clean_secret(chat_id).split(",") if c.strip()}
        self.analyst = analyst
        self.notifier = notifier
        self.snapshot_fn = snapshot_fn       # () -> dict | None
        self.scratch = scratch
        self.news_fn = news_fn               # () -> dict | None  (BTC news + Fear&Greed)
        from .convo import Conversation
        self.convo = Conversation(cfg.convo_path, ttl_s=cfg.convo_ttl_s,
                                  max_turns=cfg.convo_max_turns)
        self._offset = 0

    # ── routing (pure given injected deps) ──
    def handle_text(self, text: str) -> str:
        from .analyst import numeric_summary
        text = (text or "").strip()
        low = text.lower()
        snap = self.snapshot_fn() or {}
        if low in ("/start", "/help", "help", "?", "/?", "commands"):
            return _HELP
        if low in ("/raw",):
            return numeric_summary(snap)
        if low in ("/btc", "/status", "/now"):
            if not snap:
                return "no snapshot yet — monitor is warming up"
            return self.analyst.answer("Give a concise read of BTC right now.", snap)
        if low in ("/chart", "/plot"):
            png, cap = self._build_chart(snap)
            self.notifier.send_photo(png, cap)
            return ""        # photo already sent; no extra text
        if low == "/news":
            from .websearch import news_line
            return news_line(self.news_fn()) if self.news_fn else "news disabled"
        if low.startswith("/search "):
            q, after, before = _parse_search_dates(text[8:].strip())
            return self.analyst.search(q, snap, after=after, before=before)
        if low == "/notes":
            files = self.scratch.list() if self.scratch else []
            return "scratch files:\n" + ("\n".join(files) if files else "(none)")
        if low.startswith("/get "):
            if not self.scratch:
                return "no scratch workspace"
            try:
                return "```\n" + self.scratch.read(text[5:].strip())[:3500] + "\n```"
            except Exception as e:  # noqa: BLE001
                return f"can't read that file: {e}"
        if low.startswith("/"):
            return _HELP
        if not snap:
            return "no snapshot yet — monitor is warming up"
        # natural-language chart request -> send a plot (LLM picks indicators for the ask)
        if "show me" in low or any(w in low for w in ("chart", "plot", "graph", "draw")):
            png, cap = self._build_chart(snap, question=text)
            self.notifier.send_photo(png, cap)
            return ""
        # conversational answer, with 24h short-term memory for follow-ups
        reply = self.analyst.answer(text, snap, history=self.convo.recent())
        self.convo.add("user", text)
        self.convo.add("assistant", reply)
        return reply

    def _build_chart(self, snap, question=None):
        from .plotter import build_btc_chart
        from .signal_tuner import load_tuned
        picks = []
        if snap:
            try:                              # let the LLM choose which indicators to chart
                picks = self.analyst.pick_indicators(snap, question=question)
            except Exception:  # noqa: BLE001 - fall back to backtest-leading
                picks = []
        return build_btc_chart(self.cfg, load_tuned(self.cfg.tuned_signals_path),
                               snapshot=snap or None, indicators=picks or None)

    # ── network ──
    def _process_update(self, update: dict) -> None:
        msg = update.get("message") or update.get("channel_post") or {}
        chat = str((msg.get("chat") or {}).get("id", ""))
        text = msg.get("text", "")
        if not text:
            return
        if self.chat_ids and chat not in self.chat_ids:
            log.info("ignoring message from non-allowlisted chat %s", chat)
            return
        try:
            reply = self.handle_text(text)
            if reply:                       # /chart sends a photo itself and returns ""
                self.notifier.send(reply)
        except Exception as e:  # noqa: BLE001 - a bad question must not kill the loop
            log.warning("handler error: %s", e)

    def poll(self, stop) -> None:
        """Long-poll getUpdates until ``stop`` is set. No-op if no token."""
        if not (self.token and self.chat_ids):
            log.warning("telegram listener disabled (no token/chat id)")
            return
        import requests  # lazy
        url = f"https://api.telegram.org/bot{self.token}/getUpdates"
        log.info("telegram listener started (allowed chats: %s)",
                 ", ".join(sorted(self.chat_ids)))
        while not stop.is_set():
            try:
                r = requests.get(url, params={"offset": self._offset, "timeout": 25},
                                 timeout=35)
                r.raise_for_status()
                for upd in r.json().get("result", []):
                    self._offset = max(self._offset, upd.get("update_id", 0) + 1)
                    self._process_update(upd)
            except Exception as e:  # noqa: BLE001 - keep polling through transient errors
                log.warning("getUpdates error: %s", e)
                stop.wait(5)
