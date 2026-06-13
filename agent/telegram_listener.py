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
    "*On my own:* I monitor BTC 24/7 (order book, volume, volatility, RSI/EMA). To avoid "
    "noise I only ping when *several* signals are out-of-norm at once (confluence) and at "
    "most about once every 30 min — with an LLM read. I also read the news every ~8h on my "
    "own, keep a copy (`/digest`), and only ping if I judge it market-moving.\n"
    "\n"
    "💬 *You're the boss:* I *always* answer your messages — sensitivity / `balanced` / mute only "
    "affect my *proactive* pings, never my replies. I remember our chats & searches across days "
    "(~1 week) for context; `/clear` wipes that, `/clear old` keeps just today.\n"
    "\n"
    "*Commands*\n"
    "`/origins` — 🛰️ control panel: tap which signals may ping, how many must agree, the "
    "cadence, the preset, mute — all live\n"
    "`/digest` — my latest autonomous news read (kept copy)\n"
    "`/btc` or `/status` — my read of BTC right now\n"
    "`/raw` — just the numbers, no LLM\n"
    "`/chart` — price + the leading indicators, plotted\n"
    "`/derivs` (or `/liq`) — funding · open interest · long/short · taker flow\n"
    "`/levels [%]` — reentry & sell zones; add a target (e.g. `/levels 1`) to scale a\n"
    "          buy→sell pair that nets ~that % after Binance fees + spread.\n"
    "          `/sell 1` anchors on the sell instead; or 'sell at 65k, keep 1%'\n"
    "`/onchain` — MVRV · NUPL · SOPR · netflow (CryptoQuant)\n"
    "`/news` — BTC headlines + Fear&Greed (day/week/month)\n"
    "`/alert <metric> <op> <value>` — custom trigger, e.g. `/alert rsi > 70` "
    "(or _ping me when rsi above 70_) · `/alerts` · `/delalert <id>`\n"
    "`/search <query>` — search the web, read the top articles, summarize\n"
    "`/sensitivity [low|normal|high]` — how easily I alert (low = fewest) · `/mute` · `/unmute`\n"
    "`/notes` — list scratch files · `/get <file>` — read one\n"
    "`/memory` — what I remember · `/clear` — forget it (`/clear old` = keep today)\n"
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


def _parse_target_pct(text: str, *, allow_bare: bool = False) -> float | None:
    """Pull a target margin out of a levels request: '1%', '0.5 percent', 'want 2'.
    With allow_bare (the /levels command path) also accepts a trailing bare number."""
    for pat in (r"(\d+(?:\.\d+)?)\s*%",
                r"(\d+(?:\.\d+)?)\s*(?:percent|pct)\b",
                r"(?:want|expect|target|net|margin|profit|gain|make|around|about|~)\s*\$?"
                r"(\d+(?:\.\d+)?)"):
        m = re.search(pat, text)
        if m:
            break
    else:
        m = re.search(r"(\d+(?:\.\d+)?)\s*$", text.strip()) if allow_bare else None
    if not m:
        return None
    v = float(m.group(1))
    return v if 0 < v <= 50 else None       # sane bound; ignore absurd inputs


_PX_SUF = {"k": 1e3, "m": 1e6, None: 1}


def _parse_levels_args(text: str, *, default_anchor: str = "buy", allow_bare: bool = False):
    """-> (target_pct, anchor, anchor_price). Understands 'sell at 65k', 'buy near 60000',
    a bare 'at 65000', the side word, and the target %; an explicit 'side at price' pins
    the anchored side. [^\\d] guards stop a 'sell … at' span from eating the target number."""
    low = text.lower()
    anchor_price = explicit_side = None
    m = re.search(r"\b(sell|buy)\b[^\d]{0,14}?(?:\bat\b|@|\bnear\b|\baround\b)"
                  r"\s*\$?(\d+(?:\.\d+)?)\s*([km])?", low)
    if m:
        explicit_side = "sell" if m.group(1) == "sell" else "buy"
        v = float(m.group(2)) * _PX_SUF[m.group(3)]
        if v >= 1000:
            anchor_price = v
        low = low[:m.start()] + " " + low[m.end():]
    else:
        m2 = re.search(r"(?:\bat\b|@|\bnear\b|\baround\b)\s*\$?(\d+(?:\.\d+)?)\s*([km])?", low)
        if m2 and float(m2.group(1)) * _PX_SUF[m2.group(2)] >= 1000:
            anchor_price = float(m2.group(1)) * _PX_SUF[m2.group(2)]
            low = low[:m2.start()] + " " + low[m2.end():]
    if explicit_side:
        anchor = explicit_side
    elif "sell" in low and not any(k in low for k in ("buy", "entry", "reentry", "reenter")):
        anchor = "sell"
    else:
        anchor = default_anchor
    return _parse_target_pct(low, allow_bare=allow_bare), anchor, anchor_price


_DIR_DN = ("below", "under", "drops", "falls", "less than", "lower", "crosses below", "<")


def _nl_to_rule(text: str):
    """Best-effort 'ping me when rsi above 70' -> (metric, op, value), else None."""
    low = text.lower()
    num = re.search(r"(-?\d+(?:\.\d+)?)", low)
    if not num:
        return None
    op = "<" if any(w in low for w in _DIR_DN) else ">"
    for name in ("long_short", "oi_change", "range_pos", "vol_z", "change_24h",
                 "funding", "trend", "atr", "price", "rsi", "ls"):
        if name in low or name.replace("_", " ") in low:
            return name, op, float(num.group(1))
    return None


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
        from .alerts import AlertStore
        from .memory import Memory
        self.memory = Memory(cfg.memory_path, ttl_s=cfg.memory_ttl_s,
                             max_chat_turns=cfg.memory_max_turns)
        self.alerts = AlertStore(cfg.user_alerts_path)
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
            self._send_charts(snap)
            return ""        # photo(s) already sent; no extra text
        if low in ("/derivs", "/liq", "/funding", "/oi"):
            from .plotter import build_derivs_chart
            png, cap = build_derivs_chart(self.cfg)
            self.notifier.send_photo(png, cap)
            return ""
        if low.split(" ", 1)[0] in ("/levels", "/entry", "/sell", "/buy"):
            from .levels import levels_text
            if not snap:
                return "no snapshot yet — warming up"
            dflt = "sell" if low.split(" ", 1)[0] == "/sell" else "buy"
            tgt, anc, ap = _parse_levels_args(text, default_anchor=dflt, allow_bare=True)
            return levels_text(snap, target_pct=tgt, fee_pct=self.cfg.fee_pct,
                               anchor=anc, anchor_price=ap)
        if low in ("/onchain", "/mvrv", "/sopr", "/nupl"):
            from .onchain import onchain_text
            return onchain_text((snap or {}).get("onchain") or {})
        if low.split(" ", 1)[0] in ("/origins", "/panel", "/sources", "/widget", "/controls"):
            from .origins import widget_text
            return widget_text(self._prefs_full())   # text view; _process_update sends buttons
        if low in ("/digest", "/newsdigest"):
            return self._digest_text()
        if low.split(" ", 1)[0] in ("/sensitivity", "/sens"):
            return self._sensitivity_cmd(text)
        if low in ("/mute", "/silence"):
            self._update_prefs(muted=True)
            return ("🔇 *Muted* — no proactive alerts. Your Q&A and the daily digest still "
                    "work. `/unmute` to resume.")
        if low == "/unmute":
            from .sensitivity import describe
            p = self._update_prefs(muted=False)
            return "🔔 *Unmuted.*\n" + describe(p["sensitivity"], p["muted"], p["overrides"],
                                                p["disabled"], p.get("confluence"), p.get("cadence"))
        if low == "/news":
            from .websearch import news_line
            return news_line(self.news_fn()) if self.news_fn else "news disabled"
        if low.startswith("/search "):
            q, after, before = _parse_search_dates(text[8:].strip())
            reply = self.analyst.search(q, snap, after=after, before=before)
            self.memory.add_search(q, summary=reply[:300], after=after, before=before)
            return reply
        if low.split(" ", 1)[0] in ("/clear", "/forget"):
            scope = "old" if "old" in low or "previous" in low or "days" in low else "all"
            n = self.memory.clear(scope)
            if n < 0:                               # rewrite failed -> don't claim a false wipe
                return "⚠️ couldn't clear memory (storage error) — nothing was deleted. Try again."
            kept = "kept today's" if scope == "old" else "wiped all"
            return (f"🧹 *Memory cleared* — {kept} ({n} record{'s' if n != 1 else ''} dropped). "
                    "I forget conversations & searches automatically after a week anyway.")
        if low in ("/memory", "/mem"):
            s = self.memory.stats()
            return (f"🧠 *Memory* — {s['chats']} chat turn(s) · {s['searches']} search(es), "
                    f"auto-erased after {s['ttl_days']:g}d"
                    + (f" · since {s['oldest'][:10]}" if s['oldest'] else "")
                    + ".\n`/clear` wipes it · `/clear old` keeps just today.")
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
        if low == "/alerts":
            rules = self.alerts.load()
            if not rules:
                return "🔔 no triggers set — e.g. `/alert rsi > 70` or _ping me when rsi above 70_"
            return "🔔 *Triggers:*\n" + "\n".join(
                f"#{r['id']}  `{r['metric']} {r['op']} {r['value']}`" for r in rules)
        if low.startswith("/alert "):
            return self._add_alert(text[7:], snap)
        if low.startswith("/delalert "):
            ids = re.findall(r"\d+", text)
            return ("✅ removed" if ids and self.alerts.remove(int(ids[0]))
                    else "usage: `/delalert <id>` (see `/alerts`)")
        if low == "/clearalerts":
            self.alerts.clear()
            return "✅ all triggers cleared"
        if low.startswith("/"):
            return _HELP
        if not snap:
            return "no snapshot yet — monitor is warming up"
        # reentry/sell price request -> sat-stacking levels (structure-based)
        if any(p in low for p in ("reentry", "re-entry", "reenter", "entry price",
                                  "sell price", "buy price", "good buy", "good sell",
                                  "where to buy", "where to sell", "add sats")):
            from .levels import levels_text
            tgt, anc, ap = _parse_levels_args(text)
            return levels_text(snap, target_pct=tgt, fee_pct=self.cfg.fee_pct,
                               anchor=anc, anchor_price=ap)
        # natural-language trigger: "ping me when rsi above 70"
        if any(p in low for p in ("ping me when", "alert me when", "notify me when",
                                  "let me know when", "tell me when")):
            rule = _nl_to_rule(text)
            if not rule:
                return "tell me like: _ping me when rsi above 70_  (or `/alert rsi > 70`)"
            return self._add_alert(f"{rule[0]} {rule[1]} {rule[2]}", snap)
        # natural-language chart request -> send the LLM-orchestrated patchwork
        if "show me" in low or any(w in low for w in ("chart", "plot", "graph", "draw")):
            self._send_charts(snap, question=text)
            return ""
        # conversational answer, with multi-day memory (conversation + recent searches)
        reply = self.analyst.answer(text, snap, history=self.memory.recent_chat(),
                                    searches=self.memory.recent_searches())
        self.memory.add_chat("user", text)
        self.memory.add_chat("assistant", reply)
        srch = getattr(self.analyst, "last_search", None)   # remember any autonomous search it ran
        if srch and srch.get("query"):
            self.memory.add_search(srch["query"], summary=reply[:300],
                                   after=srch.get("after"), before=srch.get("before"))
        return reply

    def _send_charts(self, snap, question=None) -> int:
        """Render the LLM's patchwork (1+ images, each price + chosen panels) and send
        each as a photo. Returns the number sent. Falls back to one default image."""
        from .plotter import build_btc_chart
        from .signal_tuner import load_tuned
        groups = []
        if snap:
            try:                              # the LLM orchestrates the layout (image groups)
                groups = self.analyst.pick_indicators(snap, question=question)
            except Exception:  # noqa: BLE001 - fall back to backtest-leading
                groups = []
        tuned = load_tuned(self.cfg.tuned_signals_path)
        sent = 0
        for g in (groups or [None]):          # [None] -> one default (backtest-leading) image
            png, cap = build_btc_chart(self.cfg, tuned, snapshot=snap or None, indicators=g)
            if png:
                self.notifier.send_photo(png, cap)
                sent += 1
        return sent

    def _prefs_full(self):
        from .sensitivity import read_prefs
        return read_prefs(self.cfg.prefs_path, default_level=self.cfg.sensitivity,
                          default_confluence=self.cfg.confluence_min,
                          default_cadence=self.cfg.alert_cadence_s)

    def _update_prefs(self, **changes):
        """Read current prefs, apply changes, write ALL fields so none clobbers another.
        The monitor picks the file up on its next scan (~15s)."""
        from .sensitivity import write_prefs
        cur = self._prefs_full()
        cur.update(changes)
        return write_prefs(self.cfg.prefs_path, sensitivity=cur["sensitivity"], muted=cur["muted"],
                           overrides=cur["overrides"], disabled=cur["disabled"],
                           confluence=cur["confluence"], cadence=cur["cadence"])

    def _digest_text(self) -> str:
        """Show the latest autonomous news digest (the cached copy the monitor keeps)."""
        import json
        try:
            with open(self.cfg.news_digest_path) as f:
                d = json.load(f)
        except Exception:  # noqa: BLE001
            return ("📰 no autonomous digest yet — I read the news every "
                    f"{self.cfg.news_digest_hours:g}h and keep the latest copy. "
                    "Ask me `/news` for live headlines now.")
        tag = "🚨 flagged" if d.get("alert") else "🟢 quiet"
        body = (d.get("body") or d.get("raw") or "(empty)").strip()
        return f"📰 *Last news digest* — {tag} · _{d.get('iso', '')}_\n{body}"

    def _sensitivity_cmd(self, text: str) -> str:
        from .sensitivity import PRESETS, KEY_ALIAS, SIGNAL_ALIAS, MIN_KEYS, describe
        args = text.split()[1:]
        fmt = lambda p: describe(p["sensitivity"], p["muted"], p["overrides"], p["disabled"],
                                 p.get("confluence"), p.get("cadence"))
        ok = lambda p: "✅ updated — applies within ~15s\n" + fmt(p)
        if not args:
            return fmt(self._prefs_full())
        sub = args[0].lower()
        if sub in PRESETS:                                    # low | normal | high
            return ok(self._update_prefs(sensitivity=sub))
        if sub == "reset":
            return "✅ manual overrides cleared.\n" + fmt(
                self._update_prefs(overrides={}, disabled=[]))
        if sub == "set" and len(args) >= 3:
            key = KEY_ALIAS.get(args[1].lower())
            if not key:
                return ("unknown bar `%s`. keys: imbalance funding vol ret rsi_top rsi_bottom "
                        "oi near cooldown rearm" % args[1])
            try:
                val = float(args[2])
            except ValueError:
                return "value must be a number, e.g. `/sensitivity set imbalance 0.95`"
            stored = val * 60 if key in MIN_KEYS else val     # cooldown/rearm given in minutes
            cur = self._prefs_full()
            return ok(self._update_prefs(overrides={**cur["overrides"], key: stored}))
        if sub in ("off", "on") and len(args) >= 2:
            sig = SIGNAL_ALIAS.get(args[1].lower())
            if not sig:
                return ("unknown signal `%s`. signals: imbalance funding oi long_short volume "
                        "peak bottom spike_up spike_down" % args[1])
            dis = set(self._prefs_full()["disabled"])
            dis.add(sig) if sub == "off" else dis.discard(sig)
            return ok(self._update_prefs(disabled=sorted(dis)))
        return ("*Manual sensitivity:*\n"
                "`/sensitivity low|normal|high` — pick a preset\n"
                "`/sensitivity set <key> <value>` — e.g. `set imbalance 0.95` (raises the bar)\n"
                "`/sensitivity off <signal>` — e.g. `off imbalance` (silence one signal) · `on <signal>`\n"
                "`/sensitivity reset` — clear all manual changes\n\n" + fmt(self._prefs_full()))

    def _add_alert(self, spec: str, snap) -> str:
        from .alerts import SUPPORTED, parse_rule, resolve_metric
        p = parse_rule(spec)
        if not p:
            return ("format: `/alert <metric> <op> <value>` — e.g. `/alert rsi > 70`\n"
                    f"metrics: {SUPPORTED}")
        metric, op, value = p
        if snap and resolve_metric(snap, metric) is None:
            return f"unknown metric `{metric}`.\nmetrics: {SUPPORTED}"
        r = self.alerts.add(metric, op, value)
        return (f"✅ trigger *#{r['id']}* set: `{metric} {op} {value}` — I'll ping each time it "
                f"crosses. `/alerts` to list · `/delalert {r['id']}` to remove.")

    # ── the /origins inline-keyboard widget (live, no restart) ──
    _WIDGET_CMDS = ("/origins", "/panel", "/sources", "/widget", "/controls")

    def _tg_api(self, method: str, payload: dict) -> dict | None:
        """One raw Bot API call (sendMessage with keyboard, edit, answerCallback). No-op
        without a token; never raises — Telegram UI must not crash the loop."""
        if not self.token:
            return None
        try:
            import requests  # lazy
            r = requests.post(f"https://api.telegram.org/bot{self.token}/{method}",
                              json=payload, timeout=10)
            return r.json()
        except Exception as e:  # noqa: BLE001
            log.warning("telegram %s failed: %s", method, e)
            return None

    def _send_widget(self, chat: str) -> bool:
        """Send the interactive /origins panel to ``chat``. Returns False if it couldn't
        (no token) so the caller can fall back to the text view."""
        from .origins import keyboard, widget_text
        prefs = self._prefs_full()
        res = self._tg_api("sendMessage", {
            "chat_id": chat, "text": widget_text(prefs), "parse_mode": "Markdown",
            "disable_web_page_preview": True, "reply_markup": keyboard(prefs)})
        return bool(res and res.get("ok"))

    def _handle_callback(self, cbq: dict) -> None:
        """A widget button tap: apply it to prefs, re-render the panel in place, ack."""
        from .origins import apply_callback, keyboard, widget_text
        data = cbq.get("data", "")
        cbid = cbq.get("id")
        msg = cbq.get("message") or {}
        chat = str((msg.get("chat") or {}).get("id", ""))
        mid = msg.get("message_id")
        if self.chat_ids and chat not in self.chat_ids:
            self._tg_api("answerCallbackQuery", {"callback_query_id": cbid})
            return
        if data == "og:x":                              # close -> drop the keyboard
            self._tg_api("editMessageReplyMarkup", {"chat_id": chat, "message_id": mid})
            self._tg_api("answerCallbackQuery", {"callback_query_id": cbid, "text": "closed"})
            return
        new, toast = apply_callback(data, self._prefs_full())
        if new is not None:
            from .origins import _KEYS
            self._update_prefs(**{k: new[k] for k in _KEYS})
        prefs = self._prefs_full()
        self._tg_api("editMessageText", {
            "chat_id": chat, "message_id": mid, "text": widget_text(prefs),
            "parse_mode": "Markdown", "disable_web_page_preview": True,
            "reply_markup": keyboard(prefs)})
        self._tg_api("answerCallbackQuery", {"callback_query_id": cbid, "text": toast or ""})

    # ── network ──
    def _process_update(self, update: dict) -> None:
        cbq = update.get("callback_query")
        if cbq:
            try:
                self._handle_callback(cbq)
            except Exception as e:  # noqa: BLE001 - a widget tap must not kill the poll loop
                log.warning("callback error: %s", e)
            return
        msg = update.get("message") or update.get("channel_post") or {}
        chat = str((msg.get("chat") or {}).get("id", ""))
        text = msg.get("text", "")
        if not text:
            return
        if self.chat_ids and chat not in self.chat_ids:
            log.info("ignoring message from non-allowlisted chat %s", chat)
            return
        cmd = text.strip().lower().split(maxsplit=1)[0]
        if cmd in self._WIDGET_CMDS:                    # send live buttons, not text
            if not self._send_widget(chat):
                self.notifier.send(self.handle_text("/origins"))   # text fallback (no token)
            return
        try:
            reply = self.handle_text(text)
        except Exception as e:  # noqa: BLE001 - a bad question must not kill the loop OR go unanswered
            log.warning("handler error: %s", e)
            reply = "⚠️ I hit a snag on that one — try again, or send `/help`."
        if reply:                           # /chart etc. send a photo themselves and return ""
            self.notifier.send(reply)

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
