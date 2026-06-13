"""Read-only BTC market analyst (LLM). NO trading authority — it describes and
contextualizes market state; it never emits buy/sell/hold or order sizing.

Two entry points, both fed COMPUTED NUMBERS (never chart images):
  * ``event_read(metrics, signals)`` — a short read when the monitor catches an
    out-of-the-norm event (peak / bottom / spike), pushed to Telegram.
  * ``answer(question, snapshot)``  — answers an operator's Telegram question over
    the latest snapshot.

The LLM may optionally write notes/metrics to the sandboxed ``Scratch`` workspace
(temp files only). Any LLM/network error fails safe to a deterministic numeric
summary, so the bot always replies.
"""
from __future__ import annotations

import json
import logging

from .config import AnalysisConfig
from .scratch import Scratch, ScratchError

log = logging.getLogger("satoshistacker.analyst")

_SYSTEM = (
    "You are a precise BTC market microstructure & technical analyst for one "
    "operator. You receive COMPUTED NUMBERS (price, order-book depth/imbalance, "
    "volume surge, RSI, EMAs, ATR/volatility, 24h range), plus recent BTC headlines "
    "and a Fear&Greed reading, and optionally web ``search_results`` you requested "
    "(each may include the article's full ``content`` — read it, don't just skim titles). "
    "Explain in plain, concise language what is happening RIGHT NOW — e.g. likely "
    "local top/bottom, exhaustion, breakout, volume/volatility spike, thin book — tie "
    "in the news/sentiment when relevant, and state caveats. "
    "If you need fresh context (a catalyst behind a move, an ETF/macro/BTC story), set "
    "``search`` to ONE concise query and you will be re-invoked with results; otherwise "
    "set it to \"\". "
    "If the user constrains the time window (before/after/between dates, or relative like "
    "'last week' / 'since the halving'), also set ``search_after`` and/or ``search_before`` "
    "as YYYY-MM-DD — resolve relative phrases using the provided ``today``. 'before X' => "
    "search_before=X; 'after X' => search_after=X; 'between A and B' => both. "
    "HARD RULE: you are read-only. You must NOT give buy/sell/hold advice, price "
    "targets to act on, or position sizing. Describe, don't direct. "
    "You MAY save notes or computed tables to scratch files. Respond with STRICT "
    "JSON only:\n"
    '{"reply":"<=140 words, plain text","search":"<query or empty>",'
    '"search_after":"<YYYY-MM-DD or empty>","search_before":"<YYYY-MM-DD or empty>",'
    '"files":[{"name":"notes.md","content":"..."}]}  (search/dates/files optional).'
)


def numeric_summary(m: dict) -> str:
    """Deterministic one-glance card — the /raw reply and the LLM fail-safe."""
    if not m:
        return "📊 no snapshot yet — monitor is warming up"
    t, v, b = m.get("technicals", {}), m.get("volume", {}), m.get("order_book", {})
    price = m.get("price", 0)
    rsi = t.get("rsi_14", 50)
    rlabel = "overbought" if rsi >= 70 else "oversold" if rsi <= 30 else "neutral"
    rows = [
        f"RSI(14)     {rsi:<6}{rlabel}",
        f"Trend       {t.get('ema_trend_pct', 0):+.2f}%  (EMA 9/21)",
        f"24h range   {t.get('range_position_pct', 0):.0f}%   "
        f"{t.get('pct_from_high_24h', 0):+.1f}% high / {t.get('pct_from_low_24h', 0):+.1f}% low",
        f"Move        1m {t.get('ret_1m_pct', 0):+.2f}%  5m {t.get('ret_5m_pct', 0):+.2f}%  "
        f"1h {t.get('ret_1h_pct', 0):+.2f}%",
        f"Volume      {v.get('surge_x', 1)}x avg   (z {v.get('z', 0):+.1f})",
        f"Volatility  ATR {t.get('atr_pct', 0):.2f}%  ·  realized {t.get('realized_vol_pct', 0):.3f}%",
    ]
    if b.get("ok"):
        band = b.get("bands", {}).get("1.0", {})
        rows.append(f"Book ±1%     bid {band.get('bid', 0)} / ask {band.get('ask', 0)}  "
                    f"(lean {band.get('imbalance', 0):+.2f})")
    tuned = m.get("tuned", {})
    if tuned.get("top") or tuned.get("bottom"):
        from .signal_tuner import _pretty
        for side, arrow in (("top", ">="), ("bottom", "<=")):
            s = tuned.get(side)
            if s:
                rows.append(f"{'Top-watch' if side == 'top' else 'Bot-watch':<11} "
                            f"{_pretty(s['name'])} {s['value']} (fires {arrow}{s['threshold']})")
    hhmm = (m.get("iso", "") or "")[11:16]
    out = (f"📊 *BTC*  `${price:,.0f}`   _{hhmm}Z_\n"
           "```\n" + "\n".join(rows) + "\n```")
    if m.get("events"):
        out += "⚠️ *fired:* " + ", ".join(m["events"])
    return out


def _features(m: dict) -> dict:
    """Compact numeric payload for the LLM (keeps tokens small)."""
    return {k: m.get(k) for k in ("symbol", "price", "iso", "technicals",
                                  "volume", "order_book", "events")}


def format_search(query: str, results: list[dict]) -> str:
    """Plain-text rendering of web results — the /search reply when no LLM."""
    if not results:
        return f"🔎 {query}: no results"
    lines = [f"🔎 {query}"]
    for r in results[:5]:
        lines.append(f"• {r.get('title','')}".strip())
        sn = (r.get("snippet") or "").strip()
        if sn:
            lines.append(f"  {sn[:160]}")
    return "\n".join(lines)


class MockAnalyst:
    """Deterministic analyst for tests / no-key runs: echoes the numeric summary."""

    def __init__(self, scratch: Scratch | None = None) -> None:
        self.scratch = scratch

    def event_read(self, m: dict, signals) -> str:
        names = ", ".join(getattr(s, "name", str(s)) for s in signals)
        return f"[mock] {names}\n{numeric_summary(m)}"

    def answer(self, question: str, m: dict) -> str:
        return f"[mock answer to: {question[:60]}]\n{numeric_summary(m)}"

    def search(self, query: str, m: dict, after=None, before=None) -> str:
        win = f" [{after or '…'}→{before or '…'}]" if (after or before) else ""
        return f"[mock search: {query[:60]}{win}]"


class LLMAnalyst:
    """Qwen (OpenAI-compatible) read-only analyst. Fail-safe to numeric summary."""

    def __init__(self, cfg: AnalysisConfig, api_key: str,
                 scratch: Scratch | None = None, *, max_tokens: int = 450,
                 news_fn=None, search_fn=None) -> None:
        self.cfg = cfg
        self._api_key = api_key
        self.scratch = scratch
        self.max_tokens = max_tokens
        self.news_fn = news_fn        # () -> dict | None  (cached BTC news + Fear&Greed)
        self.search_fn = search_fn    # (query) -> list[dict]  (web results)

    def _llm(self, payload: dict) -> dict | None:
        """One raw LLM call -> parsed JSON dict (or {'reply': prose}); None on error."""
        try:
            from openai import OpenAI  # lazy
            client = OpenAI(base_url=self.cfg.base_url, api_key=self._api_key)
            resp = client.chat.completions.create(
                model=self.cfg.model,
                messages=[{"role": "system", "content": _SYSTEM},
                          {"role": "user", "content": json.dumps(payload)}],
                temperature=0.2,
                timeout=self.cfg.request_timeout_s,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or ""
            try:
                return json.loads(raw)
            except Exception:  # noqa: BLE001 - model returned prose, wrap it
                return {"reply": raw.strip()}
        except Exception as e:  # noqa: BLE001 - never crash the bot
            log.warning("analyst LLM error: %s", e)
            return None

    def _respond(self, payload: dict, fallback: str, *, allow_search: bool = True) -> str:
        if not self._api_key:
            return fallback
        import datetime
        news = None
        if self.news_fn:
            try:
                news = self.news_fn()
            except Exception:  # noqa: BLE001
                news = None
        p = {**payload, "today": datetime.date.today().isoformat()}
        if news:
            p["news"] = news
        d = self._llm(p)
        if d is None:
            return fallback
        # bounded autonomous search: one round only (optionally date-bounded)
        q = str(d.get("search", "")).strip()
        if q and allow_search and self.search_fn:
            after = str(d.get("search_after", "")).strip() or None
            before = str(d.get("search_before", "")).strip() or None
            try:
                results = self.search_fn(q, after, before)[:6]
            except Exception:  # noqa: BLE001
                results = []
            tag = q + (f" [{after or '…'}→{before or '…'}]" if (after or before) else "")
            d2 = self._llm({**p, "search_query": q, "search_after": after,
                            "search_before": before, "search_results": results})
            if d2 is not None:
                d = d2
                d["_searched"] = tag
        reply = str(d.get("reply", "")).strip() or fallback
        if d.get("_searched"):
            reply += f"\n🔎 searched: {d['_searched']}"
        saved = self._save_files(d.get("files") or [])
        if saved:
            reply += "\n📝 saved: " + ", ".join(saved)
        return reply

    def _save_files(self, files) -> list[str]:
        if not self.scratch or not isinstance(files, list):
            return []
        saved = []
        for fobj in files[:5]:
            try:
                name = str(fobj["name"]); content = str(fobj.get("content", ""))
                saved.append(self.scratch.write(name, content))
            except (ScratchError, KeyError, TypeError) as e:
                log.warning("scratch write rejected: %s", e)
        return saved

    def event_read(self, m: dict, signals) -> str:
        triggers = [{"name": getattr(s, "name", str(s)),
                     "kind": getattr(s, "kind", ""),
                     "detail": getattr(s, "detail", "")} for s in signals]
        payload = {"task": "event_read", "triggers": triggers, "state": _features(m)}
        return self._respond(payload, fallback=numeric_summary(m))

    def answer(self, question: str, m: dict) -> str:
        payload = {"task": "answer_question", "question": question[:500],
                   "state": _features(m)}
        return self._respond(payload, fallback=numeric_summary(m))

    def search(self, query: str, m: dict, after=None, before=None) -> str:
        """Explicit /search: do the (optionally date-bounded) lookup ourselves, then
        have the LLM synthesize over the article bodies."""
        results = []
        if self.search_fn:
            try:
                results = self.search_fn(query, after, before)[:6]
            except Exception:  # noqa: BLE001
                results = []
        payload = {"task": "web_search_synthesis", "query": query[:300],
                   "search_after": after, "search_before": before,
                   "search_results": results, "state": _features(m)}
        return self._respond(payload, fallback=format_search(query, results),
                             allow_search=False)  # already searched


def build_analyst(cfg: AnalysisConfig, api_key: str, scratch: Scratch | None,
                  *, max_tokens: int = 450, news_fn=None, search_fn=None):
    """Real analyst if a key is present and analysis enabled, else the mock."""
    if cfg.enabled and api_key:
        return LLMAnalyst(cfg, api_key, scratch, max_tokens=max_tokens,
                          news_fn=news_fn, search_fn=search_fn)
    return MockAnalyst(scratch)
