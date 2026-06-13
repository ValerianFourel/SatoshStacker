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
    "You are a precise BTC market analyst for one operator. You receive a "
    "TIMEFRAME-LABELLED snapshot of computed numbers. Read each key's timeframe "
    "carefully and NEVER mix them up (a 5m reading is not a daily one):\n"
    "• `as_of_time` — when this data is from; anchor everything to it, flag if stale.\n"
    "• `price_now` — the LIVE price. `returns_pct` are over 1m / 5m / 1h / 24h.\n"
    "• `ranges` — accurate 24h and 7d high/low + position within the 24h range.\n"
    "• `technicals_<tf>` — RSI / EMA-trend / ATR on THAT candle timeframe.\n"
    "• `multi_timeframe` — RSI & trend on 5m/1h/4h/1d; use the timeframe matching the "
    "horizon being asked about.\n"
    "• `order_book` — depth/imbalance now. `futures` — perp funding (per 8h) + open interest.\n"
    "• `tuned_signals` — the backtest-best top/bottom indicators and their current readings.\n"
    "• `stacking_levels` — structural reentry (buy) & sell zones from support/resistance.\n"
    "• `news.sentiment` — Fear&Greed for day/week/month; `headlines` are recent.\n"
    "ORDER OF PRIORITY: read PRICE, CANDLES, order book, funding & OI FIRST; treat news "
    "& sentiment as SECONDARY context, not the lead. Say which timeframe a reading is on. "
    "Explain plainly what is happening RIGHT NOW (likely top/bottom, exhaustion, breakout, "
    "volume/volatility spike, thin book, crowded funding) with caveats. "
    "Use `plot` to chart whatever is most telling: it can be a FLAT list of indicator names "
    "(one image) OR a list of GROUPS like [[...],[...]] for a PATCHWORK of several images — "
    "orchestrate it yourself: put related metrics together, split across images when there are "
    "many, choose any of `available_indicators` (each image = price + its panels). [] = default. "
    "If you need fresh context set `search` to ONE concise query (else \"\"); for a time "
    "window set `search_after`/`search_before` as YYYY-MM-DD (resolve relative phrases via "
    "`today`). GOAL: the operator stacks satoshis (accumulate BTC; lower entries = more "
    "sats/$). When asked for a reentry/sell price, present `stacking_levels` (structural "
    "support/resistance) as levels of interest for stacking — caveat: not financial advice, "
    "the bot never trades. Otherwise stay descriptive (no directive buy/sell/hold or sizing). "
    "You MAY save notes to scratch files. Respond with STRICT JSON only:\n"
    '{"reply":"<=140 words","plot":[["rsi_14","macd_hist"],["obv_slope_14","vwap_dist_20"]],'
    '"search":"<query or empty>","search_after":"<YYYY-MM-DD or empty>",'
    '"search_before":"<YYYY-MM-DD or empty>","files":[{"name":"notes.md","content":"..."}]}'
    " (plot may instead be a flat list of names for one image)."
)


def numeric_summary(m: dict) -> str:
    """Deterministic one-glance card — the /raw reply and the LLM fail-safe."""
    if not m:
        return "📊 no snapshot yet — monitor is warming up"
    t, v, b = m.get("technicals", {}), m.get("volume", {}), m.get("order_book", {})
    price = m.get("price", 0)
    rsi = t.get("rsi_14", 50)
    rlabel = "overbought" if rsi >= 70 else "oversold" if rsi <= 30 else "neutral"
    fut, tm = m.get("futures", {}), m.get("time", {})
    rows = [
        f"24h range   ${t.get('low_24h', 0):,.0f} – ${t.get('high_24h', 0):,.0f}   "
        f"({t.get('range_position_pct', 0):.0f}% up)",
        f"7d range    ${t.get('low_7d', 0):,.0f} – ${t.get('high_7d', 0):,.0f}",
        f"RSI(14)     {rsi:<6}{rlabel}",
        f"Trend       {t.get('ema_trend_pct', 0):+.2f}%  (EMA 9/21)",
        f"Move        1m {t.get('ret_1m_pct', 0):+.2f}%  1h {t.get('ret_1h_pct', 0):+.2f}%  "
        f"24h {t.get('change_24h_pct', 0):+.2f}%",
        f"Volume      {v.get('surge_x', 1)}x avg   (z {v.get('z', 0):+.1f})",
        f"Volatility  ATR {t.get('atr_pct', 0):.2f}%  ·  realized {t.get('realized_vol_pct', 0):.3f}%",
    ]
    if b.get("ok"):
        band = b.get("bands", {}).get("1.0", {})
        rows.append(f"Book ±1%     bid {band.get('bid', 0)} / ask {band.get('ask', 0)}  "
                    f"(lean {band.get('imbalance', 0):+.2f})")
    if fut.get("funding_rate_pct") is not None:
        rows.append(f"Funding     {fut['funding_rate_pct']:+.4f}%/8h  ({fut.get('funding_annualized_pct')}%/yr)")
    if fut.get("open_interest") is not None:
        oic = fut.get("oi_change_24h_pct")
        rows.append(f"Open int.   {fut['open_interest']:,.0f} BTC"
                    + (f"  ({oic:+.1f}% 24h)" if oic is not None else ""))
    if fut.get("long_short_ratio") is not None:
        rows.append(f"Long/Short  {fut['long_short_ratio']:.2f}  (retail accounts)")
    tuned = m.get("tuned", {})
    if tuned.get("top") or tuned.get("bottom"):
        from .signal_tuner import _pretty
        for side, arrow in (("top", ">="), ("bottom", "<=")):
            s = tuned.get(side)
            if s:
                rows.append(f"{'Top-watch' if side == 'top' else 'Bot-watch':<11} "
                            f"{_pretty(s['name'])} {s['value']} (fires {arrow}{s['threshold']})")
    hhmm = (tm.get("scan_iso", "") or m.get("iso", ""))[11:16]
    out = (f"📊 *BTC*  `${price:,.0f}`   _{t.get('change_24h_pct', 0):+.2f}% 24h_\n"
           "```\n" + "\n".join(rows) + "\n```"
           f"_live · as of {hhmm}Z_")
    if m.get("events"):
        out += "\n⚠️ *fired:* " + ", ".join(m["events"])
    return out


def _stacking_levels(m: dict):
    try:
        from .levels import suggest_levels
        return suggest_levels(m)
    except Exception:  # noqa: BLE001
        return None


def _features(m: dict) -> dict:
    """Timeframe-LABELLED payload for the LLM — every block tagged with its horizon
    so the model never confuses a 5m reading with a daily one."""
    t = m.get("technicals", {})
    tf = m.get("time", {}).get("trend_tf", "1h")
    return {
        "as_of_time": m.get("time", {}),
        "price_now": m.get("price"),
        "returns_pct": {"1m": t.get("ret_1m_pct"), "5m": t.get("ret_5m_pct"),
                        "1h": t.get("ret_1h_pct"), "24h": t.get("change_24h_pct")},
        "ranges": {"high_24h": t.get("high_24h"), "low_24h": t.get("low_24h"),
                   "range_position_pct": t.get("range_position_pct"),
                   "high_7d": t.get("high_7d"), "low_7d": t.get("low_7d")},
        f"technicals_{tf}": {"rsi_14": t.get("rsi_14"),
                             "ema_trend_pct": t.get("ema_trend_pct"),
                             "atr_pct": t.get("atr_pct"),
                             "realized_vol_pct": t.get("realized_vol_pct")},
        "multi_timeframe": m.get("multi_tf"),
        "volume": m.get("volume"),
        "order_book": m.get("order_book"),
        "futures": m.get("futures"),
        "tuned_signals": m.get("tuned"),
        "stacking_levels": _stacking_levels(m),   # structural reentry/sell zones
        "onchain": m.get("onchain") or None,       # MVRV/NUPL/SOPR/netflow (cycle-level)
        "events_fired": m.get("events"),
    }


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
        self.last_plot: list = []

    def pick_indicators(self, m: dict, question: str | None = None) -> list:
        return []

    def event_read(self, m: dict, signals) -> str:
        names = ", ".join(getattr(s, "name", str(s)) for s in signals)
        return f"[mock] {names}\n{numeric_summary(m)}"

    def answer(self, question: str, m: dict, history=None) -> str:
        return f"[mock answer to: {question[:60]}]\n{numeric_summary(m)}"

    def search(self, query: str, m: dict, after=None, before=None) -> str:
        win = f" [{after or '…'}→{before or '…'}]" if (after or before) else ""
        return f"[mock search: {query[:60]}{win}]"

    def news_digest(self, m: dict) -> str:
        return "QUIET: [mock] no significant news"


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
        self.last_plot: list = []     # indicators the LLM chose to chart on the last call

    @staticmethod
    def _valid_plot_spec(plot) -> list:
        """Normalize the LLM's `plot` into image-GROUPS (list[list[name]]). Accepts a
        flat list of names (ONE image) or a list of groups (a PATCHWORK of images).
        Caps at 4 images x 5 panels; drops unknown names."""
        from .signal_tuner import PERIODS
        if not isinstance(plot, list) or not plot:
            return []
        groups = ([plot] if all(isinstance(x, str) for x in plot)
                  else [g for g in plot if isinstance(g, list)])
        out = []
        for g in groups[:4]:
            names = [str(n) for n in g if str(n) in PERIODS][:5]
            if names:
                out.append(names)
        return out

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
        from .signal_tuner import PERIODS
        p = {**payload, "today": datetime.date.today().isoformat(),
             "available_indicators": list(PERIODS)}
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
        self.last_plot = self._valid_plot_spec(d.get("plot"))   # LLM's patchwork plot spec
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

    def answer(self, question: str, m: dict, history=None) -> str:
        payload = {"task": "answer_question", "question": question[:500],
                   "recent_conversation": (history or [])[-8:], "state": _features(m)}
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

    def news_digest(self, m: dict) -> str:
        """Autonomous periodic news read (the analyst decides whether to ping). News is
        auto-attached; one search round allowed. The reply MUST start with 'ALERT:' (worth
        pinging) or 'QUIET:' (cache only). Conservative by design; fail-safe to QUIET."""
        payload = {"task": "news_digest", "state": _features(m), "instruction": (
            "Read the latest BTC news + sentiment (auto-attached); if useful run ONE search "
            "for fresh context. Summarize the key developments in <=120 words. Then DECIDE "
            "whether anything is significant enough to PROACTIVELY alert the operator (who is "
            "accumulating BTC and bearish on USD price). Begin your `reply` with 'ALERT:' if "
            "yes, otherwise 'QUIET:'. Be conservative — only ALERT on genuinely market-moving "
            "news (ETF flows, regulation, macro shocks, large liquidations, hacks).")}
        return self._respond(payload, fallback="QUIET: news read unavailable")

    def pick_indicators(self, m: dict, question: str | None = None) -> list:
        """Ask the LLM which up-to-3 indicators to chart (honouring ``question`` if given)."""
        self.last_plot = []
        payload = {"task": "pick_chart_indicators", "state": _features(m)}
        if question:
            payload["request"] = question[:300]
        self._respond(payload, fallback="", allow_search=False)
        return self.last_plot


def build_analyst(cfg: AnalysisConfig, api_key: str, scratch: Scratch | None,
                  *, max_tokens: int = 450, news_fn=None, search_fn=None):
    """Real analyst if a key is present and analysis enabled, else the mock."""
    if cfg.enabled and api_key:
        return LLMAnalyst(cfg, api_key, scratch, max_tokens=max_tokens,
                          news_fn=news_fn, search_fn=search_fn)
    return MockAnalyst(scratch)
