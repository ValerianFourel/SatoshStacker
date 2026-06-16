"""News + sentiment + general web search context for the read-only analyst.

Two keyless feeds (mirroring the trading agent's approach — no API key, fail-safe):
  * ``fetch_news`` — BTC headlines (Yahoo Finance RSS) + Fear&Greed (alternative.me).
  * ``web_search`` — a general query. Prefers a configured provider for quality
    (Tavily, then Serper), else falls back to keyless DuckDuckGo HTML. Always
    returns a list (possibly empty) — a failed search never raises.

This is OBSERVE-only context for the analyst's prose. Fetched text is untrusted;
it can never trigger an action (the watch service has no order path at all).
"""
from __future__ import annotations

import datetime
import html as _htmllib
import logging
import os
import re
import time
import xml.etree.ElementTree as ET

import requests

log = logging.getLogger("satoshistacker.websearch")

_UA = {"User-Agent": "Mozilla/5.0"}
_TAG = re.compile(r"<[^>]+>")


def _strip(s: str) -> str:
    """Strip HTML tags and decode entities (&amp; &#91; &#x27; ...)."""
    return _htmllib.unescape(_TAG.sub("", s or "")).strip()


def _as_date(x) -> datetime.date | None:
    """Accept a date, datetime, or 'YYYY-MM-DD' string -> date (or None)."""
    if x is None:
        return None
    if isinstance(x, datetime.datetime):
        return x.date()
    if isinstance(x, datetime.date):
        return x
    try:
        return datetime.date.fromisoformat(str(x)[:10])
    except Exception:  # noqa: BLE001
        return None


def fetch_news(max_items: int = 6, *, asset: str = "BTC") -> dict:
    """``asset`` headlines (Yahoo RSS for <asset>-USD) + Fear&Greed day/week/month sentiment.
    No key. Fear&Greed is the crypto-market-wide index (BTC-driven) — a general gauge even
    for alts like XMR."""
    out: dict = {"headlines": [], "fear_greed": None, "sentiment": None}
    try:
        r = requests.get("https://feeds.finance.yahoo.com/rss/2.0/headline",
                         params={"s": f"{asset}-USD", "region": "US", "lang": "en-US"},
                         headers=_UA, timeout=10)
        root = ET.fromstring(r.text)
        out["headlines"] = [(it.findtext("title") or "").strip()
                            for it in root.findall(".//item")[:max_items]
                            if it.findtext("title")]
    except Exception:  # noqa: BLE001 - news is advisory only
        pass
    try:  # 30 days of Fear&Greed -> today / 7-day / 30-day sentiment
        d = requests.get("https://api.alternative.me/fng/", params={"limit": 30},
                         timeout=8).json()["data"]
        vals = [int(x["value"]) for x in d]            # d[0] = today, descending
        out["fear_greed"] = {"value": vals[0], "label": d[0]["value_classification"]}
        out["sentiment"] = {
            "day": vals[0],
            "week_avg": round(sum(vals[:7]) / min(7, len(vals))),
            "month_avg": round(sum(vals[:30]) / len(vals)),
        }
    except Exception:  # noqa: BLE001
        pass
    return out


def news_line(news: dict) -> str:
    """Human rendering of fetch_news output (for /news + fallbacks)."""
    if not news:
        return "no news"
    fg, s = news.get("fear_greed"), news.get("sentiment")
    head = ""
    if fg:
        head = f"😨 *Fear & Greed:* {fg['value']} ({fg['label']})"
        if s:
            head += f"\n   day {s['day']} · week {s['week_avg']} · month {s['month_avg']}"
        head += "\n\n📰 *Headlines:*\n"
    hs = news.get("headlines") or []
    body = "\n".join(f"• {h}" for h in hs[:6]) if hs else "(no headlines)"
    return head + body


# ───────────────────────────── general web search ───────────────────────────

def _tavily(q, key, n, timeout, after=None, before=None) -> list[dict]:
    # include_raw_content => Tavily returns the cleaned article body, so on a server
    # with a Tavily key you get full article text with no separate fetch step.
    body = {"api_key": key, "query": q, "max_results": n, "search_depth": "basic",
            "include_raw_content": True}
    if after:
        body["start_date"] = after.isoformat()   # Tavily YYYY-MM-DD date filters
    if before:
        body["end_date"] = before.isoformat()
    r = requests.post("https://api.tavily.com/search", timeout=timeout, json=body)
    r.raise_for_status()
    out = []
    for x in r.json().get("results", [])[:n]:
        item = {"title": _strip(x.get("title", "")),
                "snippet": _strip(x.get("content", "")), "url": x.get("url", "")}
        raw = x.get("raw_content")
        if raw:
            item["content"] = _strip(raw)[:4000]
        out.append(item)
    return out


def _serper_tbs(after, before) -> str:
    """Google 'tbs' custom-date-range token used by Serper."""
    if not (after or before):
        return ""
    parts = ["cdr:1"]
    if after:
        parts.append("cd_min:" + after.strftime("%m/%d/%Y"))
    if before:
        parts.append("cd_max:" + before.strftime("%m/%d/%Y"))
    return ",".join(parts)


def _serper(q, key, n, timeout, after=None, before=None) -> list[dict]:
    payload = {"q": q, "num": n}
    tbs = _serper_tbs(after, before)
    if tbs:
        payload["tbs"] = tbs
    r = requests.post("https://google.serper.dev/search", timeout=timeout,
                      headers={"X-API-KEY": key, "Content-Type": "application/json"},
                      json=payload)
    r.raise_for_status()
    return [{"title": _strip(x.get("title", "")), "snippet": _strip(x.get("snippet", "")),
             "url": x.get("link", "")} for x in r.json().get("organic", [])[:n]]


_DDG_A = re.compile(r'result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.S)
_DDG_S = re.compile(r'result__snippet"[^>]*>(.*?)</a>', re.S)


def _ddg_df(after, before) -> str:
    """DuckDuckGo 'df' date-range token: YYYY-MM-DD..YYYY-MM-DD (open ends filled)."""
    if not (after or before):
        return ""
    lo = (after or datetime.date(2009, 1, 1)).isoformat()   # bitcoin genesis as a floor
    hi = (before or datetime.date.today()).isoformat()
    return f"{lo}..{hi}"


def _duckduckgo(q, n, timeout, after=None, before=None) -> list[dict]:
    params = {"q": q}
    df = _ddg_df(after, before)
    if df:
        params["df"] = df
    r = requests.get("https://html.duckduckgo.com/html/", params=params,
                     headers=_UA, timeout=timeout)
    r.raise_for_status()
    links = _DDG_A.findall(r.text)
    snips = _DDG_S.findall(r.text)
    out = []
    for i, (url, title) in enumerate(links[:n]):
        out.append({"title": _strip(title), "url": url,
                    "snippet": _strip(snips[i]) if i < len(snips) else ""})
    return out


def _within(pub, after, before) -> bool:
    d = _as_date(pub)
    if d is None:
        return True                       # keep undated results
    if after and d < after:
        return False
    if before and d > before:
        return False
    return True


def _searxng(q, base, n, timeout, after=None, before=None) -> list[dict]:
    # SearXNG's API has no arbitrary date range, so we client-filter on publishedDate
    # when results carry one (undated results are kept).
    r = requests.get(base.rstrip("/") + "/search", headers=_UA, timeout=timeout,
                     params={"q": q, "format": "json"})
    r.raise_for_status()
    raw = r.json().get("results", [])
    if after or before:
        raw = [x for x in raw if _within(x.get("publishedDate"), after, before)]
    return [{"title": _strip(x.get("title", "")), "snippet": _strip(x.get("content", "")),
             "url": x.get("url", "")} for x in raw[:n]]


def web_search(query: str, *, max_results: int = 5, timeout: float = 12.0,
               after=None, before=None) -> list[dict]:
    """Search the web for ``query``, optionally restricted to articles dated
    >= ``after`` and/or <= ``before`` (date / datetime / 'YYYY-MM-DD'). Provider
    order: self-hosted SearXNG -> Tavily -> Serper -> keyless DuckDuckGo.
    Returns [{title, snippet, url}], [] on failure."""
    query = (query or "").strip()
    if not query:
        return []
    after, before = _as_date(after), _as_date(before)
    searx = os.getenv("SEARXNG_URL", "")
    tav, ser = os.getenv("TAVILY_API_KEY", ""), os.getenv("SERPER_API_KEY", "")
    providers = []
    if searx.strip():
        providers.append(lambda: _searxng(query, searx.strip(), max_results, timeout, after, before))
    if tav.strip():
        providers.append(lambda: _tavily(query, tav.strip(), max_results, timeout, after, before))
    if ser.strip():
        providers.append(lambda: _serper(query, ser.strip(), max_results, timeout, after, before))
    providers.append(lambda: _duckduckgo(query, max_results, timeout, after, before))
    for fn in providers:
        try:
            res = fn()
            if res:
                return res
        except Exception as e:  # noqa: BLE001 - try the next provider, then give up
            log.warning("web search provider failed: %s", e)
    return []


# ─────────────────────────── read the actual article ────────────────────────

_SCRIPT = re.compile(r"<(script|style|noscript)[^>]*>.*?</\1>", re.S | re.I)
_PARA = re.compile(r"<p[^>]*>(.*?)</p>", re.S | re.I)


def _html_to_text(html: str) -> str:
    """Dependency-free fallback extractor: real paragraph text only (tags/scripts/nav
    stripped). Returns "" if a page has no article-like <p> body (e.g. a JS dashboard)
    rather than dumping menu/footer boilerplate."""
    html = _SCRIPT.sub(" ", html)
    paras = [_strip(p) for p in _PARA.findall(html)]
    return "\n".join(p for p in paras if len(p) > 40)


def fetch_article(url: str, *, max_chars: int = 2500, timeout: float = 12.0) -> str:
    """Fetch a URL and extract the main readable text. Uses trafilatura if installed
    (best quality), else a dependency-free fallback. Returns "" on any failure."""
    if not url or not url.startswith(("http://", "https://")):
        return ""
    try:
        r = requests.get(url, headers=_UA, timeout=timeout)
        r.raise_for_status()
        html = r.text
    except Exception as e:  # noqa: BLE001
        log.warning("article fetch failed (%s): %s", url, e)
        return ""
    try:
        import trafilatura  # optional, lazy — `pip install trafilatura` for clean text
        txt = trafilatura.extract(html, include_comments=False, include_tables=False)
        if txt:
            return txt[:max_chars]
    except Exception:  # noqa: BLE001 - not installed or extraction failed -> fallback
        pass
    return _html_to_text(html)[:max_chars]


def search_and_read(query: str, *, max_results: int = 5, fetch_n: int = 2,
                    max_chars: int = 2500, timeout: float = 12.0,
                    after=None, before=None) -> list[dict]:
    """Search (optionally date-bounded by ``after``/``before``), then OPEN and READ the
    top ``fetch_n`` results (full article body added as ``content``). This is what the
    analyst gets, so it reads articles, not snippets."""
    results = web_search(query, max_results=max_results, timeout=timeout,
                         after=after, before=before)
    for r in results[:fetch_n]:
        if r.get("content"):                 # provider already returned the body (e.g. Tavily)
            continue
        body = fetch_article(r.get("url", ""), max_chars=max_chars, timeout=timeout)
        if body:
            r["content"] = body
    return results


class NewsCache:
    """TTL cache so the bot doesn't refetch news on every scan/question. ``asset`` picks the
    headline feed (BTC, XMR, …)."""

    def __init__(self, ttl_s: int = 600, *, asset: str = "BTC", clock=time.time) -> None:
        self.ttl_s = ttl_s
        self.asset = asset
        self._clock = clock
        self._at = 0.0
        self._data: dict = {}

    def get(self) -> dict:
        now = self._clock()
        if now - self._at >= self.ttl_s or not self._data:
            try:
                self._data = fetch_news(asset=self.asset)
                self._at = now
            except Exception:  # noqa: BLE001 - reuse last on failure
                pass
        return self._data
