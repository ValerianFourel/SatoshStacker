"""Tests for the read-only BTC watch service: metric math, the anomaly detector
(fire-once + cooldown + re-arm), the scratch sandbox containment, a full monitor
cycle, and the Telegram listener routing + operator-chat gating.

Deterministic — no network, no LLM (data injected, MockAnalyst used)."""
import json

import pytest

from agent.analyst import MockAnalyst, numeric_summary
from agent.config import WatchConfig
from agent.market_monitor import (AnomalyDetector, MarketMonitor, book_metrics,
                                  compute_metrics)
from agent.scratch import Scratch, ScratchError
from agent.telegram_listener import TelegramListener


# ── helpers ──
def trend_klines(n=60, start=100_000.0, end=110_000.0):
    out = []
    for i in range(n):
        c = start + (end - start) * i / (n - 1)
        out.append([float(i), c, c * 1.001, c * 0.999, c, 10.0])
    return out


def flat_fast(n=70, price=110_000.0, vol=10.0, last_vol=None):
    rows = [[float(i), price, price, price, price, vol] for i in range(n)]
    if last_vol is not None:
        rows[-1][5] = last_vol
    return rows


def book(mid=110_000.0, bid_q=5.0, ask_q=5.0):
    return {"bids": [[mid - 1, bid_q], [mid - 50, bid_q * 2]],
            "asks": [[mid + 1, ask_q], [mid + 50, ask_q * 2]]}


def make_metrics(**tech):
    base = dict(pct_from_high_24h=-5.0, pct_from_low_24h=20.0, rsi_14=50.0,
                ret_z=0.0, ret_1m_pct=0.0)
    base.update(tech)
    return {"price": 100_000.0, "technicals": base,
            "volume": {"z": 0.0, "surge_x": 1.0},
            "order_book": {"ok": True, "bands": {"1.0": {"imbalance": 0.0}}}}


# ── metric math ──
def test_compute_metrics_shape_and_book():
    m = compute_metrics(price=110_000.0, klines_fast=flat_fast(),
                        klines_trend=trend_klines(), order_book=book(),
                        cfg=WatchConfig(), now_ts=1_000.0)
    assert m["price"] == 110_000.0
    assert m["technicals"]["rsi_14"] == 100.0          # strictly rising -> 100
    assert m["order_book"]["ok"] and m["order_book"]["spread_bps"] > 0
    # balanced book -> imbalance ~0
    assert abs(m["order_book"]["bands"]["1.0"]["imbalance"]) < 0.01


def test_book_metrics_imbalance_sign():
    bm = book_metrics(book(bid_q=20.0, ask_q=1.0))
    assert bm["bands"]["1.0"]["imbalance"] > 0.5         # bid-heavy


# ── anomaly detector ──
def test_peak_fires_once_then_cooldown_then_rearm():
    cfg = WatchConfig()
    det = AnomalyDetector(cfg)
    peak = make_metrics(pct_from_high_24h=0.0, rsi_14=80.0)
    fired = det.evaluate(peak, now_ts=0.0)
    assert [s.name for s in fired] == ["peak"]
    # same condition immediately -> disarmed, no re-fire
    assert det.evaluate(peak, now_ts=1.0) == []
    # condition clears -> re-arms, but still within cooldown -> no fire
    det.evaluate(make_metrics(), now_ts=2.0)
    assert det.evaluate(peak, now_ts=3.0) == []
    # past cooldown AND re-armed -> fires again
    det.evaluate(make_metrics(), now_ts=cfg.alert_cooldown_s + 10)
    fired2 = det.evaluate(peak, now_ts=cfg.alert_cooldown_s + 20)
    assert [s.name for s in fired2] == ["peak"]


def test_volume_spike_and_bottom():
    det = AnomalyDetector(WatchConfig())
    vol = make_metrics()
    vol["volume"] = {"z": 4.5, "surge_x": 6.0}
    assert [s.name for s in det.evaluate(vol, now_ts=0.0)] == ["volume_spike"]
    bottom = make_metrics(pct_from_low_24h=0.1, rsi_14=20.0)
    assert "bottom" in [s.name for s in det.evaluate(bottom, now_ts=0.0)]


def test_detector_state_roundtrip():
    det = AnomalyDetector(WatchConfig())
    det.evaluate(make_metrics(pct_from_high_24h=0.0, rsi_14=80.0), now_ts=5.0)
    dumped = det.dump_state()
    det2 = AnomalyDetector(WatchConfig())
    det2.load_state(json.loads(json.dumps(dumped)))     # survives serialization
    # restored as disarmed within cooldown -> no immediate re-fire
    assert det2.evaluate(make_metrics(pct_from_high_24h=0.0, rsi_14=80.0),
                         now_ts=6.0) == []


# ── scratch sandbox ──
def test_scratch_write_read_and_containment(tmp_path):
    sc = Scratch(str(tmp_path / "scratch"))
    rel = sc.write("notes/x.md", "hello")
    assert sc.read(rel) == "hello"
    assert rel in sc.list()
    for bad in ("../escape.txt", "/etc/passwd", "a/../../b"):
        with pytest.raises(ScratchError):
            sc.write(bad, "nope")


# ── full monitor cycle ──
class FakeNotifier:
    def __init__(self):
        self.sent = []

    def send(self, text):
        self.sent.append(text)


def test_monitor_run_once_fires_event_and_writes_snapshot(tmp_path):
    cfg = WatchConfig(snapshot_path=str(tmp_path / "snap.json"),
                      state_path=str(tmp_path / "state.json"))
    note = FakeNotifier()

    def klines_fn(tf, lim):
        return flat_fast() if tf == cfg.kline_tf else trend_klines()

    mon = MarketMonitor(cfg, notifier=note, analyst=MockAnalyst(),
                        price_fn=lambda: 110_000.0, klines_fn=klines_fn,
                        book_fn=book, clock=lambda: 100.0)
    m = mon.run_once()
    assert "peak" in m["events"]                         # rising-to-high + RSI 100
    assert len(note.sent) == 1 and "peak" in note.sent[0]
    saved = json.load(open(cfg.snapshot_path))
    assert saved["events"] == m["events"]
    # second cycle: condition persists but disarmed -> no duplicate alert
    mon.run_once()
    assert len(note.sent) == 1


# ── telegram listener ──
def test_listener_routing():
    snap = compute_metrics(price=110_000.0, klines_fast=flat_fast(),
                           klines_trend=trend_klines(), order_book=book(),
                           cfg=WatchConfig(), now_ts=1.0)
    lis = TelegramListener(WatchConfig(), token="t", chat_id="42",
                           analyst=MockAnalyst(), notifier=FakeNotifier(),
                           snapshot_fn=lambda: snap)
    assert "read-only" in lis.handle_text("/help")
    assert "BTC $" in lis.handle_text("/raw")
    assert "mock answer" in lis.handle_text("is this a top?")
    # the guide is discoverable (bare "help") and actually explains the features
    for trigger in ("help", "?", "/start"):
        assert "Search by date" in lis.handle_text(trigger)
    assert "Commands" in lis.handle_text("help")


def test_listener_ignores_non_operator_chat():
    note = FakeNotifier()
    lis = TelegramListener(WatchConfig(), token="t", chat_id="42",
                           analyst=MockAnalyst(), notifier=note,
                           snapshot_fn=lambda: {})
    lis._process_update({"message": {"chat": {"id": 999}, "text": "hi"}})
    assert note.sent == []                                # stranger ignored
    lis._process_update({"message": {"chat": {"id": 42}, "text": "/help"}})
    assert len(note.sent) == 1                            # operator answered


# ── news + web search ──
def test_web_search_empty_and_format():
    from agent.analyst import format_search
    from agent.websearch import news_line, web_search
    assert web_search("") == []                            # no query -> no network
    assert "no results" in format_search("btc etf", [])
    out = format_search("btc", [{"title": "ETF inflows", "snippet": "big", "url": "u"}])
    assert "ETF inflows" in out
    assert "Fear&Greed: 55" in news_line(
        {"fear_greed": {"value": 55, "label": "Greed"}, "headlines": ["BTC up"]})


def test_newscache_ttl(monkeypatch):
    import agent.websearch as ws
    n = {"v": 0}

    def fake_fetch(max_items=6):
        n["v"] += 1
        return {"headlines": [f"h{n['v']}"], "fear_greed": None}

    monkeypatch.setattr(ws, "fetch_news", fake_fetch)
    t = {"now": 1000.0}
    c = ws.NewsCache(ttl_s=100, clock=lambda: t["now"])
    assert c.get()["headlines"] == ["h1"]
    t["now"] = 1050.0
    assert c.get()["headlines"] == ["h1"]                  # within TTL -> cached
    t["now"] = 1200.0
    assert c.get()["headlines"] == ["h2"]                  # past TTL -> refetched


def test_analyst_attaches_news_and_does_one_search_round():
    from agent.analyst import LLMAnalyst
    from agent.config import AnalysisConfig
    calls = []
    a = LLMAnalyst(AnalysisConfig(), api_key="x",
                   news_fn=lambda: {"headlines": ["BTC dips"], "fear_greed": None},
                   search_fn=lambda q, after=None, before=None:
                   [{"title": "macro", "snippet": "s", "url": "u"}])
    seq = [{"search": "why is btc dropping"}, {"reply": "macro selloff", "search": ""}]
    a._llm = lambda payload: (calls.append(payload), seq[len(calls) - 1])[1]
    out = a.answer("why down?", {"price": 63_000})
    assert "macro selloff" in out and "searched" in out
    assert "news" in calls[0]                              # news auto-attached
    assert calls[1]["search_results"][0]["title"] == "macro"   # results fed back


def test_listener_news_and_search_routing():
    lis = TelegramListener(
        WatchConfig(), token="t", chat_id="42", analyst=MockAnalyst(),
        notifier=FakeNotifier(), snapshot_fn=lambda: {"price": 1},
        news_fn=lambda: {"fear_greed": {"value": 50, "label": "Neutral"},
                         "headlines": ["BTC flat"]})
    assert "Fear&Greed: 50" in lis.handle_text("/news")
    assert "mock search" in lis.handle_text("/search why is btc up")


# ── article reading + self-hosted search ──
def test_html_to_text_drops_scripts_and_boilerplate():
    from agent.websearch import _html_to_text
    html = ("<html><head><script>var x = steal()</script><style>.a{}</style></head>"
            "<body><nav>menu</nav>"
            "<p>This is the first substantial paragraph of the actual article body.</p>"
            "<p>tiny</p>"
            "<p>A second real paragraph long enough to be kept by the extractor here.</p>"
            "</body></html>")
    txt = _html_to_text(html)
    assert "steal" not in txt and "menu" not in txt
    assert "first substantial paragraph" in txt and "second real paragraph" in txt


def test_fetch_article_guards_bad_urls():
    from agent.websearch import fetch_article
    assert fetch_article("") == ""
    assert fetch_article("not-a-url") == ""
    assert fetch_article("ftp://x/y") == ""               # non-http, no network


def test_search_and_read_attaches_article_bodies(monkeypatch):
    import agent.websearch as ws
    monkeypatch.setattr(ws, "web_search", lambda q, **k: [
        {"title": "a", "snippet": "s", "url": "http://x/1"},
        {"title": "b", "snippet": "s", "url": "http://x/2"},
        {"title": "c", "snippet": "s", "url": "http://x/3"}])
    monkeypatch.setattr(ws, "fetch_article", lambda url, **k: f"BODY:{url}")
    res = ws.search_and_read("q", fetch_n=2)
    assert res[0]["content"] == "BODY:http://x/1"          # top 2 opened & read
    assert res[1]["content"] == "BODY:http://x/2"
    assert "content" not in res[2]                         # 3rd left as snippet


def test_searxng_provider_preferred_when_configured(monkeypatch):
    import agent.websearch as ws
    for k in ("TAVILY_API_KEY", "SERPER_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SEARXNG_URL", "http://searx.local")
    monkeypatch.setattr(ws, "_searxng",
                        lambda q, base, n, t, after=None, before=None:
                        [{"title": "SX", "snippet": "", "url": "u"}])
    assert ws.web_search("btc")[0]["title"] == "SX"        # self-hosted used first


# ── date-filtered search ──
def test_provider_date_token_formatting():
    import datetime as dt
    from agent.websearch import _ddg_df, _serper_tbs
    a, b = dt.date(2026, 1, 1), dt.date(2026, 2, 1)
    assert _serper_tbs(a, b) == "cdr:1,cd_min:01/01/2026,cd_max:02/01/2026"
    assert _serper_tbs(None, None) == ""
    assert _ddg_df(a, b) == "2026-01-01..2026-02-01"
    assert _ddg_df(a, None).startswith("2026-01-01..")     # open end filled


def test_web_search_threads_dates_to_provider(monkeypatch):
    import datetime as dt

    import agent.websearch as ws
    for k in ("TAVILY_API_KEY", "SERPER_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SEARXNG_URL", "http://sx")
    cap = {}

    def fake_sx(q, base, n, t, after=None, before=None):
        cap["after"], cap["before"] = after, before
        return [{"title": "x", "snippet": "", "url": "u"}]

    monkeypatch.setattr(ws, "_searxng", fake_sx)
    ws.web_search("btc", after="2026-01-01", before="2026-02-01")
    assert cap["after"] == dt.date(2026, 1, 1)              # strings normalized to dates
    assert cap["before"] == dt.date(2026, 2, 1)


def test_analyst_passes_llm_extracted_dates_to_search():
    from agent.analyst import LLMAnalyst
    from agent.config import AnalysisConfig
    cap = {}

    def fake_search(q, after=None, before=None):
        cap.update(q=q, after=after, before=before)
        return [{"title": "t", "snippet": "s", "url": "u"}]

    a = LLMAnalyst(AnalysisConfig(), api_key="x", search_fn=fake_search)
    seq = [{"search": "btc etf flows", "search_after": "2026-01-01",
            "search_before": "2026-02-01"}, {"reply": "flows were positive"}]
    calls = []
    a._llm = lambda p: (calls.append(p), seq[len(calls) - 1])[1]
    out = a.answer("etf flows between jan and feb 2026?", {"price": 1})
    assert cap["after"] == "2026-01-01" and cap["before"] == "2026-02-01"
    assert "flows were positive" in out and "2026-01-01" in out  # window shown
    assert "today" in calls[0]                              # today injected for relative dates


def test_first_launch_onboarding_sends_once(tmp_path):
    from agent.btcwatch import _first_launch_onboarding
    note = FakeNotifier()
    marker = str(tmp_path / "onboarded")
    assert _first_launch_onboarding(note, marker) is True      # first launch -> sends
    assert len(note.sent) == 1 and "Welcome" in note.sent[0]
    assert _first_launch_onboarding(note, marker) is False     # restart -> no resend
    assert len(note.sent) == 1


def test_listener_search_date_tokens():
    lis = TelegramListener(WatchConfig(), token="t", chat_id="42", analyst=MockAnalyst(),
                           notifier=FakeNotifier(), snapshot_fn=lambda: {"price": 1})
    out = lis.handle_text("/search btc etf flows between:2026-01-01..2026-02-01")
    assert "btc etf flows" in out and "2026-01-01" in out and "2026-02-01" in out
    assert lis.handle_text("/search halving news after:2024-04-01").count("2024-04-01")
