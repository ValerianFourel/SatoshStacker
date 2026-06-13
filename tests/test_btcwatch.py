"""Tests for the read-only BTC watch service: metric math, the anomaly detector
(fire-once + cooldown + re-arm), the scratch sandbox containment, a full monitor
cycle, and the Telegram listener routing + operator-chat gating.

Deterministic — no network, no LLM (data injected, MockAnalyst used)."""
import json

import numpy as np
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
        self.photos = []

    def send(self, text):
        self.sent.append(text)

    def send_photo(self, png, caption=""):
        self.photos.append((png, caption))


def test_monitor_run_once_fires_event_and_writes_snapshot(tmp_path):
    cfg = WatchConfig(snapshot_path=str(tmp_path / "snap.json"),
                      state_path=str(tmp_path / "state.json"),
                      tuned_signals_path=str(tmp_path / "none.json"),  # no tuned -> RSI rule
                      alert_charts=False,                              # hermetic: no chart fetch
                      daily_update_tzs=())                             # no digest in this test
    note = FakeNotifier()

    def klines_fn(tf, lim):
        return flat_fast() if tf == cfg.kline_tf else trend_klines()

    mon = MarketMonitor(cfg, notifier=note, analyst=MockAnalyst(),
                        price_fn=lambda: 110_000.0, klines_fn=klines_fn, book_fn=book,
                        ticker_fn=lambda: None, funding_fn=lambda: None, clock=lambda: 100.0)
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
    raw = lis.handle_text("/raw")
    assert "BTC" in raw and "$" in raw and "RSI" in raw
    assert "mock answer" in lis.handle_text("is this a top?")
    # the guide is discoverable (bare "help") and actually explains the features
    for trigger in ("help", "?", "/start"):
        assert "Search by date" in lis.handle_text(trigger)
    assert "Commands" in lis.handle_text("help")


# ── sharing / allowlist, memory, daily digest, NL charts ──
def test_listener_allowlist_multiple_chats():
    note = FakeNotifier()
    lis = TelegramListener(WatchConfig(), token="t", chat_id="42,99", analyst=MockAnalyst(),
                           notifier=note, snapshot_fn=lambda: {})
    lis._process_update({"message": {"chat": {"id": 99}, "text": "/help"}})
    assert len(note.sent) == 1                     # 99 is on the allowlist
    lis._process_update({"message": {"chat": {"id": 7}, "text": "hi"}})
    assert len(note.sent) == 1                     # stranger still ignored


def test_notifier_broadcasts_to_each_chat(monkeypatch):
    from agent.notify import Notifier
    n = Notifier(token="tok", chat_id="11,22,33")
    posts = []
    monkeypatch.setattr("requests.post",
                        lambda url, **k: posts.append(k.get("json", {}).get("chat_id")))
    n.send("hi")
    assert posts == ["11", "22", "33"]             # broadcast to all three


def test_conversation_memory_ttl_and_cap(tmp_path):
    from agent.convo import Conversation
    t = {"now": 1000.0}
    c = Conversation(str(tmp_path / "c.json"), ttl_s=100, max_turns=4, clock=lambda: t["now"])
    c.add("user", "q1")
    c.add("assistant", "a1")
    assert [x["text"] for x in c.recent()] == ["q1", "a1"]
    t["now"] = 1200.0                              # past the 100s TTL
    assert c.recent() == []


def test_natural_language_chart_request(monkeypatch):
    note = FakeNotifier()
    lis = TelegramListener(WatchConfig(), token="t", chat_id="42", analyst=MockAnalyst(),
                           notifier=note, snapshot_fn=lambda: {"price": 1})
    monkeypatch.setattr(lis, "_build_chart", lambda snap, question=None: (b"PNG", "cap"))
    assert lis.handle_text("show me a chart of rsi") == ""     # photo, no text
    assert note.photos


def test_daily_digest_fires_once_per_day(tmp_path):
    import datetime
    try:
        from zoneinfo import ZoneInfo
        ny = ZoneInfo("America/New_York")
    except Exception:
        import pytest
        pytest.skip("no tzdata")
    now = datetime.datetime(2026, 6, 15, 9, 30, tzinfo=ny).timestamp()  # 9:30 NY
    cfg = WatchConfig(state_path=str(tmp_path / "st.json"),
                      snapshot_path=str(tmp_path / "sn.json"),
                      daily_update_tzs=("America/New_York",), alert_charts=False)
    note = FakeNotifier()
    mon = MarketMonitor(cfg, notifier=note, analyst=MockAnalyst(), price_fn=lambda: 1,
                        klines_fn=lambda a, b: [], book_fn=lambda: {},
                        ticker_fn=lambda: None, funding_fn=lambda: None, clock=lambda: now)
    m = {"price": 1, "technicals": {}, "volume": {}, "order_book": {}, "time": {}}
    mon._maybe_daily_update(now, m)
    assert any("Daily BTC briefing" in s for s in note.sent)
    fired = len(note.sent)
    mon._maybe_daily_update(now, m)                # same day -> no resend
    assert len(note.sent) == fired


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
    nl = news_line({"fear_greed": {"value": 55, "label": "Greed"}, "headlines": ["BTC up"]})
    assert "55" in nl and "Greed" in nl and "BTC up" in nl


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
    news = lis.handle_text("/news")
    assert "50" in news and "Neutral" in news
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


# ── signal tuner (battery backtest for top/bottom calling) ──
def test_tuner_ranks_battery_on_cyclical_data():
    from agent.signal_tuner import tune_signals
    n, period = 320, 40
    klines = []
    for i in range(n):
        c = 100 + 10 * np.sin(i * 2 * np.pi / period)
        klines.append([float(i), c, c + 0.5, c - 0.5, c, 1000.0])
    res = tune_signals(klines, swing_w=8, tol=4)
    assert res["n_tops"] > 3 and res["n_bottoms"] > 3
    assert len(res["top_leaderboard"]) >= 10        # the whole battery is scored
    assert res["best_top"]["auc"] > 0.7             # a strong top-caller is found
    assert res["best_bottom"]["auc"] > 0.7          # ...and a bottom-caller


def test_compute_metrics_includes_tuned_values():
    tuned = {"best_top": {"name": "rsi_14", "threshold": 70.0, "auc": 0.8},
             "best_bottom": {"name": "rsi_14", "threshold": 30.0, "auc": 0.8}}
    m = compute_metrics(price=110_000.0, klines_fast=flat_fast(),
                        klines_trend=trend_klines(), order_book=book(),
                        cfg=WatchConfig(), now_ts=1.0, tuned=tuned)
    assert m["tuned"]["top"]["name"] == "rsi_14"
    assert m["tuned"]["top"]["value"] > 90          # rising trend -> RSI ~100


def test_detector_uses_tuned_signals_over_defaults():
    det = AnomalyDetector(WatchConfig())
    m = make_metrics()                               # RSI 50, not near 24h high
    m["tuned"] = {"top": {"name": "rsi_14", "value": 85.0,
                          "threshold": 70.0, "auc": 0.8}}
    assert "peak" in [s.name for s in det.evaluate(m, now_ts=0.0)]
    det2 = AnomalyDetector(WatchConfig())
    m2 = make_metrics()
    m2["tuned"] = {"bottom": {"name": "stoch_14", "value": 8.0,
                              "threshold": 20.0, "auc": 0.8}}
    assert "bottom" in [s.name for s in det2.evaluate(m2, now_ts=0.0)]


# ── time-accurate 24h + funding/OI ──
def test_compute_metrics_uses_official_24h_ticker():
    ticker = {"low": 62_830.0, "high": 64_394.0, "change_pct": -0.10, "last": 63_618.0}
    m = compute_metrics(price=63_618.0, klines_fast=flat_fast(price=63_618.0),
                        klines_trend=trend_klines(start=62_000.0, end=64_000.0),
                        order_book=book(mid=63_618.0), cfg=WatchConfig(),
                        now_ts=1_700_000_000.0, ticker24=ticker)
    t = m["technicals"]
    assert t["low_24h"] == 62_830.0 and t["high_24h"] == 64_394.0   # official, not 8-day
    assert t["change_24h_pct"] == -0.10
    assert "time" in m and m["time"]["candle_age_s"] >= 0            # time-aware


def test_compute_metrics_adds_funding_and_oi():
    funding = {"funding_rate": 0.0006, "mark": 63_595.0,
               "open_interest": 97_867.0, "oi_change_24h_pct": 1.5}
    m = compute_metrics(price=63_618.0, klines_fast=flat_fast(price=63_618.0),
                        klines_trend=trend_klines(start=62_000.0, end=64_000.0),
                        order_book=book(mid=63_618.0), cfg=WatchConfig(),
                        now_ts=1_700_000_000.0, funding=funding)
    assert m["futures"]["funding_rate_pct"] == 0.06                 # 0.0006 * 100
    assert m["futures"]["open_interest"] == 97_867.0


def test_detector_funding_and_oi_flags():
    det = AnomalyDetector(WatchConfig())
    m = make_metrics()
    m["futures"] = {"funding_rate_pct": 0.08, "funding_annualized_pct": 87.6,
                    "oi_change_24h_pct": 1.0}
    assert "funding_extreme" in [s.name for s in det.evaluate(m, now_ts=0.0)]
    det2 = AnomalyDetector(WatchConfig())
    m2 = make_metrics()
    m2["futures"] = {"oi_change_24h_pct": 12.0}
    assert "oi_spike" in [s.name for s in det2.evaluate(m2, now_ts=0.0)]


# ── plotting ──
def test_chart_renders_png_with_leading_indicators():
    from agent.plotter import build_btc_chart
    kl = [[float(i * 3_600_000), 100 + i, 101 + i, 99 + i, 100 + i, 10.0] for i in range(60)]
    tuned = {"best_top": {"name": "rsi_14", "threshold": 70, "auc": 0.7},
             "best_bottom": {"name": "mfi_14", "threshold": 30, "auc": 0.6}}
    png, cap = build_btc_chart(WatchConfig(), tuned, klines_fn=lambda tf, lim: kl)
    # PNG bytes if matplotlib is installed; None (fail-safe) if not — both acceptable
    assert png is None or (isinstance(png, bytes) and png[:4] == b"\x89PNG")
    assert "RSI(14)" in cap and "MFI(14)" in cap


def test_analyst_picks_indicators_via_llm():
    from agent.analyst import LLMAnalyst
    from agent.config import AnalysisConfig
    a = LLMAnalyst(AnalysisConfig(), api_key="x")
    a._llm = lambda p: {"reply": "", "plot": ["rsi_14", "macd_hist", "not_real"]}
    assert a.pick_indicators({"price": 1}) == ["rsi_14", "macd_hist"]   # invalid dropped


def test_build_chart_uses_llm_picked_indicators():
    from agent.plotter import build_btc_chart
    kl = [[float(i * 3_600_000), 100 + i, 101 + i, 99 + i, 100 + i, 10.0] for i in range(60)]
    png, cap = build_btc_chart(WatchConfig(), None, klines_fn=lambda tf, lim: kl,
                               indicators=["rsi_14", "cci_20"])
    assert "RSI(14)" in cap and "CCI(20)" in cap and "LLM-picked" in cap


def test_features_are_timeframe_labelled():
    from agent.analyst import _features
    snap = compute_metrics(price=110_000.0, klines_fast=flat_fast(),
                           klines_trend=trend_klines(), order_book=book(),
                           cfg=WatchConfig(), now_ts=1.0)
    f = _features(snap)
    assert {"returns_pct", "ranges", "as_of_time", "multi_timeframe"} <= set(f)
    assert any(k.startswith("technicals_") for k in f)     # readings tagged by timeframe


def test_listener_chart_command_sends_photo(monkeypatch):
    note = FakeNotifier()
    lis = TelegramListener(WatchConfig(), token="t", chat_id="42", analyst=MockAnalyst(),
                           notifier=note, snapshot_fn=lambda: {"price": 1})
    monkeypatch.setattr(lis, "_build_chart", lambda snap: (b"\x89PNG-data", "📈 chart"))
    assert lis.handle_text("/chart") == ""        # photo sent, no text reply
    assert note.photos and note.photos[0][0] == b"\x89PNG-data"


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
