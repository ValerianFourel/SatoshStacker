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
    det = AnomalyDetector(cfg)                            # level=None -> raw cfg thresholds
    peak = make_metrics(pct_from_high_24h=0.0, rsi_14=80.0)
    assert [s.name for s in det.evaluate(peak, now_ts=0.0)] == ["peak"]
    assert det.evaluate(peak, now_ts=1.0) == []          # same condition -> disarmed
    # a single off-the-bar scan must NOT re-arm (hysteresis): clear then re-touch fast = silent
    det.evaluate(make_metrics(), now_ts=cfg.alert_cooldown_s + 10)
    assert det.evaluate(peak, now_ts=cfg.alert_cooldown_s + 11) == []
    # only a SUSTAINED clear (>= rearm_clear_s) re-arms; then past cooldown it fires again
    base = cfg.alert_cooldown_s + 11
    det.evaluate(make_metrics(), now_ts=base + 5)                        # clear streak starts
    det.evaluate(make_metrics(), now_ts=base + cfg.rearm_clear_s + 6)    # clear long enough -> re-arm
    fired2 = det.evaluate(peak, now_ts=base + cfg.rearm_clear_s + 7)
    assert [s.name for s in fired2] == ["peak"]


def test_hysteresis_parked_at_bar_fires_once():
    # a reading parked at the bar (active every scan) past cooldown must NOT re-fire
    cfg = WatchConfig()
    det = AnomalyDetector(cfg)
    peak = make_metrics(pct_from_high_24h=0.0, rsi_14=80.0)
    assert [s.name for s in det.evaluate(peak, now_ts=0.0)] == ["peak"]
    for t in range(1, 6):
        det.evaluate(peak, now_ts=cfg.alert_cooldown_s * t)             # still pinned, scans pass
    assert det._state["peak"]["armed"] is False                        # never re-armed while active
    # one brief flicker off (< rearm) then back on -> still silent
    det.evaluate(make_metrics(), now_ts=cfg.alert_cooldown_s * 6)
    assert det.evaluate(peak, now_ts=cfg.alert_cooldown_s * 6 + 1) == []


def test_sensitivity_preset_is_quieter():
    from agent.sensitivity import resolve
    assert resolve("low")["fund"] > resolve("high")["fund"]            # stricter bar
    assert resolve("low")["cooldown"] > resolve("high")["cooldown"]   # longer cooldown
    # funding 0.06%/8h fires at 'high' (bar 0.05) but NOT at 'low' (bar 0.12)
    m = make_metrics()
    m["futures"] = {"funding_rate_pct": 0.06, "funding_annualized_pct": 65.7,
                    "oi_change_24h_pct": 1.0, "long_short_ratio": 1.0}
    hi = AnomalyDetector(WatchConfig(), level="high")
    lo = AnomalyDetector(WatchConfig(), level="low")
    assert "funding_extreme" in [s.name for s in hi.evaluate(m, now_ts=0.0)]
    assert "funding_extreme" not in [s.name for s in lo.evaluate(m, now_ts=0.0)]


def test_muted_suppresses_all_alerts():
    det = AnomalyDetector(WatchConfig(), level="high", muted=True)
    peak = make_metrics(pct_from_high_24h=0.0, rsi_14=80.0)
    assert det.evaluate(peak, now_ts=0.0) == []                        # muted -> nothing fires


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
                      user_alerts_path=str(tmp_path / "alerts.json"),  # isolated alerts
                      prefs_path=str(tmp_path / "prefs.json"),         # isolated prefs
                      sensitivity="high",                              # old loose bars for this craft
                      confluence_min=1,                                # this test exercises single-signal firing
                      news_digest_hours=1e9,                           # don't fire a digest in this test
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


def test_listener_poll_startup_does_not_crash():
    import threading
    lis = TelegramListener(WatchConfig(), token="t", chat_id="42", analyst=MockAnalyst(),
                           notifier=FakeNotifier(), snapshot_fn=lambda: {})
    stop = threading.Event()
    stop.set()                                     # stop immediately
    lis.poll(stop)                                 # must run the startup log line + exit cleanly


def test_conversation_memory_ttl_and_cap(tmp_path):
    from agent.memory import Memory
    t = {"now": 1000.0}
    c = Memory(str(tmp_path / "m.jsonl"), ttl_s=100, max_chat_turns=4, clock=lambda: t["now"])
    c.add_chat("user", "q1")
    c.add_chat("assistant", "a1")
    assert [x["text"] for x in c.recent_chat()] == ["q1", "a1"]
    t["now"] = 1200.0                              # past the 100s TTL
    assert c.recent_chat() == []                   # auto-erased


def test_natural_language_chart_request(monkeypatch):
    note = FakeNotifier()
    lis = TelegramListener(WatchConfig(), token="t", chat_id="42", analyst=MockAnalyst(),
                           notifier=note, snapshot_fn=lambda: {"price": 1})
    import agent.plotter as pl
    monkeypatch.setattr(pl, "build_btc_chart", lambda *a, **k: (b"PNG", "cap"))
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


def test_derivs_chart_renders():
    from agent.plotter import build_derivs_chart
    kl = [[float(i * 3_600_000), 100 + i, 101 + i, 99 + i, 100 + i, 10.0] for i in range(40)]
    derivs = {"funding_rate": [(i * 3_600_000, 0.0001) for i in range(40)],
              "open_interest": [(i * 3_600_000, 90_000 + i) for i in range(40)],
              "long_short_ratio": [(i * 3_600_000, 1.5) for i in range(40)],
              "taker_buy_sell": [(i * 3_600_000, 1.1) for i in range(40)]}
    png, cap = build_derivs_chart(WatchConfig(), klines_fn=lambda tf, lim: kl,
                                  derivs_fn=lambda: derivs)
    assert png is None or png[:4] == b"\x89PNG"
    assert "derivatives" in cap.lower() and "Coinglass" in cap


def test_levels_target_scales_with_fees_and_spread():
    from agent.levels import suggest_levels
    m = {"price": 60000, "technicals": {"atr_pct": 1.0, "low_24h": 59000, "high_24h": 61000,
                                        "low_7d": 57000, "high_7d": 63000},
         "order_book": {"spread_bps": 2.0}}        # 0.02% spread
    tg = suggest_levels(m, target_pct=1.0, fee_pct=0.1)["target"]
    assert abs(tg["cost_pct"] - 0.22) < 1e-6        # 2*0.1 fee + 0.02 spread
    assert abs(tg["gross_pct"] - 1.22) < 1e-6       # target + cost
    assert tg["sell"] > tg["buy"]
    assert abs(tg["sell"] / tg["buy"] - 1.0122) < 1e-3
    # a bigger target -> a wider buy/sell gap
    tg3 = suggest_levels(m, target_pct=3.0, fee_pct=0.1)["target"]
    assert tg3["sell"] - tg3["buy"] > tg["sell"] - tg["buy"]
    assert "target" not in suggest_levels(m)        # no target -> no block (back-compat)


def test_levels_text_shows_target_and_parser():
    from agent.levels import levels_text
    from agent.telegram_listener import _parse_target_pct
    m = {"price": 60000, "technicals": {"atr_pct": 1.0, "low_24h": 59000, "high_24h": 61000},
         "order_book": {"spread_bps": 2.0}}
    txt = levels_text(m, target_pct=1.0, fee_pct=0.1)
    assert "1% net" in txt and "Buy" in txt and "Sell" in txt and "fees" in txt
    assert "want" not in levels_text(m).lower()      # no target -> structural only
    assert _parse_target_pct("/levels 1", allow_bare=True) == 1.0
    assert _parse_target_pct("/levels 0.5%", allow_bare=True) == 0.5
    assert _parse_target_pct("/levels", allow_bare=True) is None
    assert _parse_target_pct("i want a reentry and sell for about 2%") == 2.0
    assert _parse_target_pct("give me a sell price, i want 1.5 percent") == 1.5
    assert _parse_target_pct("reentry price please") is None
    assert _parse_target_pct("/levels 999", allow_bare=True) is None   # absurd -> ignored


def test_levels_sell_anchored():
    from agent.levels import suggest_levels, levels_text
    from agent.telegram_listener import _parse_levels_args
    m = {"price": 60000, "technicals": {"atr_pct": 1.0, "low_24h": 59000, "high_24h": 61000,
                                        "low_7d": 57000, "high_7d": 63000},
         "order_book": {"spread_bps": 2.0}}
    # sell-anchored at nearest resistance: buy = sell / (1+gross)
    tg = suggest_levels(m, target_pct=1.0, fee_pct=0.1, anchor="sell")["target"]
    assert tg["anchor"] == "sell" and tg["buy"] < tg["sell"]
    assert abs(tg["sell"] / tg["buy"] - 1.0122) < 1e-3
    # explicit sell price overrides the anchor; buy backs out from it
    tg2 = suggest_levels(m, target_pct=1.0, fee_pct=0.1, anchor="sell", anchor_price=65000)["target"]
    assert tg2["sell"] == 65000 and abs(tg2["buy"] - 65000 / 1.0122) < 5
    # text: sell-first + the alert hint flips to >=
    txt = levels_text(m, target_pct=1.0, anchor="sell")
    assert "anchored on the SELL" in txt and "price >=" in txt
    assert "price <=" in levels_text(m, target_pct=1.0, anchor="buy")
    # parser: command, explicit price, NL, side disambiguation
    assert _parse_levels_args("/sell 1", default_anchor="sell", allow_bare=True) == (1.0, "sell", None)
    assert _parse_levels_args("/levels 1", allow_bare=True) == (1.0, "buy", None)
    assert _parse_levels_args("/sell 1 at 65000", default_anchor="sell",
                              allow_bare=True) == (1.0, "sell", 65000.0)
    assert _parse_levels_args("sell at 65k, keep 1%") == (1.0, "sell", 65000.0)
    assert _parse_levels_args("where do i buy to keep 1% if i sell at 65000") == (1.0, "sell", 65000.0)
    assert _parse_levels_args("good sell price, want 1%") == (1.0, "sell", None)
    assert _parse_levels_args("reentry price, want 2%") == (2.0, "buy", None)


def test_sensitivity_prefs_and_listener_commands(tmp_path):
    from agent.sensitivity import read_prefs, write_prefs
    p = str(tmp_path / "prefs.json")
    assert read_prefs(p, default_level="normal") == {
        "sensitivity": "normal", "muted": False, "overrides": {}, "disabled": [],
        "confluence": 2, "cadence": 1800}
    write_prefs(p, sensitivity="high")
    assert read_prefs(p)["sensitivity"] == "high"
    write_prefs(p, muted=True)
    d = read_prefs(p)
    assert d["sensitivity"] == "high" and d["muted"] is True          # mute doesn't clobber level
    # listener commands drive the same file
    lis = TelegramListener(WatchConfig(prefs_path=p), token="t", chat_id="42",
                           analyst=MockAnalyst(), notifier=FakeNotifier(), snapshot_fn=lambda: {})
    assert "low" in lis.handle_text("/sensitivity low") and read_prefs(p)["sensitivity"] == "low"
    lis.handle_text("/mute")
    assert read_prefs(p)["muted"] is True
    lis.handle_text("/unmute")
    assert read_prefs(p)["muted"] is False
    assert "Manual sensitivity" in lis.handle_text("/sensitivity wat")  # bad token -> help
    assert read_prefs(p)["sensitivity"] == "low"                      # ...and no change


def test_manual_overrides_and_disable_signal(tmp_path):
    from agent.sensitivity import read_prefs
    p = str(tmp_path / "prefs.json")
    lis = TelegramListener(WatchConfig(prefs_path=p), token="t", chat_id="42",
                           analyst=MockAnalyst(), notifier=FakeNotifier(), snapshot_fn=lambda: {})
    lis.handle_text("/sensitivity set imbalance 0.95")               # raise one bar manually
    assert read_prefs(p)["overrides"]["imb"] == 0.95
    lis.handle_text("/sensitivity off imbalance")                   # silence book_imbalance entirely
    assert "book_imbalance" in read_prefs(p)["disabled"]
    lis.handle_text("/sensitivity set cooldown 90")                 # minutes -> seconds
    assert read_prefs(p)["overrides"]["cooldown"] == 5400
    # the detector honours both: bar raised AND the signal disabled
    pr = read_prefs(p)
    det = AnomalyDetector(WatchConfig(), level="high",
                          overrides=pr["overrides"], disabled=tuple(pr["disabled"]))
    assert det._thr()["imb"] == 0.95 and det._thr()["cooldown"] == 5400
    m = make_metrics()
    m["order_book"] = {"ok": True, "bands": {"1.0": {"imbalance": 0.99}}}  # over the bar...
    assert "book_imbalance" not in [s.name for s in det.evaluate(m, now_ts=0.0)]  # ...but off
    # re-enable + reset clears everything
    lis.handle_text("/sensitivity on imbalance")
    assert "book_imbalance" not in read_prefs(p)["disabled"]
    lis.handle_text("/sensitivity reset")
    assert read_prefs(p)["overrides"] == {} and read_prefs(p)["disabled"] == []


def test_monitor_refreshes_prefs_live(tmp_path):
    from agent.sensitivity import write_prefs
    p = str(tmp_path / "prefs.json")
    cfg = WatchConfig(prefs_path=p, sensitivity="normal", alert_charts=False,
                      snapshot_path=str(tmp_path / "s.json"), state_path=str(tmp_path / "st.json"),
                      tuned_signals_path=str(tmp_path / "none.json"), daily_update_tzs=(),
                      user_alerts_path=str(tmp_path / "a.json"))
    mon = MarketMonitor(cfg, notifier=FakeNotifier(), analyst=MockAnalyst(),
                        price_fn=lambda: 1.0, klines_fn=lambda a, b: trend_klines(),
                        book_fn=lambda: {}, ticker_fn=lambda: None, funding_fn=lambda: None,
                        clock=lambda: 100.0)
    assert mon.detector.level == "normal" and mon.detector.muted is False  # seeded from cfg
    write_prefs(p, sensitivity="low", muted=True)
    mon._refresh_prefs()
    assert mon.detector.level == "low" and mon.detector.muted is True      # picked up live


def test_onchain_text_status():
    from agent.onchain import onchain_text
    assert "unavailable" in onchain_text({})                 # no plan access -> honest status
    t = onchain_text({"mvrv": 2.1, "sopr": 1.0})
    assert "MVRV" in t and "2.1" in t


def test_detector_long_short_extreme():
    det = AnomalyDetector(WatchConfig())
    m = make_metrics()
    m["futures"] = {"long_short_ratio": 2.5}
    assert "long_short_extreme" in [s.name for s in det.evaluate(m, now_ts=0.0)]


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
    assert a.pick_indicators({"price": 1}) == [["rsi_14", "macd_hist"]]  # flat -> one image


def test_plot_spec_supports_patchwork():
    from agent.analyst import LLMAnalyst
    from agent.config import AnalysisConfig
    a = LLMAnalyst(AnalysisConfig(), api_key="x")
    a._llm = lambda p: {"reply": "", "plot": [["rsi_14", "macd_hist"], ["obv_slope_14", "x"], []]}
    assert a.pick_indicators({"price": 1}) == [["rsi_14", "macd_hist"], ["obv_slope_14"]]


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
    import agent.plotter as pl
    monkeypatch.setattr(pl, "build_btc_chart", lambda *a, **k: (b"\x89PNG-data", "chart"))
    note = FakeNotifier()
    lis = TelegramListener(WatchConfig(), token="t", chat_id="42", analyst=MockAnalyst(),
                           notifier=note, snapshot_fn=lambda: {"price": 1})
    assert lis.handle_text("/chart") == ""        # photo sent, no text reply
    assert note.photos and note.photos[0][0] == b"\x89PNG-data"


# ── custom user triggers ──
def test_alert_parse_and_resolve():
    from agent.alerts import parse_rule, resolve_metric
    assert parse_rule("rsi > 70") == ("rsi", ">", 70.0)
    assert parse_rule("price<60000") == ("price", "<", 60_000.0)
    assert parse_rule("funding >= 0.05") == ("funding", ">=", 0.05)
    assert parse_rule("gibberish") is None
    snap = {"price": 63_000, "technicals": {"rsi_14": 55},
            "futures": {"long_short_ratio": 1.6}, "multi_tf": {"4h": {"rsi_14": 61}}}
    assert resolve_metric(snap, "rsi") == 55
    assert resolve_metric(snap, "price") == 63_000
    assert resolve_metric(snap, "ls") == 1.6
    assert resolve_metric(snap, "rsi_4h") == 61
    assert resolve_metric(snap, "bogus") is None


def test_alert_fires_once_then_rearms():
    from agent.alerts import evaluate
    rules = [{"id": 1, "metric": "rsi", "op": ">", "value": 70, "armed": True}]
    assert len(evaluate(rules, {"technicals": {"rsi_14": 80}})) == 1     # crosses up -> fire
    assert evaluate(rules, {"technicals": {"rsi_14": 80}}) == []         # still high -> no spam
    evaluate(rules, {"technicals": {"rsi_14": 60}})                      # drops -> re-arm
    assert len(evaluate(rules, {"technicals": {"rsi_14": 80}})) == 1     # crosses again -> fire


def test_alert_store_and_listener_commands(tmp_path):
    cfg = WatchConfig(user_alerts_path=str(tmp_path / "a.json"))
    snap = {"price": 63_000, "technicals": {"rsi_14": 55}, "futures": {}, "volume": {}}
    lis = TelegramListener(cfg, token="t", chat_id="42", analyst=MockAnalyst(),
                           notifier=FakeNotifier(), snapshot_fn=lambda: snap)
    assert "set" in lis.handle_text("/alert rsi > 70")
    assert "#1" in lis.handle_text("/alerts")
    assert "set" in lis.handle_text("ping me when price below 60000")   # natural language
    assert "removed" in lis.handle_text("/delalert 1")


def test_monitor_fires_user_alert(tmp_path):
    from agent.alerts import AlertStore
    apath = str(tmp_path / "a.json")
    AlertStore(apath).add("rsi", ">", 70, clock=lambda: 0.0)
    cfg = WatchConfig(snapshot_path=str(tmp_path / "s.json"),
                      state_path=str(tmp_path / "st.json"), user_alerts_path=apath,
                      tuned_signals_path=str(tmp_path / "none.json"),
                      daily_update_tzs=(), alert_charts=False)
    note = FakeNotifier()
    mon = MarketMonitor(cfg, notifier=note, analyst=MockAnalyst(), price_fn=lambda: 1,
                        klines_fn=lambda a, b: [], book_fn=lambda: {},
                        ticker_fn=lambda: None, funding_fn=lambda: None, clock=lambda: 0.0)
    mon._check_user_alerts({"technicals": {"rsi_14": 80}})
    assert any("Trigger fired" in s for s in note.sent)


# ── sat-stacking levels tool ──
def test_levels_suggests_reentry_and_sell():
    from agent.levels import levels_text, suggest_levels
    snap = compute_metrics(price=63_500.0, klines_fast=flat_fast(price=63_500.0),
                           klines_trend=trend_klines(start=61_000.0, end=64_000.0),
                           order_book=book(mid=63_500.0), cfg=WatchConfig(), now_ts=1.0,
                           ticker24={"low": 62_830.0, "high": 64_394.0, "change_pct": 0.1})
    lv = suggest_levels(snap)
    assert lv["reentry"][0] < lv["reentry"][1] <= lv["price"]   # buy zone below price
    assert lv["sell"][0] >= lv["price"]                         # sell zone above price
    txt = levels_text(snap)
    assert "Reenter" in txt and "Sell" in txt and "stack" in txt.lower()


def test_listener_levels_command_and_nl():
    snap = compute_metrics(price=63_500.0, klines_fast=flat_fast(price=63_500.0),
                           klines_trend=trend_klines(start=61_000.0, end=64_000.0),
                           order_book=book(mid=63_500.0), cfg=WatchConfig(), now_ts=1.0,
                           ticker24={"low": 62_830.0, "high": 64_394.0, "change_pct": 0.1})
    lis = TelegramListener(WatchConfig(), token="t", chat_id="42", analyst=MockAnalyst(),
                           notifier=FakeNotifier(), snapshot_fn=lambda: snap)
    assert "Reenter" in lis.handle_text("/levels")
    assert "Reenter" in lis.handle_text("what's a good reentry and sell price?")


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


# ── confluence (>=2 signals must agree) ──
def _multi_active():
    """A snapshot with two distinct out-of-norm signals active: peak + volume spike."""
    m = make_metrics(pct_from_high_24h=0.0, rsi_14=80.0)   # peak
    m["volume"] = {"z": 4.5, "surge_x": 6.0}               # volume_spike
    return m


def test_confluence_requires_multiple_signals():
    det = AnomalyDetector(WatchConfig(), confluence=2)
    one = make_metrics(pct_from_high_24h=0.0, rsi_14=80.0)        # only 'peak' active
    assert det.evaluate(one, now_ts=0.0) == []                   # 1 < 2 -> no ping (the fix)
    fired = det.evaluate(_multi_active(), now_ts=10.0)
    assert {s.name for s in fired} == {"peak", "volume_spike"}   # whole agreeing bundle pings
    assert det.evaluate(_multi_active(), now_ts=11.0) == []       # disarmed within the episode


def test_confluence_cadence_then_rearm():
    cfg = WatchConfig()
    det = AnomalyDetector(cfg, confluence=2)         # cadence -> cooldown 1800, rearm 600
    assert len(det.evaluate(_multi_active(), now_ts=0.0)) == 2    # episode 1 fires
    # cluster persists PAST the cadence but never disperses -> still silent (one episode)
    assert det.evaluate(_multi_active(), now_ts=cfg.alert_cooldown_s + 5) == []
    base = cfg.alert_cooldown_s + 10
    det.evaluate(make_metrics(), now_ts=base)                              # dispersal streak starts
    det.evaluate(make_metrics(), now_ts=base + cfg.rearm_clear_s + 1)      # clear long enough -> re-arm
    fired2 = det.evaluate(_multi_active(), now_ts=base + cfg.rearm_clear_s + 2)
    assert len(fired2) == 2                                       # fresh cluster pings again


def test_confluence_custom_cadence_throttles():
    det = AnomalyDetector(WatchConfig(), confluence=2, cadence=1800)
    assert len(det.evaluate(_multi_active(), now_ts=0.0)) == 2
    # disperse to re-arm, but only 100s elapsed (< 1800 cadence) -> still throttled
    det.evaluate(make_metrics(), now_ts=10.0)
    det.evaluate(make_metrics(), now_ts=700.0)                   # re-armed (>600 clear)
    assert det.evaluate(_multi_active(), now_ts=701.0) == []     # cadence not yet elapsed


def test_confluence_one_preserves_single_fire():
    det = AnomalyDetector(WatchConfig(), confluence=1)           # off -> classic behaviour
    one = make_metrics(pct_from_high_24h=0.0, rsi_14=80.0)
    assert [s.name for s in det.evaluate(one, now_ts=0.0)] == ["peak"]


# ── /origins widget (pure logic) ──
def test_origins_widget_text_and_keyboard():
    from agent.origins import ORIGINS, keyboard, widget_text
    prefs = {"sensitivity": "high", "muted": False, "overrides": {}, "disabled": [],
             "confluence": 2, "cadence": 1800}
    txt = widget_text(prefs)
    assert "Update controls" in txt and "Confluence" in txt and "≥2" in txt and "30m" in txt
    flat = [b["callback_data"] for row in keyboard(prefs)["inline_keyboard"] for b in row]
    for canon, _e, _l in ORIGINS:
        assert f"og:t:{canon}" in flat                           # every origin is a toggle
    assert "og:c:+" in flat and "og:d:-" in flat and "og:x" in flat


def test_origins_apply_callback_toggle_and_clamps():
    from agent.origins import apply_callback
    prefs = {"sensitivity": "low", "muted": False, "overrides": {}, "disabled": [],
             "confluence": 2, "cadence": 1800}
    off, toast = apply_callback("og:t:book_imbalance", prefs)
    assert "book_imbalance" in off["disabled"] and "silenced" in toast
    on, _ = apply_callback("og:t:book_imbalance", off)
    assert "book_imbalance" not in on["disabled"]
    assert apply_callback("og:c:+", prefs)[0]["confluence"] == 3
    lo = apply_callback("og:c:-", prefs)[0]
    assert lo["confluence"] == 1 and apply_callback("og:c:-", lo)[0]["confluence"] == 1   # clamp
    assert apply_callback("og:d:-", prefs)[0]["cadence"] == 1500          # -5 min
    assert apply_callback("og:p", prefs)[0]["sensitivity"] == "normal"    # cycle preset
    assert apply_callback("og:m", prefs)[0]["muted"] is True
    assert apply_callback("og:r", prefs) == (None, "")                    # refresh -> no change
    assert apply_callback("og:x", prefs)[0] is None                       # close -> no change
    assert apply_callback("bogus", prefs)[0] is None


# ── autonomous 8h news digest ──
class _DigestAnalyst:
    def __init__(self, reply):
        self.reply = reply
        self.last_plot = []

    def news_digest(self, m):
        return self.reply


def _digest_monitor(tmp_path, analyst, *, clock):
    cfg = WatchConfig(state_path=str(tmp_path / "st.json"), snapshot_path=str(tmp_path / "sn.json"),
                      news_digest_path=str(tmp_path / "nd.json"), news_digest_hours=8.0,
                      daily_update_tzs=(), alert_charts=False,
                      tuned_signals_path=str(tmp_path / "none.json"),
                      user_alerts_path=str(tmp_path / "a.json"), prefs_path=str(tmp_path / "p.json"))
    note = FakeNotifier()
    mon = MarketMonitor(cfg, notifier=note, analyst=analyst, price_fn=lambda: 1,
                        klines_fn=lambda a, b: [], book_fn=lambda: {}, ticker_fn=lambda: None,
                        funding_fn=lambda: None, clock=clock)
    return cfg, note, mon


def test_news_digest_alert_pings_caches_and_throttles(tmp_path):
    now = 5_000_000.0
    cfg, note, mon = _digest_monitor(tmp_path, _DigestAnalyst("ALERT: ETF outflows accelerating"),
                                     clock=lambda: now)
    m = {"price": 1, "technicals": {}, "volume": {}, "order_book": {}, "time": {}}
    mon._maybe_news_digest(now, m)
    assert any("news watch" in s for s in note.sent)             # it decided to ping
    cached = json.load(open(cfg.news_digest_path))
    assert cached["alert"] is True and "ETF" in cached["body"]   # copy kept
    before = len(note.sent)
    mon._maybe_news_digest(now + 60, m)                          # < 8h later -> throttled
    assert len(note.sent) == before


def test_news_digest_quiet_is_silent_but_cached(tmp_path):
    now = 5_000_000.0
    cfg, note, mon = _digest_monitor(tmp_path, _DigestAnalyst("QUIET: nothing notable"),
                                     clock=lambda: now)
    mon._maybe_news_digest(now, {"price": 1})
    assert note.sent == []                                       # QUIET -> no ping
    assert json.load(open(cfg.news_digest_path))["alert"] is False


def test_news_digest_muted_caches_but_silent(tmp_path):
    now = 5_000_000.0
    cfg, note, mon = _digest_monitor(tmp_path, _DigestAnalyst("ALERT: big news"), clock=lambda: now)
    mon.detector.muted = True                                    # muted suppresses the ping
    mon._maybe_news_digest(now, {"price": 1})
    assert note.sent == []                                       # silenced
    assert json.load(open(cfg.news_digest_path))["alert"] is True  # ...but still kept a copy


# ── listener: /origins + /digest routing + callback gating ──
def test_listener_origins_and_digest_routing(tmp_path, monkeypatch):
    class _Resp:
        def json(self):
            return {"ok": True, "result": {}}
    posts = []
    monkeypatch.setattr("requests.post", lambda url, **k: posts.append(url) or _Resp())
    from agent.sensitivity import read_prefs
    p = str(tmp_path / "p.json")
    lis = TelegramListener(WatchConfig(prefs_path=p, news_digest_path=str(tmp_path / "nd.json")),
                           token="t", chat_id="42", analyst=MockAnalyst(),
                           notifier=FakeNotifier(), snapshot_fn=lambda: {})
    assert "Update controls" in lis.handle_text("/origins")      # text view
    assert "no autonomous digest" in lis.handle_text("/digest")  # nothing cached yet
    # a button tap from the operator persists to the same prefs file
    lis._handle_callback({"id": "1", "data": "og:t:book_imbalance",
                          "message": {"chat": {"id": 42}, "message_id": 7}})
    assert "book_imbalance" in read_prefs(p)["disabled"]
    lis._handle_callback({"id": "2", "data": "og:c:+",
                          "message": {"chat": {"id": 42}, "message_id": 7}})
    assert read_prefs(p)["confluence"] == 3
    # a tap from a stranger changes nothing
    lis._handle_callback({"id": "3", "data": "og:c:+",
                          "message": {"chat": {"id": 999}, "message_id": 7}})
    assert read_prefs(p)["confluence"] == 3
    # the /origins command sends an interactive panel (sendMessage hit)
    lis._process_update({"message": {"chat": {"id": 42}, "text": "/origins"}})
    assert any("sendMessage" in u for u in posts)


def test_upgrade_notice_fires_once_and_bakes_prefs(tmp_path):
    from agent.btcwatch import _maybe_upgrade_notice
    from agent.sensitivity import read_prefs, write_prefs
    p = str(tmp_path / "prefs.json")
    write_prefs(p, sensitivity="high", overrides={"imb": 0.86})    # an OLD-format prefs file...
    # simulate the live file having no confluence/cadence keys
    raw = json.load(open(p)); raw.pop("confluence", None); raw.pop("cadence", None)
    json.dump(raw, open(p, "w"))
    cfg = WatchConfig(prefs_path=p, upgrade_marker=str(tmp_path / "up.marker"))
    note = FakeNotifier()
    assert _maybe_upgrade_notice(cfg, note) is True                # existing install -> notify
    assert len(note.sent) == 1 and "What changed" in note.sent[0]
    baked = read_prefs(p)
    assert baked["confluence"] == 2 and baked["sensitivity"] == "high"  # defaults baked, prefs kept
    assert _maybe_upgrade_notice(cfg, note) is False               # restart -> no resend
    assert len(note.sent) == 1


# ── multi-day jsonl memory (conversation + searches) ──
def test_memory_jsonl_chat_search_and_clear(tmp_path):
    from agent.memory import Memory
    clk = {"t": 1_000_000.0}
    m = Memory(str(tmp_path / "m.jsonl"), ttl_s=7 * 86400, clock=lambda: clk["t"])
    m.add_chat("user", "is this a top?")
    m.add_chat("assistant", "looks oversold")
    m.add_search("etf flows", summary="net outflows", after="2026-06-01")
    assert [x["text"] for x in m.recent_chat()] == ["is this a top?", "looks oversold"]
    s = m.recent_searches()
    assert len(s) == 1 and s[0]["query"] == "etf flows" and s[0]["after"] == "2026-06-01"
    lines = [l for l in open(m.path).read().splitlines() if l.strip()]   # real jsonl
    assert len(lines) == 3 and all(json.loads(l)["kind"] in ("chat", "search") for l in lines)
    st = m.stats()
    assert st["chats"] == 2 and st["searches"] == 1
    assert m.clear("all") == 3
    assert m.recent_chat() == [] and m.recent_searches() == []


def test_memory_clear_old_keeps_today(tmp_path):
    import calendar
    from agent.memory import Memory
    today = calendar.timegm((2026, 6, 13, 12, 0, 0, 0, 0, 0))
    clk = {"t": today - 86400}                       # yesterday
    m = Memory(str(tmp_path / "m.jsonl"), ttl_s=30 * 86400, clock=lambda: clk["t"])
    m.add_chat("user", "old msg")
    clk["t"] = today
    m.add_chat("user", "new msg")
    assert m.clear("old") == 1                        # only yesterday's dropped
    assert [x["text"] for x in m.recent_chat()] == ["new msg"]


def test_memory_auto_erase_and_compacts(tmp_path):
    from agent.memory import Memory
    clk = {"t": 1000.0}
    m = Memory(str(tmp_path / "m.jsonl"), ttl_s=100, clock=lambda: clk["t"])
    m.add_chat("user", "stale")
    clk["t"] = 1050.0
    m.add_chat("user", "fresh")
    clk["t"] = 1120.0                                 # 'stale' 120s old (>100), 'fresh' 70s (<100)
    assert [x["text"] for x in m.recent_chat()] == ["fresh"]
    lines = [l for l in open(m.path).read().splitlines() if l.strip()]
    assert len(lines) == 1 and json.loads(lines[0])["text"] == "fresh"   # file self-compacted


def test_listener_memory_clear_and_search_recording(tmp_path):
    mp = str(tmp_path / "m.jsonl")
    snap = {"price": 1, "technicals": {}, "volume": {}, "order_book": {}, "time": {}}
    lis = TelegramListener(WatchConfig(memory_path=mp), token="t", chat_id="42",
                           analyst=MockAnalyst(), notifier=FakeNotifier(), snapshot_fn=lambda: snap)
    assert "mock search" in lis.handle_text("/search etf flows after:2026-06-01")
    s = lis.memory.recent_searches()
    assert len(s) == 1 and "etf flows" in s[0]["query"] and s[0]["after"] == "2026-06-01"
    lis.handle_text("is this a top?")                            # plain Q -> records chat turns
    assert any(x["text"] == "is this a top?" for x in lis.memory.recent_chat())
    assert len(lis.memory.recent_searches()) == 1                # answer() reset last_search; no dup
    assert "Memory" in lis.handle_text("/memory")
    assert "cleared" in lis.handle_text("/clear").lower()
    assert lis.memory.recent_chat() == [] and lis.memory.recent_searches() == []


def test_listener_always_answers_even_on_handler_error():
    class _Boom(MockAnalyst):
        def answer(self, *a, **k):
            raise RuntimeError("llm exploded")
    note = FakeNotifier()
    lis = TelegramListener(WatchConfig(), token="t", chat_id="42", analyst=_Boom(),
                           notifier=note, snapshot_fn=lambda: {"price": 1, "technicals": {},
                           "volume": {}, "order_book": {}, "time": {}})
    lis._process_update({"message": {"chat": {"id": 42}, "text": "give me an update"}})
    assert len(note.sent) == 1 and "snag" in note.sent[0].lower()   # never silently dropped


def test_memory_clear_reports_failure_not_false_success(tmp_path, monkeypatch):
    from agent.memory import Memory
    m = Memory(str(tmp_path / "m.jsonl"))
    m.add_chat("user", "secret note")
    monkeypatch.setattr("os.replace", lambda *a, **k: (_ for _ in ()).throw(OSError("read-only fs")))
    assert m.clear("all") == -1                       # rewrite failed -> honest sentinel, not a lie
    monkeypatch.undo()
    assert [x["text"] for x in m.recent_chat()] == ["secret note"]   # data really did survive


def test_listener_clear_failure_message(tmp_path, monkeypatch):
    lis = TelegramListener(WatchConfig(memory_path=str(tmp_path / "m.jsonl")), token="t",
                           chat_id="42", analyst=MockAnalyst(), notifier=FakeNotifier(),
                           snapshot_fn=lambda: {"price": 1})
    lis.memory.add_chat("user", "hi")
    monkeypatch.setattr("os.replace", lambda *a, **k: (_ for _ in ()).throw(OSError("ro")))
    out = lis.handle_text("/clear")
    assert "couldn't clear" in out.lower() and "nothing was deleted" in out.lower()
