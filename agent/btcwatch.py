"""SatoshiStacker BTC watch — a standalone, read-only market monitor + LLM analyst
+ Telegram Q&A. SEPARATE process from the trader: public Binance data only, no API
keys to the exchange, no orders, ever.

    python -m agent.btcwatch                 # run monitor + Telegram Q&A (24/7)
    python -m agent.btcwatch --once          # one scan, print snapshot, exit
    python -m agent.btcwatch --status        # print last snapshot, exit
    python -m agent.btcwatch --test-telegram # send a test message, exit

Anomaly events (peak/bottom/volume & volatility spikes) auto-trigger an LLM read
pushed to Telegram. The operator can also ask questions any time.
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import threading

log = logging.getLogger("satoshistacker.btcwatch")


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:  # noqa: BLE001 - env may already be set in the container
        pass


def _chat_allowlist() -> str:
    """Comma-joined chat ids allowed to use the bot: operator + WATCH_ALLOWED_CHATS.
    Alerts broadcast to all of them; only these chats may query."""
    ids = [os.getenv("TELEGRAM_CHAT_ID", "")] + os.getenv("WATCH_ALLOWED_CHATS", "").split(",")
    return ",".join(c.strip() for c in ids if c.strip())


def _prune_old_data(cfg) -> None:
    """Erase pulled-data / scratch / state files older than data_retention_days, so the
    bot never hoards more than ~3 months. Live candles are in-memory; the snapshot/tuned/
    conversation files are overwritten, so only genuinely stale files get removed."""
    import time
    cutoff = time.time() - cfg.data_retention_days * 86_400
    dirs = {cfg.scratch_dir, os.path.dirname(cfg.snapshot_path) or "."}
    removed = 0
    for d in dirs:
        try:
            for name in os.listdir(d):
                p = os.path.join(d, name)
                if os.path.isfile(p) and os.path.getmtime(p) < cutoff:
                    os.remove(p)
                    removed += 1
        except Exception:  # noqa: BLE001
            continue
    if removed:
        log.info("data retention: pruned %d file(s) older than %dd", removed,
                 cfg.data_retention_days)


def _first_launch_onboarding(notifier, marker_path: str) -> bool:
    """Send the onboarding tip ONCE — the first time the service ever launches. The
    marker file makes it idempotent across restarts. Returns True if it sent."""
    from .telegram_listener import ONBOARDING
    if os.path.exists(marker_path):
        return False
    notifier.send(ONBOARDING)
    try:
        os.makedirs(os.path.dirname(marker_path) or ".", exist_ok=True)
        with open(marker_path, "w") as f:
            f.write("onboarded\n")
    except Exception as e:  # noqa: BLE001 - a marker write failure must not crash startup
        log.warning("could not write onboarding marker: %s", e)
    return True


_UPGRADE_NOTICE = (
    "🆕 *What changed — quieter, smarter alerts.*\n"
    "You told me I was pinging too much, so:\n"
    "• 🧩 *Confluence* — I now ping only when *several* signals are out-of-norm *at once* "
    "(default ≥2), not on any single one. A lone book-imbalance blip won't bug you.\n"
    "• ⏱️ *Cadence* — at most about *one ping every 30 min*, and a cluster has to clear "
    "before it can fire again.\n"
    "• 📰 *News on my own* — I read the news every ~8h, keep a copy (`/digest`), and only "
    "ping if I judge it market-moving.\n"
    "\n"
    "🛰️ Tune all of it live with `/origins` — tap which signals may ping, set how many must "
    "agree, the cadence, the preset, or mute. Want the old behaviour back? Set confluence to "
    "1 in `/origins`. Full guide: `/help`."
)


def _maybe_upgrade_notice(cfg, notifier) -> bool:
    """One-time 'what changed' message for an EXISTING install when the confluence/cadence
    behaviour first ships (its default flips alerting materially, so the operator is told).
    Idempotent via a marker; also bakes the new defaults into the prefs file so it's explicit."""
    if os.path.exists(cfg.upgrade_marker):
        return False
    notifier.send(_UPGRADE_NOTICE)
    try:
        from .sensitivity import (DEFAULT_CADENCE_S, DEFAULT_CONFLUENCE, read_prefs,
                                  write_prefs)
        p = read_prefs(cfg.prefs_path, default_level=cfg.sensitivity,
                       default_confluence=cfg.confluence_min, default_cadence=cfg.alert_cadence_s)
        write_prefs(cfg.prefs_path, confluence=p.get("confluence", DEFAULT_CONFLUENCE),
                    cadence=p.get("cadence", DEFAULT_CADENCE_S))      # make the active config explicit
        os.makedirs(os.path.dirname(cfg.upgrade_marker) or ".", exist_ok=True)
        with open(cfg.upgrade_marker, "w") as f:
            f.write("notified\n")
    except Exception as e:  # noqa: BLE001 - a marker/prefs write failure must not crash startup
        log.warning("could not write upgrade marker: %s", e)
    return True


def _build():
    from .analyst import build_analyst, numeric_summary
    from .config import AnalysisConfig, WatchConfig
    from .market_monitor import MarketMonitor
    from .notify import Notifier
    from .scratch import Scratch
    from .secrets import clean_secret

    from .websearch import NewsCache, search_and_read

    cfg = WatchConfig.from_env()
    acfg = AnalysisConfig()
    scratch = Scratch(cfg.scratch_dir)
    api_key = clean_secret(os.getenv(acfg.api_key_env))
    news_fn = NewsCache(cfg.news_ttl_s).get if cfg.news_enabled else None
    search_fn = ((lambda q, after=None, before=None: search_and_read(
                    q, max_results=cfg.search_max_results, fetch_n=cfg.fetch_articles,
                    max_chars=cfg.article_max_chars, after=after, before=before))
                 if cfg.web_search_enabled else None)
    # SEPARATE analyst per thread: the monitor thread and the listener thread each get their
    # own instance so per-call scratch state (last_plot, last_search) can't race across threads.
    def _mk_analyst():
        return build_analyst(acfg, api_key, scratch, max_tokens=cfg.analyst_max_tokens,
                             news_fn=news_fn, search_fn=search_fn)
    notifier = Notifier(chat_id=_chat_allowlist())   # broadcast to operator + shared users
    monitor = MarketMonitor(cfg, notifier=notifier, analyst=_mk_analyst())
    return cfg, monitor, _mk_analyst(), notifier, scratch, news_fn, numeric_summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        "satoshistacker-btcwatch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Once running, message the Telegram bot `/help` for the full usage guide "
               "(commands, plain-language questions, and date-filtered web search).")
    p.add_argument("--once", action="store_true", help="one scan, print, exit")
    p.add_argument("--status", action="store_true", help="print last snapshot, exit")
    p.add_argument("--test-telegram", action="store_true", help="send a test msg, exit")
    p.add_argument("--tune", action="store_true",
                   help="backtest momentum oscillators for top/bottom calling, save winners, exit")
    p.add_argument("--weeks", type=float, default=None, help="--tune lookback (weeks)")
    p.add_argument("--tf", default=None, help="--tune candle timeframe (e.g. 1h, 4h)")
    args = p.parse_args(argv)

    _load_env()
    from .secrets import install_log_redaction
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    install_log_redaction()

    cfg, monitor, analyst, notifier, scratch, news_fn, numeric_summary = _build()

    if args.status:
        print(numeric_summary(monitor.latest_snapshot() or {}))
        return 0
    if args.tune:
        import datetime
        from .signal_tuner import leaderboard_text, run_tune
        # default sweeps 5m/1h/4h/1d; --tf pins one. Live detector uses the trend-TF winner.
        tfs = [args.tf] if args.tf else ["5m", "1h", "4h", "1d"]
        live_tf = args.tf or cfg.trend_tf
        res = run_tune(cfg.symbol, timeframes=tfs, live_tf=live_tf,
                       weeks=args.weeks or cfg.tune_weeks,
                       lookback_days=cfg.tune_lookback_days,   # <= ~3 months
                       out_path=cfg.tuned_signals_path,
                       stamp=datetime.datetime.now(datetime.timezone.utc).isoformat())
        txt = leaderboard_text(res)
        print(txt + f"\n\nsaved -> {cfg.tuned_signals_path} (live detector uses the {live_tf} winners)")
        notifier.send("🔬 *Signal tune complete* — live detector now uses these.\n" + txt[:3500])
        return 0
    if args.test_telegram:
        notifier.send("✅ SatoshiStacker BTC watch: Telegram OK (read-only monitor).")
        return 0
    if args.once:
        m = monitor.run_once()
        print(numeric_summary(m))
        if m.get("events"):
            print("EVENTS:", ", ".join(m["events"]))
        return 0

    # ── 24/7: monitor thread + Telegram listener ──
    from .telegram_listener import TelegramListener
    stop = threading.Event()

    def _sig(*_):  # graceful shutdown on SIGTERM/SIGINT
        log.info("shutdown signal — stopping")
        stop.set()
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    _prune_old_data(cfg)            # data-retention sweep at startup (<= ~3 months)
    mon_thread = threading.Thread(target=monitor.run, args=(stop,), daemon=True)
    mon_thread.start()

    notifier.send("🟢 *BTC watch online* — tracking live; I'll ping you on tops, bottoms "
                  "& unusual moves. Type /help any time.")
    was_first = _first_launch_onboarding(notifier, cfg.onboarded_marker)  # one-time, first launch
    if not was_first:                       # existing install -> tell them what changed (once)
        _maybe_upgrade_notice(cfg, notifier)

    if cfg.poll_telegram:
        listener = TelegramListener(
            cfg, token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            chat_id=_chat_allowlist(), analyst=analyst,
            notifier=notifier, snapshot_fn=monitor.latest_snapshot, scratch=scratch,
            news_fn=news_fn)
        listener.poll(stop)          # blocks in main thread until stop
    else:
        while not stop.is_set():
            stop.wait(3600)

    stop.set()
    mon_thread.join(timeout=5)
    log.info("stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
