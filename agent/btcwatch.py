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
    analyst = build_analyst(acfg, api_key, scratch, max_tokens=cfg.analyst_max_tokens,
                            news_fn=news_fn, search_fn=search_fn)
    notifier = Notifier()
    monitor = MarketMonitor(cfg, notifier=notifier, analyst=analyst)
    return cfg, monitor, analyst, notifier, scratch, news_fn, numeric_summary


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
        res = run_tune(cfg.symbol, timeframe=args.tf or cfg.tune_timeframe,
                       weeks=args.weeks or cfg.tune_weeks,
                       out_path=cfg.tuned_signals_path,
                       stamp=datetime.datetime.now(datetime.timezone.utc).isoformat())
        txt = leaderboard_text(res)
        print(txt + f"\n\nsaved -> {cfg.tuned_signals_path} (live detector will use these)")
        notifier.send("🔬 *Signal tune complete* — live detector now uses these.\n" + txt)
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

    mon_thread = threading.Thread(target=monitor.run, args=(stop,), daemon=True)
    mon_thread.start()

    notifier.send("🟢 SatoshiStacker BTC watch online (read-only). Send /help.")
    _first_launch_onboarding(notifier, cfg.onboarded_marker)  # one-time, first launch only

    if cfg.poll_telegram:
        listener = TelegramListener(
            cfg, token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            chat_id=os.getenv("TELEGRAM_CHAT_ID", ""), analyst=analyst,
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
