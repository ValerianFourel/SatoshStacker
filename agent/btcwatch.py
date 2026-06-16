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


def _coins_spec(base):
    """[(coin, symbol, market)] to watch. BTC (spot) always; XMR (futures — spot delisted) when
    WATCH_XMR is on (default). Extra coins via WATCH_COINS='name:SYMBOL/USDT:market,...'."""
    spec = [(base.coin, base.symbol, base.market)]            # base coin = btc spot
    if os.getenv("WATCH_XMR", "true").strip().lower() in ("1", "true", "yes", "on"):
        spec.append(("xmr", os.getenv("WATCH_XMR_SYMBOL", "XMR/USDT"), "futures"))
    for item in os.getenv("WATCH_COINS", "").split(","):       # optional extra coins
        parts = [x.strip() for x in item.split(":")]
        if len(parts) == 3 and parts[0] and parts[0] not in {c for c, _s, _m in spec}:
            spec.append((parts[0], parts[1], parts[2]))
    return spec


def _build():
    """Build the shared resources + a per-coin bundle (cfg, monitor, news, listener-analyst)."""
    from .alerts import AlertStore
    from .analyst import build_analyst, numeric_summary
    from .config import AnalysisConfig, WatchConfig, coin_config
    from .market_monitor import MarketMonitor
    from .notify import Notifier
    from .scratch import Scratch
    from .secrets import clean_secret
    from .telegram_listener import _Coin
    from .websearch import NewsCache, search_and_read

    base = WatchConfig.from_env()
    acfg = AnalysisConfig()
    scratch = Scratch(base.scratch_dir)
    api_key = clean_secret(os.getenv(acfg.api_key_env))
    search_fn = ((lambda q, after=None, before=None: search_and_read(
                    q, max_results=base.search_max_results, fetch_n=base.fetch_articles,
                    max_chars=base.article_max_chars, after=after, before=before))
                 if base.web_search_enabled else None)
    notifier = Notifier(chat_id=_chat_allowlist())   # broadcast to operator + shared users

    def mk_analyst(news_fn):  # SEPARATE analyst per thread so last_plot/last_search can't race
        return build_analyst(acfg, api_key, scratch, max_tokens=base.analyst_max_tokens,
                             news_fn=news_fn, search_fn=search_fn)

    coins, lcoins = [], {}
    for name, symbol, market in _coins_spec(base):
        ccfg = coin_config(base, coin=name, symbol=symbol, market=market)
        news_fn = (NewsCache(base.news_ttl_s, asset=symbol.split("/")[0]).get
                   if base.news_enabled else None)
        monitor = MarketMonitor(ccfg, notifier=notifier, analyst=mk_analyst(news_fn))
        coins.append({"name": name, "cfg": ccfg, "monitor": monitor})
        lcoins[name] = _Coin(name, ccfg, mk_analyst(news_fn), monitor.latest_snapshot,
                             AlertStore(ccfg.user_alerts_path), news_fn)
    return base, coins, lcoins, notifier, scratch, numeric_summary


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

    base, coins, lcoins, notifier, scratch, numeric_summary = _build()
    first = coins[0]                 # the base coin (btc) — CLI one-shots act on it
    cfg, monitor = first["cfg"], first["monitor"]

    if args.status:
        for c in coins:
            print(f"=== {c['name'].upper()} ({c['cfg'].symbol}) ===")
            print(numeric_summary(c["monitor"].latest_snapshot() or {}))
        return 0
    if args.tune:
        import datetime
        from .signal_tuner import leaderboard_text, run_tune
        tfs = [args.tf] if args.tf else ["5m", "15m", "1h", "4h", "1d"]
        for c in coins:                 # tune every watched coin (each on its own market)
            cc = c["cfg"]
            live_tf = args.tf or cc.trend_tf
            res = run_tune(cc.symbol, timeframes=tfs, live_tf=live_tf,
                           weeks=args.weeks or cc.tune_weeks, market=cc.market,
                           lookback_days=cc.tune_lookback_days, out_path=cc.tuned_signals_path,
                           stamp=datetime.datetime.now(datetime.timezone.utc).isoformat())
            txt = leaderboard_text(res)
            print(f"\n=== {cc.symbol} ({cc.market}) ===\n" + txt + f"\nsaved -> {cc.tuned_signals_path}")
            notifier.send(f"🔬 *Signal tune complete* ({cc.symbol}) — live detector now uses these.\n"
                          + txt[:3000])
        return 0
    if args.test_telegram:
        notifier.send("✅ SatoshiStacker watch: Telegram OK (read-only; "
                      + " + ".join(c["cfg"].symbol for c in coins) + ").")
        return 0
    if args.once:
        for c in coins:
            m = c["monitor"].run_once()
            print(f"=== {c['cfg'].symbol} ===")
            print(numeric_summary(m))
            if m.get("events"):
                print("EVENTS:", ", ".join(m["events"]))
        return 0

    # ── 24/7: one monitor thread PER COIN + a single multi-coin Telegram listener ──
    from .telegram_listener import TelegramListener
    stop = threading.Event()

    def _sig(*_):  # graceful shutdown on SIGTERM/SIGINT
        log.info("shutdown signal — stopping")
        stop.set()
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    _prune_old_data(base)           # data-retention sweep at startup (<= ~3 months)
    threads = []
    for c in coins:
        t = threading.Thread(target=c["monitor"].run, args=(stop,), daemon=True)
        t.start()
        threads.append(t)

    syms = " + ".join(c["cfg"].symbol for c in coins)
    notifier.send(f"🟢 *Watch online* — tracking {syms}. I'll ping on tops, bottoms & unusual "
                  "moves. Target a coin with e.g. `/xmr`, `/chart xmr 4h`. Type /help any time.")
    was_first = _first_launch_onboarding(notifier, cfg.onboarded_marker)  # one-time, first launch
    if not was_first:                       # existing install -> tell them what changed (once)
        _maybe_upgrade_notice(cfg, notifier)

    if cfg.poll_telegram:
        listener = TelegramListener(
            cfg, token=os.getenv("TELEGRAM_BOT_TOKEN", ""), chat_id=_chat_allowlist(),
            analyst=lcoins[first["name"]].analyst, notifier=notifier,
            snapshot_fn=monitor.latest_snapshot, scratch=scratch,
            news_fn=lcoins[first["name"]].news_fn, coins=lcoins)
        listener.poll(stop)          # blocks in main thread until stop
    else:
        while not stop.is_set():
            stop.wait(3600)

    stop.set()
    for t in threads:
        t.join(timeout=5)
    log.info("stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
