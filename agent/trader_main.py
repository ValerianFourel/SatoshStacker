"""CLI for the active satoshi-stacker trader (testnet first).

It manages whatever capital is in the Binance account, in proportions of the pot.

    python -m agent.trader_main --mode testnet --once     # one 4h decision
    python -m agent.trader_main --mode testnet            # 24/7 loop (reports daily/weekly)
    python -m agent.trader_main --status                  # pot + benchmarks
    python -m agent.trader_main --report daily            # send a report now
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import time

from .config import AgentConfig
from .exchange import build_exchange
from .notify import Notifier
from .secrets import clean_secret, install_log_redaction
from .trader import SatoshiTrader, TraderStore, _lot

log = logging.getLogger("satoshistacker.trader")


def _build(args):
    cfg = AgentConfig.from_env()
    cfg = dataclasses.replace(cfg, mode=(args.mode or cfg.mode))
    ex = build_exchange(cfg, store_path=(args.db + ".paperbook.json"))
    key = clean_secret(os.getenv("LLM_API_KEY"))
    model = os.getenv("TRADER_MODEL", "qwen/qwen3.5-plus-20260420")  # smart; 1 call/4h
    client = None
    if key:
        from openai import OpenAI
        client = OpenAI(base_url=os.getenv("LLM_BASE_URL"), api_key=key)
    store = TraderStore(args.db)
    news_on = os.getenv("NEWS_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")
    trader = SatoshiTrader(exchange=ex, store=store, notifier=Notifier(), llm_client=client,
                           model=model, symbol=cfg.symbol, news_enabled=news_on,
                           stack_usdc=float(os.getenv("STACK_USDC", "2000")),
                           dca_days=int(os.getenv("DCA_BENCHMARK_DAYS", "30")), cycle_hours=4)
    return cfg, trader


def _status(trader):
    px = trader.ex.get_price()
    usdc, btc, pot, avg = trader._snapshot(px)
    hodl, dca = _lot(trader.store.get("hodl")), _lot(trader.store.get("dca"))
    ae = f"${avg:,.0f}" if btc > 0 else "n/a"
    print(f"  BOT  {btc:.6f} BTC  avg {ae:>9}  +${usdc:,.0f} cash  = ${pot:,.0f}  "
          f"({pot/px:.6f} sats-equiv)")
    print(f"  DCA  {dca.btc:.6f} BTC  avg ${dca.avg:>9,.0f}  (${dca.value(px):,.0f})")
    print(f"  HODL {hodl.btc:.6f} BTC  avg ${hodl.avg:>9,.0f}  (${hodl.value(px):,.0f})")
    print(f"  BTC ${px:,.0f}  | goal: most sats / lowest avg entry")


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    install_log_redaction()
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:  # noqa: BLE001
        pass

    p = argparse.ArgumentParser("satoshistacker-trader")
    p.add_argument("--mode", choices=["dry_run", "testnet", "live"])
    p.add_argument("--db", default="trader.db")
    p.add_argument("--once", action="store_true")
    p.add_argument("--cycles", type=int)
    p.add_argument("--status", action="store_true")
    p.add_argument("--report", choices=["daily", "weekly"])
    args = p.parse_args(argv)

    cfg, trader = _build(args)

    if args.status:
        _status(trader); return 0
    if args.report:
        (trader.daily_report if args.report == "daily" else trader.weekly_report)()
        print(f"sent {args.report} report"); return 0
    if cfg.mode == "live" and os.getenv("LIVE_TRADING_CONFIRMED") != "yes":
        print("live requires LIVE_TRADING_CONFIRMED=yes; refusing"); return 2

    print(f"satoshi-stacker trader [{cfg.mode}] {cfg.symbol} — smart-LLM sequential, "
          f"manages the live pot in proportions")
    if args.once:
        print(trader.run_cycle()); return 0
    i = 0
    while True:
        try:
            log.info("cycle %d: %s", i, trader.run_cycle())
        except Exception as e:  # noqa: BLE001
            from .secrets import redact
            log.error("cycle error: %s", redact(e))
        i += 1
        if args.cycles and i >= args.cycles:
            break
        time.sleep(max(60, 4 * 3600))  # 4h cadence
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
