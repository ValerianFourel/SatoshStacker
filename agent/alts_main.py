"""CLI for the multi-asset alt-stacker (SOL/ETH/HYPE), testnet first.

    python -m agent.alts_main --mode dry_run --once      # one paper cycle
    python -m agent.alts_main --mode dry_run --cycles 5  # 5 paper cycles
    python -m agent.alts_main --mode testnet             # 24/7 loop (4h decisions)
    python -m agent.alts_main --status                   # pot + positions + benchmarks
    python -m agent.alts_main --preflight                # connectivity + safety, no trading
    python -m agent.alts_main --clear-halt               # clear a kill-switch halt
    python -m agent.alts_main --reset                    # wipe state db (+ paper book)

Live trading requires ALL of:
  * --mode live AND env LIVE_TRADING_CONFIRMED=yes
  * the alt gate marker file exists (you reviewed M-dry-run + a backtest)
  * the API key cannot withdraw (verified live; refuses otherwise)
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import sys
import time
from pathlib import Path

from .alts import AltConfig, AltStacker, AltStore, _pos
from .multi_exchange import build_multi_exchange
from .notify import Notifier
from .secrets import clean_secret, install_log_redaction, redact

log = logging.getLogger("satoshistacker.alts")


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:  # noqa: BLE001
        pass


def _build(args) -> tuple[AltConfig, AltStacker]:
    cfg = AltConfig.from_env()
    if args.mode:
        cfg = dataclasses.replace(cfg, mode=args.mode)
    if args.db:
        cfg = dataclasses.replace(cfg, db_path=args.db)
    # Binance testnet typically only has USDT pairs — default the quote accordingly
    if cfg.mode == "testnet" and cfg.venue == "binance" and not os.getenv("ALT_QUOTE"):
        cfg = dataclasses.replace(cfg, quote="USDT")
    paper_book = str(Path(cfg.db_path).with_suffix(".paperbook.json"))
    ex = build_multi_exchange(cfg, store_path=paper_book)
    store = AltStore(cfg.db_path)
    key = clean_secret(os.getenv("LLM_API_KEY"))
    client = None
    if key:
        from openai import OpenAI
        client = OpenAI(base_url=os.getenv("LLM_BASE_URL") or None, api_key=key)
    from .multi_exchange import public_feeds
    _, ohlcv_fn = public_feeds(cfg.venue)   # venue-aware public candles (Binance | Kraken)
    stacker = AltStacker(cfg=cfg, exchange=ex, store=store,
                         notifier=Notifier(), llm_client=client, ohlcv_source=ohlcv_fn)
    return cfg, stacker


def _safety_checks(cfg: AltConfig, ex) -> list[str]:
    errors: list[str] = []
    if cfg.mode == "live":
        if os.getenv("LIVE_TRADING_CONFIRMED", "no") != "yes":
            errors.append("live requires env LIVE_TRADING_CONFIRMED=yes")
        if not Path(cfg.gate_marker).exists():
            errors.append(f"alt gate marker missing: {cfg.gate_marker} "
                          "(review dry-run + a backtest before live)")
        try:
            if ex.can_withdraw():
                errors.append("API key CAN WITHDRAW — refusing. Use a spot-only, "
                              "withdrawals-disabled, IP-restricted key.")
        except Exception as e:  # noqa: BLE001 - fail safe
            errors.append(f"could not verify withdrawal permission (fail-safe refuse): "
                          f"{redact(e)}")
    return errors


def _preflight(cfg: AltConfig, stacker: AltStacker) -> int:
    print(f"Preflight [mode={cfg.mode}] venue={cfg.venue} quote={cfg.quote} "
          f"assets={', '.join(cfg.bases)}")
    errors = _safety_checks(cfg, stacker.ex)
    for base in cfg.bases:
        try:
            px = stacker.ex.price(base)
            print(f"  ✓ price {base}/{cfg.quote} = {px:,.4f}")
        except Exception as e:  # noqa: BLE001
            errors.append(f"price feed failed for {base}: {redact(e)} "
                          f"(is {base}/{cfg.quote} listed on this venue?)")
    try:
        print(f"  ✓ cash (USDC+USDT) = ${stacker.ex.cash():,.2f}")
    except Exception as e:  # noqa: BLE001
        errors.append(f"balance fetch failed: {redact(e)}")
    if cfg.mode in ("testnet", "live"):
        try:
            w = stacker.ex.can_withdraw()
            print(f"  {'✗' if w else '✓'} withdrawals disabled: {not w}")
            if w and cfg.mode == "live":
                errors.append("API key CAN WITHDRAW — must be withdrawals-disabled")
        except Exception as e:  # noqa: BLE001
            if cfg.mode == "live":
                errors.append(f"withdrawal check failed (fail-safe): {redact(e)}")
            else:
                print(f"    (testnet withdrawal check unavailable: {redact(e)})")
    if errors:
        print("PREFLIGHT FAILED:", file=sys.stderr)
        for e in errors:
            print(f"  ✗ {e}", file=sys.stderr)
        return 2
    print("PREFLIGHT OK — connectivity and safety checks passed.")
    return 0


def _status(cfg: AltConfig, stacker: AltStacker) -> None:
    prices = {}
    for b in cfg.bases:
        try:
            prices[b] = stacker.ex.price(b)
        except Exception:  # noqa: BLE001
            continue
    cash = stacker.ex.cash()
    pot = cash + sum(_pos(stacker.store.get(f"pos:{b}")).value(prices[b]) for b in prices)
    print(f"mode={cfg.mode} venue={cfg.venue} quote={cfg.quote} "
          f"halted={stacker.store.is_halted()} ({stacker.store.get('halt_reason')})")
    print(f"pot=${pot:,.2f}  cash=${cash:,.2f}")
    for b in cfg.bases:
        pos = _pos(stacker.store.get(f"pos:{b}"))
        px = prices.get(b)
        spec = cfg.spec(b)
        ae = f"${pos.avg:,.2f}" if pos.units > 0 else "n/a"
        pxs = f"${px:,.4f}" if px else "n/a"
        extra = (f" realizedUSDC=${pos.realized_quote:+,.2f}"
                 if spec.objective == "accumulate_quote" else "")
        print(f"  {b} [{spec.objective}] {pos.units:.6f} u  avg {ae}  px {pxs}  "
              f"max_frac {spec.max_fraction}{extra}")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    install_log_redaction()
    _load_env()

    p = argparse.ArgumentParser("satoshistacker-alts")
    p.add_argument("--mode", choices=["dry_run", "testnet", "live"])
    p.add_argument("--db", default=None)
    p.add_argument("--once", action="store_true")
    p.add_argument("--cycles", type=int, default=None)
    p.add_argument("--status", action="store_true")
    p.add_argument("--report", choices=["daily"])
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--test-telegram", action="store_true")
    p.add_argument("--clear-halt", action="store_true")
    p.add_argument("--reset", action="store_true")
    args = p.parse_args(argv)

    cfg = AltConfig.from_env()
    if args.mode:
        cfg = dataclasses.replace(cfg, mode=args.mode)
    if args.db:
        cfg = dataclasses.replace(cfg, db_path=args.db)

    if args.reset:
        for f in (Path(cfg.db_path), Path(cfg.db_path).with_suffix(".paperbook.json")):
            if f.exists():
                f.unlink()
        print(f"reset: wiped {cfg.db_path} (+ paper book)")
        return 0

    cfg, stacker = _build(args)

    if args.clear_halt:
        stacker.store.clear_halt(); print("halt cleared"); return 0
    if args.status:
        _status(cfg, stacker); return 0
    if args.test_telegram:
        stacker.notify.send(f"✅ alt-stacker test [{cfg.mode}] — alerts wired.")
        print("sent test notification"); return 0
    if args.report == "daily":
        stacker.daily_report(); print("sent daily report"); return 0
    if args.preflight:
        return _preflight(cfg, stacker)

    errors = _safety_checks(cfg, stacker.ex)
    if errors:
        print("REFUSING TO START — safety checks failed:", file=sys.stderr)
        for e in errors:
            print(f"  ✗ {e}", file=sys.stderr)
        return 2

    print(f"alt-stacker starting [{cfg.mode}] {', '.join(cfg.bases)}/{cfg.quote}")
    if args.once:
        print(redact(stacker.run_cycle())); return 0
    i = 0
    while True:
        try:
            log.info("cycle %d: %s", i, stacker.run_cycle())
        except Exception as e:  # noqa: BLE001 - never crash the loop
            log.error("cycle error: %s", redact(e))
        i += 1
        if args.cycles and i >= args.cycles:
            break
        time.sleep(max(60, cfg.cycle_hours * 3600))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
