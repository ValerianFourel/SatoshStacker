"""CLI entrypoint: mode handling + fail-closed startup safety checks.

    python -m agent.main --mode dry_run --cycles 5     # paper, 5 cycles
    python -m agent.main --status                       # show state, no trading
    python -m agent.main --clear-halt                   # manual halt clear
    python -m agent.main --mode live                    # refuses unless ALL gates pass

Live trading requires ALL of:
  * --mode live AND env LIVE_TRADING_CONFIRMED=yes
  * the backtest gate marker file exists (the accumulation backtest passed)
  * the API key cannot withdraw (verified live; refuses otherwise)
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import sys
from pathlib import Path

from .analysis import MockAdvisor, QwenAdvisor
from .config import AgentConfig
from .exchange import build_exchange
from .loop import Agent
from .notify import Notifier
from .orders import OrderManager
from .risk import RiskGate
from .secrets import clean_secret, install_log_redaction, redact
from .state import Store


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:  # noqa: BLE001
        pass


def _safety_checks(cfg: AgentConfig, ex) -> list[str]:
    """Return a list of FATAL reasons to refuse startup (empty = ok)."""
    errors: list[str] = []
    if cfg.mode == "live":
        if os.getenv("LIVE_TRADING_CONFIRMED", "no") != "yes":
            errors.append("live requires env LIVE_TRADING_CONFIRMED=yes")
        if not Path(cfg.backtest_gate_marker).exists():
            errors.append(f"backtest gate marker missing: {cfg.backtest_gate_marker} "
                          "(the accumulation backtest must pass before live)")
        try:
            if ex.can_withdraw():
                errors.append("API key CAN WITHDRAW — refusing. Use a spot-only, "
                              "withdrawals-disabled, IP-restricted key.")
        except Exception as e:  # noqa: BLE001 - fail safe
            errors.append(f"could not verify withdrawal permission (fail-safe refuse): {e}")
    return errors


def _build_agent(cfg: AgentConfig) -> Agent:
    store = Store(cfg.db_path)
    paper_book = str(Path(cfg.db_path).with_suffix(".paperbook.json"))
    ex = build_exchange(cfg, store_path=paper_book)
    risk = RiskGate(cfg.risk, store)
    om = OrderManager(cfg, ex, store, risk)
    notifier = Notifier()
    # advisor: real Qwen only if enabled AND a key is present; else deterministic mock
    key = clean_secret(os.getenv(cfg.analysis.api_key_env, ""))
    if cfg.analysis.enabled and key:
        advisor = QwenAdvisor(cfg.analysis, key)
    else:
        advisor = MockAdvisor()
    return Agent(cfg=cfg, store=store, ex=ex, risk=risk, advisor=advisor,
                 om=om, notifier=notifier)


def _preflight(agent: Agent) -> int:
    """Read-only checks before a real run: connectivity + startup safety. No orders."""
    print(f"Preflight [mode={agent.cfg.mode}] symbol={agent.cfg.symbol}")
    errors = _safety_checks(agent.cfg, agent.ex)  # live gating (no-op for dry_run/testnet)

    try:
        px = agent.ex.get_price()
        print(f"  ✓ price feed: {agent.cfg.symbol} = ${px:,.2f}")
    except Exception as e:  # noqa: BLE001
        errors.append(f"price feed failed: {redact(e)}")
    try:
        bal = agent.ex.fetch_balance()
        shown = {k: bal[k] for k in list(bal)[:6]}
        print(f"  ✓ balances: {shown}")
    except Exception as e:  # noqa: BLE001
        errors.append(f"balance fetch failed: {redact(e)}")
    if agent.cfg.mode in ("testnet", "live"):
        try:
            w = agent.ex.can_withdraw()
            print(f"  {'✗' if w else '✓'} withdrawals disabled: {not w}")
            if w and agent.cfg.mode == "live":
                errors.append("API key CAN WITHDRAW — must be a spot-only, "
                              "withdrawals-disabled, IP-restricted key")
            elif w and agent.cfg.mode == "testnet":
                print("    (testnet: the withdrawal-restriction endpoint is often "
                      "unavailable and testnet keys hold no real funds — informational)")
        except Exception as e:  # noqa: BLE001
            if agent.cfg.mode == "live":
                errors.append(f"withdrawal-permission check failed (fail-safe): {redact(e)}")
            else:
                print(f"    (testnet withdrawal check unavailable: {redact(e)})")

    if errors:
        print("PREFLIGHT FAILED:", file=sys.stderr)
        for e in errors:
            print(f"  ✗ {e}", file=sys.stderr)
        return 2
    print("PREFLIGHT OK — connectivity and safety checks passed; safe to start.")
    return 0


def _print_status(agent: Agent) -> None:
    s, cfg = agent.store, agent.cfg
    dep, btc = s.total_deployed(), s.total_btc()
    print(f"mode={cfg.mode} strategy={cfg.ladder.strategy} symbol={cfg.symbol}")
    print(f"phase={s.get_meta('mode_phase','laddering')} "
          f"generation={s.get_meta('generation','-')} "
          f"halted={s.is_halted()} ({s.get_meta('halt_reason')})")
    print(f"budget=${cfg.ladder.budget_usdc:,.0f} deployed=${dep:,.2f} "
          f"({100*dep/max(cfg.ladder.budget_usdc,1):.1f}%) remaining=${cfg.ladder.budget_usdc-dep:,.2f}")
    avg = f"${dep/btc:,.0f}" if btc > 0 else "n/a"
    print(f"BTC stacked={btc:.8f} avg_cost={avg}")
    resting = s.rungs_by_status("resting")
    print(f"resting rungs: {len(resting)}")
    for r in resting:
        print(f"  {r['client_order_id']}: ${r['usdc']:,.2f} @ ${r['price']:,.0f}")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    install_log_redaction()  # scrub any credential that reaches the logs
    _load_env()
    p = argparse.ArgumentParser("satoshistacker")
    p.add_argument("--mode", choices=["dry_run", "testnet", "live"])
    p.add_argument("--db", default=None)
    p.add_argument("--once", action="store_true", help="run a single cycle and exit")
    p.add_argument("--cycles", type=int, default=None, help="run N cycles and exit")
    p.add_argument("--status", action="store_true", help="print state and exit")
    p.add_argument("--preflight", action="store_true",
                   help="check connectivity + startup safety, no trading")
    p.add_argument("--test-telegram", action="store_true",
                   help="send a test Telegram notification and exit")
    p.add_argument("--clear-halt", action="store_true", help="clear a halt (manual)")
    p.add_argument("--reset", action="store_true", help="WIPE state db and exit")
    args = p.parse_args(argv)

    cfg = AgentConfig.from_env()
    overrides: dict = {}
    if args.mode:
        overrides["mode"] = args.mode
    if args.db:
        overrides["db_path"] = args.db
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)

    if args.reset:
        for ext in ("", ".paperbook.json"):
            f = Path(cfg.db_path + ext if ext else cfg.db_path)
            if f.exists():
                f.unlink()
        print(f"reset: wiped {cfg.db_path} (+ paper book)")
        return 0

    agent = _build_agent(cfg)

    if args.clear_halt:
        agent.store.clear_halt()
        print("halt cleared")
        return 0
    if args.status:
        _print_status(agent)
        return 0
    if args.test_telegram:
        agent.notifier.send(f"✅ SatoshiStacker test message [{cfg.mode}] — alerts are wired.")
        print("sent test notification (check Telegram, or stdout if no token set)")
        return 0
    if args.preflight:
        return _preflight(agent)

    errors = _safety_checks(cfg, agent.ex)
    if errors:
        print("REFUSING TO START — safety checks failed:", file=sys.stderr)
        for e in errors:
            print(f"  ✗ {e}", file=sys.stderr)
        return 2

    print(f"SatoshiStacker starting [{cfg.mode}] — reconciling against exchange first…")
    if args.once:
        print(redact(agent.run_cycle()))  # stdout is outside the logging filter
    else:
        agent.run(max_cycles=args.cycles)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
