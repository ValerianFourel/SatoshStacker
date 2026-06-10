"""SatoshiStacker — deterministic BTC/USDC accumulation agent.

The deterministic spine (ladder + deploy-by-deadline) lives in `ladder.py`
and is intentionally free of exchange/LLM imports so the backtest can
exercise the *exact* production logic.
"""
