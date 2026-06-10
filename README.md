# SatoshStacker

A 24/7 Bitcoin accumulation system — two strategies sharing one codebase.

### 1. Accumulate-only ladder agent (`agent/`)
A re-anchoring limit-buy ladder + deploy-by-deadline guard that beats a static ladder
and naive DCA across real BTC drawdowns (3-round adversarial audit). Never sells; the
LLM (Qwen) is a veto/shrink-only advisor. Modes: `dry_run` → `testnet` → gated `live`.
See **CLAUDE.md** and **backtest/AUDIT.md**.

### 2. Active satoshi-stacker trader (`agent/trader.py`, `agent/trader_main.py`)
A tactical BTC↔USDC trader that manages **whatever capital is in the Binance account**,
in proportions of the pot, to **stack the most sats / lower the average entry**. Each 4h:
momentum (weekly-tuned RSI) + **live news** (Yahoo Finance RSS) + **Fear & Greed sentiment**
→ a smart-LLM (Qwen) decision → rebalance. Tracks average entry, runs DCA + buy-and-hold
shadow benchmarks, and sends **daily + weekly Telegram reports**.

### 3. Weekly self-tuning (`backtest/`)
`indicators.py` ranks technicals by Information Coefficient across timeframes;
`weekly_tune.py` has the best Qwen model pick the leading technicals + parameters and
writes `agent/technicals.json`, which the trader reads.

## Quickstart
```bash
pip install -r requirements.txt
cp .env.example .env        # fill in Binance + OpenRouter + Telegram keys
python -m agent.trader_main --mode testnet --once     # one tactical decision (testnet)
python -m agent.main        --mode dry_run            # the accumulate-only ladder (paper)
pytest -q                                             # 67 tests
```

Deployment (Docker / systemd / VPS): **DEPLOY.md**. All secrets live only in `.env` (gitignored).
