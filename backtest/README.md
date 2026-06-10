# Milestone 1 — Accumulation backtest

**Question:** does the agent's deterministic spine accumulate more BTC, from a fixed
$2,000 USDC budget over a window, than the honest baselines it must beat — a **static
resting ladder**, **naive equal-interval DCA**, and **lump-sum-now**? Per the spec, if
it can't beat the static ladder *and* DCA, the correct deliverable is the static ladder.

## Data
`fetch_data.py` pulls **real** Binance public klines (no key needed):
`BTCUSDT_1d.parquet` (3,220 daily candles, 2017-08-17 → 2026-06-10, $2,817–$126,200)
and `BTCUSDT_4h.parquet`. BTCUSDT is the price proxy (USDT≈USDC≈$1 for *price*; the
USDC de-peg risk is a *live* concern handled in `agent/risk.py`, not here).

## Honesty guarantees (why these numbers can be trusted)
- **No look-ahead.** Every fill/decision uses only the current bar's open/low/close.
- **Realistic maker fills.** A resting limit buy at level `L` fills on bar `b` iff
  `low_b ≤ L`, at price `min(L, open_b)` (the limit, or a better gap-down open).
- **Fees.** maker = taker = 0.001 by default (Binance spot), charged in received BTC.
  Ladders pay maker; DCA/lump pay taker.
- **Forward floor rule.** Each window's floor = `anchor_open × frac` for
  `frac ∈ {0.55, 0.65}` — never hindsight knowledge of the actual bottom.
- **Rising windows included** (`bull_2020H2`, `bull_2023H1`) so the agent is also judged
  where laddering is *expected* to lose to lump-sum.

## Strategies
| name | description | fees |
|---|---|---|
| `lump_sum` | deploy entire budget at window open | taker |
| `dca` / `dca_weekly` | equal market buys, evenly spaced / every 7 bars | taker |
| `static_ladder` | weighted resting maker ladder, **no deadline** (the "no software" baseline) | maker |
| `adaptive` | spine = weighted ladder + per-day cap + **deploy-by-deadline** DCA of leftover | maker + taker |
| `ladder_reanchor` | **trailing ladder**: reserves powder, re-anchors a lower ladder when price breaks its floor | maker + taker |
| `ladder_hybrid` | 50% static ladder + 50% weekly DCA sleeve | maker + taker |

## Run
```bash
python3 backtest/fetch_data.py      # one-time: pull real history
python3 backtest/engine.py          # baselines + spine scoreboard + verdict
python3 backtest/variants.py        # design search vs the stronger weekly-DCA baseline
```

## Headline result (after adversarial audit — see `AUDIT.md`)
- The single-anchor spine + deadline **never loses to the *no-deadline* static resting
  ladder** (ties when the floor is hit, wins when it isn't — the deadline guard stops USDC
  being stranded) and **beats lump-sum in drawdowns** (~1.4× BTC), but **loses to DCA in
  deep bears** because a floor above the eventual bottom leaves it fully deployed too high.
- `ladder_reanchor` (trailing) fixes the DCA gap: it **beats every honest fixed-cadence
  DCA across drawdown windows (12/16 cells, median BTC ratio 1.06)** — verified causal
  (bit-for-bit identical to a streaming sim), friction-robust, and replicated
  out-of-sample on disjoint rolling windows.
- **Honesty correction:** reanchor beats the static ladder only on **total stack /
  protracted bears**, *not* per-cell (static wins 9/16 sharp-V cells; reanchor wins the
  total stack 1.868 vs 1.753 BTC). The edge is **regime-conditional** (drawdowns only;
  lags lump-sum in bull runs, as expected). Its one structural weakness is the protracted
  multi-quarter grind — the regime closest to a "slow grind to $35k" thesis — where DCA
  is a genuinely strong competitor.

See `results/scoreboard.csv` and `results/variants.csv` for raw per-cell numbers, and
**`AUDIT.md`** for the full adversarial robustness review.
