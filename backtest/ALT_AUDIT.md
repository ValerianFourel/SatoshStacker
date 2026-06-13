# Alt-stacker backtest, candle/technicals study & honest verdict

Sister of the BTC `backtest/AUDIT.md`. Records what the make-USD alt-stacker can and can't do.

## Goal & asset journey
- **Goal:** MAKE USD by trading the alts (end in dollars), NOT accumulate them. Gold (PAXG) is
  the exception — accumulate ounces as a hedge. (BTC, separate bots, accumulates satoshis.)
- **Assets:** SOL/ETH/HYPE → dropped (long-only spot can't make USD on fallers). XMR → dropped
  (operator is European; XMR not EU-tradeable). **Settled on TRX + BNB** (EU-tradeable on Kraken).

## Tooling (reuses the exact production sizing spine)
`backtest/study_fetch.py` pulls deep Binance public history (1h/4h/8h/12h/1d; prices ≈ Kraken,
live trades on Kraken). `backtest/alt_engine.py` replays it through `agent.alts.clamp_targets` +
`plan_trades` + `Position`, with `--timeframe`, `--strategy {trend,meanrev}`, and `--allocator
{momentum,llm}` (LLM = exact live `decide_asset`; point-in-time Fear&Greed wired in, leak-free).
`backtest/study_candles.py` sweeps timeframe × strategy and ranks indicators by Information
Coefficient (the same IC engine the BTC trader self-tunes with).

## Make-USD gate
Bar = bot ends with MORE USD than holding cash AND beats DCA (HODL ends in crypto = opposite of
the goal). Spot is long-only, so in a downtrend "beat cash" is genuinely hard.

## Study findings — TRX & BNB, last 365 days

```
TRX  meanrev / trend (bot % USD)        BNB  meanrev / trend
  1d   -7.0%  / +3.1%   dca +4.4%        1d  -19.4% / -3.4%   dca -22.1%
 12h  -13.2% / -5.6%                     12h -15.7% / -5.4%
  8h  -15.6% / -6.1%                      8h -13.4% / -7.1%
  4h  -26.0% / -11.7%                     4h -33.1% / -10.3%
  1h  -73.1% / -43.3%                     1h -79.1% / -41.2%
Top technicals by |IC| (daily, ~6-bar fwd): BNB rsi_14 +0.17, rsi_21 +0.16; TRX rsi_21 +0.15,
  momentum_24 +0.14, ema_cross_8_21 +0.14.   IC collapses to ~0.04 (noise) at 1h.
```

1. **Candle: DAILY wins decisively.** Finer timeframes are progressively worse; **1h is
   catastrophic** (−43% to −79%) — fee drag (0.26%/round-trip) + intraday noise. Use 1d (12h second).
   4h/1h are traps. (The operator's 4h instinct is too fast for these assets.)
2. **Signal: TREND-FOLLOWING, not mean-reversion.** Indicators have **positive IC** (momentum
   persists). The dip-buy swing rule traded *against* the edge and lost everywhere; the
   trend-follower (`momentum_trend_target`: long confirmed strength, cash on weakness) beats it
   at every timeframe. Best signals: BNB → RSI(14/21); TRX → EMA-cross(8/21) + momentum_24; daily.

## Verdict (honest, do not overstate)
Even **best-config (TRX+BNB, daily, trend-following)** over the last year ended **−1.5% USD**: it
**beat DCA (−8.8%)** and stayed mostly in cash, but **did NOT beat holding cash → GATE FAILS, no
marker.** It is a **defensive "lose-less, end-in-cash" tool, not a USD printer.** Spot long-only
cannot reliably extract USD without a choppy regime or shorting/perps (operator excluded those).
**Recommendation: do not deploy real size expecting profit; treat as capital-preservation at best;
if run, use a DAILY cadence and the trend-following signals — never 4h/1h.**

## Caveats
Single historical path; equal-weight DCA/HODL baselines; Kraken/Binance fee (0.26%); IC = linear
predictive power, not a guarantee. LLM-in-the-loop runs are illustrative (single path; the LLM can
recognize an asset's price history — hindsight); news/headlines via GDELT is still a future add.
