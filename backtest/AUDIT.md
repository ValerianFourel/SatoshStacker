# Milestone-1 adversarial audit — does the re-anchoring ladder *really* beat DCA?

A 5-skeptic + 1-synthesizer multi-agent audit was run with a single mandate: **refute**
the claim that the re-anchoring (trailing) ladder accumulates more BTC than naive DCA
across real BTC drawdown windows. Each skeptic ran its own experiments against the
committed code. Result: **conclusion holds, high confidence — with one honesty correction.**

## What survived every attack (the claim is real)

| Attack axis | What was tried | Outcome |
|---|---|---|
| **Look-ahead / future leak** | Reimplemented reanchor as a strict left-to-right streaming sim where bar *i* sees only bars `0..i`; deadline commits fixed per-bar amounts at *future* opens | **Bit-for-bit identical** (max diff `0.0` BTC). Strategy is provably causal. |
| **Fill-model optimism** | `fill@L` (no gap bonus), pierce-margin, close-confirmation (no wick fills), +0.5% adverse ladder slippage | Still **12/16**, median ~1.06. BTC has ~no overnight gaps so `min(L,open)==L`; the gap branch fires **0 times**. |
| **Fee asymmetry** | Flat/higher fees, realistic maker<taker (BNB), bps spread, fixed per-order cost | Proportional fees are **ratio-neutral** (both deploy the full $2000). Realistic maker<taker **helps** reanchor (ladder fills are maker; DCA is 100% taker). Only an unrealistic ~$2–3 *fixed* per-order fee flips it. |
| **Baseline sandbagging** | 54 honest DCA variants: every cadence 1–30d, value-averaging, dip-buying; granted DCA the ladder's maker+limit fill | reanchor beats **every** fixed cadence (11–13/16); even an impossible daily-low-fill DCA only narrows to 11/16. Weekly is representative, not weak. |
| **Overfitting / cherry-picking** | Param sweeps (gen_fraction, floor, deadline, n_tranches, geom_ratio, max_gen); **disjoint rolling quarter-start windows 2018–2026** | Out-of-sample on rolling drawdown windows: **31/42** wins (median 1.064); non-overlapping **18/24** (binomial p≈0.011); under a *causal* trailing-drawdown entry filter **17/24**. Curated median (1.062) ≈ rolling-population median (1.064): **not** cherry-picked. |

Reproduce the corrected head-to-head numbers: `python3 backtest/variants.py`.

## The honesty correction (baked into the claim)

1. **"Beats the static ladder" is true only on TOTAL STACK / in protracted bears — NOT
   per cell.** Direct head-to-head over the 16 drawdown cells: **static wins 9, reanchor
   wins 7** (median rea/static = 0.998). Reanchor leads only on the **aggregate stack**
   (1.868 vs 1.753 BTC, mean ratio 1.11) by dominating long grinds (2018 +27–35%, 2022
   bear +52% vs static) while the static ladder wins sharp V-recoveries (covid −11–17%).
   They are **complementary**. Do not market reanchor as strictly dominating the static
   ladder.

2. **The edge is REGIME-CONDITIONAL.** It exists *in drawdowns* (your thesis). On
   unfiltered rolling windows mean ratio < 1.0; in *rising* markets reanchor loses badly
   (11/64 wins, median 0.85) — irreducible: nothing that waits for lower beats lump-sum
   when price only rises. The deploy-by-deadline guard caps that damage.

3. **One genuine structural loss vs DCA: the protracted multi-quarter grind**
   (`2022_bear_full` @ floor 0.65, ratio 0.85). Reanchor under-deploys and force-DCAs the
   leftover *above* the eventual bottom at the 70% deadline. Mitigation = deeper floor or
   earlier deadline (a tuning choice). **This is the regime closest to your "slow grind to
   $35k" thesis — so DCA is a genuinely strong competitor for your specific scenario, and
   reanchor's edge there is modest.**

4. **Where the capital actually goes.** In `2022_ftx` and `2024_summer` ~**87–92%** of
   reanchor's budget is the *taker* deadline-DCA, not maker-ladder fills — i.e. reanchor ≈
   delayed-DCA there, and the headline edge is carried by the *other* windows.

## Required before LIVE (not needed for the backtest verdict)

- **MIN_NOTIONAL guard.** Deep bears make reanchor emit a tail of sub-$5 orders
  (`2022_bear_full`: 126/149 orders < $5, smallest $2.67) that Binance would reject.
  Honest modelling (reject + deadline mop-up) leaves the headline unchanged, but the live
  order layer must aggregate/defer sub-min rungs. Tracked for Milestone 2.
- **Stricter live fill model** (queue position / partial fills) — neutral to the verdict
  but more honest for live expectations.

## Bottom line (spec gate)

> A **causal, never-stranding** re-anchoring ladder accumulates **more BTC than every
> honest fixed-cadence DCA across BTC drawdown windows (12/16, median +6%)**, and a
> **larger total stack than the static ladder** by winning protracted bears — though the
> static ladder wins more individual V-shaped cells. The DCA gate is solidly met; the
> static-ladder gate is met on total stack. **Ship it (with the qualified wording).**
