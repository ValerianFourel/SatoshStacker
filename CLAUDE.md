# SatoshiStacker — BTC/USDC accumulation agent

A 24/7 agent whose single job is to **accumulate the most BTC possible from a fixed USDC
budget** over a defined window, for an operator who is **bearish on BTC's USD price** and
expects a grind toward a ~$35,000 floor. **Accumulate-only — it never sells BTC back.**
Spot only. No leverage/margin/futures.

> **The honest baseline this agent must beat.** The trivial version of this thesis is a
> static resting limit-buy ladder on Binance (GTC bids weighted toward a floor — no running
> software, maker fees, zero slippage). This agent only earns its existence if it beats
> **both** (a) that static ladder and (b) naive equal-interval DCA, on accumulated BTC, in
> backtest. **Milestone 1 settled this — see "Backtest verdict" below.**

## Status: Milestone 2 built (dry-run paper trader), awaiting review before testnet

Chosen config (operator, 2026-06-10): strategy **reanchor**, **DashScope / Alibaba Model
Studio** (intl) Qwen endpoint, **3-month** window (now → 2026-09-10, deploy-by 2026-08-20),
**$2,000 / $35k floor**.

| # | Milestone | State |
|---|---|---|
| 1 | Accumulation backtest vs static-ladder / DCA / lump-sum on real BTC drawdowns | ✅ done, audited |
| 2 | Dry-run paper trader: full loop, ladder, deadline, risk gate, state, Telegram, tests | ✅ done — 50 tests, **safety-audited GO** |
| 3 | Binance Spot Testnet, real maker limit orders, run ≥2 weeks | 🛠️ **ops kit delivered** — awaiting your testnet keys to run |
| 4 | Live (gated): only after review + backtest gate passes; start with a small portion | — |

**Milestone-3 deployment kit (ready to run on a ~$5–9/mo VPS):** `Dockerfile` +
`docker-compose.yml` + `.dockerignore` (lean `requirements-runtime.txt`, no pandas),
`deploy/satoshistacker.service` (systemd, auto-restart, SIGTERM graceful shutdown that
keeps resting bids), and **`DEPLOY.md`** (full runbook: VPS, Binance Spot Testnet keys,
Telegram bot, `.env`, preflight, run, monitor, halt/upgrade, and the path to live). New CLI:
`--preflight` (connectivity + safety checks, no trading), `--test-telegram`. Config is now
fully `.env`-driven (`AgentConfig.from_env`). Use `SYMBOL=BTC/USDT` on testnet (liquidity),
`BTC/USDC` for live. **Your next action:** create testnet keys + Telegram bot, fill `.env`,
run `--preflight`, then `docker compose up -d`.

**M2 adversarial safety audit (3 rounds, ~17 agents).** Round 1 confirmed dry-run
isolation, LLM shrink-only containment, and live-start gating, and **falsified 3 invariants**;
round 2 caught that the first fixes were **incomplete** (the value of re-auditing fixes, not
just original code); round 3 confirmed **GO** (only a LOW defense-in-depth note, since closed).
Final state — all fixed + regression-tested:
- **Crash-restart torn-write** (could double-buy / lose a fill): write-ahead `pending` rungs;
  `reconcile` is authoritative against the exchange via `fetch_order_by_client_id`.
- **Secrets in plaintext**: `agent/secrets.py` `redact()` on every `str(e)`/stdout/report sink,
  a root-logger filter + format-level `RedactingFormatter` (scrubs tracebacks), and a
  telegram-URL/quoted-JSON/header-pattern set.
- **Deadline stranding**: sub-MIN_NOTIONAL surfaced as un-deployable dust; the dust guard
  predicate exactly matches the risk gate; every forced tranche ≥ MIN_NOTIONAL.
- **Per-day cap crash-atomicity**: `committed_today` is derived from rung rows (`placed_day`
  column), not an incremental counter, so a crash can't evade the cap.
- Hardening: `clamp_multiplier` floor pinned to [0,1]; advisor + de-peg calls fail-safe.

Milestone-2 entrypoints: `python -m agent.main --mode dry_run --cycles N` (paper, live
public prices), `--status`, `--clear-halt`, `--reset`. Replay the live loop over history:
`python3 backtest/replay.py 2022_bear_full`.

## Alt-stacker — multi-asset LLM trader (SOL / ETH / HYPE / GOLD) on KRAKEN — separate from BTC

A second, independent agent. **Operator's goal: MAKE USD by trading ZEC/USD — NOT accumulate it;
accumulate OUNCES of gold (hedge). BTC is the opposite — the separate BTC bots accumulate
satoshis.** (Asset journey: SOL/ETH/HYPE dropped — can't make USD on fallers; XMR & TRX/BNB
dropped; settled on **ZEC (Zcash)**, EU-tradeable on Kraken — `ZEC/USD`, unlike Monero.) Built by
generalizing the active BTC `agent/trader.py` into a **portfolio sharing ONE USD cash pot** on
Kraken. Venue split: **alts on Kraken, BTC on Binance**. Own `ALT_MODE`/`ALT_VENUE` env. **Decoupled
cadence: decide every `ALT_CYCLE_HOURS`, news/F&G every `ALT_NEWS_EVERY_HOURS` (8h).** ⚠️ The study
(below) says **DAILY beats every intraday cadence decisively** — use `ALT_CYCLE_HOURS=24`, NOT scalp.

- **Objectives** (`AssetSpec.objective`): `accumulate_quote` = **make USD** (**ZEC** — grow the
  quote; spot-only; ride confirmed uptrends, bank profits, cash in downtrends, never bag-hold), and
  `store_of_value` (PAXG/XAUT **gold** — accumulate ounces; units-benchmarked). `accumulate_base`
  (most units) exists but no alt uses it. Max exposure ≤0.6 ("LLM decides"); cage stops extremes.
  Default `ALT_ASSETS=ZEC:accumulate_quote:0.6,PAXG:store_of_value:0.5`.
- **LLM has FULL decision authority** (operator's choice) but only INSIDE a deterministic cage
  (`Rails`): per-asset `max_fraction`, per-cycle + per-day turnover caps, total-allocation clamp,
  USDC de-peg halt, price-sanity bad-tick guard, MIN_NOTIONAL, consecutive-API-failure halt,
  kill switch, spot-only, withdrawals-disabled key. A broken/malformed LLM reply ⇒ **HOLD**
  (no churn). This is a deliberate departure from the BTC bot's advisory-only LLM — documented,
  not audited to the same bar yet (see status).
- **One SEQUENTIAL LLM call per asset** (each decision + rationale observable), each fed that
  asset's own momentum, **per-asset sentiment** (Yahoo headlines + market Fear&Greed), and its
  **weekly self-tuned technicals** (`agent/technicals_<BASE>.json` via `tune.run_tune`).
- **Per-asset shadow benchmarks** (DCA + HODL on units for SOL/ETH; realized-USDC vs hold-HYPE
  for HYPE) in the daily/weekly Telegram report.

New modules: `agent/multi_exchange.py` (shared-cash paper + ccxt multi-symbol, **venue-aware**:
binance|kraken, public price/candle feeds), `agent/alts.py` (the `AltStacker`, cage, sequential
decisions, tune, benchmarks, **reconcile-against-exchange**), `agent/alts_main.py` (CLI).
Tests: `tests/test_alts.py` (cage + rebalance + audit-fix regressions, 18 tests).
Entrypoints: `python -m agent.alts_main --mode dry_run --once|--cycles N`, `--status`,
`--preflight`, `--clear-halt`, `--reset`. Config is `.env`-driven (`ALT_*`, `AltConfig.from_env`).
Deploy unit: `deploy/satoshistacker-alts.service`.

> **HYPE venue — resolved on Kraken.** HYPE is **not on Binance spot** (`HYPE/USDC`/`HYPE/USDT`
> both 404), but **Kraken lists SOL/ETH/HYPE — all vs USD**. So `ALT_VENUE=kraken` runs the whole
> alt-stacker on one venue with a shared **USD** cash pot ("accumulate USDC via HYPE" → "accumulate
> USD via HYPE"). Trade-off: **Kraken has no spot testnet**, so its path is `dry_run` → small `live`
> (no testnet rehearsal). `ALT_VENUE=binance` remains available for a SOL/ETH-only testnet run.
> De-peg halt auto-disables for a fiat (USD) quote; Kraken live keys = `KRAKEN_API_KEY/SECRET`.

**Adversarial safety audit (27 agents) — done; all confirmed findings fixed + regression-tested.**
21 findings (1 CRITICAL, 8 HIGH, 8 MED, 4 LOW); the CRITICAL + ~9 were one root cause (no
reconcile + wall-clock coid + position persisted after the order = torn-write double-buy). Fixes:
- **Crash/double-trade:** `_reconcile_positions` makes the **exchange the source of truth for held
  units** every cycle (assumes a dedicated account/sub-account); deterministic per-cycle-bucket
  `clientOrderId`; fills are read from the actual order, not the requested size.
- **Malformed-LLM containment:** a NaN/"nan"/inf/bool `target_fraction` now **HOLDs** (was clamping
  to 1.0 = full deploy).
- **Per-day turnover** derived from the trades log (crash-atomic), not an evadable counter.
- **Anomaly halts:** one degraded-cycle counter + single alert; de-peg **fails closed** (skips
  trading if the peg can't be verified) and is skipped for a fiat quote; bad-ticks now alert.
- **Redaction:** `Notifier.send` runs `redact()`; untrusted LLM `stance` is sanitized before Markdown.

**Status:** dry-run paper trader, audited + fixed, **19 alt tests (full suite 87 passing)**;
verified live on Kraken (SOL/ETH/HYPE/PAXG — incl. HYPE @ ~$58, gold PAXG @ ~$4,120). **Still no historical backtest gate**
for the alt thesis — required before live, mirroring BTC Milestone 1. Live gated identically
(`--mode live` + `LIVE_TRADING_CONFIRMED=yes` + `backtest/results/ALT_GATE_PASSED` + key cannot
withdraw).

## BTC watch — read-only monitor + LLM analyst + Telegram Q&A (`agent.btcwatch`)

A **separate, non-trading** service (operator request): it continuously watches BTC on
**public Binance data only — no keys, no orders, ever**, and uses the LLM purely to
*describe* the market (this extends the bot's existing "LLM never trades" rule to a
talk-only assistant). Two ways the LLM gets called:
- **Proactive:** `market_monitor.py` computes order-book depth/imbalance, volume-surge,
  realized-vol/ATR, RSI/EMA/24h-range every `WATCH_SCAN_INTERVAL_S`; an `AnomalyDetector`
  fires on **out-of-the-norm events (peak / bottom / volume & price spikes)** with a
  per-signal cooldown + re-arm (one episode = one alert, survives restart). On fire it
  auto-calls the analyst and pushes the read to Telegram.
- **On-demand:** `telegram_listener.py` long-polls `getUpdates` **gated to `TELEGRAM_CHAT_ID`**
  (strangers ignored); `/btc`·`/status` → LLM read, `/raw` → numbers only, free text → answer.

**News + web search + article reading (`websearch.py`).** Every analyst call auto-attaches
**BTC headlines (Yahoo RSS) + Fear&Greed (alternative.me)** — keyless, TTL-cached. The analyst
can also **search the web** (autonomously — returns a `search` query, re-invoked once with
results, bounded to one round — or on demand via `/search <q>` / `/news`) and **open & read
the top `WATCH_FETCH_ARTICLES` results' full article bodies**, not just snippets. Provider order
**self-hosted SearXNG (`SEARXNG_URL`) → Tavily → Serper → keyless DuckDuckGo**; article text via
**trafilatura if installed → built-in regex fallback** (Tavily returns body directly). For a
server, run SearXNG + `pip install trafilatura` (fully keyless) or set `TAVILY_API_KEY`.
Searches can be **date-bounded** (before / after / between): the LLM extracts the window from
a natural-language ask (relative dates resolved via an injected `today`), or pin it explicitly
with `/search … after:YYYY-MM-DD before:YYYY-MM-DD` (or `between:A..B`) — translated per provider
(Tavily start/end_date, Serper `tbs`, DuckDuckGo `df`, SearXNG client-side `publishedDate` filter).
All off-box fetches are untrusted + failure-safe and can never trigger an action.

The analyst is read-only but may write notes/metrics to a **sandboxed scratch dir**
(`scratch.py`, path-traversal-proof — never the repo/.env/exchange). Any LLM/network error
fails safe to a deterministic numeric summary. All metric math + the detector + the
chat-gate are unit-tested with zero network (`tests/test_btcwatch.py`, 17 tests).
Run: `python -m agent.btcwatch [--once|--status|--test-telegram]`; deploy
`deploy/satoshistacker-btcwatch.service`. Config: `WATCH_*` in `.env`.

## Backtest verdict (Milestone 1)

Tested on **real Binance daily klines, 2017-08 → 2026-06** across 8 historical drawdown
windows + 2 rising windows, with a 5-agent adversarial audit (full record: `backtest/AUDIT.md`).

- **vs the static resting ladder (gate a):** the deploy-by-deadline guard means we **never
  strand USDC** — we tie the static ladder when the floor is hit, win when it isn't.
- **vs naive DCA (gate b):** the naive single-anchor ladder **loses** to DCA in deep bears.
  The **re-anchoring (trailing) ladder** fixes this — it **beats every honest fixed-cadence
  DCA across drawdown windows (12/16 cells, median +6% BTC)**, verified causal and
  friction-robust, replicated out-of-sample on disjoint rolling windows.
- **Honest caveats (do not overstate):** the edge is **regime-conditional** — it holds in
  drawdowns (the operator's thesis) and **lags lump-sum in rising markets** (irreducible).
  Reanchor beats the static ladder on **total stack / protracted bears**, *not* per-cell.
  Its weak spot is the long multi-quarter grind, which is the regime closest to a "slow
  grind to $35k" thesis — so **DCA remains a strong competitor for that exact scenario**.

**Conclusion:** the agent meets the spec gate (beats static ladder AND DCA in drawdowns).
The recommended production strategy is the **re-anchoring ladder + deploy-by-deadline guard**.

## Architecture

```
agent/
  config.py     # dataclasses: ladder/risk/schedule/analysis + env loading (no net/keys) [done]
  ladder.py     # deterministic ladder + reanchor generation + min-notional + LLM clamp  [done]
  exchange.py   # Exchange ABC; PaperExchange (dry_run) + CcxtExchange (testnet/live)     [done]
  risk.py       # per-day commit cap, de-peg halt, anomaly, fail-closed                   [done]
  orders.py     # idempotent maker rungs (clientOrderId) + deadline market DCA            [done]
  state.py      # SQLite Store; reconcile-on-startup (exchange = source of truth)         [done]
  loop.py       # Agent.run_cycle state machine: laddering -> deadline; graceful shutdown [done]
  analysis.py   # Qwen3.6-Plus advisory — veto/shrink ONLY, fail-safe, journal            [done]
  notify.py     # Telegram alerts + daily accumulation summary (console fallback)         [done]
  main.py       # CLI, mode handling, fail-closed startup safety checks                   [done]
  # ── BTC watch (read-only; agent.btcwatch) — separate process, no keys/orders ──
  market_monitor.py # public order book/volume/vol/technicals -> snapshot + anomaly detector [done]
  analyst.py        # read-only LLM (event reads + Q&A); NEVER buy/sell; news+search+scratch   [done]
  websearch.py      # keyless BTC news (Yahoo RSS) + Fear&Greed; web search Tavily/Serper/DDG  [done]
  telegram_listener.py # inbound getUpdates, operator-chat-gated, routes to analyst           [done]
  scratch.py        # sandboxed scratch dir the analyst may write (path-traversal-proof)      [done]
  btcwatch.py       # entrypoint: monitor thread + Telegram Q&A, graceful shutdown            [done]
backtest/       # backtest + design search + adversarial audit + live-loop replay  [done]
tests/          # ladder math + deadline + clamp + full agent safety (99 passing)  [done]
```

ccxt and the OpenAI SDK are imported **lazily** (only on the testnet/live and real-LLM
paths), so the dry-run paper trader and the whole test suite run with zero exchange/LLM
dependencies. `PaperExchange` keeps its own persisted book (an independent source of truth)
so reconcile-on-restart is genuinely exercised.

The deterministic spine (`ladder.py`, `config.py`) is **import-clean** (no ccxt/LLM/network)
so the backtest exercises the *exact* production sizing logic, not a reimplementation.

## Non-negotiable safety (fail closed)

1. **Modes, default `dry_run`:** `dry_run` (paper) → `testnet` (Binance Spot Testnet) →
   `live`. `live` requires BOTH `--mode live` AND env `LIVE_TRADING_CONFIRMED=yes`.
2. **No live until the backtest gate passes** (it now does) and the operator reviews M1–M3.
3. **The LLM never places / sizes / cancels / approves orders.** All sizing is deterministic
   in `ladder.py`. The Qwen advisor may only make buying *more cautious* (shrink a tranche,
   never below 50%); it can never increase/speed up buying, buy above the ladder, sell, or
   block the deploy-by-deadline guard. Broken/stale/malformed advisory ⇒ **spine size, no
   extra deployment in either direction** (see `clamp_multiplier`).
4. **API key:** spot only, **withdrawals disabled**, IP-restricted to the VPS. Startup
   verifies the key cannot withdraw and refuses to run if it can. Keys from `.env`, never logged.
5. **Crash-safe state (SQLite):** persist ladder, orders, fills, USDC remaining, schedule.
   On restart, reconcile against the exchange (**exchange = source of truth**) before acting.
   Idempotent orders via `clientOrderId`.
6. **Anomaly halt + alert** (then require manual clear): USDC de-peg (price ≠ ~1), an order
   deviating absurdly from the ladder, or repeated API failures.

## Analysis layer tradeoff (web search)

`analysis.py` may optionally run a **guarded** web search (hard timeout, capped queries) to
add macro/news context to its rationale. **This sends queries off the box.** Fetched text is
treated as untrusted context for the *rationale only* — it can never change deployment beyond
the deterministic ladder, and on any error returns a safe `proceed`-at-spine-size default.
Disabled by default (`ANALYSIS_WEB_SEARCH=false`).

## Carried into Milestone 2 (from the audit)

- Add a **MIN_NOTIONAL guard** to the order layer: deep bears make the trailing ladder emit
  sub-$5 rungs Binance would reject; aggregate/defer them and let the deadline mop up.
- Consider a **deeper floor / earlier deadline** for the protracted-grind regime.
- Log the per-tranche **maker-vs-forced-DCA split** so live behaviour is transparent.

## Run the backtest

```bash
pip install -r requirements.txt
# BTC ladder (Binance):
python3 backtest/fetch_data.py    # one-time: pull real Binance history
python3 backtest/engine.py        # baselines + spine scoreboard + verdict
python3 backtest/variants.py      # design search + audit-corrected honesty checks
# Alt-stacker (Kraken; ZEC/USD make-USD + GOLD hedge) — reuses the exact alts sizing spine:
python3 backtest/study_fetch.py --tfs 15m,30m,1h,4h,1d ZEC   # one-time: deep multi-tf Binance history
python3 backtest/study_scalp.py            # ZEC: scalp vs daily, taker vs maker fee (make-USD?)
python3 backtest/study_candles.py          # which candle + which technicals (TRX/BNB legacy study)
python3 backtest/alt_engine.py --assets ZEC --timeframe 1d --strategy trend --write-gate
python3 -m pytest tests/ -q       # 90 tests (incl. 22 alt: cage, spine, audit-fixes, gold)
```

**ZEC scalp study (`backtest/study_fetch.py` + `study_scalp.py` + `study_candles.py`; full record
`backtest/ALT_AUDIT.md`).** Deep Binance public history 15m→1d; sweeps timeframe × strategy
(trend vs mean-reversion) × fee (taker 0.26% / maker 0.16%) and ranks indicators by IC. Findings
(ZEC, last 180d):
- **INTRADAY SCALPING LOSES — the finer the worse.** 15m trend −54%(taker)/−30%(maker), 30m −27/−8,
  1h +6/+19, 4h +24/+26; mean-reversion −39% to −94% everywhere. Round-trips pay 2× the per-side fee
  → fees annihilate scalps. (Forcing a tiny rebalance band changes nothing — scalping loses on
  merit.) **Do NOT scalp.**
- **ONLY DAILY trend-following made USD:** 1d trend +28.5%(taker)/**+30.4%(maker)**, beating cash
  (+0%) and DCA (+28.8%). Signal: positive IC (momentum persists), strongest at daily (ZEC rsi_28
  IC≈0.20). The opposite of scalping.
- **⚠️ Verification in progress** (adversarial workflow): the +30% daily-trend result may be ZEC's
  recent rally — an out-of-sample / overfit check is running before this is trusted for live.
