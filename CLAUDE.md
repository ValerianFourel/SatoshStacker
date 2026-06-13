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

## BTC watch — read-only monitor + LLM analyst + Telegram Q&A (`agent.btcwatch`)

A **separate, non-trading** service (operator request): it continuously watches BTC on
**public Binance data only — no keys, no orders, ever**, and uses the LLM purely to
*describe* the market (this extends the bot's existing "LLM never trades" rule to a
talk-only assistant). Two ways the LLM gets called:
- **Proactive:** `market_monitor.py` computes order-book depth/imbalance, volume-surge,
  realized-vol/ATR, RSI/EMA/24h-range every `WATCH_SCAN_INTERVAL_S`; an `AnomalyDetector`
  fires on **out-of-the-norm events (peak / bottom / volume & price spikes)**. By default it
  uses **confluence** (`WATCH_CONFLUENCE_MIN=2`): it pings the whole agreeing bundle only when
  **≥N signals are out-of-norm at once**, then stays quiet for a global **cadence**
  (`WATCH_ALERT_CADENCE_S`, ~30m) and until the cluster disperses + re-arms (one episode = one
  alert, survives restart). `confluence=1` restores the classic per-signal model. On fire it
  auto-calls the analyst and pushes the read to Telegram.
- **`/origins` widget:** an inline-keyboard control panel (`origins.py`) to tune — live, no
  restart — which signals may ping, how many must agree (confluence ±), the cadence (±), the
  preset, and mute. Taps mutate `watch_prefs.json`; the monitor picks them up within ~15s.
  Callbacks are operator-chat-gated and never crash the poll loop.
- **Autonomous news (`_maybe_news_digest` + `analyst.news_digest`):** every
  `WATCH_NEWS_DIGEST_HOURS` (~8h) the analyst reads the news regardless, caches a copy to
  `news_digest_path` (`/digest` to read it), and pings **only if it itself decides** the news
  is material (reply starts `ALERT`; `QUIET` is cached silently). Mute suppresses the ping but
  the copy is still kept. An upgrade-notice fires once on existing installs explaining the
  confluence/cadence/news change and pointing at `/origins`.
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
chat-gate are unit-tested with zero network (`tests/test_btcwatch.py`).

**Signal tuning (`--tune`, `signal_tuner.py`).** Backtests a family-diverse BATTERY across
**multiple timeframes (5m/1h/4h/1d)** and reports the best top/bottom caller PER FAMILY (combine
across families, don't stack redundant ones):
- *momentum*: RSI, Stochastic, StochRSI, MACD-hist, CCI, Williams %R, ROC
- *volatility*: Bollinger %B + width, Keltner position, ATR%
- *trend*: EMA crosses (21/50, 50/200), SMA-200 distance, Supertrend
- *volume*: MFI, OBV slope, VWAP distance, CVD slope

It labels real swing tops/bottoms (±swing_w extrema) and scores each by rank-AUC + best-F1
threshold. The `live_tf` (= monitor trend TF) winners go to `agent/watch_signals.json`; the live
detector fires peaks/bottoms off the tuned signal+threshold (RSI defaults if absent).
`python -m agent.btcwatch --tune [--weeks 8 --tf 1h]` — verified live (4h cleanest, AUC ~0.8;
e.g. 4h top=vwap_dist/rsi, bottom=rsi). NOTE: crypto-native flags (funding, OI, liquidations,
on-chain) are not yet in the battery — they need Binance-futures/Coinglass/Glassnode feeds.

Run: `python -m agent.btcwatch [--once|--status|--test-telegram|--tune]`; deploy
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
  signal_tuner.py   # --tune: backtest oscillator battery -> best top/bottom callers -> json   [done]
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
python3 -m pytest tests/ -q       # BTC accumulation + active trader + btcwatch tests
```
