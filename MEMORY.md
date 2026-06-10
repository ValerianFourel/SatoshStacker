# SatoshiStacker — running decision log

Append-only notes on *why* things are the way they are. Newest first.

## 2026-06-10 — testnet keys validated; DashScope key valid but model not activated; Ed25519 support
- **Binance testnet (HMAC) keys work** — preflight passed, account funded (~10k USDT testnet). M3 can run now.
- **DashScope/Model Studio key + workspace endpoint are valid** (auth reaches the model gate), but EVERY
  Qwen model returns `AccessDenied.Unpurchased` → no model activated on the account/workspace (ap-southeast-1)
  yet. Action for operator: activate Qwen (qwen3.6-plus) in Model Studio. Until then the advisor is a SAFE
  no-op (QwenAdvisor catches the 403 → Verdict.safe spine size; agent runs the deterministic ladder unchanged
  — verified). LLM endpoint = workspace `https://ws-oi3ic6git9pc3c45.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1`.
- **Ed25519 Binance key support added:** `build_exchange._binance_secret` reads the secret from a
  `*_SECRET_FILE` PEM path (recommended for Ed25519, which is a multi-line private-key PEM) or inline (with
  `\n` unescaping). Ed25519 needs BOTH an API-Key string AND a private-key PEM. Operator pasted only a
  63-char API-Key-shaped string into the live SECRET slot (preserved as a comment, awaiting clarification:
  testnet or live? where is the .pem?). Testnet does NOT need Ed25519 (HMAC already works).
- Installed `ccxt` + `openai` into the conda env for local testnet/LLM validation.

## 2026-06-10 — Milestone 3 deployment kit delivered (awaiting testnet keys)
- Operator chose: build M3 ops now, they'll create Binance Spot Testnet keys, Telegram alerts on.
- Delivered: `Dockerfile` + `docker-compose.yml` + `.dockerignore` (lean `requirements-runtime.txt`
  = ccxt/openai/dotenv/requests, NO pandas — agent has zero module-level pandas/numpy imports;
  ccxt+openai are lazy), `deploy/satoshistacker.service` (systemd, Restart=always, SIGTERM→graceful
  keeps resting bids), `DEPLOY.md` (VPS + testnet keys + Telegram + .env + preflight + run + monitor
  + go-live runbook).
- New CLI: `--preflight` (connectivity + startup safety, no trading — tested), `--test-telegram`.
  Config is now fully `.env`-driven via `AgentConfig.from_env` (SYMBOL, window dates, caps, analysis).
- **Testnet symbol = BTC/USDT** (testnet BTCUSDC liquidity is thin); live = BTC/USDC. De-peg guard
  is inert on a USDT symbol (correct). The actual ≥2-week testnet RUN is the operator's step (needs
  their keys); preflight validates the key can't withdraw before any run.

## 2026-06-10 — Milestone 2 built + adversarially safety-audited
- Operator chose: strategy **reanchor**, **OpenRouter** Qwen endpoint, **3-month** window
  (now → 2026-09-10, deploy-by 2026-08-20), **$2,000 / $35k floor**.
- Built the full agent (config, ladder, exchange[Paper+lazy-ccxt], state[SQLite+reconcile],
  risk, orders, analysis[fail-safe Qwen+mock], notify, loop, main). ccxt/openai imported
  lazily so dry-run + tests need neither. `python -m agent.main --mode dry_run`.
- A 6-agent adversarial safety audit confirmed dry-run isolation, LLM shrink-only
  containment, and live-start gating, but **falsified 3 invariants** — all then fixed +
  regression-tested (43 tests pass):
  1. **Torn-write double-buy / lost fill (HIGH):** order placed before persisted + reconcile
     only scanned DB rows. Fix: **write-ahead** `pending` rungs (persist clientOrderId before
     placing) + `reconcile` now authoritative against the exchange via
     `Exchange.fetch_order_by_client_id`; deadline tranche recovers by clientOrderId too.
  2. **Secrets in plaintext (HIGH):** raw `str(e)` journaled/logged could carry the
     `X-MBX-APIKEY`. Fix: `agent/secrets.py` `redact()` on every str(e) sink + a root-logger
     `RedactingFilter`.
  3. **Deadline stranded sub-MIN_NOTIONAL dust forever (MEDIUM):** Fix: leftover < min_notional
     surfaced as un-deployable dust (Binance rejects sub-min anyway); every forced tranche now
     sized ≥ MIN_NOTIONAL so leftover ≥ min always fully deploys.
  Hardening: `clamp_multiplier` floor pinned to [0,1]; advisor + de-peg calls wrapped fail-safe.
- The per-day cap now correctly meters **placement** of resting notional (you can't cap fills
  of already-resting GTC orders), so it actually binds (e.g. $400/day default → ~5 rungs/day).
- **Re-audit found the first fixes INCOMPLETE; fixed again (47 tests):** (1) `loop.py` price-fetch
  `report['error']` was the one sink not redacted → leaked to `--once` stdout; now `redact(e)` +
  `main` prints `redact(report)`. (2) dust guard `< min_notional - 1e-6` was looser than the gate's
  strict `< min_notional` → reopened the strand in a 1e-6 band; now matches the gate exactly.
  (3) `committed_today` was an incremental meta counter (crash could evade the cap); now **derived
  from rung rows** via a `placed_day` column (crash-atomic, self-healing). (4) redact() now masks the
  telegram `/bot<id>:<token>/` URL and a `RedactingFormatter` scrubs `log.exception` tracebacks.
- **Lesson:** for money code, re-audit the fixes, not just the original code — the second pass caught
  that each first fix had an edge it missed.


## 2026-06-10 — Milestone 1 complete + adversarially audited
- Built the accumulation backtest on **real** Binance daily data (2017→2026). The pure
  deterministic spine lives in `agent/ladder.py` + `agent/config.py` and is shared by the
  backtest, so we test production sizing code, not a copy.
- **Finding:** a naive single-anchor ladder beats the static (no-deadline) resting ladder
  and lump-sum in drawdowns, but **loses to DCA in deep bears** (floor above the eventual
  bottom → deployed too high). The **re-anchoring/trailing ladder** (reserves powder,
  re-anchors a lower ladder when price breaks its floor) **beats every honest fixed-cadence
  DCA across drawdown windows: 12/16 cells, median +6% BTC.**
- **5-agent adversarial audit** (try-to-refute): edge is **causal** (bit-for-bit identical
  to a streaming sim, max diff 0.0 BTC), **friction-robust** (survives pessimistic fills,
  adverse slippage, asymmetric fees, min-notional), **not sandbagged** (beats every cadence
  1–30d), **not cherry-picked** (replicates OOS on disjoint rolling windows, p≈0.011).
- **Honesty corrections (binding):** (1) reanchor beats the static ladder only on TOTAL
  STACK / protracted bears, NOT per-cell (static wins 9/16 sharp-V cells). (2) Edge is
  REGIME-CONDITIONAL — drawdowns only; lags lump-sum in bull markets. (3) Worst case is the
  protracted multi-quarter grind (closest to the $35k-grind thesis) where DCA competes hard.
- **Decision:** production strategy = **re-anchoring ladder + deploy-by-deadline guard**.
  Carried to M2: MIN_NOTIONAL guard, consider deeper floor / earlier deadline, log
  maker-vs-forced-DCA split.

## Design rules locked
- Accumulate-only (`enable_trim=false`). Spot only. Default mode `dry_run`.
- LLM (Qwen3.6-Plus) is veto/shrink-ONLY and fail-safe; all sizing deterministic in `ladder.py`.
- Deploy-by-deadline + spine ALWAYS win over the advisor; broken advisory ⇒ spine size.
- Backtest honesty: no look-ahead, realistic maker fills, forward floor rule, rising windows included.
