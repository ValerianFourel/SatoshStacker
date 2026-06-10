# Deploying SatoshiStacker (Milestone 3 — testnet, then Milestone 4 — live)

A headless Python agent for a **~$5–9/mo CPU VPS** (Hetzner CX22, Contabo, etc. — no GPU).
This is the **testnet** runbook; the final section covers the extra gates for going live.

> **Safety recap.** Default mode is `dry_run`. `testnet` places **real maker limit orders on
> the Binance Spot Testnet** (fake money). `live` is refused unless ALL of: `--mode live`,
> env `LIVE_TRADING_CONFIRMED=yes`, the backtest-gate marker exists, and the API key
> **cannot withdraw** (verified at startup). The LLM advisor can only ever *shrink* a buy.

---

## 0. Symbol note (read first)
The agent trades **BTC/USDC** in production, but the Binance **Spot Testnet** has reliable
liquidity on **BTCUSDT**, not always BTCUSDC. For Milestone 3 set `SYMBOL=BTC/USDT` to
validate real maker fills mechanically; switch back to `SYMBOL=BTC/USDC` for live. (The
USDC de-peg guard is inert on a USDT symbol — correct, there is no USDC to de-peg.)

## 1. Provision the VPS
Any ~$5–9/mo Ubuntu box works, **or use the OCI Always Free tier (recommended, $0)** — see 1B.
1. Create an Ubuntu 24.04 box (1 vCPU / 1–2 GB RAM is plenty).
2. SSH in, create a user, basic firewall:
   ```bash
   sudo adduser --disabled-password --gecos "" satoshi
   sudo ufw allow OpenSSH && sudo ufw enable
   ```
3. Note the VPS's **public IPv4** — you'll IP-restrict the API key to it.

## 1B. Oracle Cloud (OCI) Always Free — Oracle Linux 9 (VM.Standard.E2.1.Micro)
Free, always-on, x86_64, 1 OCPU / 1 GB — plenty for this agent. In the create-instance wizard:
- **Image/Shape:** Oracle Linux 9, `VM.Standard.E2.1.Micro` (Always-Free-eligible).
- **Networking:** Create new VCN + **public subnet**; **Automatically assign public IPv4 address = Yes**.
- **SSH keys:** paste your `~/.ssh/id_ed25519.pub`, or "Generate a key pair" and **download the private key**.
- No inbound ports beyond SSH (the agent is outbound-only). For the *live* key later, convert the
  ephemeral public IP to a **reserved** IP and restrict the Binance key to it.

Then deploy (run on your Mac where the code lives, then on the instance as user `opc`):
```bash
# 0) ON YOUR MAC — copy code + .env to the box (no git remote needed):
rsync -av --exclude '.git' --exclude '.venv' --exclude '__pycache__' --exclude '*.db' \
  --exclude '*.paperbook.json' --exclude 'backtest/data' \
  ~/Pharsale/SatoshiStacker/  opc@<PUBLIC_IP>:/home/opc/SatoshiStacker/
scp ~/Pharsale/SatoshiStacker/.env  opc@<PUBLIC_IP>:/home/opc/SatoshiStacker/.env

# 1) ON THE INSTANCE:
sudo dnf -y install python3.11 git
sudo mkdir -p /opt/satoshistacker && sudo rsync -a /home/opc/SatoshiStacker/ /opt/satoshistacker/
cd /opt/satoshistacker
python3.11 -m venv .venv
.venv/bin/pip install -U pip && .venv/bin/pip install -r requirements-runtime.txt
chmod 600 .env
.venv/bin/python -m agent.main --mode testnet --preflight      # MUST print PREFLIGHT OK
.venv/bin/python -m agent.main --mode testnet --test-telegram   # if Telegram configured

# 2) run 24/7 under systemd (dedicated user, auto-restart, graceful SIGTERM):
sudo useradd -r -s /sbin/nologin satoshi 2>/dev/null || true
sudo chown -R satoshi:satoshi /opt/satoshistacker
sudo cp deploy/satoshistacker.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now satoshistacker
journalctl -u satoshistacker -f
```
The shipped `deploy/satoshistacker.service` already targets `/opt/satoshistacker`, user `satoshi`, and
`.venv/bin/python` — no edits needed. (Optional on a 1 GB box: add a 1 GB swap file —
`sudo fallocate -l 1G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile`.)

## 2. Binance Spot Testnet API keys
1. Go to **https://testnet.binance.vision/**, sign in with GitHub, **Generate HMAC_SHA256 Key**.
2. Copy the **API Key** and **Secret** (testnet keys cannot withdraw and hold only fake funds).
3. (Live later: create the key at binance.com with **Enable Spot Trading ONLY**, **Enable
   Withdrawals OFF**, and **restrict access to your VPS IP**.)

## 3. Telegram alerts
1. In Telegram, message **@BotFather** → `/newbot` → copy the **bot token**.
2. Message your new bot once (say "hi"), then get your **chat id**:
   ```bash
   curl -s "https://api.telegram.org/bot<TOKEN>/getUpdates" | grep -o '"chat":{"id":[0-9-]*'
   ```
   (or message **@userinfobot**). Put the token + chat id in `.env`.

## 4. Configure `.env`
```bash
cp .env.example .env && nano .env
```
Set at minimum:
```
MODE=testnet
SYMBOL=BTC/USDT
BUDGET_USDC=2000
FLOOR_PRICE=35000
DEPLOY_BY=2026-08-20T00:00:00Z
WINDOW_END=2026-09-10T00:00:00Z
MAX_DEPLOY_PER_DAY_USDC=400
BINANCE_TESTNET_API_KEY=...
BINANCE_TESTNET_API_SECRET=...
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=qwen/qwen3.6-plus
LLM_API_KEY=...                 # optional on testnet; without it the advisor is a no-op
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
DB_PATH=/data/satoshistacker.db
```

## 5. Deploy — Option A: Docker (recommended)
```bash
# install docker, then from the repo dir:
docker compose run --rm satoshistacker --preflight     # MUST pass before starting
docker compose run --rm satoshistacker --test-telegram # confirm the alert reaches your phone
docker compose up -d --build                           # start (auto-restarts on crash/reboot)
docker compose logs -f                                 # watch
docker compose run --rm satoshistacker --status        # state snapshot
```

## 5. Deploy — Option B: systemd + venv
```bash
sudo mkdir -p /opt/satoshistacker /data && sudo chown -R satoshi /opt/satoshistacker /data
sudo -u satoshi git clone <repo> /opt/satoshistacker && cd /opt/satoshistacker
sudo -u satoshi python3.12 -m venv .venv
sudo -u satoshi .venv/bin/pip install -r requirements-runtime.txt
sudo -u satoshi cp .env.example .env && sudo -u satoshi nano .env   # DB_PATH=/data/satoshistacker.db
sudo -u satoshi .venv/bin/python -m agent.main --mode testnet --preflight   # MUST pass
sudo cp deploy/satoshistacker.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now satoshistacker
journalctl -u satoshistacker -f
```

## 6. Verify it's working
- `--preflight` printed `PREFLIGHT OK` (price feed + balances reachable; key can't withdraw).
- Telegram received the start message and (within a cycle) a daily summary.
- `--status` shows the resting ladder rungs growing as the per-day cap allows (~$400/day).
- Fills appear in the journal and as Telegram alerts when testnet price touches a rung.

## 7. Run for ≥2 weeks (Milestone 3 acceptance)
Watch that: maker limit orders rest and fill on the testnet; re-anchoring fires if price
breaks a generation floor; the daily summary reports BTC stacked / avg cost / % deployed;
restarts (`docker compose restart` / `systemctl restart`) **reconcile** cleanly and never
double-buy or strand USDC; a graceful stop leaves resting bids in place.

## 8. Operations
- **Halt / resume.** Anomalies (USDC de-peg, absurd order, repeated API failures) auto-halt
  and alert. After investigating: `--clear-halt` (Docker: `docker compose run --rm
  satoshistacker --clear-halt`), then the loop resumes.
- **Upgrade.** `git pull` → `docker compose up -d --build` (or `pip install -r
  requirements-runtime.txt && systemctl restart satoshistacker`). State persists in `/data`.
- **Backups.** Snapshot `/data/satoshistacker.db` (+ `.paperbook.json` in dry_run). It holds
  the ladder, fills, schedule, and halt state; the exchange remains the source of truth on
  restart.

## 9. Going LIVE (Milestone 4 — only after you review M1–M3)
1. Re-run the backtest gate (`python3 backtest/variants.py`) and keep `backtest/results/GATE_PASSED`.
2. Create a **live** Binance key: **spot trading only, withdrawals OFF, IP-restricted** to the VPS.
3. In `.env`: `MODE=live`, `SYMBOL=BTC/USDC`, `LIVE_TRADING_CONFIRMED=yes`, `BINANCE_API_KEY/SECRET`.
4. **Start small:** set `BUDGET_USDC` to a small portion first; scale up only on your command.
5. `--preflight` must pass (it refuses to start if the key can withdraw or any gate is unmet).
6. `docker compose up -d` with `command: ["--mode","live"]` (or edit the systemd `ExecStart`).

## Troubleshooting
- **Preflight fails "key CAN withdraw"** → your key has withdrawal permission; rotate to a
  spot-only, withdrawals-disabled key. The agent refuses to run live otherwise (by design).
- **`ModuleNotFoundError: ccxt`** (Option B) → `pip install -r requirements-runtime.txt`.
- **No Telegram messages** → re-check `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID`; `--test-telegram`.
- **Orders not filling on testnet** → testnet liquidity is thin; confirm `SYMBOL=BTC/USDT`
  and that price has actually traded down to a rung level.
