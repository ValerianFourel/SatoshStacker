"""Active satoshi-stacker — a tactical BTC<->USDC trader (testnet first).

It plays with WHATEVER capital is actually in the Binance account RIGHT NOW: each
cycle it reads the live balances (USDC/USDT + BTC), treats the total as "the pot",
and the LLM decides a TARGET FRACTION of the pot to hold in BTC (0=all cash, 1=all
BTC). It rebalances the real balances toward that target. Sole objective: STACK THE
MOST BTC / lower the average entry. The LLM is told NOTHING about any price target.

Tracks the average BTC entry (average-cost basis) and runs DCA + buy-and-hold shadow
benchmarks (both seeded from the SAME starting pot) so the daily/weekly Telegram
reports show whether the bot actually beats just-DCA / just-holding. Reads
agent/technicals.json (weekly self-tune) for the tuned RSI period + context note.

Lean (numpy only). Average-cost accounting: buying lower drops the avg entry; selling
keeps the remaining lot's avg unchanged but frees cash to rebuy lower (= more sats).
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

from .exchange import Exchange, public_ohlcv
from .notify import Notifier
from .secrets import redact

ROOT = Path(__file__).resolve().parent.parent
TAKER = 0.001
REBAL_BAND = 0.10  # don't trade unless target differs from current by > this

TACTICAL_SYSTEM = (
    "You are a tactical crypto trader whose SOLE objective is to maximize the ENDING "
    "amount of BITCOIN (satoshis) held in a portfolio of USDC + BTC. You gain satoshis "
    "by holding USDC BEFORE price falls (then rebuying more BTC lower) and holding BTC at/"
    "after bottoms and in uptrends. You are given momentum metrics, the current pot (your "
    "USDC and BTC and average entry), and you decide a TARGET fraction of total pot value "
    "to hold in BTC right now (0.0=all USDC, 1.0=all BTC). Round-trip costs ~0.2%, so only "
    "move decisively. You are also given RECENT BTC NEWS HEADLINES and a crypto FEAR & "
    "GREED index (0=extreme fear, 100=extreme greed): extreme fear often marks bottoms "
    "(favor accumulating BTC), extreme greed often precedes pullbacks (favor cash) — but "
    "let momentum confirm. Respond STRICT JSON only:\n"
    '{"target_btc_fraction":<0..1>,"stance":"long_btc|long_cash|hold","note":"<short>"}'
)


def live_news(max_items: int = 6) -> dict:
    """Live BTC headlines (Yahoo Finance RSS) + Fear&Greed sentiment. No API key.
    For the LIVE agent 'now' IS the point in time, so a simple feed is honest. Fail-safe."""
    import xml.etree.ElementTree as ET
    out: dict = {"headlines": [], "fear_greed": None}
    try:
        r = requests.get("https://feeds.finance.yahoo.com/rss/2.0/headline",
                         params={"s": "BTC-USD", "region": "US", "lang": "en-US"},
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        root = ET.fromstring(r.text)
        out["headlines"] = [(it.findtext("title") or "").strip()
                            for it in root.findall(".//item")[:max_items]
                            if it.findtext("title")]
    except Exception:  # noqa: BLE001 - news is advisory only
        pass
    try:
        d = requests.get("https://api.alternative.me/fng/", params={"limit": 1},
                         timeout=8).json()["data"][0]
        out["fear_greed"] = {"value": int(d["value"]), "label": d["value_classification"]}
    except Exception:  # noqa: BLE001
        pass
    return out


def load_technicals() -> tuple[int, str]:
    try:
        d = json.loads((ROOT / "agent" / "technicals.json").read_text())
        return int(d.get("suggested", {}).get("rsi_period", 14)), d.get("context_note", "")
    except Exception:  # noqa: BLE001
        return 14, ""


def momentum(closes: np.ndarray, highs: np.ndarray, price: float, rsi_n: int,
             bpd: int = 6) -> dict:
    def rsi(c, n):
        if len(c) < n + 1:
            return 50.0
        d = np.diff(c[-(n + 1):])
        up, dn = d[d > 0].sum(), -d[d < 0].sum()
        return 100.0 if dn == 0 else float(100 - 100 / (1 + (up / n) / (dn / n)))

    def sma(c, n):
        return float(np.mean(c[-n:])) if len(c) >= n else float(np.mean(c))

    fast, slow = sma(closes, bpd * 3), sma(closes, bpd * 10)
    r24 = float(price / closes[-bpd] - 1) * 100 if len(closes) > bpd else 0.0
    r72 = float(price / closes[-bpd * 3] - 1) * 100 if len(closes) > bpd * 3 else 0.0
    wh = float(highs.max()) if len(highs) else price
    return {"rsi": round(rsi(closes, rsi_n), 1), "rsi_period": rsi_n,
            "trend_pct": round((fast / slow - 1) * 100, 2) if slow else 0.0,
            "ret_24h_pct": round(r24, 2), "ret_72h_pct": round(r72, 2),
            "drawdown_from_high_pct": round((price - wh) / wh * 100, 1) if wh else 0.0}


class TraderStore:
    def __init__(self, path: str) -> None:
        self.conn = sqlite3.connect(path)
        self.conn.execute("CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS trades(ts INT, side TEXT, usd REAL,"
                          " btc REAL, price REAL, coid TEXT)")
        self.conn.commit()

    def get(self, k, default=None):
        r = self.conn.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
        return json.loads(r[0]) if r else default

    def set(self, k, v):
        self.conn.execute("INSERT INTO meta(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE "
                          "SET v=excluded.v", (k, json.dumps(v)))
        self.conn.commit()

    def trade(self, side, usd, btc, price, coid):
        self.conn.execute("INSERT INTO trades VALUES(?,?,?,?,?,?)",
                          (int(time.time()), side, usd, btc, price, coid))
        self.conn.commit()

    def recent_trades(self, n=8):
        return self.conn.execute("SELECT ts,side,usd,btc,price FROM trades ORDER BY ts "
                                 "DESC LIMIT ?", (n,)).fetchall()


@dataclass
class Lot:
    """Shadow portfolio (benchmarks) with average-cost basis."""
    usdc: float = 0.0
    btc: float = 0.0
    cost: float = 0.0

    @property
    def avg(self): return self.cost / self.btc if self.btc > 1e-12 else 0.0
    def value(self, px): return self.usdc + self.btc * px

    def buy(self, usd, px):
        usd = min(usd, self.usdc)
        if usd < 1:
            return
        self.btc += usd * (1 - TAKER) / px
        self.cost += usd
        self.usdc -= usd

    def asdict(self): return self.__dict__


def _lot(d): return Lot(**d) if d else Lot()


class SatoshiTrader:
    def __init__(self, *, exchange: Exchange, store: TraderStore, notifier: Notifier,
                 llm_client, model: str, symbol: str, dca_days: int = 30,
                 cycle_hours: int = 4, news_enabled: bool = True) -> None:
        self.ex, self.store, self.notify = exchange, store, notifier
        self.client, self.model, self.symbol = llm_client, model, symbol
        self.dca_days, self.cycle_hours = dca_days, cycle_hours
        self.news_enabled = news_enabled
        self.rsi_n, self.weekly_ctx = load_technicals()

    # ----- live account balances = the pot -----
    def balances(self) -> tuple[float, float]:
        bal = self.ex.fetch_balance()
        usdc = float(bal.get("USDT", 0) or 0) + float(bal.get("USDC", 0) or 0)
        return usdc, float(bal.get("BTC", 0) or 0)

    # ----- LLM decision (portfolio-aware, in pot proportions) -----
    def decide(self, price, mom, usdc, btc, avg_entry, cur_frac, news=None) -> tuple[float, str, str]:
        feats = {"price": round(price, 2), "pot_usd": round(usdc + btc * price, 2),
                 "usdc": round(usdc, 2), "btc": round(btc, 6),
                 "btc_fraction_now": round(cur_frac, 2),
                 "avg_entry": round(avg_entry) if btc > 0 else None, **mom,
                 "recent_news": (news or {}).get("headlines", []),
                 "fear_greed": (news or {}).get("fear_greed")}
        try:
            sysmsg = TACTICAL_SYSTEM + (f"\n\nTHIS WEEK'S TUNED GUIDANCE: {self.weekly_ctx}"
                                        if self.weekly_ctx else "")
            r = self.client.chat.completions.create(
                model=self.model, temperature=0.0, timeout=30, max_tokens=400,
                messages=[{"role": "system", "content": sysmsg},
                          {"role": "user", "content": json.dumps(feats)}])
            t = r.choices[0].message.content or ""
            i, j = t.find("{"), t.rfind("}")
            d = json.loads(t[i:j + 1]) if 0 <= i < j else {}
            tgt = max(0.0, min(1.0, float(d.get("target_btc_fraction", cur_frac))))
            return tgt, str(d.get("stance", "?")), str(d.get("note", ""))[:80]
        except Exception as e:  # noqa: BLE001 - keep current allocation on any error
            return cur_frac, "error", f"{type(e).__name__}: hold"

    # ----- one 4h cycle -----
    def run_cycle(self, now: datetime | None = None) -> dict:
        now = now or datetime.now(timezone.utc)
        try:
            price = self.ex.get_price()
            usdc, btc = self.balances()
            ohlc = np.array(public_ohlcv(self.symbol, "4h", 200))
            closes, highs = ohlc[:, 3], ohlc[:, 1]
        except Exception as e:  # noqa: BLE001
            return {"error": redact(e)}
        pot = usdc + btc * price

        # first run: snapshot the starting pot + seed cost basis + benchmarks
        if not self.store.get("started"):
            self.store.set("cost_basis", btc * price)        # seed existing BTC at start px
            self.store.set("start", {"pot": pot, "price": price, "ts": now.isoformat()})
            self.store.set("hodl", {"usdc": pot, "btc": 0.0, "cost": 0.0})  # all-in at start
            hodl = _lot(self.store.get("hodl")); hodl.buy(pot, price)
            self.store.set("hodl", hodl.asdict())
            self.store.set("dca", {"usdc": pot, "btc": 0.0, "cost": 0.0})
            self.store.set("dca_slice", pot / max(1, int(self.dca_days * 24 / self.cycle_hours)))
            self.store.set("started", True)
            self.notify.send(f"🚀 satoshi-stacker started — pot ${pot:,.0f} "
                             f"({btc:.6f} BTC + ${usdc:,.0f} USDC) @ ${price:,.0f}")

        cost_basis = float(self.store.get("cost_basis", btc * price))
        cur = (btc * price) / pot if pot > 0 else 0.0
        avg_entry = cost_basis / btc if btc > 1e-12 else 0.0

        mom = momentum(closes, highs, price, self.rsi_n)
        news = live_news() if self.news_enabled else {}
        target, stance, note = self.decide(price, mom, usdc, btc, avg_entry, cur, news)

        # rebalance the REAL pot toward the target fraction
        action = "hold"
        if target - cur > REBAL_BAND:
            spend = min((target - cur) * pot, usdc)
            if spend >= 5:
                self._order("buy", spend, price, now)
                cost_basis += spend
                action = f"BUY ${spend:,.0f}"
        elif cur - target > REBAL_BAND:
            sell_val = min((cur - target) * pot, btc * price)
            if sell_val >= 5:
                self._order("sell", sell_val, price, now)
                cost_basis *= max(0.0, 1 - (sell_val / price) / btc)  # avg-cost: remove at avg
                action = f"SELL ${sell_val:,.0f}"
        self.store.set("cost_basis", cost_basis)

        # DCA benchmark deploys its per-cycle slice of the starting pot
        dca = _lot(self.store.get("dca"))
        slice_ = float(self.store.get("dca_slice", 0) or 0)
        if dca.usdc > 0 and slice_ > 0:
            dca.buy(min(slice_, dca.usdc), price)
            self.store.set("dca", dca.asdict())

        self._maybe_reports(now, price)
        return {"ts": now.isoformat(), "price": price, "pot": round(pot),
                "btc_frac": round(cur, 2), "stance": stance, "target": target,
                "action": action, "avg_entry": round(avg_entry), "note": note}

    def _order(self, side, usd, price, now):
        coid = f"ss-{side}-{int(now.timestamp())}"
        try:
            if side == "buy":
                self.ex.create_market_buy_quote(usd, coid)
            else:
                self.ex.create_market_sell(usd / price, coid)
            self.store.trade(side, usd, usd / price, price, coid)
        except Exception as e:  # noqa: BLE001
            self.store.set("last_error", redact(e))

    # ----- reports -----
    def _snapshot(self, price):
        usdc, btc = self.balances()
        pot = usdc + btc * price
        cb = float(self.store.get("cost_basis", btc * price))
        return usdc, btc, pot, (cb / btc if btc > 1e-12 else 0.0)

    def daily_report(self, price=None):
        price = price or self.ex.get_price()
        usdc, btc, pot, avg = self._snapshot(price)
        hodl, dca = _lot(self.store.get("hodl")), _lot(self.store.get("dca"))
        ae = f"${avg:,.0f}" if btc > 0 else "n/a"
        beat = (pot / price) >= hodl.value(price) / price and (pot / price) >= dca.value(price) / price
        self.notify.send(
            f"📅 *Daily — satoshi-stacker* (BTC ${price:,.0f})\n"
            f"🤖 BOT: {btc:.6f} BTC  avg {ae}  +${usdc:,.0f} USDC  = ${pot:,.0f} "
            f"({(pot/price):.6f} sats-equiv)\n"
            f"   DCA:  {dca.btc:.6f} BTC avg ${dca.avg:,.0f}  (${dca.value(price):,.0f})\n"
            f"   HODL: {hodl.btc:.6f} BTC avg ${hodl.avg:,.0f}  (${hodl.value(price):,.0f})\n"
            f"goal = most sats / lowest avg entry. "
            f"{'✅ beating both' if beat else '⚠️ behind a benchmark'}")

    def weekly_report(self, price=None):
        price = price or self.ex.get_price()
        usdc, btc, pot, avg = self._snapshot(price)
        hodl, dca = _lot(self.store.get("hodl")), _lot(self.store.get("dca"))
        rsi_n, ctx = load_technicals()
        tr = " ".join(f"{r[1][0].upper()}${r[2]:.0f}" for r in self.store.recent_trades(6)) or "none"
        ae = f"${avg:,.0f}" if btc > 0 else "n/a"
        vib = pot / price
        self.notify.send(
            f"🗓️ *Weekly — satoshi-stacker* (BTC ${price:,.0f})\n"
            f"🤖 BOT: {btc:.6f} BTC @ avg {ae}, ${usdc:,.0f} cash "
            f"({vib / max(hodl.value(price) / price, 1e-9) * 100 - 100:+.1f}% sats-equiv vs HODL)\n"
            f"   DCA:  {dca.btc:.6f} BTC avg ${dca.avg:,.0f}  (${dca.value(price):,.0f})\n"
            f"   HODL: {hodl.btc:.6f} BTC avg ${hodl.avg:,.0f}  (${hodl.value(price):,.0f})\n"
            f"pot ${pot:,.0f}, BTC {100*btc*price/max(pot,1):.0f}% / cash {100*usdc/max(pot,1):.0f}%\n"
            f"recent trades: {tr}\n"
            f"🔧 self-tune: RSI period {rsi_n} — {ctx[:130]}")

    def _maybe_reports(self, now, price):
        today = now.strftime("%Y-%m-%d")
        if self.store.get("last_daily") != today:
            self.daily_report(price); self.store.set("last_daily", today)
        wk = now.strftime("%Y-W%U")
        if self.store.get("last_weekly") != wk:
            self.weekly_report(price); self.store.set("last_weekly", wk)
