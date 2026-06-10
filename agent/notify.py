"""Telegram alerts + daily accumulation summary, with a console fallback.

`requests` is imported lazily and any send failure is swallowed (notifications
must never crash or block the trading loop). With no bot token configured,
messages print to stdout so dry-run is fully observable.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("satoshistacker.notify")


class Notifier:
    def __init__(self, token: str | None = None, chat_id: str | None = None) -> None:
        from .secrets import clean_secret
        self.token = clean_secret(token or os.getenv("TELEGRAM_BOT_TOKEN"))
        self.chat_id = clean_secret(chat_id or os.getenv("TELEGRAM_CHAT_ID"))

    def send(self, text: str) -> None:
        if not (self.token and self.chat_id):
            log.info("[notify] %s", text)
            print(f"[notify] {text}")
            return
        try:
            import requests
            requests.post(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                json={"chat_id": self.chat_id, "text": text,
                      "parse_mode": "Markdown", "disable_web_page_preview": True},
                timeout=10)
        except Exception as e:  # noqa: BLE001 - never let notifications break the loop
            log.warning("telegram send failed: %s", e)
            print(f"[notify-fallback] {text}")

    def heartbeat(self, mode: str, price: float) -> None:
        self.send(f"💓 SatoshiStacker alive [{mode}] BTC=${price:,.0f}")

    def daily_summary(self, *, mode: str, deployed: float, budget: float,
                      btc: float, avg_cost: float, price: float,
                      remaining: float) -> None:
        ac = "n/a" if btc <= 0 else f"${avg_cost:,.0f}"
        val = btc * price + remaining
        self.send(
            f"📊 *Daily accumulation* [{mode}]\n"
            f"BTC stacked: `{btc:.8f}`\n"
            f"Avg cost: `{ac}`\n"
            f"USDC deployed: `${deployed:,.0f}` / `${budget:,.0f}` "
            f"({100*deployed/max(budget,1):.0f}%)\n"
            f"USDC remaining: `${remaining:,.0f}`\n"
            f"BTC price: `${price:,.0f}`  |  portfolio ≈ `${val:,.0f}`")
