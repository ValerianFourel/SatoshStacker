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

    def _chat_ids(self) -> list[str]:
        """One or more chat ids (comma-separated) — broadcast to each."""
        return [c.strip() for c in (self.chat_id or "").split(",") if c.strip()]

    def send(self, text: str, reply_markup: dict | None = None) -> None:
        from .secrets import redact
        text = redact(text)  # scrub any credential before it reaches a log/stdout/Telegram
        chats = self._chat_ids()
        if not (self.token and chats):
            log.info("[notify] %s", text)
            print(f"[notify] {text}")
            return
        for cid in chats:
            try:
                import requests
                body = {"chat_id": cid, "text": text,
                        "parse_mode": "Markdown", "disable_web_page_preview": True}
                if reply_markup:                      # inline keyboard (e.g. an alarm's ✓ Seen button)
                    body["reply_markup"] = reply_markup
                requests.post(
                    f"https://api.telegram.org/bot{self.token}/sendMessage",
                    json=body, timeout=10)
            except Exception as e:  # noqa: BLE001 - never let notifications break the loop
                log.warning("telegram send failed: %s", e)
                print(f"[notify-fallback] {text}")

    def send_photo(self, png: bytes | None, caption: str = "") -> None:
        """Send a PNG chart with a caption. Falls back to a text send if no image /
        no Telegram. Never raises."""
        from .secrets import redact
        if not png:
            if caption:
                self.send(caption)
            return
        caption = redact(caption)[:1024]
        chats = self._chat_ids()
        if not (self.token and chats):
            log.info("[notify-photo] %s", caption)
            print(f"[notify-photo] {caption} (<{len(png)} bytes png>)")
            return
        for cid in chats:
            try:
                import requests
                requests.post(
                    f"https://api.telegram.org/bot{self.token}/sendPhoto",
                    data={"chat_id": cid, "caption": caption, "parse_mode": "Markdown"},
                    files={"photo": ("chart.png", png, "image/png")}, timeout=20)
            except Exception as e:  # noqa: BLE001
                log.warning("telegram sendPhoto failed: %s", e)

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
