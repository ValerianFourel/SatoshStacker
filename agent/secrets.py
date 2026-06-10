"""Secret redaction for logs, journal entries, and notifications.

Exception strings from ccxt/HTTP layers can embed credentials (Binance sends the
API key as the ``X-MBX-APIKEY`` header on every request, and request URLs/headers
can appear in error text). Nothing that reaches the SQLite journal, the rotating
log, or stdout may contain a raw secret — so every ``str(e)`` / log message is
routed through ``redact()``.
"""
from __future__ import annotations

import logging
import os
import re

# env vars whose *values* are secrets and must be scrubbed verbatim if present
_SECRET_ENV_VARS = (
    "BINANCE_API_KEY", "BINANCE_API_SECRET",
    "BINANCE_TESTNET_API_KEY", "BINANCE_TESTNET_API_SECRET",
    "LLM_API_KEY", "TELEGRAM_BOT_TOKEN",
)

_REDACTED = "***REDACTED***"

# key=value / "key":"value" / header style credential patterns. The delimiter
# group allows optional surrounding quotes so quoted-JSON values are masked too.
_PATTERNS = (
    re.compile(r"(api[_-]?key|signature|secret|token|password|apikey|"
               r"x-mbx-apikey)(\s*[\"']?\s*[=:]\s*[\"']?)([^\s&\"'}]+)", re.I),
    # header form without a =/: delimiter, e.g. "X-MBX-APIKEY <value>"
    re.compile(r"(x-mbx-apikey)(\s+)([A-Za-z0-9._\-]{6,})", re.I),
    re.compile(r"(Bearer)(\s+)([A-Za-z0-9._\-]+)", re.I),
)
# Telegram bot-token URL shape (/bot<id>:<token>/) has no key= delimiter
_TELEGRAM_URL = re.compile(r"(/bot)(\d+:[A-Za-z0-9_\-]{6,})")


def clean_secret(value: str | None) -> str:
    """Normalize a secret read from env. Strips whitespace and treats a value that
    is actually a leaked inline comment (starts with ``#``) as empty — guards the
    python-dotenv gotcha where ``KEY=   # comment`` yields the comment as the value,
    so the agent never runs with a garbage credential."""
    if not value:
        return ""
    v = value.strip()
    return "" if v.startswith("#") else v


def redact(text: object) -> str:
    """Return ``text`` as a string with any credentials masked."""
    s = "" if text is None else str(text)
    for var in _SECRET_ENV_VARS:
        val = os.getenv(var)
        if val and len(val) >= 6:
            s = s.replace(val, _REDACTED)
    s = _TELEGRAM_URL.sub(lambda m: m.group(1) + _REDACTED, s)
    for pat in _PATTERNS:
        s = pat.sub(lambda m: f"{m.group(1)}{m.group(2)}{_REDACTED}", s)
    return s


class RedactingFilter(logging.Filter):
    """Logging filter that scrubs secrets from every record's message."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        try:
            record.msg = redact(record.getMessage())
            record.args = ()
        except Exception:  # noqa: BLE001 - never let logging break the loop
            pass
        return True


class RedactingFormatter(logging.Formatter):
    """Wraps another formatter and redacts its full rendered output — including
    exception tracebacks (exc_info), which a record filter cannot reach."""

    def __init__(self, base: logging.Formatter) -> None:
        super().__init__()
        self._base = base

    def format(self, record: logging.LogRecord) -> str:
        try:
            return redact(self._base.format(record))
        except Exception:  # noqa: BLE001
            return redact(record.getMessage())


def install_log_redaction() -> None:
    """Scrub secrets on the root logger: a message filter PLUS a format-level
    wrapper so even ``log.exception`` tracebacks are redacted."""
    root = logging.getLogger()
    filt = RedactingFilter()
    root.addFilter(filt)
    for h in root.handlers:
        h.addFilter(filt)
        base = h.formatter or logging.Formatter()
        if not isinstance(base, RedactingFormatter):
            h.setFormatter(RedactingFormatter(base))
