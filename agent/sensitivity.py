"""Anomaly sensitivity presets + a tiny persisted prefs file the user can change live.

Less sensitive = stricter bars + longer cooldown + a longer re-arm window, so a reading
*parked near a threshold* doesn't keep re-firing. 'high' ≈ the original defaults; 'low'
(the default) only fires on genuinely stretched readings. Change live with /sensitivity.
"""
from __future__ import annotations

import json
import logging
import os

log = logging.getLogger("satoshistacker.sensitivity")

# least -> most alerts. Bigger bar / longer cooldown+rearm = quieter.
PRESETS = {
    "low": dict(rsi_ob=77.0, rsi_os=23.0, vol_z=4.5, ret_z=4.8, near=0.10, imb=0.86,
                fund=0.12, oi=15.0, ls_long=2.8, ls_short=0.36, cooldown=7200, rearm=900),
    "normal": dict(rsi_ob=74.0, rsi_os=26.0, vol_z=3.8, ret_z=4.0, near=0.18, imb=0.74,
                   fund=0.08, oi=11.0, ls_long=2.4, ls_short=0.45, cooldown=3600, rearm=600),
    "high": dict(rsi_ob=72.0, rsi_os=28.0, vol_z=3.0, ret_z=3.5, near=0.25, imb=0.60,
                 fund=0.05, oi=8.0, ls_long=2.0, ls_short=0.60, cooldown=1800, rearm=300),
}
ORDER = ("low", "normal", "high")          # quiet -> chatty

# confluence: how many signals must be out-of-norm AT ONCE to ping; cadence: min gap (s).
DEFAULT_CONFLUENCE = 2
DEFAULT_CADENCE_S = 1800
CONFLUENCE_RANGE = (1, 6)                    # widget clamp
CADENCE_RANGE_S = (300, 14_400)             # 5 min … 4 h widget clamp

# friendly name -> canonical threshold key (for `/sensitivity set <key> <value>`)
KEY_ALIAS = {
    "imb": "imb", "imbalance": "imb", "book": "imb",
    "fund": "fund", "funding": "fund",
    "vol": "vol_z", "volume": "vol_z", "vol_z": "vol_z",
    "ret": "ret_z", "return": "ret_z", "ret_z": "ret_z",
    "rsi_ob": "rsi_ob", "overbought": "rsi_ob", "rsi_top": "rsi_ob", "top": "rsi_ob",
    "rsi_os": "rsi_os", "oversold": "rsi_os", "rsi_bottom": "rsi_os", "bottom": "rsi_os",
    "oi": "oi", "near": "near", "ls_long": "ls_long", "ls_short": "ls_short",
    "cooldown": "cooldown", "rearm": "rearm",
}
MIN_KEYS = {"cooldown", "rearm"}           # set/displayed in minutes, stored in seconds

# friendly name -> canonical Signal.name (for `/sensitivity off <signal>`)
SIGNAL_ALIAS = {
    "imbalance": "book_imbalance", "book": "book_imbalance", "book_imbalance": "book_imbalance",
    "funding": "funding_extreme", "funding_extreme": "funding_extreme",
    "oi": "oi_spike", "oi_spike": "oi_spike",
    "long_short": "long_short_extreme", "ls": "long_short_extreme",
    "long_short_extreme": "long_short_extreme",
    "volume": "volume_spike", "volume_spike": "volume_spike",
    "peak": "peak", "top": "peak", "bottom": "bottom",
    "spike_up": "price_spike_up", "price_spike_up": "price_spike_up",
    "spike_down": "price_spike_down", "price_spike_down": "price_spike_down",
}


def resolve(level: str | None) -> dict:
    return PRESETS.get((level or "").lower(), PRESETS["low"])


def effective(level: str | None, overrides: dict | None) -> dict:
    """Preset bars with the user's manual per-key overrides merged on top."""
    base = dict(resolve(level))
    for k, v in (overrides or {}).items():
        if k in base:
            base[k] = v
    return base


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def read_prefs(path: str, *, default_level: str = "low",
               default_confluence: int = DEFAULT_CONFLUENCE,
               default_cadence: int = DEFAULT_CADENCE_S) -> dict:
    """{'sensitivity','muted','overrides','disabled','confluence','cadence'}.
    Missing/garbage -> safe defaults."""
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception:  # noqa: BLE001
        d = {}
    lvl = str(d.get("sensitivity", default_level)).lower()
    ov = {}
    for k, v in (d.get("overrides") or {}).items():
        if k in PRESETS["low"] and isinstance(v, (int, float)):
            ov[k] = float(v)
    dis = [s for s in (d.get("disabled") or []) if isinstance(s, str)]
    try:
        conf = int(d.get("confluence", default_confluence))
    except (TypeError, ValueError):
        conf = default_confluence
    try:
        cad = int(d.get("cadence", default_cadence))
    except (TypeError, ValueError):
        cad = default_cadence
    return {"sensitivity": lvl if lvl in PRESETS else default_level,
            "muted": bool(d.get("muted", False)), "overrides": ov, "disabled": dis,
            "confluence": _clamp(conf, *CONFLUENCE_RANGE),
            "cadence": _clamp(cad, *CADENCE_RANGE_S)}


def write_prefs(path: str, **changes) -> dict:
    d = read_prefs(path)
    if "sensitivity" in changes:
        changes["sensitivity"] = str(changes["sensitivity"]).lower()
    d.update(changes)
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(d, f)
        os.replace(tmp, path)
    except Exception as e:  # noqa: BLE001
        log.warning("could not persist prefs: %s", e)
    return d


def describe(level: str, muted: bool, overrides: dict | None = None,
             disabled: list | None = None, confluence: int | None = None,
             cadence: int | None = None) -> str:
    p = effective(level, overrides)
    overrides, disabled = overrides or {}, disabled or []
    bar = {"low": "🟢 quiet", "normal": "🟡 balanced", "high": "🔴 chatty"}.get(level, level)
    mute = "  ·  🔇 *MUTED* (no proactive alerts)" if muted else ""
    star = lambda k: "*" if k in overrides else ""   # mark a manually-set bar
    out = [f"🎚 *Sensitivity:* {bar} (`{level}`){mute}",
           (f"   fires when: RSI ≥{p['rsi_ob']:g}{star('rsi_ob')}/≤{p['rsi_os']:g}{star('rsi_os')} · "
            f"vol z≥{p['vol_z']:g}{star('vol_z')} · |imbalance|≥{p['imb']:g}{star('imb')} · "
            f"|funding|≥{p['fund']:g}{star('fund')}% · OI≥{p['oi']:g}{star('oi')}%"),
           f"   cooldown {int(p['cooldown']) // 60}m · re-arms after {int(p['rearm']) // 60}m clear"]
    if confluence is not None:
        c = max(1, int(confluence))
        cad = int(cadence if cadence is not None else DEFAULT_CADENCE_S) // 60
        out.append(f"   🧩 *confluence:* need ≥{c} signal{'s' if c != 1 else ''} at once · "
                   f"≤1 ping / {cad}m" if c > 1 else
                   f"   🧩 *confluence:* off (any single signal can ping) · ≤1 ping / {cad}m")
    if disabled:
        out.append("   🚫 *off:* " + ", ".join(disabled))
    if overrides:
        out.append("   ✋ *manual* (★): " + ", ".join(f"{k}={v:g}" for k, v in overrides.items()))
    out.append("   `/origins` to tune which signals ping & how many must agree · "
               "`/sensitivity low|normal|high` · `set <key> <val>` · `/mute`")
    return "\n".join(out)
