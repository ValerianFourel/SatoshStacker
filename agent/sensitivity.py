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


def resolve(level: str | None) -> dict:
    return PRESETS.get((level or "").lower(), PRESETS["low"])


def read_prefs(path: str, *, default_level: str = "low") -> dict:
    """{'sensitivity': <level>, 'muted': bool}. Missing/garbage -> safe defaults."""
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception:  # noqa: BLE001
        d = {}
    lvl = str(d.get("sensitivity", default_level)).lower()
    return {"sensitivity": lvl if lvl in PRESETS else default_level,
            "muted": bool(d.get("muted", False))}


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


def describe(level: str, muted: bool) -> str:
    p = resolve(level)
    bar = {"low": "🟢 quiet", "normal": "🟡 balanced", "high": "🔴 chatty"}.get(level, level)
    mute = "  ·  🔇 *MUTED* (no proactive alerts)" if muted else ""
    return (f"🎚 *Sensitivity:* {bar} (`{level}`){mute}\n"
            f"   fires when: RSI ≥{p['rsi_ob']:g}/≤{p['rsi_os']:g} · vol z≥{p['vol_z']:g} · "
            f"|imbalance|≥{p['imb']:g} · |funding|≥{p['fund']:g}% · OI≥{p['oi']:g}%\n"
            f"   cooldown {p['cooldown'] // 60}m · re-arms after {p['rearm'] // 60}m clear\n"
            f"   change: `/sensitivity low|normal|high`  ·  `/mute` · `/unmute`")
