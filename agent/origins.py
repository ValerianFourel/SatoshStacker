"""The `/origins` control widget — an inline-keyboard panel to tune, live, which
signals may ping you, how many must agree (confluence), and how often (cadence).

Pure here (text + keyboard + a prefs mutation from a button tap); the Telegram I/O
(send / edit / answerCallbackQuery) lives in ``telegram_listener``. State is the same
``watch_prefs.json`` the monitor already reads on its next scan, so taps apply within
~15s with no restart.
"""
from __future__ import annotations

from .sensitivity import (ALARM_COOLDOWN_RANGE_S, CADENCE_RANGE_S, CONFLUENCE_RANGE, ORDER,
                          _clamp)

# canonical Signal.name -> (emoji, short label). These are the "origins" of proactive
# alerts; toggling one adds/removes it from prefs['disabled'].
ORIGINS = [
    ("peak", "📈", "Top / peak"),
    ("bottom", "📉", "Bottom"),
    ("volume_spike", "📊", "Volume spike"),
    ("price_spike_up", "⚡", "Spike up"),
    ("price_spike_down", "⚡", "Spike down"),
    ("book_imbalance", "📖", "Book imbalance"),
    ("funding_extreme", "💸", "Funding"),
    ("oi_spike", "🔭", "Open interest"),
    ("long_short_extreme", "⚖️", "Long/short"),
]
_LABEL = {c: l for c, _e, l in ORIGINS}

CADENCE_STEP_S = 300       # ± 5 min per tap
_KEYS = ("sensitivity", "muted", "overrides", "disabled", "confluence", "cadence",
         "alarm_cooldown")

# callback_data scheme (all < 64 bytes): og:<coin>:<action>[:<arg>]
P = "og"


def _preset_label(level: str) -> str:
    return {"low": "🟢 quiet", "normal": "🟡 balanced", "high": "🔴 chatty"}.get(level, level)


def widget_text(prefs: dict, coin: str = "btc") -> str:
    """The panel header — current confluence / cadence / preset / mute, in plain words.
    The per-origin on/off state is shown on the buttons themselves."""
    conf = max(1, int(prefs.get("confluence", 2)))
    cad_m = int(prefs.get("cadence", 1800)) // 60
    acd_m = int(prefs.get("alarm_cooldown", 900)) // 60
    level = prefs.get("sensitivity", "low")
    muted = bool(prefs.get("muted", False))
    disabled = prefs.get("disabled") or []
    conf_line = (f"need *≥{conf}* signals out-of-norm at once"
                 if conf > 1 else "*any single* signal can ping (off)")
    lines = [
        f"🛰️ *{coin.upper()} — update controls* — what pings you, and how often",
        f"🧩 Confluence: {conf_line}",
        f"⏱️ Cadence: at most *1 ping / {cad_m}m* (proactive)",
        f"🔔 Alarm cooldown: " + (f"*{acd_m}m* between re-fires of a trigger" if acd_m else "*off*"),
        f"🎚️ Sensitivity bars: {_preset_label(level)}   "
        + ("🔇 *muted*" if muted else "🔔 live"),
    ]
    on = len(ORIGINS) - len([d for d in disabled if d in _LABEL])
    lines.append(f"\n_{on}/{len(ORIGINS)} signals active — tap one to silence/enable it._")
    return "\n".join(lines)


def keyboard(prefs: dict, coin: str = "btc") -> dict:
    """Telegram inline_keyboard dict reflecting the current prefs (callback data carries coin)."""
    conf = max(1, int(prefs.get("confluence", 2)))
    cad_m = int(prefs.get("cadence", 1800)) // 60
    acd_m = int(prefs.get("alarm_cooldown", 900)) // 60
    level = prefs.get("sensitivity", "low")
    muted = bool(prefs.get("muted", False))
    disabled = set(prefs.get("disabled") or [])

    def btn(text, data):
        return {"text": text, "callback_data": data}

    def cb(s):                                # og:<coin>:<action...>
        return f"{P}:{coin}:{s}"

    rows = [
        [btn("➖", cb("c:-")), btn(f"🧩 confluence ≥{conf}", cb("r")), btn("➕", cb("c:+"))],
        [btn("➖", cb("d:-")), btn(f"⏱️ cadence {cad_m}m", cb("r")), btn("➕", cb("d:+"))],
        [btn("➖", cb("a:-")), btn(f"🔔 alarm cooldown {acd_m}m", cb("r")), btn("➕", cb("a:+"))],
    ]
    # origin toggles, two per row
    pair = []
    for canon, emoji, label in ORIGINS:
        mark = "🚫" if canon in disabled else "✅"
        pair.append(btn(f"{mark} {emoji} {label}", cb(f"t:{canon}")))
        if len(pair) == 2:
            rows.append(pair)
            pair = []
    if pair:
        rows.append(pair)
    rows.append([btn(f"🎚️ {_preset_label(level)}", cb("p")),
                 btn("🔇 muted — unmute" if muted else "🔔 live — mute", cb("m"))])
    rows.append([btn("✖️ close", cb("x"))])
    return {"inline_keyboard": rows}


def apply_callback(data: str, prefs: dict) -> tuple[dict | None, str]:
    """Apply a button tap (og:<coin>:<action>[:<arg>]) to a prefs dict. Returns
    (updated_prefs | None, toast). None => no change to persist (refresh/close/unknown).
    Coin routing is the caller's job; this only mutates the passed prefs."""
    p = dict(prefs)
    p["disabled"] = list(p.get("disabled") or [])
    p["confluence"] = max(1, int(p.get("confluence", 2)))
    p["cadence"] = int(p.get("cadence", 1800))
    p["alarm_cooldown"] = int(p.get("alarm_cooldown", 900))
    parts = (data or "").split(":")
    if len(parts) < 3 or parts[0] != P:
        return None, ""
    action = parts[2]                         # parts[1] is the coin (handled by the caller)
    if action == "t" and len(parts) >= 4:
        canon = parts[3]
        if canon not in _LABEL:
            return None, ""
        dis = set(p["disabled"])
        if canon in dis:
            dis.discard(canon)
            toast = f"✅ {_LABEL[canon]} on"
        else:
            dis.add(canon)
            toast = f"🚫 {_LABEL[canon]} silenced"
        p["disabled"] = sorted(dis)
        return p, toast
    if action == "c":
        p["confluence"] = _clamp(p["confluence"] + (1 if parts[-1] == "+" else -1),
                                 *CONFLUENCE_RANGE)
        msg = (f"≥{p['confluence']} must agree" if p["confluence"] > 1
               else "confluence off")
        return p, f"🧩 {msg}"
    if action == "d":
        p["cadence"] = _clamp(p["cadence"] + (CADENCE_STEP_S if parts[-1] == "+"
                                              else -CADENCE_STEP_S), *CADENCE_RANGE_S)
        return p, f"⏱️ 1 ping / {p['cadence'] // 60}m"
    if action == "a":
        p["alarm_cooldown"] = _clamp(p["alarm_cooldown"] + (CADENCE_STEP_S if parts[-1] == "+"
                                     else -CADENCE_STEP_S), *ALARM_COOLDOWN_RANGE_S)
        m = p["alarm_cooldown"] // 60
        return p, (f"🔔 alarm cooldown {m}m" if m else "🔔 alarm cooldown off")
    if action == "p":
        i = ORDER.index(p["sensitivity"]) if p.get("sensitivity") in ORDER else 0
        p["sensitivity"] = ORDER[(i + 1) % len(ORDER)]
        return p, f"🎚️ {_preset_label(p['sensitivity'])}"
    if action == "m":
        p["muted"] = not bool(p.get("muted", False))
        return p, ("🔇 muted" if p["muted"] else "🔔 live")
    return None, ""           # og:r (refresh), og:x (close), unknown -> no persist
