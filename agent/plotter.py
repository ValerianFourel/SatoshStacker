"""Render BTC chart PNGs (price + the leading indicators) for Telegram.

The "leading indicators" are the backtest winners from signal_tuner (the tuned
best top/bottom callers) — i.e. the ones that actually called tops/bottoms best,
not a guess. matplotlib runs headless (Agg); fail-safe (returns None on any error
so a chart problem never blocks the alert/answer).
"""
from __future__ import annotations

import io
import logging

log = logging.getLogger("satoshistacker.plotter")

# which families/indicators get reference bands drawn
_BANDS = {"rsi_14": (70, 30), "rsi_21": (70, 30), "stoch_rsi_14": (80, 20),
          "stoch_14": (80, 20), "mfi_14": (80, 20), "williams_14": (80, 20),
          "bb_pctb_20": (100, 0)}


def render_chart(klines: list, *, title: str, panels: list, marks: dict | None = None) -> bytes | None:
    """klines rows [ts,o,h,l,c,v]; panels = [(label, np.array, ref_bands_or_None)].
    Returns PNG bytes (or None on failure)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
        import numpy as np
        from datetime import datetime, timezone

        a = np.array(klines, dtype=float)
        c = a[:, 4]
        x = [datetime.fromtimestamp(ts / 1000, tz=timezone.utc) for ts in a[:, 0]]
        n = 1 + len(panels)
        fig, axes = plt.subplots(n, 1, figsize=(9, 2.0 + 1.5 * n), sharex=True,
                                 gridspec_kw={"height_ratios": [2.4] + [1] * len(panels),
                                              "hspace": 0.12})
        if n == 1:
            axes = [axes]
        # price
        ax = axes[0]
        ax.plot(x, c, color="#f7931a", lw=1.4)
        ax.fill_between(x, c, c.min(), color="#f7931a", alpha=0.06)
        ax.set_title(title, fontsize=11, loc="left")
        ax.grid(alpha=0.2)
        ax.set_ylabel("price")
        if marks:
            for lbl, (px, color) in marks.items():
                ax.axhline(px, color=color, ls="--", lw=0.8, alpha=0.7)
        # indicator panels
        for i, (label, series, bands) in enumerate(panels):
            axp = axes[i + 1]
            axp.plot(x, series, lw=1.1, color="#2b6cb0")
            axp.set_ylabel(label, fontsize=9)
            axp.grid(alpha=0.2)
            if bands:
                hi, lo = bands
                axp.axhline(hi, color="#d9534f", ls=":", lw=0.8, alpha=0.7)
                axp.axhline(lo, color="#5cb85c", ls=":", lw=0.8, alpha=0.7)
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %Hh"))
        fig.autofmt_xdate(rotation=0, ha="center")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return buf.getvalue()
    except Exception as e:  # noqa: BLE001 - a chart failure must never block a message
        log.warning("chart render failed: %s", e)
        return None


def build_btc_chart(cfg, tuned: dict | None, *, snapshot: dict | None = None,
                    klines_fn=None) -> tuple[bytes | None, str]:
    """Fetch candles, plot price + the tuned leading indicators, return (png, caption)."""
    from .signal_tuner import _pretty, compute_one
    import numpy as np
    if klines_fn is None:
        from .exchange import public_klines
        klines_fn = lambda tf, lim: public_klines(cfg.symbol, tf, lim)  # noqa: E731
    tf = cfg.trend_tf
    kl = klines_fn(tf, 180)
    a = np.array(kl, dtype=float)
    o, h, l, c, v = a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5]

    panels, names = [], []
    for side in ("best_top", "best_bottom"):
        sig = (tuned or {}).get(side)
        if sig and sig["name"] not in names:
            names.append(sig["name"])
            series = compute_one(sig["name"], o, h, l, c, v)
            panels.append((_pretty(sig["name"]), series, _BANDS.get(sig["name"])))
    if not panels:  # no tuned signals yet -> default to RSI(14)
        panels.append(("RSI(14)", compute_one("rsi_14", o, h, l, c, v), (70, 30)))
        names.append("rsi_14")

    marks = None
    if snapshot:
        t = snapshot.get("technicals", {})
        marks = {"24h high": (t.get("high_24h"), "#d9534f"),
                 "24h low": (t.get("low_24h"), "#5cb85c")}
        marks = {k: vv for k, vv in marks.items() if vv[0]}
    title = f"BTC {tf} — price + leading indicators: {', '.join(_pretty(n) for n in names)}"
    png = render_chart(kl, title=title, panels=panels, marks=marks)
    cap = "📈 *BTC* — " + " · ".join(_pretty(n) for n in names) + f"  (`{tf}`, backtest-leading)"
    return png, cap
