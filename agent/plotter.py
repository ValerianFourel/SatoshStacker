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
_FAMILY_BANDS = {"rsi": (70, 30), "mfi": (80, 20), "stoch": (80, 20), "stoch_rsi": (80, 20),
                 "williams": (80, 20), "bb_pctb": (100, 0), "cci": (100, -100)}
TF_CHOICES = ("5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d")


def _bands_for(name: str):
    """Reference bands for an oscillator panel — exact name, else by family (so swept names
    like rsi_15 / mfi_21 still get their 70/30 / 80/20 guides)."""
    if name in _BANDS:
        return _BANDS[name]
    import re
    m = re.fullmatch(r"([a-z_]+?)_\d+", name or "")
    return _FAMILY_BANDS.get(m.group(1)) if m else None


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


def render_series_chart(title: str, price_klines: list, panels: list) -> bytes | None:
    """Price (top) + each panel as its OWN time-series (own x-axis). panels =
    [(label, [(ms, value), ...], hline_or_None)]. Fail-safe -> None."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
        import numpy as np
        from datetime import datetime, timezone

        pa = np.array(price_klines, dtype=float)
        px = [datetime.fromtimestamp(t / 1000, tz=timezone.utc) for t in pa[:, 0]]
        n = 1 + len(panels)
        fig, axes = plt.subplots(n, 1, figsize=(9, 2.0 + 1.4 * n),
                                 gridspec_kw={"height_ratios": [2.2] + [1] * len(panels),
                                              "hspace": 0.18})
        if n == 1:
            axes = [axes]
        axes[0].plot(px, pa[:, 4], color="#f7931a", lw=1.4)
        axes[0].set_title(title, fontsize=11, loc="left")
        axes[0].grid(alpha=0.2)
        axes[0].set_ylabel("price")
        for i, (label, series, hline) in enumerate(panels):
            ax = axes[i + 1]
            if series:
                t = [datetime.fromtimestamp(ms / 1000, tz=timezone.utc) for ms, _ in series]
                ax.plot(t, [v for _, v in series], lw=1.1, color="#2b6cb0")
            else:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes,
                        fontsize=8, color="#999")
            ax.set_ylabel(label, fontsize=9)
            ax.grid(alpha=0.2)
            if hline is not None:
                ax.axhline(hline, color="#888", ls="--", lw=0.8)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %Hh"))
        fig.autofmt_xdate(rotation=0, ha="center")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return buf.getvalue()
    except Exception as e:  # noqa: BLE001
        log.warning("derivs chart render failed: %s", e)
        return None


def build_derivs_chart(cfg, *, klines_fn=None, derivs_fn=None,
                       timeframe=None) -> tuple[bytes | None, str]:
    """Crypto-native derivatives chart (Binance perp, no key): price + funding-rate
    history + open interest + long/short ratio + taker buy/sell flow. ``timeframe`` picks
    the candle/period window (defaults to ``cfg.trend_tf``)."""
    from .exchange import public_derivs_history, public_klines
    period = timeframe if timeframe in TF_CHOICES else (
        cfg.trend_tf if cfg.trend_tf in TF_CHOICES else "1h")
    if klines_fn is None:
        klines_fn = lambda tf, lim: public_klines(cfg.symbol, tf, lim)  # noqa: E731
    if derivs_fn is None:
        derivs_fn = lambda: public_derivs_history(cfg.symbol, period=period, limit=96)  # noqa: E731
    kl = klines_fn(period, 96)
    d = derivs_fn()
    panels = [
        ("funding %/8h", [(t, v * 100) for t, v in d.get("funding_rate", [])], 0.0),
        ("open interest", d.get("open_interest", []), None),
        ("long/short", d.get("long_short_ratio", []), 1.0),
        ("taker buy/sell", d.get("taker_buy_sell", []), 1.0),
    ]
    png = render_series_chart(
        f"BTC derivatives ({period}) — funding · OI · long/short · taker flow", kl, panels)
    cap = ("📉 *BTC derivatives* — funding, open interest, long/short ratio, taker buy/sell "
           f"(`{period}`, Binance perp). _Liquidation map needs a Coinglass key._")
    return png, cap


def build_btc_chart(cfg, tuned: dict | None, *, snapshot: dict | None = None,
                    klines_fn=None, indicators=None, timeframe=None) -> tuple[bytes | None, str]:
    """Fetch candles and plot price + indicators on a CHOSEN timeframe, return (png, caption).

    ``timeframe`` (5m/15m/1h/4h/…) selects the candle window; defaults to ``cfg.trend_tf``.
    ``indicators`` (validated battery names) = the LLM's picks; if absent, fall back to the
    backtest-leading tuned top/bottom callers *for that timeframe* (per_timeframe winners),
    else the live winners, else RSI(14)."""
    from .signal_tuner import PERIODS, _pretty, compute_one
    import numpy as np
    if klines_fn is None:
        from .exchange import public_klines
        klines_fn = lambda tf, lim: public_klines(cfg.symbol, tf, lim)  # noqa: E731
    tf = timeframe if timeframe in TF_CHOICES else cfg.trend_tf
    kl = klines_fn(tf, 180)
    a = np.array(kl, dtype=float)
    o, h, l, c, v = a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5]

    # the tuned signals FOR this timeframe (from a retune's per_timeframe block) if present,
    # else the live-tf winners — so the chart shows the signal associated with the TF being plotted.
    tf_tuned = (((tuned or {}).get("per_timeframe", {}) or {}).get(tf)) or tuned or {}

    names: list[str] = []
    if indicators:                                       # LLM-picked
        names = [n for n in indicators if n in PERIODS][:3]
    if not names:                                        # backtest-leading (this TF's winners)
        for side in ("best_top", "best_bottom"):
            sig = tf_tuned.get(side)
            if sig and sig.get("name") and sig["name"] not in names:
                names.append(sig["name"])
    if not names:
        names = ["rsi_14"]
    names = names[:3]
    panels = [(_pretty(n), compute_one(n, o, h, l, c, v), _bands_for(n)) for n in names]

    marks = None
    if snapshot:
        t = snapshot.get("technicals", {})
        marks = {k: vv for k, vv in {"24h high": (t.get("high_24h"), "#d9534f"),
                                     "24h low": (t.get("low_24h"), "#5cb85c")}.items()
                 if vv[0]}
    src = "LLM-picked" if indicators else "backtest-leading"
    title = f"BTC {tf} — price + {', '.join(_pretty(n) for n in names)}"
    png = render_chart(kl, title=title, panels=panels, marks=marks)
    cap = "📈 *BTC* — " + " · ".join(_pretty(n) for n in names) + f"  (`{tf}`, {src})"
    return png, cap
