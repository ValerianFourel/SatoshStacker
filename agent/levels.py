"""Sat-stacking levels: a *good reentry (buy) zone* and a *sell/rotate zone*, derived
from market STRUCTURE (24h/7d range, ATR, order-book walls) — not freelance advice.

Goal framing: stacking satoshis. Buying lower = more sats per dollar; selling into
strength and rebuying lower grows the stack. These are levels of interest only; the
watch bot is read-only and never places orders.
"""
from __future__ import annotations


def suggest_levels(m: dict, *, target_pct: float | None = None, fee_pct: float = 0.1,
                   anchor: str = "buy", anchor_price: float | None = None) -> dict | None:
    """Reentry/sell zones from market structure. If ``target_pct`` is given, also a
    concrete buy/sell pair scaled so the round trip nets ~target% AFTER Binance fees
    (``fee_pct`` per leg) + the live spread.

    ``anchor="buy"`` (default) fixes the buy at the nearest support and solves for the
    sell; ``anchor="sell"`` fixes the sell at the nearest resistance and solves for the
    buy. ``anchor_price`` overrides the anchored side with an explicit level."""
    t = m.get("technicals", {})
    b = m.get("order_book", {})
    price = m.get("price")
    if not price:
        return None
    atr = (t.get("atr_pct") or 0.5) / 100 * price            # ATR in dollars
    low24, high24 = t.get("low_24h"), t.get("high_24h")
    low7, high7 = t.get("low_7d"), t.get("high_7d")
    bid_wall = (b.get("top_bid_wall") or [None])[0]
    ask_wall = (b.get("top_ask_wall") or [None])[0]

    supports = sorted({x for x in (low24, low7, bid_wall, price - atr, price - 2 * atr)
                       if x and x < price}, reverse=True)
    resist = sorted({x for x in (high24, high7, ask_wall, price + atr, price + 2 * atr)
                     if x and x > price})
    reentry_hi = supports[0] if supports else price * 0.99
    reentry_lo = supports[-1] if len(supports) > 1 else price - 2 * atr
    sell_lo = resist[0] if resist else price * 1.01
    sell_hi = resist[-1] if len(resist) > 1 else price + 2 * atr
    out = {
        "price": price,
        "reentry": (round(min(reentry_lo, reentry_hi)), round(max(reentry_lo, reentry_hi))),
        "sell": (round(min(sell_lo, sell_hi)), round(max(sell_lo, sell_hi))),
        "atr": round(atr),
        "support": {"24h_low": low24, "7d_low": low7, "bid_wall": bid_wall},
        "resistance": {"24h_high": high24, "7d_high": high7, "ask_wall": ask_wall},
    }
    if target_pct is not None:
        spread_pct = (b.get("spread_bps") or 1.0) / 100.0    # cost of crossing the book
        cost_pct = 2 * fee_pct + spread_pct                  # 2 legs of fee + the spread
        gross_pct = target_pct + cost_pct                    # required buy->sell move
        ratio = 1 + gross_pct / 100
        if anchor == "sell":                                 # fix the sell, solve for the buy
            sell = anchor_price or sell_lo or price * 1.01   # nearest resistance by default
            buy = sell / ratio
        else:                                                # fix the buy, solve for the sell
            buy = anchor_price or reentry_hi or price        # nearest support by default
            sell = buy * ratio
        out["target"] = {
            "anchor": "sell" if anchor == "sell" else "buy",
            "want_pct": round(target_pct, 3), "fee_pct": fee_pct,
            "spread_pct": round(spread_pct, 3), "cost_pct": round(cost_pct, 3),
            "gross_pct": round(gross_pct, 3),
            "buy": round(buy), "sell": round(sell),
        }
    return out


def _named(d: dict) -> str:
    return " · ".join(f"{k.replace('_', ' ')} ${v:,.0f}" for k, v in d.items() if v)


def levels_text(m: dict, *, target_pct: float | None = None, fee_pct: float = 0.1,
                anchor: str = "buy", anchor_price: float | None = None) -> str:
    lv = suggest_levels(m, target_pct=target_pct, fee_pct=fee_pct,
                        anchor=anchor, anchor_price=anchor_price)
    if not lv:
        return "no snapshot yet — can't compute levels"
    r_lo, r_hi = lv["reentry"]
    s_lo, s_hi = lv["sell"]
    extra = ""
    sl = lv["reentry"][0]                       # deepest entry = the real sat gain
    if sl and lv["price"] and sl < lv["price"]:
        more = (lv["price"] / sl - 1) * 100
        extra = f"  (~{more:.1f}% more sats/$ at the low)"
    head = ""
    if lv.get("target"):
        tg = lv["target"]
        net = tg["want_pct"]
        cost = (f"   move needed `{tg['gross_pct']:g}%` = {net:g}% you keep "
                f"+ {2 * tg['fee_pct']:g}% fees (2×{tg['fee_pct']:g}%) "
                f"+ {tg['spread_pct']:g}% spread\n")
        if tg["anchor"] == "sell":              # sell fixed (at resistance / your price)
            head = (
                f"🎯 *You want ~{net:g}% net*, anchored on the SELL → here's the pair:\n"
                f"🔴 *Sell ≥* `${tg['sell']:,.0f}`   🟢 *Buy ≤* `${tg['buy']:,.0f}`\n"
                + cost +
                f"   _set_ `/alert price >= {tg['sell']}` _to get pinged at the sell._\n\n")
        else:                                   # buy fixed (at support / your price)
            head = (
                f"🎯 *You want ~{net:g}% net*, anchored on the BUY → here's the pair:\n"
                f"🟢 *Buy ≤* `${tg['buy']:,.0f}`   🔴 *Sell ≥* `${tg['sell']:,.0f}`\n"
                + cost +
                f"   _set_ `/alert price <= {tg['buy']}` _to get pinged at the buy._\n\n")
    return (
        head +
        f"📐 *Structure* — now `${lv['price']:,.0f}`  (ATR ≈ ${lv['atr']:,.0f})\n"
        f"🟢 *Reenter / add sats:* `${r_lo:,.0f} – ${r_hi:,.0f}`{extra}\n"
        f"   support: {_named(lv['support'])}\n"
        f"🔴 *Sell / rotate:* `${s_lo:,.0f} – ${s_hi:,.0f}`\n"
        f"   resistance: {_named(lv['resistance'])}\n"
        f"_Structure-based levels of interest — buying lower stacks more sats. "
        f"Not financial advice; the bot doesn't trade._")
