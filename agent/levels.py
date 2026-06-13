"""Sat-stacking levels: a *good reentry (buy) zone* and a *sell/rotate zone*, derived
from market STRUCTURE (24h/7d range, ATR, order-book walls) — not freelance advice.

Goal framing: stacking satoshis. Buying lower = more sats per dollar; selling into
strength and rebuying lower grows the stack. These are levels of interest only; the
watch bot is read-only and never places orders.
"""
from __future__ import annotations


def suggest_levels(m: dict) -> dict | None:
    """Return reentry/sell zones + the support/resistance they're built from, or None."""
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
    return {
        "price": price,
        "reentry": (round(min(reentry_lo, reentry_hi)), round(max(reentry_lo, reentry_hi))),
        "sell": (round(min(sell_lo, sell_hi)), round(max(sell_lo, sell_hi))),
        "atr": round(atr),
        "support": {"24h_low": low24, "7d_low": low7, "bid_wall": bid_wall},
        "resistance": {"24h_high": high24, "7d_high": high7, "ask_wall": ask_wall},
    }


def _named(d: dict) -> str:
    return " · ".join(f"{k.replace('_', ' ')} ${v:,.0f}" for k, v in d.items() if v)


def levels_text(m: dict) -> str:
    lv = suggest_levels(m)
    if not lv:
        return "no snapshot yet — can't compute levels"
    r_lo, r_hi = lv["reentry"]
    s_lo, s_hi = lv["sell"]
    extra = ""
    sl = lv["reentry"][0]                       # deepest entry = the real sat gain
    if sl and lv["price"] and sl < lv["price"]:
        more = (lv["price"] / sl - 1) * 100
        extra = f"  (~{more:.1f}% more sats/$ at the low)"
    return (
        f"🎯 *Sat-stacking levels* — now `${lv['price']:,.0f}`  (ATR ≈ ${lv['atr']:,.0f})\n"
        f"🟢 *Reenter / add sats:* `${r_lo:,.0f} – ${r_hi:,.0f}`{extra}\n"
        f"   support: {_named(lv['support'])}\n"
        f"🔴 *Sell / rotate:* `${s_lo:,.0f} – ${s_hi:,.0f}`\n"
        f"   resistance: {_named(lv['resistance'])}\n"
        f"_Structure-based levels of interest — buying lower stacks more sats. "
        f"Not financial advice; the bot doesn't trade._")
