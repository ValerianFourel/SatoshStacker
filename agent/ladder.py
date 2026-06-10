"""Deterministic accumulation ladder + deploy-by-deadline schedule.

This is the **spine** of the agent. Every sizing decision originates here and
is fully deterministic given (anchor price, floor, budget, config). The LLM
advisor and the live order layer consume this output but can never change it
in a way that deploys *more* or *faster* (see analysis.py / risk.py).

Design choices (documented because they decide whether the agent beats a
static ladder):

* **Price spacing** of rungs is linear from just below the anchor down to the
  floor, with the lowest rung sitting exactly at the floor.
* **Capital weighting** is heavier toward lower rungs (bearish tilt):
  - ``linear``    : rung i (1=top .. N=floor) gets weight proportional to i.
  - ``geometric`` : rung i gets weight proportional to ``ratio**(i-1)`` — a
    sharper tilt toward the floor.
* The **deploy-by-deadline** guard is the agent's one honest structural edge
  over a static resting ladder: if the bearish call is wrong and price never
  reaches the lower rungs, leftover USDC is force-DCA'd rather than stranded.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import LadderConfig


@dataclass(frozen=True)
class Rung:
    """One resting limit-buy level.

    Attributes:
        price: limit price for the maker buy.
        usdc:  capital allocated to this rung.
        weight: normalized allocation weight (sums to 1 across the ladder).
        index: 1 = nearest the anchor, n_tranches = at the floor.
    """
    index: int
    price: float
    usdc: float
    weight: float


def _weights(n: int, weighting: str, ratio: float) -> list[float]:
    """Return normalized weights for rungs 1..n, heavier toward the floor (n)."""
    if n <= 0:
        return []
    if weighting == "linear":
        raw = [float(i) for i in range(1, n + 1)]
    elif weighting == "geometric":
        raw = [ratio ** (i - 1) for i in range(n)]  # rung n (floor) is heaviest
    else:
        raise ValueError(f"unknown weighting: {weighting!r}")
    total = sum(raw)
    return [r / total for r in raw]


def build_ladder(anchor_price: float, cfg: LadderConfig,
                 *, budget_override: float | None = None,
                 floor_override: float | None = None) -> list[Rung]:
    """Build the deterministic buy ladder from ``anchor_price`` down to the floor.

    Args:
        anchor_price: current/reference BTC price; rungs are placed below it.
        cfg: ladder configuration (budget, floor, n_tranches, weighting).
        budget_override: deploy this much instead of ``cfg.budget_usdc`` (used by
            the re-anchoring planner to ladder a fraction of the remaining budget).
        floor_override: ladder down to this floor instead of ``cfg.floor_price``
            (used by re-anchoring: each generation's floor = anchor * floor_frac).

    Returns:
        A list of ``Rung`` ordered top (index 1, highest price) to bottom
        (index n, at the floor). Allocations sum to the effective budget.

    Edge cases:
        * If ``anchor_price <= floor`` the market is already at/below the floor;
          we collapse to a single rung at ``anchor_price`` holding the full
          budget (nothing lower to ladder into).
    """
    n = cfg.n_tranches
    floor = cfg.floor_price if floor_override is None else floor_override
    budget = cfg.budget_usdc if budget_override is None else budget_override

    if n <= 0:
        raise ValueError("n_tranches must be >= 1")
    if anchor_price <= floor:
        return [Rung(index=1, price=anchor_price, usdc=budget, weight=1.0)]

    weights = _weights(n, cfg.weighting, cfg.geometric_ratio)
    span = anchor_price - floor
    rungs: list[Rung] = []
    for i in range(1, n + 1):
        # rung i price: linear from just below anchor (i=1) down to floor (i=n)
        price = anchor_price - span * (i / n)
        w = weights[i - 1]
        rungs.append(Rung(index=i, price=price, usdc=budget * w, weight=w))
    return rungs


def next_generation(anchor_price: float, remaining_budget: float,
                    cfg: LadderConfig) -> tuple[list[Rung], float]:
    """Build one re-anchoring *generation*: ladder ``gen_fraction`` of the
    remaining budget from ``anchor_price`` down to ``anchor_price * floor_frac``.

    Returns ``(rungs, generation_floor)``. The generation deliberately deploys
    only a fraction of the remaining budget so dry powder is reserved for lower
    re-anchored generations (the trailing-ladder edge from Milestone 1).
    """
    gen_floor = anchor_price * cfg.floor_frac
    gen_budget = remaining_budget * cfg.gen_fraction
    rungs = build_ladder(anchor_price, cfg, budget_override=gen_budget,
                         floor_override=gen_floor)
    return rungs, gen_floor


def filter_min_notional(rungs: list[Rung], min_notional: float) -> tuple[list[Rung], float]:
    """Split rungs into placeable (>= ``min_notional``) and a deferred USDC pool.

    Binance rejects orders below MIN_NOTIONAL (~$10). Rather than place
    un-fillable dust orders, sub-minimum rungs are deferred — their capital
    stays in the remaining budget for a future re-anchored generation or the
    deploy-by-deadline mop-up (verified backtest-neutral in the M1 audit).

    Returns ``(placeable_rungs, deferred_usdc)``.
    """
    placeable = [r for r in rungs if r.usdc >= min_notional]
    deferred = sum(r.usdc for r in rungs if r.usdc < min_notional)
    return placeable, deferred


def deadline_dca_tranches(leftover_usdc: float, n_buckets: int) -> list[float]:
    """Split leftover USDC into ``n_buckets`` equal forced-DCA tranches.

    Used by the deploy-by-deadline guard. Returns an empty list if there is
    nothing left to deploy or no buckets remain.
    """
    if leftover_usdc <= 0 or n_buckets <= 0:
        return []
    each = leftover_usdc / n_buckets
    return [each] * n_buckets


def clamp_multiplier(multiplier: float, floor: float) -> float:
    """Clamp an LLM size multiplier to [floor, 1.0].

    Enforces the hard asymmetry: the advisor may only *shrink* (values <1),
    never *increase* (values >1 are ignored -> 1.0), and never below ``floor``.
    Any non-finite / nonsensical value fails safe to 1.0 (spine size).
    """
    # Hard-clamp the floor itself to [0, 1] so an operator-misconfigured
    # min_tranche_fraction > 1 can NEVER lift a multiplier above the spine.
    floor = max(0.0, min(float(floor), 1.0))
    try:
        m = float(multiplier)
    except (TypeError, ValueError):
        return 1.0
    if not (m == m) or m in (float("inf"), float("-inf")):  # NaN/inf guard
        return 1.0
    if m > 1.0:
        return 1.0
    if m < floor:
        return floor
    return m
