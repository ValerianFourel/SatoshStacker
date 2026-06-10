"""Tests for the deterministic ladder spine (the most safety-critical math)."""
import math

import pytest

from agent.config import LadderConfig
from agent.ladder import (
    Rung, build_ladder, clamp_multiplier, deadline_dca_tranches,
)


def _cfg(**kw):
    base = dict(budget_usdc=2000.0, floor_price=35_000.0, n_tranches=8,
                weighting="geometric", geometric_ratio=1.5)
    base.update(kw)
    return LadderConfig(**base)


def test_allocations_sum_to_budget():
    rungs = build_ladder(60_000.0, _cfg())
    assert math.isclose(sum(r.usdc for r in rungs), 2000.0, rel_tol=1e-9)


def test_weights_sum_to_one():
    rungs = build_ladder(60_000.0, _cfg())
    assert math.isclose(sum(r.weight for r in rungs), 1.0, rel_tol=1e-9)


@pytest.mark.parametrize("weighting", ["linear", "geometric"])
def test_heavier_toward_floor(weighting):
    """Lower rungs (closer to floor) must get >= capital than higher rungs."""
    rungs = build_ladder(60_000.0, _cfg(weighting=weighting))
    usdc = [r.usdc for r in rungs]  # index 0 = top, last = floor
    assert usdc == sorted(usdc), "allocations must be non-decreasing toward the floor"
    assert usdc[-1] > usdc[0], "floor rung must be heaviest"


def test_geometric_sharper_than_linear():
    geo = build_ladder(60_000.0, _cfg(weighting="geometric"))
    lin = build_ladder(60_000.0, _cfg(weighting="linear"))
    # geometric tilts MORE capital to the bottom rung than linear
    assert geo[-1].usdc > lin[-1].usdc


def test_rung_prices_span_anchor_to_floor():
    anchor, floor = 60_000.0, 35_000.0
    rungs = build_ladder(anchor, _cfg(floor_price=floor))
    prices = [r.price for r in rungs]
    assert prices == sorted(prices, reverse=True), "prices descend top->floor"
    assert prices[0] < anchor, "top rung sits below the anchor"
    assert math.isclose(prices[-1], floor, rel_tol=1e-9), "bottom rung == floor"


def test_anchor_at_or_below_floor_collapses_to_single_rung():
    rungs = build_ladder(30_000.0, _cfg(floor_price=35_000.0))
    assert len(rungs) == 1
    assert rungs[0].usdc == 2000.0
    assert rungs[0].price == 30_000.0


def test_n_tranches_respected():
    for n in (1, 4, 8, 20):
        rungs = build_ladder(60_000.0, _cfg(n_tranches=n))
        assert len(rungs) == n


# ── deploy-by-deadline ──────────────────────────────────────────────────────
def test_deadline_tranches_equal_and_sum():
    t = deadline_dca_tranches(1000.0, 5)
    assert len(t) == 5
    assert all(math.isclose(x, 200.0) for x in t)
    assert math.isclose(sum(t), 1000.0)


def test_deadline_tranches_empty_when_nothing_left():
    assert deadline_dca_tranches(0.0, 5) == []
    assert deadline_dca_tranches(-10.0, 5) == []
    assert deadline_dca_tranches(1000.0, 0) == []


# ── LLM multiplier clamp (hard asymmetry: shrink-only, fail-safe to spine) ───
@pytest.mark.parametrize("val,expected", [
    (1.5, 1.0),        # cannot increase
    (1.0, 1.0),
    (0.7, 0.7),        # legitimate shrink
    (0.5, 0.5),
    (0.3, 0.5),        # below floor -> floor
    (0.0, 0.5),
    (-1.0, 0.5),
    (float("nan"), 1.0),   # fail safe to spine size
    (float("inf"), 1.0),
    (None, 1.0),
    ("garbage", 1.0),
])
def test_clamp_multiplier(val, expected):
    assert clamp_multiplier(val, 0.5) == expected
