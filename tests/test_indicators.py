"""Tests for the weekly self-tuner's indicator backtester + trader wiring."""
import numpy as np
import pandas as pd

from backtest.indicators import ic, rank_tf, signals


def test_ic_perfect_correlation():
    s = pd.Series(np.arange(200, dtype=float))
    assert ic(s, s * 2.0 + 1) > 0.99           # perfectly correlated -> IC ~1


def test_ic_too_short_is_zero():
    assert ic(pd.Series([1.0, 2, 3]), pd.Series([1.0, 2, 3])) == 0.0  # <30 samples


def test_signals_cover_expected_indicators():
    c = pd.Series(np.cumsum(np.random.RandomState(0).randn(300)) + 100.0)
    sg = signals(c)
    for k in ("rsi_7", "rsi_14", "rsi_28", "ema_cross_8_21", "macd_hist", "momentum_12"):
        assert k in sg and len(sg[k]) == len(c)


def test_rank_is_sorted_by_abs_ic():
    close = pd.Series(np.cumsum(np.random.RandomState(1).randn(600)) + 1000.0)
    ranked = rank_tf(pd.DataFrame({"close": close}), [1, 2, 3, 6])
    assert ranked == sorted(ranked, key=lambda r: -r["abs_ic"])
    assert all({"indicator", "lag", "ic", "direction", "abs_ic"} <= set(r) for r in ranked)


def test_rank_detects_momentum_in_trending_series():
    # a strongly trending series -> momentum/ema indicators should have positive IC
    close = pd.Series(np.linspace(100, 200, 400) + np.random.RandomState(2).randn(400))
    ranked = rank_tf(pd.DataFrame({"close": close}), [1, 2, 3])
    top = ranked[0]
    assert top["abs_ic"] > 0.05  # there IS predictive structure in a trend


def test_trader_reads_tuned_technicals():
    # sim_sats loads agent/technicals.json at import; the tuned rsi period flows in
    import backtest.sim_sats as s
    assert isinstance(s.RSI_PERIOD, int) and s.RSI_PERIOD >= 2
    assert isinstance(s.WEEKLY_CONTEXT, str)
