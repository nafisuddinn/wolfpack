"""Persona #1 (Trend Follower): SMA(20)/SMA(50) crossover correctness."""

from __future__ import annotations

from datetime import timezone

import pandas as pd
import pytest

from wolfpack_worker.strategies.base import StrategyContext
from wolfpack_worker.strategies.trend_follower import LOOKBACK_BARS, TrendFollower

TICKER = "SPY"


def make_bars(closes: list[float]) -> pd.DataFrame:
    idx = pd.bdate_range(start="2026-01-01", periods=len(closes), tz="UTC")
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1_000_000] * len(closes),
        },
        index=idx,
    )


def evaluate_single(closes: list[float]):
    bars = make_bars(closes)
    ctx = StrategyContext(as_of=bars.index[-1], universe=(TICKER,), bars={TICKER: bars})
    strategy = TrendFollower()
    return strategy.evaluate(ctx), bars


def test_rising_series_produces_expected_sma_values_and_long_target():
    closes = [100.0 + i for i in range(LOOKBACK_BARS)]
    targets, bars = evaluate_single(closes)

    assert len(targets) == 1
    target = targets[0]

    expected_fast = sum(closes[-20:]) / 20
    expected_slow = sum(closes[-50:]) / 50
    expected_fast_prev = sum(closes[-21:-1]) / 20
    expected_slow_prev = sum(closes[-51:-1]) / 50

    assert target.ticker == TICKER
    assert target.target_exposure == 1.0
    assert target.payload["sma_fast"] == pytest.approx(expected_fast)
    assert target.payload["sma_slow"] == pytest.approx(expected_slow)
    assert target.payload["sma_fast_prev"] == pytest.approx(expected_fast_prev)
    assert target.payload["sma_slow_prev"] == pytest.approx(expected_slow_prev)
    assert target.payload["last_close"] == pytest.approx(closes[-1])
    assert target.payload["regime"] == "fast_above"
    assert target.payload["bars_used"] == LOOKBACK_BARS
    assert target.payload["strategy"] == "sma_crossover"
    assert target.payload["params"] == {
        "fast": 20,
        "slow": 50,
        "field": "close",
        "adjustment": "split",
        "feed": "sip",
    }
    assert target.signal_ts == bars.index[-1]


def test_falling_series_produces_flat_target():
    closes = [200.0 - i for i in range(LOOKBACK_BARS)]
    targets, _ = evaluate_single(closes)

    assert len(targets) == 1
    target = targets[0]
    assert target.target_exposure == 0.0
    assert target.payload["regime"] == "fast_below"


def test_exact_tie_is_flat_not_long():
    closes = [100.0] * LOOKBACK_BARS
    targets, _ = evaluate_single(closes)

    assert len(targets) == 1
    target = targets[0]
    assert target.payload["sma_fast"] == pytest.approx(target.payload["sma_slow"])
    assert target.target_exposure == 0.0
    assert target.payload["regime"] == "fast_below"
    assert target.payload["crossed_today"] is False
    assert target.payload["cross_direction"] is None


def test_insufficient_history_excludes_ticker_without_error():
    closes = [100.0 + i for i in range(LOOKBACK_BARS - 1)]  # 50 bars, one short
    targets, _ = evaluate_single(closes)

    assert targets == []


def test_crossed_today_detected_on_upward_cross():
    # Flat for a while, then a sharp jump on the very last bar that pulls the
    # fast SMA above the slow SMA for the first time.
    closes = [100.0] * (LOOKBACK_BARS - 1) + [140.0]
    targets, _ = evaluate_single(closes)

    assert len(targets) == 1
    target = targets[0]
    assert target.payload["crossed_today"] is True
    assert target.payload["cross_direction"] == "up"
    assert target.target_exposure == 1.0
