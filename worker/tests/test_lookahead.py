"""No-lookahead safety: applies even to pure price-data strategies.

This is CLAUDE.md's non-negotiable "no lookahead bias" rule, enforced
structurally by StrategyContext, not just by convention.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from wolfpack_worker.strategies.base import LookaheadError, StrategyContext, truncate_bars
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


def test_naive_as_of_raises_lookahead_error():
    bars = make_bars([100.0] * 10)
    naive_as_of = datetime(2026, 1, 1)  # no tzinfo

    with pytest.raises(LookaheadError):
        StrategyContext(as_of=naive_as_of, universe=(TICKER,), bars={TICKER: bars})


def test_bar_after_as_of_raises_lookahead_error():
    bars = make_bars([100.0] * 10)
    as_of = bars.index[-2]  # one bar short of the last -> last bar is "future"

    with pytest.raises(LookaheadError):
        StrategyContext(as_of=as_of, universe=(TICKER,), bars={TICKER: bars})


def test_garbage_bars_after_as_of_do_not_change_evaluate_output():
    n = LOOKBACK_BARS + 40
    closes = [100.0 + i for i in range(n)]
    full_bars = make_bars(closes)

    cutoff = LOOKBACK_BARS + 9  # index of the last "real" bar for this as_of
    as_of = full_bars.index[cutoff]

    manually_truncated = full_bars.iloc[: cutoff + 1]
    ctx_manual = StrategyContext(
        as_of=as_of, universe=(TICKER,), bars={TICKER: manually_truncated}
    )
    expected = TrendFollower().evaluate(ctx_manual)

    # Corrupt every row strictly after the cutoff with extreme garbage values
    # — these must never influence the signal for `as_of`.
    garbage = full_bars.copy()
    garbage.iloc[cutoff + 1 :, garbage.columns.get_indexer(["open", "high", "low", "close"])] = (
        1e12
    )

    truncated = truncate_bars({TICKER: garbage}, as_of)
    ctx_from_truncated = StrategyContext(as_of=as_of, universe=(TICKER,), bars=truncated)
    actual = TrendFollower().evaluate(ctx_from_truncated)

    assert actual == expected


def test_walk_forward_prefix_consistency():
    """evaluate() on each chronological prefix must match what the
    orchestrator would produce with as_of set to that prefix's last ts —
    i.e. truncate_bars(full_series, as_of) is equivalent to slicing the
    series to that point from scratch, for every step of a walk forward.
    """
    n = LOOKBACK_BARS + 15
    closes = [100.0 + (i % 7) - (i * 0.3) for i in range(n)]
    full_bars = make_bars(closes)
    strategy = TrendFollower()

    for k in range(LOOKBACK_BARS, n + 1):
        as_of = full_bars.index[k - 1]

        prefix_bars = full_bars.iloc[:k]
        ctx_prefix = StrategyContext(as_of=as_of, universe=(TICKER,), bars={TICKER: prefix_bars})
        expected = strategy.evaluate(ctx_prefix)

        truncated = truncate_bars({TICKER: full_bars}, as_of)
        ctx_truncated = StrategyContext(as_of=as_of, universe=(TICKER,), bars=truncated)
        actual = strategy.evaluate(ctx_truncated)

        assert actual == expected, f"mismatch at prefix length {k}"
