"""Shared synthetic-data helpers for The Analyst's tests.

Deterministic (seeded) OHLCV generators — no network, no DB. Prices follow a
geometric random walk so every log-return-based feature is well defined.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

UNIVERSE = ("SPY", "QQQ", "AAPL", "JPM", "XOM")


def make_ohlcv(
    n: int,
    *,
    seed: int = 0,
    start: str = "2016-01-04",
    start_price: float = 100.0,
    daily_vol: float = 0.01,
    index: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if index is None:
        # Alpaca daily bars are stamped at midnight America/New_York, which
        # is 04:00/05:00 UTC; the exact hour doesn't matter here, only that
        # the index is tz-aware and strictly increasing.
        index = pd.bdate_range(start=start, periods=n, tz="UTC") + pd.Timedelta(hours=5)
    r = rng.normal(0.0002, daily_vol, size=n)
    close = start_price * np.exp(np.cumsum(r))
    # Open = previous close nudged by a small gap, so open-to-open returns
    # (the label) differ from close-to-close returns (the features).
    gap = rng.normal(0.0, daily_vol / 4, size=n)
    open_ = np.concatenate([[start_price], close[:-1]]) * np.exp(gap)
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, daily_vol / 2, size=n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, daily_vol / 2, size=n)))
    volume = rng.integers(1_000_000, 5_000_000, size=n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


def make_universe_bars(n: int, *, seed: int = 0, start: str = "2016-01-04") -> dict[str, pd.DataFrame]:
    index = pd.bdate_range(start=start, periods=n, tz="UTC") + pd.Timedelta(hours=5)
    return {
        ticker: make_ohlcv(n, seed=seed + i, index=index, start_price=50.0 + 40 * i)
        for i, ticker in enumerate(UNIVERSE)
    }
