"""Persona #1: Trend Follower — a plain SMA(20)/SMA(50) crossover.

Long-only (v1): fully invested when the fast SMA is above the slow SMA,
flat otherwise. Ties go flat (no exposure without a clear signal).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from wolfpack_worker.strategies.base import StrategyContext, TargetPosition
from wolfpack_worker.universe import ADJUSTMENT, DATA_FEED

FAST_WINDOW = 20
SLOW_WINDOW = 50
# +1 so both today's and yesterday's SMA(50) can be computed, which is what
# lets us detect a crossover ("crossed_today") rather than just a regime.
LOOKBACK_BARS = SLOW_WINDOW + 1


@dataclass
class TrendFollower:
    slug: str = "trend-follower"
    version: str = "trend_follower/v1"
    lookback_bars: int = field(default=LOOKBACK_BARS)

    def evaluate(self, ctx: StrategyContext) -> list[TargetPosition]:
        targets: list[TargetPosition] = []

        for ticker in ctx.universe:
            df = ctx.bars.get(ticker)
            if df is None or len(df) < self.lookback_bars:
                # Not enough history yet — leave this ticker out entirely,
                # per the Strategy contract (don't error).
                continue

            window = df.iloc[-self.lookback_bars :]
            closes = window["close"]

            sma_fast_series = closes.rolling(FAST_WINDOW).mean()
            sma_slow_series = closes.rolling(SLOW_WINDOW).mean()

            sma_fast = sma_fast_series.iloc[-1]
            sma_slow = sma_slow_series.iloc[-1]
            sma_fast_prev = sma_fast_series.iloc[-2]
            sma_slow_prev = sma_slow_series.iloc[-2]

            if any(pd.isna(v) for v in (sma_fast, sma_slow, sma_fast_prev, sma_slow_prev)):
                # Shouldn't happen once len(window) == lookback_bars, but
                # stay defensive rather than emitting a NaN-based signal.
                continue

            last_close = float(closes.iloc[-1])
            bar_ts = window.index[-1]

            is_above_now = sma_fast > sma_slow
            was_above_prev = sma_fast_prev > sma_slow_prev

            if is_above_now:
                target_exposure = 1.0
                regime = "fast_above"
            else:
                # Exact tie (sma_fast == sma_slow) falls here too: flat wins.
                target_exposure = 0.0
                regime = "fast_below"

            crossed_today = bool(is_above_now != was_above_prev)
            cross_direction = None
            if crossed_today:
                cross_direction = "up" if is_above_now else "down"

            spread_pct = float((sma_fast - sma_slow) / sma_slow) if sma_slow else 0.0

            payload = {
                "strategy": "sma_crossover",
                "version": self.version,
                "params": {
                    "fast": FAST_WINDOW,
                    "slow": SLOW_WINDOW,
                    "field": "close",
                    "adjustment": ADJUSTMENT,
                    "feed": DATA_FEED,
                },
                "as_of": ctx.as_of.isoformat(),
                "bar_ts": bar_ts.isoformat() if hasattr(bar_ts, "isoformat") else str(bar_ts),
                "last_close": last_close,
                "sma_fast": float(sma_fast),
                "sma_slow": float(sma_slow),
                "sma_fast_prev": float(sma_fast_prev),
                "sma_slow_prev": float(sma_slow_prev),
                "spread_pct": spread_pct,
                "regime": regime,
                "crossed_today": crossed_today,
                "cross_direction": cross_direction,
                "bars_used": int(len(window)),
            }

            targets.append(
                TargetPosition(
                    ticker=ticker,
                    target_exposure=target_exposure,
                    signal_ts=bar_ts,
                    payload=payload,
                )
            )

        return targets
