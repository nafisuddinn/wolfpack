"""Persona #2: The Contrarian — a Bollinger-band mean-reversion rule.

Long-only (v1): enters long when price closes more than 2 population standard
deviations below its SMA(20) ("oversold"), and exits back to flat once price
closes at or above the SMA(20) again. No shorting.

The z-score below is computed from raw close price levels, not log returns.
CLAUDE.md's log-returns rule applies to ML *features*; this scale-independent
threshold rule is a classic Bollinger-band mean-reversion signal and using
price levels (rather than log returns) is the correct, standard construction
for it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from wolfpack_worker.strategies.base import StrategyContext, TargetPosition
from wolfpack_worker.universe import ADJUSTMENT, DATA_FEED

logger = logging.getLogger(__name__)

WINDOW = 20
ENTRY_Z = -2.0
EXIT_Z = 0.0
STD_DDOF = 0
LOOKBACK_BARS = 60


def _decisive_index(z: pd.Series) -> int | None:
    """Index (positional, into `z`) of the most recent decisive bar, or None.

    A bar is decisive if z < ENTRY_Z (entry) or z >= EXIT_Z (exit). Bars with
    NaN z, or z in [ENTRY_Z, EXIT_Z), are not decisive.
    """

    decisive = z[(z < ENTRY_Z) | (z >= EXIT_Z)]
    if decisive.empty:
        return None
    last_label = decisive.index[-1]
    return z.index.get_loc(last_label)


@dataclass
class Contrarian:
    slug: str = "contrarian"
    version: str = "contrarian/v1"
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

            sma_series = closes.rolling(WINDOW).mean()
            sd_series = closes.rolling(WINDOW).std(ddof=STD_DDOF)
            z_series = (closes - sma_series) / sd_series
            z_series = z_series.mask(sd_series == 0)

            decisive_loc = _decisive_index(z_series)
            if decisive_loc is None:
                # No decisive bar anywhere in the window — state is
                # indeterminate. Leave this ticker out entirely (hold
                # current position), same convention as insufficient
                # history.
                logger.info(
                    "contrarian.evaluate: no decisive bar in window for %s "
                    "— holding current position.",
                    ticker,
                    extra={"reason": "state_indeterminate", "ticker": ticker},
                )
                continue

            decisive_z = float(z_series.iloc[decisive_loc])
            decisive_bar_ts = z_series.index[decisive_loc]

            last_close = float(closes.iloc[-1])
            bar_ts = window.index[-1]
            sma = float(sma_series.iloc[-1])
            sd = float(sd_series.iloc[-1])
            z = z_series.iloc[-1]
            z = float(z) if not pd.isna(z) else None
            lower_band = sma - 2 * sd

            is_today_decisive = decisive_bar_ts == bar_ts

            if decisive_z < ENTRY_Z:
                target_exposure = 1.0
                regime = "oversold_entry" if is_today_decisive else "holding_below_mean"
            elif is_today_decisive:
                target_exposure = 0.0
                regime = "reverted_exit"
            else:
                # Flat, and today's own bar isn't itself decisive. Since
                # decisive_z is guaranteed >= EXIT_Z here (it failed the
                # first branch's `< ENTRY_Z` check but is still decisive by
                # definition), the most recent decisive event was an exit;
                # we're just staying flat. Today's own z can only be `None`
                # (sd == 0, price flat at the mean) or strictly inside
                # [ENTRY_Z, EXIT_Z) — z >= EXIT_Z today would itself be
                # decisive, contradicting `is_today_decisive` being False.
                target_exposure = 0.0
                regime = "flat_above_mean" if z is None else "flat_neutral"

            # signal_fired_today: the most recent decisive bar is today's
            # bar AND the state differs from what it was excluding today.
            signal_fired_today = False
            if is_today_decisive:
                prior_z = z_series.iloc[:-1]
                prior_loc = _decisive_index(prior_z)
                prior_state_long = None
                if prior_loc is not None:
                    prior_state_long = float(prior_z.iloc[prior_loc]) < ENTRY_Z
                current_state_long = decisive_z < ENTRY_Z
                signal_fired_today = prior_state_long != current_state_long

            payload = {
                "strategy": "bollinger_mean_reversion",
                "version": self.version,
                "params": {
                    "window": WINDOW,
                    "entry_z": ENTRY_Z,
                    "exit_z": EXIT_Z,
                    "std_ddof": STD_DDOF,
                    "field": "close",
                    "adjustment": ADJUSTMENT,
                    "feed": DATA_FEED,
                },
                "as_of": ctx.as_of.isoformat(),
                "bar_ts": bar_ts.isoformat() if hasattr(bar_ts, "isoformat") else str(bar_ts),
                "last_close": last_close,
                "sma": sma,
                "sd": sd,
                "z": z,
                "lower_band": lower_band,
                "regime": regime,
                "decisive_bar_ts": (
                    decisive_bar_ts.isoformat()
                    if hasattr(decisive_bar_ts, "isoformat")
                    else str(decisive_bar_ts)
                ),
                "decisive_z": decisive_z,
                "signal_fired_today": signal_fired_today,
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
