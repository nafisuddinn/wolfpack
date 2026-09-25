"""Shared strategy contracts: the shape every persona's strategy plugs into.

This module is deliberately domain-agnostic about *how* a strategy decides a
target exposure — it only enforces the one non-negotiable rule that applies
to every persona: no lookahead. `StrategyContext` cannot be constructed with
future information, so a strategy that only ever reads from `ctx.bars` and
`ctx.as_of` structurally cannot cheat.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Protocol

import pandas as pd


class LookaheadError(ValueError):
    """Raised when a StrategyContext would let a strategy see future data.

    This must never be caught and swallowed — per CLAUDE.md's no-lookahead
    rule, any code path that could compute a signal using information from
    after `as_of` must hard-fail, not silently truncate.
    """


@dataclass(frozen=True)
class StrategyContext:
    """Everything a `Strategy.evaluate()` call is allowed to see.

    `as_of` must be tz-aware. `bars` maps ticker -> a DataFrame with a
    tz-aware, ascending DatetimeIndex and columns open/high/low/close/volume.
    Construction raises `LookaheadError` if `as_of` is naive, or if any bar
    in any ticker's DataFrame has a timestamp after `as_of`.
    """

    as_of: datetime
    universe: tuple[str, ...]
    bars: Mapping[str, pd.DataFrame]

    def __post_init__(self) -> None:
        if self.as_of.tzinfo is None or self.as_of.tzinfo.utcoffset(self.as_of) is None:
            raise LookaheadError(
                "StrategyContext.as_of must be timezone-aware — a naive "
                "datetime is ambiguous and unsafe for lookahead checks."
            )
        for ticker, df in self.bars.items():
            if df is None or len(df) == 0:
                continue
            last_ts = df.index.max()
            if last_ts > self.as_of:
                raise LookaheadError(
                    f"Bar data for {ticker!r} contains a timestamp "
                    f"({last_ts!r}) after as_of ({self.as_of!r}) — refusing "
                    "to construct a StrategyContext that could leak future "
                    "information into a strategy."
                )


@dataclass(frozen=True)
class TargetPosition:
    """A strategy's desired exposure for one ticker as of one signal time."""

    ticker: str
    target_exposure: float
    signal_ts: datetime
    payload: Mapping[str, Any]


class Strategy(Protocol):
    slug: str
    version: str
    lookback_bars: int

    def evaluate(self, ctx: StrategyContext) -> list[TargetPosition]: ...


def truncate_bars(
    bars: Mapping[str, pd.DataFrame], as_of: datetime
) -> dict[str, pd.DataFrame]:
    """Drop any bar rows with a timestamp after `as_of`, per ticker.

    This is the orchestrator's job, not `StrategyContext`'s: a `PriceStore`
    may legitimately hold data past `as_of` (e.g. the DB simply has more
    history than a given run needs), and the orchestrator is expected to
    truncate before ever constructing a `StrategyContext`. `StrategyContext`
    itself stays strict and raises on anything left over after truncation —
    defense in depth, not a substitute for this step.
    """

    truncated: dict[str, pd.DataFrame] = {}
    for ticker, df in bars.items():
        if df is None or len(df) == 0:
            truncated[ticker] = df
            continue
        truncated[ticker] = df.loc[df.index <= as_of]
    return truncated
