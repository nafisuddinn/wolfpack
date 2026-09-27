"""Persona #2 (Contrarian): 20-day Bollinger z-score mean reversion.

z = (close - SMA20) / population_std20. Entry (long) when the most recent
decisive bar in a 60-bar lookback has z < -2.0 (strictly). Exit (flat) when
the most recent decisive bar has z >= 0.0 (inclusive). Bars with z in
[-2.0, 0.0), NaN z, or sd == 0 are non-decisive and don't change state.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from wolfpack_worker.strategies.base import StrategyContext
from wolfpack_worker.strategies.contrarian import (
    ENTRY_Z,
    EXIT_Z,
    LOOKBACK_BARS,
    Contrarian,
    _decisive_index,
)

TICKER = "SPY"

# A long, constant run of 100.0 padded in front of every fixture below. Only
# the trailing ~20 bars of any 60-bar window feed a given day's z-score, and
# `_decisive_index` only cares about the *most recent* decisive bar, so an
# arbitrarily long flat prefix is a safe, signal-free way to pad fixtures out
# to (or past) LOOKBACK_BARS without disturbing the interesting tail.
PAD = [100.0] * 80


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


def make_bars_ending(closes: list[float], end: str) -> pd.DataFrame:
    idx = pd.bdate_range(end=end, periods=len(closes), tz="UTC")
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
    strategy = Contrarian()
    return strategy.evaluate(ctx), bars


# ---------------------------------------------------------------------------
# 1. Entry
# ---------------------------------------------------------------------------


def test_entry_fires_when_z_drops_below_entry_threshold():
    closes = PAD + [80.0]  # sharp drop -> z ~= -4.36 on the last bar
    targets, bars = evaluate_single(closes)

    assert len(targets) == 1
    target = targets[0]
    assert target.ticker == TICKER
    assert target.target_exposure == 1.0
    assert target.payload["z"] < ENTRY_Z
    assert target.payload["regime"] == "oversold_entry"
    assert target.payload["signal_fired_today"] is True
    assert target.signal_ts == bars.index[-1]


# ---------------------------------------------------------------------------
# 2. Holds through the (-2.0, 0.0) zone
# ---------------------------------------------------------------------------


def test_position_holds_long_while_z_recovers_but_stays_below_exit():
    closes = PAD + [80.0, 90.0]  # entry, then a partial recovery to z ~= -1.78
    targets, _ = evaluate_single(closes)

    assert len(targets) == 1
    target = targets[0]
    assert ENTRY_Z <= target.payload["z"] < EXIT_Z
    assert target.target_exposure == 1.0
    assert target.payload["regime"] == "holding_below_mean"
    assert target.payload["decisive_z"] < ENTRY_Z  # still governed by the entry bar
    assert target.payload["signal_fired_today"] is False


# ---------------------------------------------------------------------------
# 3. Exit at exactly z == 0.0
# ---------------------------------------------------------------------------


def test_exit_fires_at_exactly_zero_z():
    # 98.42105263157895 is chosen so close == SMA20 of its own 20-bar window,
    # i.e. z == 0.0 exactly (not just close to it).
    closes = PAD + [80.0, 90.0, 98.42105263157895]
    targets, _ = evaluate_single(closes)

    assert len(targets) == 1
    target = targets[0]
    assert target.payload["z"] == 0.0
    assert target.payload["decisive_z"] == 0.0
    assert target.target_exposure == 0.0
    assert target.payload["regime"] == "reverted_exit"
    assert target.payload["signal_fired_today"] is True


# ---------------------------------------------------------------------------
# 4. Exactly z == -2.0 must NOT enter (strictly less than)
# ---------------------------------------------------------------------------


def test_decisive_index_excludes_exact_entry_boundary():
    """Unit-level check on the boundary primitive itself, with an exact
    Python float -2.0 (no floating-point search noise): the spec requires
    entry only on z strictly less than -2.0, so a bar sitting exactly on the
    boundary must not be selected as decisive, and the search must fall back
    to the earlier real decisive bar.
    """

    z = pd.Series(
        [-3.0, -2.0],
        index=pd.bdate_range(start="2026-01-01", periods=2, tz="UTC"),
    )
    loc = _decisive_index(z)
    assert loc == 0  # the -3.0 bar, not the -2.0 bar
    assert z.iloc[loc] == -3.0


def test_exact_entry_boundary_does_not_fire_at_strategy_level():
    """A bar whose z lands (to double precision) exactly on -2.0 must not by
    itself cause an entry. Constructed so an earlier, genuinely decisive bar
    exists in the window, so a non-empty target with target_exposure == 0.0
    unambiguously shows the boundary bar was excluded rather than being
    treated as the (nonexistent) most recent decisive bar.
    """

    base19 = [100.0] * 18 + [100.5]
    # Solved by bisection (see PR description / test authoring notes) so that
    # appending `boundary` makes the 20-bar window's z land on -2.0 to
    # within double-precision floating point error.
    boundary = 99.76847476391757
    closes = PAD + base19 + [boundary]
    targets, _ = evaluate_single(closes)

    assert len(targets) == 1
    target = targets[0]
    assert target.payload["z"] == pytest.approx(-2.0, abs=1e-9)
    assert target.payload["z"] >= ENTRY_Z  # not strictly less than -2.0
    assert target.target_exposure == 0.0
    assert target.payload["regime"] != "oversold_entry"

    # Sanity: a value just past the boundary DOES enter, confirming this is
    # a real threshold and not an artifact of the fixture.
    just_below_targets, _ = evaluate_single(PAD + base19 + [boundary - 0.5])
    assert just_below_targets[0].target_exposure == 1.0
    assert just_below_targets[0].payload["regime"] == "oversold_entry"


# ---------------------------------------------------------------------------
# 5. sd == 0 must not produce a NaN-driven false signal
# ---------------------------------------------------------------------------


def test_constant_price_window_does_not_produce_false_signal(caplog):
    closes = [100.0] * LOOKBACK_BARS  # sd == 0 everywhere -> z masked to NaN
    with caplog.at_level(logging.INFO):
        targets, _ = evaluate_single(closes)

    # No decisive bar anywhere -> omitted entirely, never defaulted to long
    # or flat off the back of a NaN.
    assert targets == []
    assert any(
        record.__dict__.get("reason") == "state_indeterminate" for record in caplog.records
    )


# ---------------------------------------------------------------------------
# 6. Fully indeterminate state (no decisive bar anywhere in the window)
# ---------------------------------------------------------------------------


def test_fully_indeterminate_state_omits_ticker(caplog):
    # A slow, steady linear decline settles into a stable steady-state z in
    # roughly (-1.65, -1.65) for every bar once the rolling windows are warm
    # (verified independently) — comfortably inside the non-decisive zone
    # [-2.0, 0.0) for all 41 bars that have a real (non-NaN) z, and NaN for
    # the first 19 warm-up bars. No decisive bar exists anywhere.
    closes = [100.0 - i * 0.05 for i in range(LOOKBACK_BARS)]
    with caplog.at_level(logging.INFO):
        targets, _ = evaluate_single(closes)

    assert targets == []
    assert any(
        record.__dict__.get("reason") == "state_indeterminate" for record in caplog.records
    )


# ---------------------------------------------------------------------------
# 7. Window-length independence (60-bar vs. 70-bar, same trailing data)
# ---------------------------------------------------------------------------


def test_result_is_independent_of_extra_older_history():
    end = "2026-06-01"

    def run(closes):
        bars = make_bars_ending(closes, end)
        ctx = StrategyContext(as_of=bars.index[-1], universe=(TICKER,), bars={TICKER: bars})
        return Contrarian().evaluate(ctx)

    tail = [100.0] * 59 + [80.0]  # decisive entry bar at the end
    targets_60 = run(tail)  # exactly LOOKBACK_BARS bars
    targets_70 = run([100.0] * 10 + tail)  # 10 extra bars of older history

    assert len(targets_60) == 1 and len(targets_70) == 1
    assert targets_60[0] == targets_70[0]


# ---------------------------------------------------------------------------
# 8. signal_fired_today
# ---------------------------------------------------------------------------


def test_signal_fired_today_true_on_entry_day():
    targets, _ = evaluate_single(PAD + [80.0])
    assert targets[0].payload["signal_fired_today"] is True


def test_signal_fired_today_true_on_exit_day():
    targets, _ = evaluate_single(PAD + [80.0, 90.0, 98.42105263157895])
    assert targets[0].payload["signal_fired_today"] is True


def test_signal_fired_today_false_while_holding_unchanged_below_mean():
    targets, _ = evaluate_single(PAD + [80.0, 90.0])
    assert targets[0].target_exposure == 1.0
    assert targets[0].payload["signal_fired_today"] is False


def test_signal_fired_today_false_when_already_long_and_still_decisive():
    # Two consecutive decisive-entry bars: day 2 re-confirms "long" rather
    # than newly entering it, so no new signal should fire even though
    # day 2's own bar is, by itself, decisive.
    targets, _ = evaluate_single(PAD + [80.0, 60.0])
    assert targets[0].target_exposure == 1.0
    assert targets[0].payload["regime"] == "oversold_entry"
    assert targets[0].payload["signal_fired_today"] is False


def test_signal_fired_today_false_when_already_flat_and_still_decisive():
    # Two consecutive decisive-exit bars: day 2 re-confirms "flat" rather
    # than newly exiting, so no new signal should fire.
    targets, _ = evaluate_single(PAD + [80.0, 90.0, 98.42105263157895, 100.0])
    assert targets[0].target_exposure == 0.0
    assert targets[0].payload["regime"] == "reverted_exit"
    assert targets[0].payload["signal_fired_today"] is False


# ---------------------------------------------------------------------------
# Regime payload contract
# ---------------------------------------------------------------------------


def test_regime_oversold_entry_on_entry_day():
    targets, _ = evaluate_single(PAD + [80.0])
    assert targets[0].payload["regime"] == "oversold_entry"


def test_regime_holding_below_mean_while_long_and_non_decisive():
    targets, _ = evaluate_single(PAD + [80.0, 90.0])
    assert targets[0].payload["regime"] == "holding_below_mean"


def test_regime_reverted_exit_on_exit_day():
    targets, _ = evaluate_single(PAD + [80.0, 90.0, 98.42105263157895])
    assert targets[0].payload["regime"] == "reverted_exit"


def test_regime_flat_neutral_when_flat_and_today_below_mean_but_not_oversold():
    # Flat from an earlier exit, then today's bar drifts back down into the
    # non-decisive [-2.0, 0.0) zone without re-entering.
    closes = PAD + [80.0, 90.0, 98.42105263157895, 100.0, 96.0]
    targets, _ = evaluate_single(closes)
    target = targets[0]
    assert target.target_exposure == 0.0
    assert ENTRY_Z <= target.payload["z"] < EXIT_Z
    assert target.payload["regime"] == "flat_neutral"


def test_regime_flat_above_mean_when_flat_and_today_sd_is_zero():
    # Flat from an earlier exit, then a run of constant prices (sd == 0
    # today -> z masked to NaN) while the decisive state remains "exited".
    closes = PAD + [80.0, 90.0, 98.42105263157895] + [98.42105263157895] * 25
    targets, _ = evaluate_single(closes)
    target = targets[0]
    assert target.target_exposure == 0.0
    assert target.payload["z"] is None
    assert target.payload["regime"] == "flat_above_mean"


def test_regime_holding_below_mean_when_long_and_today_sd_is_zero():
    # Symmetric long-side case of the sd==0 test above: a decisive entry bar
    # (z < -2.0) followed by a plateau of constant prices that persists
    # through the end of the series (sd == 0 -> z masked to None today),
    # while the decisive state remains "entered" (still long). This must not
    # repeat the flat-side's dead-code mismatch -- the regime label must
    # reflect "still long, non-decisive today", not "reverted".
    closes = PAD + [80.0] + [80.0] * 25
    targets, _ = evaluate_single(closes)
    target = targets[0]
    assert target.target_exposure == 1.0
    assert target.payload["z"] is None
    assert target.payload["regime"] == "holding_below_mean"


# ---------------------------------------------------------------------------
# 10. Insufficient history
# ---------------------------------------------------------------------------


def test_insufficient_history_excludes_ticker_without_error():
    closes = [100.0 + i for i in range(LOOKBACK_BARS - 1)]  # one bar short
    targets, _ = evaluate_single(closes)
    assert targets == []
