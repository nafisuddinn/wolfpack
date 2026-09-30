"""Labels, dataset assembly, chronological split/embargo, walk-forward folds,
and the split-artifact data guard for The Analyst."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from analyst_helpers import make_ohlcv, make_universe_bars
from wolfpack_worker.analyst.dataset import (
    EMBARGO_SESSIONS,
    MAX_ABS_DAILY_LOG_RETURN,
    TEST_SESSIONS,
    SuspectPriceDataError,
    assert_no_split_artifacts,
    build_dataset,
    chronological_split,
    make_labels,
    walk_forward_folds,
)
from wolfpack_worker.analyst.features import FEATURE_NAMES, WARMUP_BARS

# ---------------------------------------------------------------------------
# Label
# ---------------------------------------------------------------------------


def test_label_is_sign_of_open_to_open_log_return_from_t_plus_1_to_t_plus_2():
    idx = pd.bdate_range("2021-01-04", periods=6, tz="UTC")
    opens = [10.0, 11.0, 12.0, 6.0, 6.0, 7.0]
    df = pd.DataFrame({"open": opens, "close": opens}, index=idx)
    lab = make_labels(df)
    # t=0: ln(O2/O1) = ln(12/11) > 0 -> 1
    assert lab["y"].iloc[0] == 1
    assert lab["fwd_logret"].iloc[0] == pytest.approx(math.log(12 / 11))
    # t=1: ln(O3/O2) = ln(6/12) < 0 -> 0
    assert lab["y"].iloc[1] == 0
    # t=2: ln(O4/O3) = 0 exactly -> 0 (zero is not "up")
    assert lab["y"].iloc[2] == 0
    assert lab["fwd_logret"].iloc[2] == 0.0
    # t=3: ln(7/6) > 0
    assert lab["y"].iloc[3] == 1
    # label end = the bar at t+2
    assert lab["label_end_ts"].iloc[0] == idx[2]
    assert lab["label_end_ts"].iloc[3] == idx[5]
    # last 2 bars have no label
    assert lab["y"].iloc[-2:].isna().all()
    assert lab["label_end_ts"].iloc[-2:].isna().all()


def test_label_never_uses_close_or_same_bar_open():
    """Changing C_t, O_t, and everything except O_{t+1}, O_{t+2} must not
    change y_t — the label is strictly after the decision bar."""
    df = make_ohlcv(10, seed=1)
    base = make_labels(df)
    mod = df.copy()
    mod["close"] = mod["close"] * 3
    mod.iloc[4, mod.columns.get_loc("open")] *= 5  # O_t for t=4
    new = make_labels(mod)
    assert new["y"].iloc[4] == base["y"].iloc[4]
    assert new["fwd_logret"].iloc[4] == base["fwd_logret"].iloc[4]


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------


def test_dataset_is_sorted_chronologically_and_never_shuffled():
    ds = build_dataset(make_universe_bars(400, seed=2))
    assert ds.index.equals(pd.RangeIndex(len(ds)))
    assert ds["ts"].is_monotonic_increasing
    # (ts, ticker) unique -> strictly increasing composite key
    assert not ds.duplicated(["ts", "ticker"]).any()


def test_dataset_drops_warmup_nan_rows_and_unlabeled_last_two_bars():
    n = 300
    bars = make_universe_bars(n, seed=2)
    ds = build_dataset(bars)
    per_ticker = ds.groupby("ticker").size()
    assert (per_ticker == n - (WARMUP_BARS - 1) - 2).all()
    assert not ds[list(FEATURE_NAMES)].isna().any().any()
    assert ds["y"].isin([0, 1]).all()
    last_ts = bars["SPY"].index[-1]
    assert ds["ts"].max() == bars["SPY"].index[-3]
    assert (ds["label_end_ts"] <= last_ts).all()


def test_dataset_has_no_ticker_id_column():
    ds = build_dataset(make_universe_bars(200, seed=2))
    feature_cols = [c for c in ds.columns if c in FEATURE_NAMES]
    assert feature_cols == list(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Chronological split + embargo
# ---------------------------------------------------------------------------


def test_split_test_is_last_252_sessions_and_no_train_label_reaches_test_start():
    ds = build_dataset(make_universe_bars(900, seed=3))
    split = chronological_split(ds)
    test_dates = split.test["ts"].unique()
    assert len(test_dates) == TEST_SESSIONS == 252
    assert split.test["ts"].min() == split.test_start
    assert split.test["ts"].max() == ds["ts"].max()
    # Core leakage check: no training row's label ends at/after test start.
    assert (split.train["label_end_ts"] < split.test_start).all()
    assert (split.train["ts"] < split.test_start).all()
    assert split.trained_through == split.train["label_end_ts"].max()
    assert split.trained_through < split.test_start
    # Both keep chronological order (never shuffled).
    assert split.train["ts"].is_monotonic_increasing
    assert split.test["ts"].is_monotonic_increasing
    assert split.train.index.max() < split.test.index.min()


def test_embargo_drops_exactly_the_last_two_pre_test_sessions():
    ds = build_dataset(make_universe_bars(900, seed=3))
    split = chronological_split(ds)
    all_dates = pd.DatetimeIndex(ds["ts"].unique()).sort_values()
    test_start_pos = int(all_dates.searchsorted(split.test_start))
    embargoed = all_dates[test_start_pos - EMBARGO_SESSIONS : test_start_pos]
    assert EMBARGO_SESSIONS == 2
    assert not split.train["ts"].isin(embargoed).any()
    assert split.train["ts"].max() == all_dates[test_start_pos - EMBARGO_SESSIONS - 1]


def test_split_refuses_when_not_enough_history():
    ds = build_dataset(make_universe_bars(250, seed=3))
    with pytest.raises(ValueError):
        chronological_split(ds)


# ---------------------------------------------------------------------------
# Walk-forward folds
# ---------------------------------------------------------------------------


def test_walk_forward_folds_are_expanding_and_embargoed():
    # ~2016-01 through ~2026-09
    ds = build_dataset(make_universe_bars(2800, seed=4))
    folds = list(walk_forward_folds(ds))
    names = [f.name for f in folds]
    assert names == [str(y) for y in range(2019, 2026)] + ["trailing_252"]
    prev_train_len = 0
    for f in folds:
        assert len(f.train) > 0 and len(f.test) > 0
        assert (f.train["label_end_ts"] < f.test_start).all()
        assert f.train["ts"].is_monotonic_increasing
        if f.name != "trailing_252":
            assert (f.test["ts"].dt.year == int(f.name)).all()
            assert len(f.train) > prev_train_len  # expanding window
            prev_train_len = len(f.train)
    assert folds[-1].test["ts"].nunique() == TEST_SESSIONS


# ---------------------------------------------------------------------------
# Split-artifact data guard
# ---------------------------------------------------------------------------


def test_guard_passes_on_normal_data():
    assert MAX_ABS_DAILY_LOG_RETURN == 0.25
    assert_no_split_artifacts(make_universe_bars(300, seed=1))


def test_guard_aborts_on_unadjusted_split_in_close():
    bars = make_universe_bars(300, seed=1)
    aapl = bars["AAPL"].copy()
    # A 4:1 split that wasn't back-adjusted: price quarters overnight.
    aapl.iloc[150:, :4] = aapl.iloc[150:, :4] / 4
    bars["AAPL"] = aapl
    with pytest.raises(SuspectPriceDataError, match="AAPL"):
        assert_no_split_artifacts(bars)


def test_guard_aborts_on_bad_open_even_if_close_is_fine():
    bars = make_universe_bars(300, seed=1)
    xom = bars["XOM"].copy()
    xom.iloc[200, xom.columns.get_loc("open")] *= 1.5  # ln(1.5) = 0.405
    bars["XOM"] = xom
    with pytest.raises(SuspectPriceDataError, match="XOM"):
        assert_no_split_artifacts(bars)


def test_guard_threshold_is_strictly_greater_than():
    idx = pd.bdate_range("2021-01-04", periods=3, tz="UTC")
    c = [100.0, 100.0 * math.exp(0.2499), 100.0]
    df = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1.0}, index=idx)
    assert_no_split_artifacts({"X": df})  # just under 0.25 (either direction) is allowed
    c2 = [100.0, 100.0 * math.exp(0.2501), 100.0]
    df2 = pd.DataFrame({"open": c2, "high": c2, "low": c2, "close": c2, "volume": 1.0}, index=idx)
    with pytest.raises(SuspectPriceDataError):
        assert_no_split_artifacts({"X": df2})
