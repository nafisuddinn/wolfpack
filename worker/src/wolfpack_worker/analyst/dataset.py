"""Labels, dataset assembly, chronological split + embargo, walk-forward folds,
and the split-artifact data guard for The Analyst.

Label (strictly after the decision bar t):
    y_t = 1[ ln(O_{t+2} / O_{t+1}) > 0 ]
The signal is computed after the close of day t; the order fills at day
t+1's open; the next decision can change the position at day t+2's open. So
the return the decision actually earns is open(t+1) -> open(t+2). A zero
return is labeled 0 (not "up"). The last two bars of each ticker have no
label. `label_end_ts` = timestamp of bar t+2, the last bar the label reads.

No shuffling anywhere: the dataset is sorted by (ts, ticker) with a fresh
RangeIndex, and every split is a timestamp cut. A training row is only kept
if its label ends strictly before the test window starts (and it is not in
the EMBARGO_SESSIONS sessions immediately before the test window) — without
this, the last training rows' labels would read opens from inside the test
window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Mapping

import numpy as np
import pandas as pd

from wolfpack_worker.analyst.features import FEATURE_NAMES, build_features

LABEL_DEFINITION = (
    "y_t = 1[ln(O_{t+2}/O_{t+1}) > 0] — sign of the next open-to-open log "
    "return (signal after close of t, fill at open of t+1, exit/revise at "
    "open of t+2); zero -> 0; last 2 bars unlabeled"
)
TEST_SESSIONS = 252
EMBARGO_SESSIONS = 2
WALK_FORWARD_YEARS = tuple(range(2019, 2026))
# Largest believable daily |log return| for this universe (5 large, liquid
# US tickers). Largest real moves since 2016 (Mar 2020) are ~0.12-0.16.
# Anything above this almost certainly means an unadjusted split or bad bar.
MAX_ABS_DAILY_LOG_RETURN = 0.25


class SuspectPriceDataError(RuntimeError):
    """Price history contains a move that looks like an unadjusted split or
    a bad bar. Training must abort — never train on it silently."""


def assert_no_split_artifacts(
    bars: Mapping[str, pd.DataFrame], max_abs: float = MAX_ABS_DAILY_LOG_RETURN
) -> None:
    """Raise SuspectPriceDataError if any ticker has a bar-to-bar |log
    return| > `max_abs` in either close (features) or open (labels), or any
    non-positive price."""
    problems: list[str] = []
    for ticker, df in bars.items():
        if df is None or len(df) == 0:
            continue
        for col in ("close", "open"):
            px = df[col].to_numpy(dtype=float)
            if (px <= 0).any() or np.isnan(px).any():
                problems.append(f"{ticker}: non-positive or NaN {col} price")
                continue
            r = np.diff(np.log(px))
            bad = np.flatnonzero(np.abs(r) > max_abs)
            for i in bad[:5]:
                problems.append(
                    f"{ticker}: |ln({col}[{df.index[i + 1]}] / {col}[{df.index[i]}])| "
                    f"= {abs(r[i]):.4f} > {max_abs}"
                )
    if problems:
        raise SuspectPriceDataError(
            "Refusing to train: price history looks unadjusted or corrupt "
            "(re-run the backfill; check split adjustment):\n  " + "\n  ".join(problems)
        )


def make_labels(df: pd.DataFrame) -> pd.DataFrame:
    """y, fwd_logret (= ln(O_{t+2}/O_{t+1})) and label_end_ts per bar."""
    o = df["open"].to_numpy(dtype=float)
    n = o.shape[0]
    fwd = np.full(n, np.nan)
    if n > 2:
        fwd[:-2] = np.log(o[2:] / o[1:-1])
    y = np.where(np.isnan(fwd), np.nan, (fwd > 0).astype(float))
    label_end = pd.Series(df.index, index=df.index).shift(-2)
    return pd.DataFrame(
        {"y": y, "fwd_logret": fwd, "label_end_ts": label_end.to_numpy()}, index=df.index
    )


def build_dataset(bars: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Pooled (all tickers, no ticker-ID feature) labeled dataset.

    Columns: ts, ticker, *FEATURE_NAMES, y (int), fwd_logret, label_end_ts.
    Rows with any NaN feature or no label are dropped. Sorted by (ts,
    ticker), RangeIndex — chronological, never shuffled.
    """
    feats = build_features(bars)
    frames = []
    for ticker in sorted(feats):
        f = feats[ticker]
        lab = make_labels(bars[ticker])
        frame = pd.concat([f, lab], axis=1)
        frame.insert(0, "ticker", ticker)
        frame.insert(0, "ts", frame.index)
        frames.append(frame)
    if not frames:
        raise ValueError("build_dataset: no bars")
    ds = pd.concat(frames, ignore_index=True)
    ds = ds.dropna(subset=list(FEATURE_NAMES) + ["y"])
    ds["y"] = ds["y"].astype(int)
    ds["label_end_ts"] = pd.to_datetime(ds["label_end_ts"], utc=True)
    ds = ds.sort_values(["ts", "ticker"], kind="mergesort").reset_index(drop=True)
    return ds


@dataclass(frozen=True)
class Split:
    train: pd.DataFrame
    test: pd.DataFrame
    test_start: pd.Timestamp
    trained_through: pd.Timestamp  # last training label's end bar


def session_dates(ds: pd.DataFrame) -> pd.DatetimeIndex:
    """Sorted unique bar timestamps (tz-aware) present in the dataset."""
    return pd.DatetimeIndex(ds["ts"].unique()).sort_values()


def _embargoed_train(ds: pd.DataFrame, test_start: pd.Timestamp) -> pd.DataFrame:
    dates = session_dates(ds)
    pos = int(dates.searchsorted(test_start))
    embargo_start = dates[max(pos - EMBARGO_SESSIONS, 0)]
    mask = (
        (ds["ts"] < embargo_start)
        & (ds["ts"] < test_start)
        & (ds["label_end_ts"] < test_start)
    )
    return ds.loc[mask]


def chronological_split(
    ds: pd.DataFrame, test_sessions: int = TEST_SESSIONS
) -> Split:
    """Test = the most recent `test_sessions` sessions (all tickers). Train =
    everything before, minus the embargo."""
    dates = session_dates(ds)
    if len(dates) < test_sessions + EMBARGO_SESSIONS + 1:
        raise ValueError(
            f"Need more than {test_sessions + EMBARGO_SESSIONS} labeled sessions "
            f"for a {test_sessions}-session holdout; have {len(dates)}."
        )
    test_start = pd.Timestamp(dates[-test_sessions])
    test = ds.loc[ds["ts"] >= test_start]
    train = _embargoed_train(ds, test_start)
    if train.empty:
        raise ValueError("chronological_split: empty training set after embargo")
    return Split(
        train=train,
        test=test,
        test_start=test_start,
        trained_through=pd.Timestamp(train["label_end_ts"].max()),
    )


@dataclass(frozen=True)
class Fold:
    name: str
    train: pd.DataFrame
    test: pd.DataFrame
    test_start: pd.Timestamp


def walk_forward_folds(ds: pd.DataFrame) -> Iterator[Fold]:
    """Expanding-window folds: one per calendar year in WALK_FORWARD_YEARS,
    plus the trailing TEST_SESSIONS (the deployed model's own holdout). Each
    fold trains on everything before its test start, minus the embargo.
    Reporting only — no fold's model is ever deployed."""
    years = ds["ts"].dt.year
    for year in WALK_FORWARD_YEARS:
        test = ds.loc[years == year]
        if test.empty:
            continue
        test_start = pd.Timestamp(test["ts"].min())
        train = _embargoed_train(ds, test_start)
        if train.empty:
            continue
        yield Fold(name=str(year), train=train, test=test, test_start=test_start)
    split = chronological_split(ds)
    yield Fold(
        name=f"trailing_{TEST_SESSIONS}", train=split.train, test=split.test,
        test_start=split.test_start,
    )
