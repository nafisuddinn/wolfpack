"""The Scout's labeled dataset: scout_v1 features + The Analyst's labels.

Labels are imported, not re-implemented: `analyst.dataset.make_labels`,
y_t = 1[ln(O_{t+2}/O_{t+1}) > 0] (decision after the close of t, fill at
the open of t+1, revisable at the open of t+2), label_end_ts = bar t+2.
So the Scout's rows, holdout, embargo and gate are directly comparable with
The Analyst's (design section 2), and the chronological split / embargo /
walk-forward functions in analyst.dataset apply unchanged.

Row (ts = bar t, ticker) gets the features of the session that bar belongs
to: headlines with created_at <= close_t only (scout/windows.py). Rows
before the 25-window warmup, and the last two bars (no label), are dropped.
Sorted by (ts, ticker), never shuffled.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from wolfpack_worker.analyst.dataset import make_labels
from wolfpack_worker.broker import Session
from wolfpack_worker.scout.features import get_scout_feature_spec
from wolfpack_worker.scout.windows import bar_sessions
from wolfpack_worker.universe import UNIVERSE

__all__ = ["build_scout_dataset", "make_labels"]


def build_scout_dataset(
    bars: Mapping[str, pd.DataFrame],
    articles: pd.DataFrame,
    sessions: Sequence[Session],
    spec_version: str = "scout_v1",
    universe: Sequence[str] = UNIVERSE,
) -> pd.DataFrame:
    """Columns: ts, ticker, *spec names, y (int), fwd_logret, label_end_ts."""
    spec = get_scout_feature_spec(spec_version)
    feats = spec.build_fn(articles, sessions, universe)
    names = list(spec.names)
    frames = []
    for ticker in sorted(bars):
        df = bars[ticker]
        if df is None or len(df) == 0:
            continue
        if ticker not in feats:
            raise ValueError(f"{ticker} has bars but is not in the feature universe {list(universe)}")
        pos = bar_sessions(df.index, sessions)
        f = feats[ticker].iloc[pos]
        f.index = df.index
        lab = make_labels(df)
        frame = pd.concat([f, lab], axis=1)
        frame.insert(0, "ticker", ticker)
        frame.insert(0, "ts", frame.index)
        frames.append(frame)
    if not frames:
        raise ValueError("build_scout_dataset: no bars")
    ds = pd.concat(frames, ignore_index=True)
    ds = ds.dropna(subset=names + ["y"])
    ds["y"] = ds["y"].astype(int)
    ds["label_end_ts"] = pd.to_datetime(ds["label_end_ts"], utc=True)
    ds = ds.sort_values(["ts", "ticker"], kind="mergesort").reset_index(drop=True)
    if not np.isfinite(ds[names].to_numpy(dtype=float)).all():
        raise ValueError("non-finite Scout feature value after warmup")
    return ds
