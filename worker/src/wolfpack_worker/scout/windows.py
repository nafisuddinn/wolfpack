"""Point-in-time session windows for The Scout's news features.

Decision time for session t is its close (`ctx.as_of` = close_t, from the
market calendar, so 13:00 ET early closes and the DST switch between 21:00
and 20:00 UTC closes are exact). The news window is

    W_t = (close_{t-1}, close_t]   on the article's vendor created_at.

So an article stamped exactly at the close counts for that session, and one
second later it belongs to the next session. Weekend, holiday and overnight
articles roll forward to the next session. The first session of a calendar
only bounds W_1; articles at or before its close (or after the last close)
get window -1 and are ignored.

These boundaries are what keep a post-close "why XYZ fell today" article out
of the decision made at that close.
"""

from __future__ import annotations

from typing import Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from wolfpack_worker.broker import Session

MARKET_TZ = ZoneInfo("America/New_York")


def session_closes(sessions: Sequence[Session]) -> pd.DatetimeIndex:
    for s in sessions:
        if s.close.tzinfo is None or s.close.tzinfo.utcoffset(s.close) is None:
            raise ValueError(f"session {s.date} close must be timezone-aware")
    closes = pd.DatetimeIndex(pd.to_datetime([s.close for s in sessions], utc=True))
    if len(closes) > 1 and not (np.diff(closes.asi8) > 0).all():
        raise ValueError("sessions must be sorted by close, strictly increasing")
    return closes


def assign_windows(created_at: pd.DatetimeIndex, sessions: Sequence[Session]) -> np.ndarray:
    """Window (session position) of each timestamp: i with
    close_{i-1} < created_at <= close_i, i >= 1; -1 if outside the calendar."""
    created_at = pd.DatetimeIndex(created_at)
    if len(created_at) and created_at.tz is None:
        raise ValueError("created_at must be timezone-aware")
    closes = session_closes(sessions)
    if len(created_at) == 0 or len(closes) == 0:
        return np.full(len(created_at), -1, dtype=np.int64)
    idx = closes.searchsorted(created_at.tz_convert("UTC"), side="left").astype(np.int64)
    idx[(idx == 0) | (idx >= len(closes))] = -1
    return idx


def bar_sessions(bar_index: pd.DatetimeIndex, sessions: Sequence[Session]) -> np.ndarray:
    """Position in `sessions` of each daily bar (bars are stamped at midnight
    New York time on their session date). Raises if a bar's date is not a
    calendar session: a price bar on a non-session day is a data bug."""
    bar_index = pd.DatetimeIndex(bar_index)
    if len(bar_index) and bar_index.tz is None:
        raise ValueError("bar index must be timezone-aware")
    pos = {s.date: i for i, s in enumerate(sessions)}
    dates = bar_index.tz_convert(MARKET_TZ).date if len(bar_index) else []
    out = np.empty(len(bar_index), dtype=np.int64)
    for k, d in enumerate(dates):
        if d not in pos:
            raise ValueError(f"bar dated {d} ({bar_index[k]}) is not a session in the market calendar")
        out[k] = pos[d]
    return out
