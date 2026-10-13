"""Synthetic market calendar + headlines for The Scout's tests (no network).

Sessions use real New York wall-clock rules: 09:30-16:00 ET, 13:00 ET on
given early-close days, converted to UTC with DST (so a close is 21:00 UTC
in winter and 20:00 UTC in summer). Headline text is synthetic; scores are
either VADER on that text or given explicitly.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Iterable, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from wolfpack_worker.broker import Session
from wolfpack_worker.news import articles_frame

NY = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
UNIVERSE = ("SPY", "QQQ", "AAPL", "JPM", "XOM")


def ny(d: date, hh: int, mm: int = 0) -> datetime:
    return datetime.combine(d, time(hh, mm), tzinfo=NY).astimezone(UTC)


def make_sessions(days: Iterable[date], early_close: Sequence[date] = ()) -> tuple[Session, ...]:
    early = set(early_close)
    return tuple(
        Session(date=d, open=ny(d, 9, 30), close=ny(d, 13) if d in early else ny(d, 16))
        for d in days
    )


def business_sessions(start: str, n: int, early_close: Sequence[date] = ()) -> tuple[Session, ...]:
    days = [ts.date() for ts in pd.bdate_range(start=start, periods=n)]
    return make_sessions(days, early_close)


def bar_index(sessions: Sequence[Session]) -> pd.DatetimeIndex:
    """Alpaca daily bars are stamped at midnight America/New_York (in UTC)."""
    return pd.DatetimeIndex([pd.Timestamp(datetime.combine(s.date, time(0), tzinfo=NY)).tz_convert("UTC")
                             for s in sessions])


def articles(rows: Iterable[tuple], *, first_seen: datetime | None = None, ingest_mode: str = "backfill"):
    """rows: (id, created_at datetime, symbols tuple, headline or None, [score])."""
    out = []
    for r in rows:
        i, created, symbols, headline = r[:4]
        rec = {"id": i, "created_at": pd.Timestamp(created).isoformat(), "vendor_updated_at": None,
               "headline": headline if headline is not None else f"neutral update number {i}",
               "source": "benzinga", "url": f"https://example.invalid/{i}", "symbols": list(symbols),
               "first_seen_at": pd.Timestamp(first_seen or created).isoformat(), "ingest_mode": ingest_mode}
        out.append(rec)
    df = articles_frame(out)
    scores = {r[0]: r[4] for r in rows if len(r) > 4}
    if scores:
        df["score"] = df["id"].map(scores).astype(float)
    return df


POS = "Company posts record profit, shares soar on excellent growth"
NEG = "Company faces lawsuit, terrible losses and fraud probe"


def random_articles(sessions: Sequence[Session], *, seed: int = 0, per_session: float = 3.0,
                    universe: Sequence[str] = UNIVERSE, with_scores: bool = True) -> pd.DataFrame:
    """Random headlines spread over [first open - 2 days, last close + 2 days],
    including weekends/after-hours; scores drawn uniformly in [-1, 1]."""
    rng = np.random.default_rng(seed)
    lo = pd.Timestamp(sessions[0].open) - pd.Timedelta(days=2)
    hi = pd.Timestamp(sessions[-1].close) + pd.Timedelta(days=2)
    n = int(per_session * len(sessions))
    secs = rng.integers(0, int((hi - lo).total_seconds()), size=n)
    rows = []
    for k in range(n):
        syms = tuple(sorted(rng.choice(list(universe) + ["MSFT", "TSLA"], size=rng.integers(1, 4), replace=False)))
        rows.append((10_000 + k, (lo + pd.Timedelta(seconds=int(secs[k]))).to_pydatetime(), syms, None,
                     float(np.round(rng.uniform(-1, 1), 4)) if with_scores else None))
    if not with_scores:
        rows = [r[:4] for r in rows]
    return articles(rows)


def make_bars(sessions: Sequence[Session], *, seed: int = 0, universe: Sequence[str] = UNIVERSE) -> dict:
    idx = bar_index(sessions)
    out = {}
    for k, t in enumerate(universe):
        rng = np.random.default_rng(seed + 31 * k)
        r = rng.normal(0.0003, 0.01, size=len(idx))
        close = (60.0 + 25 * k) * np.exp(np.cumsum(r))
        open_ = np.concatenate([[close[0]], close[:-1]]) * np.exp(rng.normal(0, 0.002, size=len(idx)))
        out[t] = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * 1.003,
                               "low": np.minimum(open_, close) * 0.997, "close": close,
                               "volume": rng.integers(1e6, 5e6, size=len(idx)).astype(float)}, index=idx)
    return out


SYNTH_N = 560
SECRET = "SECRET-HEADLINE-TEXT"


def synthetic(signal: bool, seed: int = 0):
    """Sessions, bars and scored headlines. One headline per ticker per
    session at 15:00 ET with tone +/-0.8; if `signal`, the label of row t
    (open t+1 -> open t+2) agrees with that tone 80% of the time."""
    sessions = business_sessions("2023-01-02", SYNTH_N)
    idx = bar_index(sessions)
    rng = np.random.default_rng(seed)
    bars, rows, aid = {}, [], 1
    for k, t in enumerate(UNIVERSE):
        tone = rng.choice([-0.8, 0.8], size=SYNTH_N)
        mag = np.abs(rng.normal(0, 0.01, size=SYNTH_N))
        sign = np.where(rng.random(SYNTH_N) < 0.8, np.sign(tone), -np.sign(tone)) if signal else rng.choice([-1, 1], SYNTH_N)
        o = np.empty(SYNTH_N)
        o[0] = o[1] = 50.0 + 10 * k
        for i in range(SYNTH_N - 2):
            o[i + 2] = o[i + 1] * np.exp(sign[i] * mag[i])  # label of row i
        close = o * np.exp(rng.normal(0, 0.002, SYNTH_N))
        bars[t] = pd.DataFrame({"open": o, "high": np.maximum(o, close) * 1.002, "low": np.minimum(o, close) * 0.998,
                                "close": close, "volume": np.full(SYNTH_N, 1e6)}, index=idx)
        for i, s in enumerate(sessions):
            rows.append((aid, s.close - timedelta(hours=1), (t,), f"{SECRET} {aid}", float(tone[i])))
            aid += 1
    return sessions, bars, articles(rows)
