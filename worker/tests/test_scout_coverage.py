"""M1 coverage / revision report (scout/coverage.py): no prices, no labels."""

from __future__ import annotations

import inspect
from datetime import date, timedelta

import pandas as pd

from scout_helpers import UNIVERSE, articles, business_sessions, ny
from wolfpack_worker.news import articles_frame
from wolfpack_worker.scout import coverage as C

S = business_sessions("2026-01-05", 40)


def _arts_with_updates(specs):
    """specs: (id, created, updated, symbols)"""
    return articles_frame([
        {"id": i, "created_at": pd.Timestamp(c).isoformat(), "vendor_updated_at": pd.Timestamp(u).isoformat(),
         "headline": "x", "source": "benzinga", "url": None, "symbols": list(sy),
         "first_seen_at": pd.Timestamp(c).isoformat(), "ingest_mode": "backfill"}
        for i, c, u, sy in specs
    ])


def test_revised_after_decision_uses_the_articles_own_window_close():
    t = 10
    c = ny(S[t].date, 15)
    arts = _arts_with_updates([
        (1, c, c + timedelta(seconds=1), ("AAPL",)),          # trivial revision before the close: not flagged
        (2, c, S[t].close + timedelta(minutes=5), ("AAPL",)),  # revised after the decision: flagged
        (3, c, c, ("XOM",)),
    ])
    assert list(C.revised_after_decision(arts, S)) == [False, True, False]
    flags = C.revision_flags(arts, S, UNIVERSE)
    assert flags["AAPL"][t] and flags["AAPL"].sum() == 1 and not flags["XOM"].any()


def test_month_gaps_detect_a_missing_backfill_month():
    sessions = business_sessions("2026-01-05", 70)  # Jan..Apr
    rows = [(i, ny(s.date, 12), ("AAPL",), None) for i, s in enumerate(sessions) if s.date.month != 2]
    assert C.month_gaps(articles(rows), sessions, UNIVERSE) == ["2026-02"]


def test_report_shape_and_timing_phases():
    t = 12
    d = S[t].date
    arts = articles([
        (1, ny(d, 8), ("AAPL",), None),            # pre-open
        (2, ny(d, 12), ("AAPL", "SPY"), None),     # in session
        (3, ny(d, 17), ("XOM",), None),            # after the close
        (4, ny(date(2026, 1, 10), 12), ("JPM",), None),  # Saturday
        (5, ny(d, 12), ("MSFT",), None),           # not in the universe: ignored
    ])
    rep = C.coverage_report(arts, S, UNIVERSE)
    assert rep["n_articles_universe"] == 4
    assert rep["labels_or_prices_used"] is False
    assert set(rep["timing_share"]) == {"pre_open", "in_session", "after_close", "non_session_day"}
    assert rep["per_ticker"]["AAPL"]["headlines"] == 2
    assert rep["revisions"]["share_revised_after_own_decision_close"] == 0.0
    assert rep["live_latency_minutes"]["n_live"] == 0


def test_report_takes_no_prices_or_labels():
    params = inspect.signature(C.coverage_report).parameters
    assert list(params) == ["articles", "sessions", "universe"]
    src = inspect.getsource(C)
    assert "make_labels" not in src and "fwd_logret" not in src
