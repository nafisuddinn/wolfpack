"""StrategyContext.news / truncate_news / requires_news / the daily refresh step.

The news a strategy can see is checked the same way bars are: a
StrategyContext cannot be built with an article published after as_of, with
a naive timestamp, or with an article first seen after the run's news
cutoff. The orchestrator truncates first (truncate_news); the context is the
defense-in-depth check that refuses anything left over.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from fakes import FakeBroker, InMemoryPriceStore, InMemoryTradeRepo
from wolfpack_worker.broker import Session
from wolfpack_worker.daily_trades import refresh_news_safely, run_persona
from wolfpack_worker.execution import FixedNotionalSizer
from wolfpack_worker.news import LocalNewsStore, NewsRefreshResult, article_from_raw, articles_frame
from wolfpack_worker.strategies.base import (
    LookaheadError,
    NewsSnapshot,
    StrategyContext,
    TargetPosition,
    truncate_news,
)

UTC = timezone.utc
AS_OF = datetime(2026, 1, 6, 21, 0, tzinfo=UTC)  # 16:00 ET close (EST)


def _rows(*specs):
    """specs: (id, created_at, first_seen_at)"""
    return articles_frame([
        {"id": i, "created_at": c, "vendor_updated_at": c, "headline": f"h{i}", "source": "benzinga",
         "url": None, "symbols": ["AAPL"], "first_seen_at": f, "ingest_mode": "live"}
        for i, c, f in specs
    ])


def _sessions(n=3, end=date(2026, 1, 6)):
    days = pd.bdate_range(end=pd.Timestamp(end), periods=n)
    return tuple(Session(date=d.date(), open=datetime(d.year, d.month, d.day, 14, 30, tzinfo=UTC),
                         close=datetime(d.year, d.month, d.day, 21, 0, tzinfo=UTC)) for d in days)


def _snap(articles, cutoff=AS_OF + timedelta(minutes=40), sessions=None, status="ok"):
    return NewsSnapshot(articles=articles, sessions=sessions or _sessions(), cutoff=cutoff, status=status)


def test_context_accepts_news_up_to_as_of():
    arts = _rows((1, "2026-01-06T21:00:00Z", "2026-01-06T21:30:00Z"))
    ctx = StrategyContext(as_of=AS_OF, universe=("AAPL",), bars={}, news=_snap(arts))
    assert len(ctx.news.articles) == 1


def test_context_rejects_an_article_published_after_as_of():
    arts = _rows((1, "2026-01-06T21:00:01Z", "2026-01-06T21:30:00Z"))  # 1s after the close
    with pytest.raises(LookaheadError, match="after as_of"):
        StrategyContext(as_of=AS_OF, universe=("AAPL",), bars={}, news=_snap(arts))


def test_context_rejects_an_article_first_seen_after_the_cutoff():
    arts = _rows((1, "2026-01-06T20:00:00Z", "2026-01-06T22:30:00Z"))
    with pytest.raises(LookaheadError, match="first seen"):
        StrategyContext(as_of=AS_OF, universe=("AAPL",), bars={},
                        news=_snap(arts, cutoff=datetime(2026, 1, 6, 21, 40, tzinfo=UTC)))


def test_context_rejects_naive_news_timestamps():
    arts = _rows((1, "2026-01-06T20:00:00Z", "2026-01-06T21:30:00Z"))
    arts["created_at"] = arts["created_at"].dt.tz_localize(None)
    with pytest.raises(LookaheadError, match="timezone"):
        StrategyContext(as_of=AS_OF, universe=("AAPL",), bars={}, news=_snap(arts))
    good = _rows((1, "2026-01-06T20:00:00Z", "2026-01-06T21:30:00Z"))
    with pytest.raises(LookaheadError, match="timezone"):
        StrategyContext(as_of=AS_OF, universe=("AAPL",), bars={}, news=_snap(good, cutoff=datetime(2026, 1, 6, 22)))


def test_context_rejects_a_calendar_that_runs_past_as_of():
    sessions = _sessions(end=date(2026, 1, 7))
    with pytest.raises(LookaheadError, match="session"):
        StrategyContext(as_of=AS_OF, universe=("AAPL",), bars={}, news=_snap(_rows(), sessions=sessions))


def test_failed_snapshot_carries_no_articles():
    with pytest.raises(ValueError):
        NewsSnapshot(articles=_rows((1, "2026-01-06T20:00:00Z", "2026-01-06T21:30:00Z")), sessions=(),
                     cutoff=None, status="failed")
    snap = NewsSnapshot.unavailable("failed", "boom")
    assert snap.status == "failed" and snap.reason == "boom" and snap.articles.empty


def test_truncate_news_drops_future_and_late_rows_and_counts_them():
    cutoff = datetime(2026, 1, 6, 21, 40, tzinfo=UTC)
    arts = _rows(
        (1, "2026-01-06T20:59:59Z", "2026-01-06T21:30:00Z"),  # kept
        (2, "2026-01-06T21:00:00Z", "2026-01-06T21:30:00Z"),  # exactly the close: kept
        (3, "2026-01-06T21:00:01Z", "2026-01-06T21:30:00Z"),  # after the close: dropped
        (4, "2026-01-06T15:00:00Z", "2026-01-07T09:00:00Z"),  # first seen after the cutoff: dropped
    )
    kept, n_after_cutoff = truncate_news(arts, AS_OF, cutoff)
    assert list(kept["id"]) == [1, 2] and n_after_cutoff == 1


# --- run_persona wiring ----------------------------------------------------------------------


@dataclass
class _Probe:
    slug: str = "probe"
    version: str = "probe/v1"
    lookback_bars: int = 5
    requires_news: bool = False
    news_lookback_sessions: int = 2
    seen: list = field(default_factory=list)

    def evaluate(self, ctx: StrategyContext) -> list[TargetPosition]:
        self.seen.append(ctx.news)
        return []


class _ExplodingStore(LocalNewsStore):
    def get_articles(self, *a, **k):
        raise AssertionError("a strategy that does not require news must not read the news store")


def _broker():
    return FakeBroker(sessions=list(_sessions(n=10)))


def _run(strategy, store, refresh):
    return run_persona(strategy=strategy, broker=_broker(), price_store=InMemoryPriceStore(),
                       trade_repo=InMemoryTradeRepo(), as_of=AS_OF, universe=("AAPL",), sizer=FixedNotionalSizer(),
                       run_id="r", dry_run=True, news_store=store, news_refresh=refresh)


def test_non_news_strategy_never_touches_news():
    p = _Probe()
    _run(p, _ExplodingStore(), NewsRefreshResult("ok", cutoff=AS_OF + timedelta(minutes=30)))
    assert p.seen == [None]


def test_news_strategy_gets_a_truncated_point_in_time_snapshot():
    store = LocalNewsStore()
    cutoff = AS_OF + timedelta(minutes=30)
    store.insert_new([
        article_from_raw({"id": 1, "created_at": "2026-01-06T20:00:00Z", "headline": "a", "source": "benzinga",
                          "symbols": ["AAPL"]}),
        article_from_raw({"id": 2, "created_at": "2026-01-06T21:10:00Z", "headline": "after close",
                          "source": "benzinga", "symbols": ["AAPL"]}),
    ], first_seen_at=cutoff, ingest_mode="live")
    store.insert_new([article_from_raw({"id": 3, "created_at": "2025-11-01T20:00:00Z", "headline": "old",
                                        "source": "benzinga", "symbols": ["AAPL"]})],
                     first_seen_at=datetime(2025, 11, 2, tzinfo=UTC), ingest_mode="backfill")
    p = _Probe(requires_news=True)
    _run(p, store, NewsRefreshResult("ok", cutoff=cutoff))
    snap = p.seen[0]
    assert snap.status == "ok" and list(snap.articles["id"]) == [1]
    assert snap.sessions[-1].close == AS_OF and len(snap.sessions) == p.news_lookback_sessions + 1
    assert snap.cutoff == cutoff


def test_news_strategy_sees_failed_refresh_as_unavailable():
    p = _Probe(requires_news=True)
    _run(p, LocalNewsStore(), NewsRefreshResult("failed", error="boom"))
    assert p.seen[0].status == "failed" and "boom" in p.seen[0].reason


def test_news_strategy_sees_missing_history_as_stale():
    store = LocalNewsStore()
    store.insert_new([article_from_raw({"id": 1, "created_at": "2026-01-06T20:00:00Z", "headline": "a",
                                        "source": "benzinga", "symbols": ["AAPL"]})],
                     first_seen_at=AS_OF, ingest_mode="live")
    p = _Probe(requires_news=True)
    _run(p, store, NewsRefreshResult("ok", cutoff=AS_OF + timedelta(minutes=30)))
    assert p.seen[0].status == "stale" and "backfill" in p.seen[0].reason


def test_refresh_news_safely_never_raises():
    class Boom:
        def fetch(self, *a):
            raise RuntimeError("network down")

    res = refresh_news_safely(lambda: Boom(), LocalNewsStore(), ("AAPL",), AS_OF)
    assert res.status == "failed"

    def broken_factory():
        raise RuntimeError("cannot build client")

    res = refresh_news_safely(broken_factory, LocalNewsStore(), ("AAPL",), AS_OF)
    assert res.status == "failed" and "cannot build client" in res.error


def test_late_arrivals_count_live_rows_first_seen_after_their_own_decision():
    from wolfpack_worker.daily_trades import count_late_arrivals

    arts = _rows(
        (1, "2026-01-05T15:00:00Z", "2026-01-06T21:30:00Z"),  # window of 01-05 (close 21:00), seen next day: late
        (2, "2026-01-06T20:00:00Z", "2026-01-06T21:30:00Z"),  # seen 30 min after its own close: on time
        (3, "2026-01-05T21:00:00Z", "2026-01-05T23:59:00Z"),  # exactly the 01-05 close, seen within 3h: on time
        (4, "2026-01-05T21:00:01Z", "2026-01-06T21:30:00Z"),  # 1s after 01-05 close -> 01-06 window: on time
    )
    closes = [s.close for s in _sessions()]
    assert count_late_arrivals(arts, closes) == 1
    arts.loc[arts["id"] == 1, "ingest_mode"] = "backfill"  # backfilled rows are never "late"
    assert count_late_arrivals(arts, closes) == 0
