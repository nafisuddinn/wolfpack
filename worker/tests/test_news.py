"""News ingestion + storage for The Scout (news.py, news_backfill.py, migration).

No network: the Alpaca fetcher is replaced by an in-memory fake; the
Supabase store is exercised against a recording fake client.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from wolfpack_worker import news
from wolfpack_worker.news import (
    ARTICLE_COLUMNS,
    LocalNewsStore,
    SupabaseNewsStore,
    article_from_raw,
    refresh_news,
)

UTC = timezone.utc
REPO_ROOT = Path(__file__).resolve().parents[2]


def raw(i, created, symbols=("AAPL",), headline=None, updated=None):
    return {
        "id": i,
        "headline": headline or f"headline {i}",
        "source": "benzinga",
        "url": f"https://example.invalid/{i}",
        "summary": "SUMMARY-NOT-STORED",
        "content": "<p>BODY-NOT-STORED</p>",
        "author": "someone",
        "created_at": created,
        "updated_at": updated or created,
        "symbols": list(symbols),
        "images": [],
    }


def art(i, created, symbols=("AAPL",), headline=None):
    return article_from_raw(raw(i, created, symbols, headline))


class FakeFetcher:
    def __init__(self, articles=(), fail=None):
        self.articles = list(articles)
        self.calls = []
        self.fail = fail

    def fetch(self, symbols, start, end):
        self.calls.append((tuple(symbols), start, end))
        if self.fail:
            raise self.fail
        return [a for a in self.articles if start <= a.created_at <= end and set(a.symbols) & set(symbols)]


# --- parsing --------------------------------------------------------------------------


def test_article_from_raw_keeps_headline_only_fields_and_utc_times():
    a = article_from_raw(raw(7, "2026-01-05T15:00:00Z", ("AAPL", "MSFT"), updated="2026-01-05T15:00:01Z"))
    assert a.id == 7 and a.symbols == ("AAPL", "MSFT") and a.source == "benzinga"
    assert a.created_at == datetime(2026, 1, 5, 15, tzinfo=UTC)
    assert a.vendor_updated_at == datetime(2026, 1, 5, 15, 0, 1, tzinfo=UTC)
    # Body/summary are never kept (headline only, per design).
    assert not any("NOT-STORED" in str(v) for v in a.__dict__.values())


def test_article_from_raw_refuses_naive_timestamps():
    with pytest.raises(ValueError, match="timezone"):
        article_from_raw(raw(1, "2026-01-05T15:00:00"))


def test_article_from_raw_accepts_datetime_objects():
    a = article_from_raw(raw(1, datetime(2026, 1, 5, 15, tzinfo=UTC)))
    assert a.created_at.tzinfo is not None


# --- local store ------------------------------------------------------------------------


def test_insert_never_overwrites_first_seen_or_headline(tmp_path):
    store = LocalNewsStore(tmp_path / "n.jsonl")
    t0 = datetime(2026, 1, 6, 22, tzinfo=UTC)
    assert store.insert_new([art(1, "2026-01-05T15:00:00Z", headline="first")], first_seen_at=t0,
                            ingest_mode="live") == 1
    later = t0 + timedelta(days=3)
    assert store.insert_new([art(1, "2026-01-05T15:00:00Z", headline="REVISED"), art(2, "2026-01-05T16:00:00Z")],
                            first_seen_at=later, ingest_mode="backfill") == 1
    df = store.get_articles(["AAPL"], datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC))
    row = df.set_index("id").loc[1]
    assert row["headline"] == "first" and row["first_seen_at"] == t0 and row["ingest_mode"] == "live"
    # Reload from disk: same (append-only JSONL).
    again = LocalNewsStore(tmp_path / "n.jsonl")
    df2 = again.get_articles(["AAPL"], datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC))
    pd.testing.assert_frame_equal(df.reset_index(drop=True), df2.reset_index(drop=True))


def test_get_articles_window_is_open_closed_on_created_at_and_filters_symbols():
    store = LocalNewsStore()
    t = datetime(2026, 1, 5, 21, tzinfo=UTC)
    store.insert_new([
        art(1, "2026-01-05T21:00:00Z"),            # exactly the end -> included
        art(2, "2026-01-05T21:00:01Z"),            # after the end -> excluded
        art(3, "2026-01-04T21:00:00Z"),            # exactly the start -> excluded (open)
        art(4, "2026-01-05T12:00:00Z", ("MSFT",)),  # other symbol -> excluded
        art(5, "2026-01-05T12:00:00Z", ("MSFT", "XOM")),
    ], first_seen_at=t, ingest_mode="backfill")
    df = store.get_articles(["AAPL", "XOM"], t - timedelta(days=1), t)
    assert sorted(df["id"]) == [1, 5]
    assert list(df.columns) == list(ARTICLE_COLUMNS)
    assert all(ts.tzinfo is not None for ts in df["created_at"])


def test_store_rejects_naive_query_bounds_and_bad_ingest_mode():
    store = LocalNewsStore()
    with pytest.raises(ValueError):
        store.get_articles(["AAPL"], datetime(2026, 1, 1), datetime(2026, 1, 2, tzinfo=UTC))
    with pytest.raises(ValueError):
        store.insert_new([], first_seen_at=datetime(2026, 1, 1, tzinfo=UTC), ingest_mode="other")


def test_store_coverage_bounds():
    store = LocalNewsStore()
    assert store.earliest_created_at() is None and store.latest_created_at() is None
    store.insert_new([art(1, "2026-01-05T15:00:00Z"), art(2, "2026-01-02T15:00:00Z")],
                     first_seen_at=datetime(2026, 1, 6, tzinfo=UTC), ingest_mode="backfill")
    assert store.earliest_created_at() == datetime(2026, 1, 2, 15, tzinfo=UTC)
    assert store.latest_created_at() == datetime(2026, 1, 5, 15, tzinfo=UTC)


# --- refresh (daily, live) ----------------------------------------------------------------


def test_refresh_news_refetches_the_feature_lookback_and_stamps_live():
    now = datetime(2026, 1, 6, 21, 40, tzinfo=UTC)
    fetcher = FakeFetcher([art(1, "2026-01-06T15:00:00Z"), art(2, "2025-12-20T15:00:00Z")])
    store = LocalNewsStore()
    res = refresh_news(fetcher=fetcher, store=store, universe=("AAPL", "SPY"), now=now)
    assert res.ok and res.status == "ok" and res.n_fetched == 2 and res.n_inserted == 2
    assert res.cutoff == now
    (symbols, start, end), = fetcher.calls
    assert symbols == ("AAPL", "SPY") and end == now and start == now - timedelta(days=news.LIVE_REFETCH_DAYS)
    df = store.get_articles(["AAPL"], now - timedelta(days=60), now)
    assert set(df["ingest_mode"]) == {"live"} and set(df["first_seen_at"]) == {now}


def test_refresh_news_failure_is_reported_not_raised():
    now = datetime(2026, 1, 6, 21, 40, tzinfo=UTC)
    res = refresh_news(fetcher=FakeFetcher(fail=RuntimeError("boom")), store=LocalNewsStore(),
                       universe=("AAPL",), now=now)
    assert not res.ok and res.status == "failed" and "boom" in res.error and res.cutoff is None


def test_refresh_news_requires_aware_now():
    with pytest.raises(ValueError):
        refresh_news(fetcher=FakeFetcher(), store=LocalNewsStore(), universe=("AAPL",), now=datetime(2026, 1, 6))


# --- Supabase store (recording fake client) -------------------------------------------------


class _Resp:
    def __init__(self, data):
        self.data = data


class _Builder:
    def __init__(self, client, table):
        self.client, self.table, self.ops = client, table, []

    def __getattr__(self, name):
        def op(*args, **kwargs):
            self.ops.append((name, args, kwargs))
            return self
        return op

    def execute(self):
        self.client.executed.append((self.table, self.ops))
        names = [o[0] for o in self.ops]
        if "upsert" in names:
            rows = next(o for o in self.ops if o[0] == "upsert")[1][0]
            return _Resp(rows[:1])  # pretend only the first row was new
        if "range" in names:
            _, (lo, _hi), _ = next(o for o in self.ops if o[0] == "range")
            return _Resp(self.client.rows if lo == 0 else [])
        return _Resp(self.client.rows[:1])


class _Client:
    def __init__(self, rows=()):
        self.rows, self.executed = list(rows), []

    def table(self, name):
        return _Builder(self, name)


def test_supabase_insert_is_on_conflict_do_nothing_with_explicit_first_seen():
    client = _Client()
    store = SupabaseNewsStore(client)
    t0 = datetime(2026, 1, 6, 22, tzinfo=UTC)
    n = store.insert_new([art(1, "2026-01-05T15:00:00Z"), art(2, "2026-01-05T16:00:00Z")], first_seen_at=t0,
                         ingest_mode="live")
    assert n == 1
    (table, ops), = client.executed
    assert table == "news_articles"
    name, args, kwargs = ops[0]
    assert name == "upsert" and kwargs["on_conflict"] == "id" and kwargs["ignore_duplicates"] is True
    rows = args[0]
    assert {r["first_seen_at"] for r in rows} == {t0.isoformat()} and {r["ingest_mode"] for r in rows} == {"live"}
    assert set(rows[0]) == {"id", "created_at", "vendor_updated_at", "headline", "source", "url", "symbols",
                            "first_seen_at", "ingest_mode"}


def test_supabase_get_articles_pages_and_parses():
    rows = [{"id": 1, "created_at": "2026-01-05T15:00:00+00:00", "vendor_updated_at": None, "headline": "h",
             "source": "benzinga", "url": None, "symbols": ["AAPL"], "first_seen_at": "2026-01-06T22:00:00+00:00",
             "ingest_mode": "live"}]
    client = _Client(rows)
    df = SupabaseNewsStore(client).get_articles(["AAPL"], datetime(2026, 1, 1, tzinfo=UTC),
                                                datetime(2026, 2, 1, tzinfo=UTC))
    assert list(df["id"]) == [1] and df["created_at"].iloc[0] == datetime(2026, 1, 5, 15, tzinfo=UTC)
    ops = [o[0] for _, oplist in client.executed for o in oplist]
    assert "ov" in ops and "gt" in ops and "lte" in ops and "range" in ops


# --- backfill CLI core ------------------------------------------------------------------------


def test_backfill_walks_monthly_chunks_and_stamps_backfill():
    from wolfpack_worker.news_backfill import backfill_news

    fetcher = FakeFetcher([art(1, "2016-01-10T15:00:00Z"), art(2, "2016-02-10T15:00:00Z"),
                           art(3, "2016-03-10T15:00:00Z")])
    store = LocalNewsStore()
    now = datetime(2026, 10, 4, tzinfo=UTC)
    out = backfill_news(fetcher=fetcher, store=store, universe=("AAPL",), since=date(2016, 1, 4),
                        until=datetime(2016, 3, 20, tzinfo=UTC), now=now, sleep=lambda s: None)
    assert out["n_inserted"] == 3
    starts = [c[1] for c in fetcher.calls]
    assert starts[0] == datetime(2016, 1, 4, tzinfo=UTC) and starts == sorted(starts)
    # Chunks tile the range with no gaps.
    for (_, s1, e1), (_, s2, _) in zip(fetcher.calls, fetcher.calls[1:]):
        assert s2 == e1
    df = store.get_articles(["AAPL"], datetime(2016, 1, 1, tzinfo=UTC), now)
    assert set(df["ingest_mode"]) == {"backfill"} and set(df["first_seen_at"]) == {now}


def test_backfill_retries_a_rate_limited_chunk():
    from wolfpack_worker.news_backfill import backfill_news

    class Flaky(FakeFetcher):
        def fetch(self, symbols, start, end):
            if not getattr(self, "failed", False):
                self.failed = True
                raise RuntimeError("429 too many requests")
            return super().fetch(symbols, start, end)

    sleeps = []
    out = backfill_news(fetcher=Flaky([art(1, "2016-01-10T15:00:00Z")]), store=LocalNewsStore(),
                        universe=("AAPL",), since=date(2016, 1, 4), until=datetime(2016, 1, 20, tzinfo=UTC),
                        now=datetime(2026, 10, 4, tzinfo=UTC), sleep=sleeps.append)
    assert out["n_inserted"] == 1 and sleeps


# --- migration ------------------------------------------------------------------------------


def test_migration_keeps_news_private_and_insert_once():
    sql = (REPO_ROOT / "supabase" / "migrations" / "20261004_0003_news_articles.sql").read_text().lower()
    assert "create table if not exists news_articles" in sql
    assert "enable row level security" in sql
    assert "to service_role" in sql
    assert "to anon" not in sql  # no anon policy at all
    assert "revoke all on table news_articles from anon, authenticated" in sql
    assert "before update on news_articles" in sql
    assert "using gin (symbols)" in sql
    assert "ingest_mode in ('backfill', 'live')" in sql


def test_news_cache_is_gitignored():
    gi = (REPO_ROOT / ".gitignore").read_text()
    assert "worker/data/" in gi
    assert str(news.DEFAULT_CACHE_PATH).startswith(str(REPO_ROOT / "worker" / "data"))
