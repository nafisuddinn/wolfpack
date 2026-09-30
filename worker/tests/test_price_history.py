"""`SupabasePriceStore.get_history` pagination + the full-history backfill.

Supabase/PostgREST caps responses at 1,000 rows by default (server-side
`max_rows`), silently — a `limit(3000)` still returns 1,000. A ~10-year
daily history is ~2,700 rows per ticker, so anything that reads full
history must page. These tests use a fake client that enforces a cap the
same way, so an un-paged implementation fails loudly here instead of
silently training on only the most recent (or oldest) ~4 years.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from analyst_helpers import make_ohlcv
from fakes import FakeBroker, InMemoryPriceStore
from wolfpack_worker.broker import Session
from wolfpack_worker.config import WorkerConfig
from wolfpack_worker.store import SupabasePriceStore


class _FakeQuery:
    def __init__(self, table: "_FakeTable") -> None:
        self._t = table
        self._filters: list = []
        self._order_desc = False
        self._range = None
        self._limit = None
        self._op = "select"
        self._payload = None

    def select(self, *_a, **_k):
        return self

    def eq(self, col, val):
        self._filters.append(lambda r: r[col] == val)
        return self

    def gte(self, col, val):
        self._filters.append(lambda r: r[col] >= val)
        return self

    def lte(self, col, val):
        self._filters.append(lambda r: r[col] <= val)
        return self

    def order(self, col, desc=False):
        assert col == "ts"
        self._order_desc = desc
        return self

    def range(self, start, end):
        self._range = (start, end)
        return self

    def limit(self, n):
        self._limit = n
        return self

    def upsert(self, records, on_conflict=None, **_k):
        self._op = "upsert"
        self._payload = records
        return self

    def execute(self):
        if self._op == "upsert":
            self._t.upsert_calls.append(len(self._payload))
            for rec in self._payload:
                self._t.rows[(rec["ticker"], rec["timeframe"], rec["ts"])] = rec
            return SimpleNamespace(data=self._payload)
        rows = [r for r in self._t.rows.values() if all(f(r) for f in self._filters)]
        rows.sort(key=lambda r: pd.Timestamp(r["ts"]), reverse=self._order_desc)
        if self._range is not None:
            s, e = self._range
            rows = rows[s : e + 1]
        if self._limit is not None:
            rows = rows[: self._limit]
        rows = rows[: self._t.max_rows]  # PostgREST server-side cap
        self._t.select_calls += 1
        return SimpleNamespace(data=[dict(r) for r in rows])


class _FakeTable:
    def __init__(self, max_rows: int) -> None:
        self.rows: dict = {}
        self.max_rows = max_rows
        self.select_calls = 0
        self.upsert_calls: list[int] = []


class FakeSupabase:
    def __init__(self, max_rows: int = 1000) -> None:
        self.prices = _FakeTable(max_rows)

    def table(self, name):
        assert name == "prices"
        return _FakeQuery(self.prices)


def _seed(client: FakeSupabase, ticker: str, df: pd.DataFrame) -> None:
    for ts, row in df.iterrows():
        client.prices.rows[(ticker, "1Day", ts.isoformat())] = {
            "ticker": ticker,
            "timeframe": "1Day",
            "ts": ts.isoformat(),
            **{k: float(row[k]) for k in ("open", "high", "low", "close")},
            "volume": int(row["volume"]),
        }


@pytest.mark.parametrize("max_rows", [1000, 300])
def test_get_history_pages_past_the_server_row_cap(max_rows):
    client = FakeSupabase(max_rows=max_rows)
    df = make_ohlcv(2700, seed=1)
    _seed(client, "SPY", df)
    _seed(client, "QQQ", make_ohlcv(50, seed=2))
    store = SupabasePriceStore(client)

    out = store.get_history("SPY", start=df.index[0], end=df.index[-1])

    assert len(out) == 2700
    assert out.index.is_monotonic_increasing and out.index.is_unique
    assert str(out.index.tz) == "UTC"
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    pd.testing.assert_series_equal(out["close"], df["close"], check_names=False, check_freq=False)
    assert client.prices.select_calls >= 2700 // max_rows


def test_get_history_respects_start_and_end_inclusive():
    client = FakeSupabase()
    df = make_ohlcv(1500, seed=1)
    _seed(client, "SPY", df)
    out = SupabasePriceStore(client).get_history("SPY", start=df.index[100], end=df.index[1299])
    assert out.index[0] == df.index[100]
    assert out.index[-1] == df.index[1299]
    assert len(out) == 1200


def test_get_history_empty():
    out = SupabasePriceStore(FakeSupabase()).get_history(
        "SPY", start=pd.Timestamp("2016-01-01", tz="UTC"), end=pd.Timestamp("2026-01-01", tz="UTC")
    )
    assert out.empty


def test_upsert_bars_is_chunked_for_large_backfills():
    client = FakeSupabase()
    df = make_ohlcv(2700, seed=1)
    SupabasePriceStore(client).upsert_bars("SPY", "1Day", df)
    assert len(client.prices.rows) == 2700
    assert max(client.prices.upsert_calls) <= 500
    # Idempotent: a second identical backfill doesn't duplicate rows.
    SupabasePriceStore(client).upsert_bars("SPY", "1Day", df)
    assert len(client.prices.rows) == 2700


# ---------------------------------------------------------------------------
# Backfill command
# ---------------------------------------------------------------------------


def _config() -> WorkerConfig:
    return WorkerConfig(
        alpaca_api_key="k",
        alpaca_secret_key="s",
        alpaca_base_url="https://paper-api.alpaca.markets",
        supabase_url="https://example.supabase.co",
        supabase_service_role_key="x",
    )


def test_backfill_fetches_from_since_through_as_of_and_upserts(monkeypatch):
    from wolfpack_worker import backfill

    as_of = datetime(2026, 9, 29, 20, 0, tzinfo=timezone.utc)
    full = make_ohlcv(30, seed=1, start="2026-08-18")
    # Alpaca might hand back a still-forming bar after as_of — must be dropped.
    calls = []

    def fake_fetch(*, ticker, config, start, end):
        calls.append((ticker, start, end))
        return full, "sip"

    monkeypatch.setattr(backfill, "fetch_daily_bars", fake_fetch)
    store = InMemoryPriceStore()
    feeds = backfill.backfill_prices(
        tickers=("SPY", "QQQ"),
        price_store=store,
        config=_config(),
        since=date(2016, 1, 4),
        as_of=as_of,
    )
    assert feeds == {"SPY": "sip", "QQQ": "sip"}
    assert [c[0] for c in calls] == ["SPY", "QQQ"]
    assert calls[0][1] == datetime(2016, 1, 4, tzinfo=timezone.utc)
    assert calls[0][2] == as_of
    assert store.bars["SPY"].index.max() <= as_of
    assert len(store.bars["SPY"]) == (full.index <= as_of).sum()


def test_backfill_raises_if_a_ticker_returns_no_bars(monkeypatch):
    from wolfpack_worker import backfill

    monkeypatch.setattr(
        backfill,
        "fetch_daily_bars",
        lambda **_k: (pd.DataFrame(columns=["open", "high", "low", "close", "volume"]), None),
    )
    with pytest.raises(RuntimeError, match="SPY"):
        backfill.backfill_prices(
            tickers=("SPY",),
            price_store=InMemoryPriceStore(),
            config=_config(),
            since=date(2016, 1, 4),
            as_of=datetime(2026, 9, 29, 20, tzinfo=timezone.utc),
        )


def test_backfill_resolves_as_of_from_latest_completed_session():
    from wolfpack_worker import backfill

    sessions = [
        Session(date(2026, 9, 28), datetime(2026, 9, 28, 13, 30, tzinfo=timezone.utc),
                datetime(2026, 9, 28, 20, 0, tzinfo=timezone.utc)),
        Session(date(2026, 9, 29), datetime(2026, 9, 29, 13, 30, tzinfo=timezone.utc),
                datetime(2026, 9, 29, 20, 0, tzinfo=timezone.utc)),
    ]
    broker = FakeBroker(sessions=sessions)
    now = datetime(2026, 9, 29, 17, 0, tzinfo=timezone.utc)  # mid-session
    assert backfill.resolve_as_of(broker, now) == sessions[0].close
