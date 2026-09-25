"""Session-boundary safety for market_data.py.

`latest_completed_session` is what keeps a same-day, still-in-progress
session's bar out of any `StrategyContext` — since `refresh_prices`/
`daily_trades` derive `as_of` from this function's return value, not from
`datetime.now()` directly, a bug here would be a genuine lookahead leak that
`StrategyContext`'s own `> as_of` check could never catch (a same-day
"partial" bar for the still-open session is timestamped at the *start* of
that session, which is always <= as_of if as_of naively used "now").

No test file previously exercised this module at all — this file closes
that gap.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from wolfpack_worker.broker import Session
from wolfpack_worker.config import WorkerConfig
from wolfpack_worker.market_data import (
    _SESSION_CLOSE_DELAY,
    latest_completed_session,
    refresh_prices,
)
from fakes import InMemoryPriceStore


def _session(d: date, open_hour: int, close_hour: int) -> Session:
    return Session(
        date=d,
        open=datetime(d.year, d.month, d.day, open_hour, tzinfo=timezone.utc),
        close=datetime(d.year, d.month, d.day, close_hour, tzinfo=timezone.utc),
    )


def test_still_open_todays_session_is_never_selected_as_completed():
    """The most direct lookahead check for this module: if `now` is during
    market hours (today's session open but not yet closed), the function
    must return YESTERDAY's session, never today's in-progress one — even
    though today's session is chronologically the most recent in the list.
    """
    yesterday = date(2026, 1, 5)
    today = date(2026, 1, 6)
    sessions = [
        _session(yesterday, 14, 21),  # fully closed
        _session(today, 14, 21),  # still open right now
    ]
    # "Now" is mid-session on `today` — well before today's close.
    now = datetime(2026, 1, 6, 17, 0, tzinfo=timezone.utc)

    result = latest_completed_session(now, sessions)

    assert result.date == yesterday


def test_session_closed_but_within_the_settlement_delay_is_not_yet_completed():
    """A session that closed 5 minutes ago is NOT safe to use yet — Alpaca's
    daily bar for it may still be a preliminary/partial value. The buffer
    (`_SESSION_CLOSE_DELAY`) exists precisely to avoid this partial-bar
    lookahead risk, so `now` just after close but inside the buffer must
    still fall back to the prior session.
    """
    yesterday = date(2026, 1, 5)
    today = date(2026, 1, 6)
    sessions = [
        _session(yesterday, 14, 21),
        _session(today, 14, 21),
    ]
    now = today_close_plus = datetime(2026, 1, 6, 21, 5, tzinfo=timezone.utc)
    assert today_close_plus - _session(today, 14, 21).close < _SESSION_CLOSE_DELAY

    result = latest_completed_session(now, sessions)

    assert result.date == yesterday


def test_session_completed_exactly_at_the_delay_boundary_is_selected():
    today = date(2026, 1, 6)
    close = _session(today, 14, 21).close
    now = close + _SESSION_CLOSE_DELAY  # exactly at the boundary, inclusive

    result = latest_completed_session(now, [_session(today, 14, 21)])

    assert result.date == today


def test_no_completed_session_in_window_raises_rather_than_guessing():
    today = date(2026, 1, 6)
    now = datetime(2026, 1, 6, 15, 0, tzinfo=timezone.utc)  # mid-session

    with pytest.raises(ValueError):
        latest_completed_session(now, [_session(today, 14, 21)])


# ---------------------------------------------------------------------------
# refresh_prices: SIP/IEX feed fallback (raise-based and empty-response-based)
# ---------------------------------------------------------------------------


class _FakeBarSet:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df


def _bars_df(ts: datetime) -> pd.DataFrame:
    index = pd.DatetimeIndex([ts], name="timestamp")
    return pd.DataFrame(
        {"open": [10.0], "high": [11.0], "low": [9.0], "close": [10.5], "volume": [1000]},
        index=index,
    )


def _empty_bars_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


class _FakeStockHistoricalDataClient:
    """Stands in for alpaca-py's `StockHistoricalDataClient` in tests.

    `feed_behavior` maps a feed's string value ("sip"/"iex") to one of:
    - "raise": simulate the SDK raising (e.g. a hard rejection)
    - a `pd.DataFrame`: simulate a successful response with that bar set
      (an empty DataFrame simulates the "no exception, but no bars either"
      case being tested here)
    """

    def __init__(self, feed_behavior: dict[str, object], calls: list[str], **_kwargs) -> None:
        self._feed_behavior = feed_behavior
        self._calls = calls

    def get_stock_bars(self, request):
        feed_value = request.feed.value
        self._calls.append(feed_value)
        outcome = self._feed_behavior[feed_value]
        if isinstance(outcome, str) and outcome == "raise":
            raise RuntimeError(f"simulated rejection for feed={feed_value!r}")
        return _FakeBarSet(outcome)


def _config() -> WorkerConfig:
    return WorkerConfig(
        alpaca_api_key="test-key",
        alpaca_secret_key="test-secret",
        alpaca_base_url="https://paper-api.alpaca.markets",
        supabase_url="https://example.supabase.co",
        supabase_service_role_key="test",
    )


def _patch_data_client(monkeypatch, feed_behavior: dict[str, object]) -> list[str]:
    calls: list[str] = []

    def _factory(**kwargs):
        return _FakeStockHistoricalDataClient(feed_behavior, calls, **kwargs)

    monkeypatch.setattr(
        "alpaca.data.historical.stock.StockHistoricalDataClient", _factory
    )
    return calls


def test_refresh_prices_uses_sip_feed_when_it_returns_bars(monkeypatch):
    session_close = datetime(2026, 1, 6, 21, 0, tzinfo=timezone.utc)
    bar_ts = datetime(2026, 1, 6, 5, 0, tzinfo=timezone.utc)
    calls = _patch_data_client(monkeypatch, {"sip": _bars_df(bar_ts)})

    result = refresh_prices(
        ticker="AAPL",
        broker=None,
        price_store=InMemoryPriceStore(),
        config=_config(),
        as_of=session_close,
        session_close=session_close,
    )

    assert calls == ["sip"]
    assert not result.empty


def test_refresh_prices_falls_back_to_iex_when_sip_raises(monkeypatch):
    session_close = datetime(2026, 1, 6, 21, 0, tzinfo=timezone.utc)
    bar_ts = datetime(2026, 1, 6, 5, 0, tzinfo=timezone.utc)
    calls = _patch_data_client(
        monkeypatch, {"sip": "raise", "iex": _bars_df(bar_ts)}
    )

    result = refresh_prices(
        ticker="AAPL",
        broker=None,
        price_store=InMemoryPriceStore(),
        config=_config(),
        as_of=session_close,
        session_close=session_close,
    )

    assert calls == ["sip", "iex"]
    assert not result.empty


def test_refresh_prices_falls_back_to_iex_when_sip_returns_empty_bars(monkeypatch, caplog):
    """The warning's core regression case: SIP responds successfully (no
    exception) but with zero bars — e.g. a SIP-ineligible account that
    doesn't raise. This must be treated the same as a rejection and retried
    against IEX, not silently returned as empty data.
    """
    session_close = datetime(2026, 1, 6, 21, 0, tzinfo=timezone.utc)
    bar_ts = datetime(2026, 1, 6, 5, 0, tzinfo=timezone.utc)
    calls = _patch_data_client(
        monkeypatch, {"sip": _empty_bars_df(), "iex": _bars_df(bar_ts)}
    )

    with caplog.at_level("WARNING"):
        result = refresh_prices(
            ticker="AAPL",
            broker=None,
            price_store=InMemoryPriceStore(),
            config=_config(),
            as_of=session_close,
            session_close=session_close,
        )

    assert calls == ["sip", "iex"]
    assert not result.empty
    assert any("returned no bars" in message for message in caplog.messages)


def test_refresh_prices_logs_warning_when_both_feeds_return_empty_bars(monkeypatch, caplog):
    """If neither feed has data, `refresh_prices` must return empty data
    loudly (a clear warning), never silently, per the module's docstring.
    """
    session_close = datetime(2026, 1, 6, 21, 0, tzinfo=timezone.utc)
    calls = _patch_data_client(
        monkeypatch, {"sip": _empty_bars_df(), "iex": _empty_bars_df()}
    )

    with caplog.at_level("WARNING"):
        result = refresh_prices(
            ticker="AAPL",
            broker=None,
            price_store=InMemoryPriceStore(),
            config=_config(),
            as_of=session_close,
            session_close=session_close,
        )

    assert calls == ["sip", "iex"]
    assert result.empty
    assert any(
        "no bars" in message and "AAPL" in message for message in caplog.messages
    )
