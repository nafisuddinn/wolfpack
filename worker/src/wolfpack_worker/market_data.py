"""Daily-bar price ingestion + session/as_of resolution.

Bars are split-adjusted, NOT dividend-adjusted (see universe.py's ADJUSTMENT
comment for why that's an accepted, documented limitation for v1).

Feed selection: tries `feed="sip"` (the consolidated tape) first; if the
Alpaca account/plan doesn't have SIP entitlement, falls back to `feed="iex"`
and logs the fallback loudly — this must never be silently swallowed, since
IEX-only data is a materially different (thinner) view of the tape.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from wolfpack_worker.broker import Broker, Session
from wolfpack_worker.config import WorkerConfig
from wolfpack_worker.store import PriceStore
from wolfpack_worker.universe import ADJUSTMENT, DATA_FEED

logger = logging.getLogger(__name__)

TIMEFRAME = "1Day"
_BACKFILL_DAYS = 120
_INCREMENTAL_LOOKBACK_DAYS = 7
_SESSION_CLOSE_DELAY = timedelta(minutes=15)


def latest_completed_session(now: datetime, sessions: list[Session]) -> Session:
    """The last session whose close + 15 minutes is at or before `now`.

    `sessions` must be sorted ascending by date and cover a wide enough
    range to include at least one completed session before `now` (callers
    typically fetch Alpaca's calendar for the trailing ~10 calendar days).
    """
    completed = [s for s in sessions if s.close + _SESSION_CLOSE_DELAY <= now]
    if not completed:
        raise ValueError(
            f"No completed session found at or before {now!r} in the "
            "provided calendar window — widen the calendar query range."
        )
    return completed[-1]


def refresh_prices(
    *,
    ticker: str,
    broker: Broker,
    price_store: PriceStore,
    config: WorkerConfig,
    as_of: datetime,
    session_close: datetime,
) -> pd.DataFrame:
    """Fetch and upsert daily bars for `ticker` through `session_close`.

    Fetches from `latest_ts - 7 days` (or does a 120-day backfill if
    `price_store` has no bars for this ticker yet) through `session_close`.
    Drops any bar with `ts > as_of` before upserting — a defensive measure
    against Alpaca returning a partial/in-progress bar for a session that
    hasn't fully closed yet. Returns the freshly upserted bars.
    """
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    latest_ts = price_store.latest_bar_ts(ticker, TIMEFRAME)
    if latest_ts is None:
        start = session_close - timedelta(days=_BACKFILL_DAYS)
    else:
        start = latest_ts - timedelta(days=_INCREMENTAL_LOOKBACK_DAYS)

    data_client = StockHistoricalDataClient(
        api_key=config.alpaca_api_key, secret_key=config.alpaca_secret_key
    )

    adjustment = Adjustment(ADJUSTMENT)
    bars_df = None
    for feed_name in (DATA_FEED, "iex"):
        request = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Day,
            start=start,
            end=session_close,
            adjustment=adjustment,
            feed=DataFeed(feed_name),
        )
        try:
            bar_set = data_client.get_stock_bars(request)
        except Exception as exc:  # noqa: BLE001 - broad: any feed rejection triggers fallback
            if feed_name == DATA_FEED:
                logger.warning(
                    "market_data.refresh_prices: feed=%r rejected for %s "
                    "(%s) — falling back to feed='iex'.",
                    feed_name,
                    ticker,
                    exc,
                )
                continue
            raise

        candidate_df = bar_set.df
        if candidate_df is None or candidate_df.empty:
            # Some accounts/plans don't raise on a SIP-ineligible request —
            # they just return a successful, empty bar set. Treat that the
            # same as an explicit rejection and retry with the next feed,
            # rather than silently returning empty data (see module
            # docstring: fail loudly, never silently).
            if feed_name == DATA_FEED:
                logger.warning(
                    "market_data.refresh_prices: feed=%r returned no bars "
                    "for %s (empty response, not an error) — falling back "
                    "to feed='iex'.",
                    feed_name,
                    ticker,
                )
                continue
            logger.warning(
                "market_data.refresh_prices: both feed=%r and feed='iex' "
                "returned no bars for %s — no price data available this run.",
                DATA_FEED,
                ticker,
            )
            bars_df = candidate_df
            break

        bars_df = candidate_df
        if feed_name != DATA_FEED:
            logger.warning(
                "market_data.refresh_prices: used feed='iex' fallback for %s "
                "— SIP feed was unavailable this run.",
                ticker,
            )
        break

    if bars_df is None or bars_df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    if isinstance(bars_df.index, pd.MultiIndex):
        bars_df = bars_df.loc[ticker]

    bars_df = bars_df[["open", "high", "low", "close", "volume"]].sort_index()
    bars_df = bars_df.loc[bars_df.index <= as_of]

    price_store.upsert_bars(ticker, TIMEFRAME, bars_df)
    return bars_df
