"""Full-history daily price backfill into the `prices` table.

    python -m wolfpack_worker.backfill --since 2016-01-04

Reuses `market_data.fetch_daily_bars` (SIP first, IEX fallback, split-
adjusted) with a start-date override, and upserts every bar from `--since`
through the latest *completed* session (same as-of resolution as the daily
job). Idempotent: rows are upserted on (ticker, timeframe, ts), so re-running
simply overwrites with Alpaca's current split-adjusted values — which is the
point: The Analyst's training re-runs this first so any split since the last
run is back-adjusted across the whole history. `prices` stays the single
source of truth; the worker stays its only writer.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from typing import Sequence

from wolfpack_worker.broker import Broker
from wolfpack_worker.config import WorkerConfig
from wolfpack_worker.market_data import TIMEFRAME, fetch_daily_bars, latest_completed_session
from wolfpack_worker.store import PriceStore
from wolfpack_worker.universe import UNIVERSE

logger = logging.getLogger(__name__)

DEFAULT_SINCE = date(2016, 1, 4)
_CALENDAR_LOOKBACK_DAYS = 10


def resolve_as_of(broker: Broker, now: datetime) -> datetime:
    """Close of the latest completed session at `now` (never an in-progress one)."""
    start = (now - timedelta(days=_CALENDAR_LOOKBACK_DAYS)).date()
    sessions = broker.get_calendar(start, now.date())
    return latest_completed_session(now, sessions).close


def backfill_prices(
    *,
    tickers: Sequence[str],
    price_store: PriceStore,
    config: WorkerConfig,
    since: date,
    as_of: datetime,
) -> dict[str, str | None]:
    """Fetch [since, as_of] for each ticker and upsert. Returns ticker -> feed used.

    Raises if any ticker comes back with no bars at all — training on a
    silently missing ticker would be worse than failing.
    """
    start = datetime(since.year, since.month, since.day, tzinfo=timezone.utc)
    feeds: dict[str, str | None] = {}
    for ticker in tickers:
        bars, feed = fetch_daily_bars(ticker=ticker, config=config, start=start, end=as_of)
        if bars.empty:
            raise RuntimeError(f"backfill: no bars returned for {ticker} from either feed")
        bars = bars.loc[bars.index <= as_of]
        price_store.upsert_bars(ticker, TIMEFRAME, bars)
        feeds[ticker] = feed
        logger.info(
            "backfill: %s %d bars %s..%s feed=%s",
            ticker, len(bars), bars.index.min(), bars.index.max(), feed,
        )
    return feeds


def run_backfill(since: date = DEFAULT_SINCE) -> tuple[dict[str, str | None], datetime]:
    """Real-service entry point (Alpaca + Supabase). Returns (feeds, as_of)."""
    from wolfpack_worker.broker import AlpacaPaperBroker
    from wolfpack_worker.config import load_config
    from wolfpack_worker.db import get_client
    from wolfpack_worker.store import SupabasePriceStore

    config = load_config()
    broker = AlpacaPaperBroker(config)
    store = SupabasePriceStore(get_client(config))
    as_of = resolve_as_of(broker, datetime.now(tz=timezone.utc))
    feeds = backfill_prices(
        tickers=UNIVERSE, price_store=store, config=config, since=since, as_of=as_of
    )
    return feeds, as_of


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--since",
        type=date.fromisoformat,
        default=DEFAULT_SINCE,
        help="First date to backfill (YYYY-MM-DD). Default: %(default)s",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    feeds, as_of = run_backfill(args.since)
    print(f"backfill complete through {as_of.isoformat()}: {feeds}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
