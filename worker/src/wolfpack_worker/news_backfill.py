"""One-off historical news backfill for The Scout.

    uv run --project worker -m wolfpack_worker.news_backfill --since 2016-01-04
    uv run --project worker -m wolfpack_worker.news_backfill --since 2016-01-04 --store local [--cache PATH]

Fetches every Alpaca/Benzinga headline mentioning any universe ticker, one
calendar month at a time (so a failure loses at most one chunk and a re-run
simply skips what is already stored), and inserts new rows with
ingest_mode = "backfill" and first_seen_at = the backfill time. Insert-once:
re-running never overwrites an existing row.

`--store supabase` (default) writes the private `news_articles` table
(requires migration 20261004_0003). `--store local` writes the gitignored
JSONL cache (worker/data/news_cache/), used for offline training when the
table is not available. Neither may ever be committed or published: the
headline text is licensed (Decision Log 2026-10-04).

Backfilled rows carry no real "first seen" time, so historical features can
only be point-in-time on the vendor's created_at; vendor revisions and
archive backfill are a documented residual leak (see scout/coverage.py).
Free tier: existing Alpaca keys, ~200 calls/min; rate-limit errors are
retried with a pause.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence

import pandas as pd

from wolfpack_worker.news import (
    DEFAULT_CACHE_PATH,
    AlpacaNewsFetcher,
    LocalNewsStore,
    NewsFetcher,
    NewsStore,
    SupabaseNewsStore,
)
from wolfpack_worker.universe import UNIVERSE

logger = logging.getLogger(__name__)

DEFAULT_SINCE = date(2016, 1, 4)
MAX_ATTEMPTS = 6
RATE_LIMIT_PAUSE_S = 30.0
CHUNK_PAUSE_S = 1.0


def month_chunks(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    """[start, end] tiled into consecutive (lo, hi) pairs at month starts.
    Adjacent chunks share their boundary instant (duplicates are ignored on
    insert), so nothing can fall between two chunks."""
    edges = [start]
    s0 = pd.Timestamp(start).tz_convert("UTC")
    m = pd.Timestamp(year=s0.year, month=s0.month, day=1, tz="UTC")
    while True:
        m = m + pd.offsets.MonthBegin(1)
        if m.to_pydatetime() >= end:
            break
        edges.append(m.to_pydatetime())
    edges.append(end)
    return list(zip(edges[:-1], edges[1:]))


def _rate_limited(exc: Exception) -> bool:
    text = str(exc).lower()
    return "429" in text or "too many requests" in text or "rate limit" in text


def backfill_news(
    *,
    fetcher: NewsFetcher,
    store: NewsStore,
    universe: Sequence[str],
    since: date,
    until: datetime,
    now: datetime,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    start = datetime(since.year, since.month, since.day, tzinfo=timezone.utc)
    if until.tzinfo is None or now.tzinfo is None:
        raise ValueError("until and now must be timezone-aware")
    total_fetched = total_new = 0
    chunks = month_chunks(start, until)
    for i, (lo, hi) in enumerate(chunks):
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                arts = fetcher.fetch(list(universe), lo, hi)
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == MAX_ATTEMPTS:
                    raise
                pause = RATE_LIMIT_PAUSE_S if _rate_limited(exc) else 5.0 * attempt
                logger.warning("news_backfill: %s..%s attempt %d failed (%s); retrying in %.0fs",
                               lo.date(), hi.date(), attempt, exc, pause)
                sleep(pause)
        n_new = store.insert_new(arts, first_seen_at=now, ingest_mode="backfill")
        total_fetched += len(arts)
        total_new += n_new
        logger.info("news_backfill: [%d/%d] %s..%s fetched %d, new %d", i + 1, len(chunks), lo.date(), hi.date(),
                    len(arts), n_new)
        if i + 1 < len(chunks):
            sleep(CHUNK_PAUSE_S)
    return {"chunks": len(chunks), "n_fetched": total_fetched, "n_inserted": total_new,
            "since": start.isoformat(), "until": until.isoformat()}


def make_store(kind: str, cache: Optional[Path] = None) -> NewsStore:
    if kind == "local":
        return LocalNewsStore(cache or DEFAULT_CACHE_PATH)
    if kind == "supabase":
        from wolfpack_worker.config import load_config
        from wolfpack_worker.db import get_client

        return SupabaseNewsStore(get_client(load_config()))
    raise ValueError(f"unknown store {kind!r}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--since", type=date.fromisoformat, default=DEFAULT_SINCE)
    parser.add_argument("--until", type=date.fromisoformat, default=None,
                        help="last date (exclusive end at 00:00 UTC); default: now")
    parser.add_argument("--store", choices=("supabase", "local"), default="supabase")
    parser.add_argument("--cache", type=Path, default=None, help=f"local JSONL path (default {DEFAULT_CACHE_PATH})")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from wolfpack_worker.config import load_config

    now = datetime.now(timezone.utc)
    until = (datetime(args.until.year, args.until.month, args.until.day, tzinfo=timezone.utc)
             if args.until else now)
    out = backfill_news(
        fetcher=AlpacaNewsFetcher.from_config(load_config()),
        store=make_store(args.store, args.cache),
        universe=UNIVERSE,
        since=args.since,
        until=until,
        now=now,
    )
    print(f"news_backfill ({args.store}): {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
