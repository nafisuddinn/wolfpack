"""Orchestrator for the daily mechanical trade-execution job.

Invoked by .github/workflows/daily-trades.yml's mechanical step as:
    uv run --project worker -m wolfpack_worker.daily_trades

For each active persona in `strategies.REGISTRY`:
1. Reconcile any still-`pending` trade rows against the broker's view of
   reality (fills, rejections, crash-after-accept recovery) — always first.
2. Refresh price data for the universe, then news headlines (The Scout's
   input) in their own failure boundary: a news failure only makes The
   Scout skip the session; the other personas trade as usual.
3. Build a lookahead-safe `StrategyContext` (plus, for strategies with
   `requires_news`, a point-in-time `NewsSnapshot`) and evaluate.
4. Plan and (unless `--dry-run`) execute any resulting order.

`--dry-run` computes and prints everything above with zero trade writes and
zero broker order submissions — safe to run against production data to
sanity-check what today's run *would* do. (As before, the price refresh
still upserts bars, and the news refresh still inserts new headlines into
the private news table: both are market data, not trades.)
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from wolfpack_worker.broker import AlpacaPaperBroker, Broker
from wolfpack_worker.config import WorkerConfig, load_config
from wolfpack_worker.db import get_client
from wolfpack_worker.execution import (
    OrderIntent,
    execute_intent,
    make_run_id,
    plan_order,
    reconcile_open_orders,
)
from wolfpack_worker.execution import FixedNotionalSizer, Sizer
from wolfpack_worker.market_data import latest_completed_session, refresh_prices
from wolfpack_worker.news import NewsFetcher, NewsRefreshResult, NewsStore, refresh_news
from wolfpack_worker.store import PriceStore, SupabasePriceStore, SupabaseTradeRepo, TradeRepo
from wolfpack_worker.strategies import REGISTRY
from wolfpack_worker.strategies.base import (
    NewsSnapshot,
    Strategy,
    StrategyContext,
    truncate_bars,
    truncate_news,
)
from wolfpack_worker.universe import UNIVERSE

logger = logging.getLogger(__name__)

_CALENDAR_LOOKBACK_DAYS = 10
# An article counts as a late arrival if WolfPack first saw it more than this
# long after the close of the session whose window it belongs to (the cron
# runs ~30-90 min after the close; GitHub can delay it further).
LATE_ARRIVAL_GRACE = timedelta(hours=3)


def refresh_news_safely(
    fetcher_factory: Callable[[], NewsFetcher],
    news_store: NewsStore,
    universe: Sequence[str],
    now: datetime,
) -> NewsRefreshResult:
    """The daily news refresh in its own failure boundary: never raises."""
    try:
        return refresh_news(fetcher=fetcher_factory(), store=news_store, universe=universe, now=now)
    except Exception as exc:  # noqa: BLE001 - a news failure must not stop other personas
        logger.warning("daily_trades: news refresh failed: %s", exc, extra={"reason": "news_refresh_failed"})
        return NewsRefreshResult(status="failed", error=f"{type(exc).__name__}: {exc}")


def count_late_arrivals(articles: pd.DataFrame, closes: Sequence[datetime],
                        grace: timedelta = LATE_ARRIVAL_GRACE) -> int:
    """Live-ingested articles first seen > grace after the close of the
    session their created_at falls in (i.e. after that session's decision
    ran). Backfilled rows are excluded: their first_seen_at is the backfill
    time by construction."""
    if articles is None or len(articles) == 0 or not len(closes):
        return 0
    live = articles.loc[articles["ingest_mode"] == "live"]
    if live.empty:
        return 0
    c = pd.DatetimeIndex(pd.to_datetime(list(closes), utc=True))
    idx = c.searchsorted(pd.DatetimeIndex(live["created_at"]), side="left")
    ok = idx < len(c)
    own_close = pd.Series(pd.NaT, index=live.index, dtype="datetime64[ns, UTC]")
    own_close[ok] = c[idx[ok]]
    late = ok & (live["first_seen_at"] > own_close + grace).to_numpy()
    return int(np.sum(late))


def build_news_snapshot(
    *,
    news_store: Optional[NewsStore],
    broker: Broker,
    as_of: datetime,
    refresh: Optional[NewsRefreshResult],
    universe: Sequence[str],
    lookback_sessions: int,
) -> NewsSnapshot:
    """Point-in-time news for one decision at `as_of` (= close of session t).

    Covers `lookback_sessions` feature windows: sessions t-L..t plus the one
    before (whose close bounds the earliest window). Unavailable ("failed" /
    "stale", with a reason) if the refresh failed, the store can't be read,
    the calendar doesn't end at as_of, or the stored history doesn't reach
    back to the lookback start (run news_backfill first).
    """
    if news_store is None or refresh is None or not refresh.ok or refresh.cutoff is None:
        why = "no news refresh ran" if refresh is None else f"news refresh {refresh.status}: {refresh.error}"
        return NewsSnapshot.unavailable("failed", why)
    try:
        start_day = (as_of - timedelta(days=2 * lookback_sessions + 20)).date()
        sessions = [s for s in broker.get_calendar(start_day, as_of.date()) if s.close <= as_of]
        if not sessions or sessions[-1].close != as_of:
            return NewsSnapshot.unavailable("stale", "market calendar does not end at as_of")
        sessions = sessions[-(lookback_sessions + 1):]
        if len(sessions) < lookback_sessions + 1:
            return NewsSnapshot.unavailable("stale", "market calendar shorter than the news lookback")
        start = sessions[0].close
        earliest = news_store.earliest_created_at()
        if earliest is None or earliest > start:
            return NewsSnapshot.unavailable(
                "stale",
                f"stored news starts {earliest!r}, after the lookback start {start!r}: run "
                "`python -m wolfpack_worker.news_backfill` first",
            )
        raw = news_store.get_articles(list(universe), start, as_of)
    except Exception as exc:  # noqa: BLE001 - store/calendar I/O only; lookahead checks happen below
        logger.warning("daily_trades: news snapshot failed: %s", exc, extra={"reason": "news_read_failed"})
        return NewsSnapshot.unavailable("failed", f"news read failed: {type(exc).__name__}: {exc}")
    arts, n_after_cutoff = truncate_news(raw, as_of, refresh.cutoff)
    return NewsSnapshot(
        articles=arts,
        sessions=tuple(sessions),
        cutoff=refresh.cutoff,
        status="ok",
        late_arrivals=count_late_arrivals(arts, [s.close for s in sessions]),
        excluded_after_cutoff=n_after_cutoff,
    )


def run_persona(
    *,
    strategy: Strategy,
    broker: Broker,
    price_store: PriceStore,
    trade_repo: TradeRepo,
    as_of: datetime,
    universe: Sequence[str],
    sizer: Sizer,
    run_id: str,
    dry_run: bool,
    news_store: Optional[NewsStore] = None,
    news_refresh: Optional[NewsRefreshResult] = None,
) -> list[OrderIntent]:
    """Run one persona's full signal->order pipeline. Pure of any I/O side
    effects besides `broker`/`trade_repo`/`price_store`/`news_store` — fully
    testable with the fakes in tests/fakes.py.
    """
    reconcile_open_orders(broker, trade_repo, as_of.date())

    persona_id = trade_repo.get_persona_id(strategy.slug)

    raw_bars = {
        ticker: price_store.get_bars(ticker, as_of, limit=strategy.lookback_bars + 10)
        for ticker in universe
    }
    bars = truncate_bars(raw_bars, as_of)
    news = None
    if getattr(strategy, "requires_news", False):
        news = build_news_snapshot(
            news_store=news_store,
            broker=broker,
            as_of=as_of,
            refresh=news_refresh,
            universe=universe,
            lookback_sessions=int(getattr(strategy, "news_lookback_sessions")),
        )
    ctx = StrategyContext(as_of=as_of, universe=tuple(universe), bars=bars, news=news)
    targets = strategy.evaluate(ctx)

    open_tickers = {order.symbol for order in broker.get_open_orders()}

    intents: list[OrderIntent] = []
    for target in targets:
        # Source of truth for "what does this persona currently hold" is its
        # own filled trades, NOT the shared Alpaca paper account directly —
        # multiple personas trade through one account, so reading Alpaca's
        # position for a ticker would blend them together (2026-09-24
        # Decision Log entry). We do compare against the broker's actual
        # position as a warn-only sanity check, without hard-enforcing it.
        current_qty = trade_repo.get_position_qty(persona_id, target.ticker)
        broker_qty = broker.get_position_qty(target.ticker)
        if broker_qty != current_qty:
            logger.warning(
                "daily_trades.run_persona: position drift for %s/%s — "
                "persona's filled-trades position is %s but the shared "
                "broker account reports %s. Not auto-correcting; investigate.",
                strategy.slug,
                target.ticker,
                current_qty,
                broker_qty,
            )

        has_open_order = target.ticker in open_tickers
        last_close = float(target.payload["last_close"])

        intent = plan_order(target, current_qty, has_open_order, last_close, sizer)
        if intent is None:
            continue

        intents.append(intent)
        if not dry_run:
            execute_intent(
                broker=broker,
                trade_repo=trade_repo,
                persona_id=persona_id,
                persona_slug=strategy.slug,
                intent=intent,
                run_id=run_id,
            )

    return intents


def _resolve_as_of(broker: Broker, now: datetime) -> tuple[datetime, date]:
    start = (now - timedelta(days=_CALENDAR_LOOKBACK_DAYS)).date()
    sessions = broker.get_calendar(start, now.date())
    session = latest_completed_session(now, sessions)
    return session.close, session.date


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print intents without writing to the DB or placing broker orders.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config: WorkerConfig = load_config()

    broker = AlpacaPaperBroker(config)
    supabase = get_client(config)
    price_store = SupabasePriceStore(supabase)
    trade_repo = SupabaseTradeRepo(supabase)
    sizer = FixedNotionalSizer()
    run_id = make_run_id()

    as_of, session_date = _resolve_as_of(broker, datetime.now(tz=timezone.utc))

    for ticker in UNIVERSE:
        refresh_prices(
            ticker=ticker,
            broker=broker,
            price_store=price_store,
            config=config,
            as_of=as_of,
            session_close=as_of,
        )

    # News (The Scout): its own failure boundary. The refresh's finish time is
    # the run's news cutoff (live features use only rows first seen by then).
    from wolfpack_worker.news import AlpacaNewsFetcher, SupabaseNewsStore

    news_store = SupabaseNewsStore(supabase)
    news_refresh = refresh_news_safely(
        lambda: AlpacaNewsFetcher.from_config(config), news_store, UNIVERSE, datetime.now(tz=timezone.utc)
    )
    print(f"wolfpack_worker.daily_trades: news refresh {news_refresh.status} "
          f"(fetched {news_refresh.n_fetched}, new {news_refresh.n_inserted})")

    total_intents = 0
    for slug, factory in REGISTRY.items():
        strategy = factory()
        intents = run_persona(
            strategy=strategy,
            broker=broker,
            price_store=price_store,
            trade_repo=trade_repo,
            as_of=as_of,
            universe=UNIVERSE,
            sizer=sizer,
            run_id=run_id,
            dry_run=args.dry_run,
            news_store=news_store,
            news_refresh=news_refresh,
        )
        total_intents += len(intents)
        for intent in intents:
            verb = "Would place" if args.dry_run else "Placed"
            print(f"{verb} {intent.side} {intent.qty} {intent.ticker} ({slug})")

    if total_intents == 0:
        print("wolfpack_worker.daily_trades: no order intents produced for this session.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
