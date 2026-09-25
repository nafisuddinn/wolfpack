"""Orchestrator for the daily mechanical trade-execution job.

Invoked by .github/workflows/daily-trades.yml's mechanical step as:
    uv run --project worker -m wolfpack_worker.daily_trades

For each active persona in `strategies.REGISTRY`:
1. Reconcile any still-`pending` trade rows against the broker's view of
   reality (fills, rejections, crash-after-accept recovery) — always first.
2. Refresh price data for the universe.
3. Build a lookahead-safe `StrategyContext` and evaluate the strategy.
4. Plan and (unless `--dry-run`) execute any resulting order.

`--dry-run` computes and prints everything above with zero DB writes and
zero broker order submissions — safe to run against production data to
sanity-check what today's run *would* do.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from typing import Sequence

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
from wolfpack_worker.store import PriceStore, SupabasePriceStore, SupabaseTradeRepo, TradeRepo
from wolfpack_worker.strategies import REGISTRY
from wolfpack_worker.strategies.base import Strategy, StrategyContext, truncate_bars
from wolfpack_worker.universe import UNIVERSE

logger = logging.getLogger(__name__)

_CALENDAR_LOOKBACK_DAYS = 10


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
) -> list[OrderIntent]:
    """Run one persona's full signal->order pipeline. Pure of any I/O side
    effects besides `broker`/`trade_repo`/`price_store` — fully testable
    with the fakes in tests/fakes.py.
    """
    reconcile_open_orders(broker, trade_repo, as_of.date())

    persona_id = trade_repo.get_persona_id(strategy.slug)

    raw_bars = {
        ticker: price_store.get_bars(ticker, as_of, limit=strategy.lookback_bars + 10)
        for ticker in universe
    }
    bars = truncate_bars(raw_bars, as_of)
    ctx = StrategyContext(as_of=as_of, universe=tuple(universe), bars=bars)
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
