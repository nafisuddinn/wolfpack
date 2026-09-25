"""Turns strategy signals into (idempotent, auditable) broker orders + trade rows.

Order of operations for every order this module places (never reordered):
1. Insert the `trades` row as `status='pending'` FIRST, with `client_order_id`
   set, via an idempotent upsert keyed on `(persona_id, ticker, signal_ts)`.
   An empty/duplicate result means some earlier run already handled this
   signal — skip submitting to the broker entirely.
2. Submit the broker order.
3. Attach the broker's order id to the trade row.

If step 2 crashes after the broker actually accepted the order (network
drop, process kill, etc.), the trade row is left with a `client_order_id`
but no `broker_order_id`. `reconcile_open_orders` (run at the start of every
`main()` invocation, before any new signals are evaluated) looks such rows
up by `client_order_id` and attaches the id — this is what prevents a
crash-after-accept from producing a duplicate order on the next run.

This module never writes `rationale`, `rationale_written_at`, or
`rationale_author` — those are write-once via a separate RPC, owned by a
different, least-privileged step (see supabase/migrations init schema +
.github/RUNBOOK.md).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Literal, Mapping, Optional, Protocol

from wolfpack_worker.broker import Broker
from wolfpack_worker.store import PendingTrade, TradeRepo
from wolfpack_worker.strategies.base import TargetPosition
from wolfpack_worker.universe import NOTIONAL_PER_POSITION

# Order/trade statuses that mean "the broker is done deciding" — once an
# order reaches one of these, reconciliation stops polling it.
_FILLED = "filled"
_REJECTED = "rejected"
_CANCELED_LIKE = {"canceled", "expired"}


class Sizer(Protocol):
    def qty_for(self, target_exposure: float, last_close: float) -> int: ...


@dataclass(frozen=True)
class FixedNotionalSizer:
    """Whole shares only, floored — never over-spends `notional_per_position`."""

    notional_per_position: float = NOTIONAL_PER_POSITION

    def qty_for(self, target_exposure: float, last_close: float) -> int:
        if last_close <= 0 or target_exposure <= 0:
            return 0
        return math.floor(self.notional_per_position * target_exposure / last_close)


@dataclass(frozen=True)
class OrderIntent:
    ticker: str
    side: Literal["buy", "sell"]
    qty: int
    signal_ts: datetime
    payload: Mapping[str, Any]


def plan_order(
    target: TargetPosition,
    current_qty: float,
    has_open_order: bool,
    last_close: float,
    sizer: Sizer,
) -> Optional[OrderIntent]:
    """Decide whether a position-STATE change is needed. long-only, v1.

    Only produces an order when flat->long (buy the desired qty) or
    long->flat (sell everything currently held). long->long or flat->flat
    never produces an order, even if the desired qty at the new exposure
    would differ from what's currently held — v1 doesn't rebalance within a
    state, only enters/exits it. An open order already resting on this
    ticker means skip (never stack orders).
    """
    if has_open_order:
        return None

    desired_qty = sizer.qty_for(target.target_exposure, last_close)
    is_currently_long = current_qty > 0
    wants_long = desired_qty > 0

    if not is_currently_long and wants_long:
        return OrderIntent(
            ticker=target.ticker,
            side="buy",
            qty=desired_qty,
            signal_ts=target.signal_ts,
            payload=target.payload,
        )

    if is_currently_long and not wants_long:
        return OrderIntent(
            ticker=target.ticker,
            side="sell",
            qty=int(current_qty),
            signal_ts=target.signal_ts,
            payload=target.payload,
        )

    return None


def make_run_id() -> str:
    """`gh-{GITHUB_RUN_ID}-{GITHUB_RUN_ATTEMPT}` in CI, else a local timestamp.

    Used only for auditing which run acted — never for idempotency (that's
    `(persona_id, ticker, signal_ts)`, via `client_order_id`).
    """
    run_id = os.environ.get("GITHUB_RUN_ID")
    attempt = os.environ.get("GITHUB_RUN_ATTEMPT")
    if run_id and attempt:
        return f"gh-{run_id}-{attempt}"
    return f"local-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"


def make_client_order_id(slug: str, ticker: str, bar_date: date) -> str:
    return f"wp1-{slug}-{ticker}-{bar_date:%Y%m%d}"


def execute_intent(
    *,
    broker: Broker,
    trade_repo: TradeRepo,
    persona_id: str,
    persona_slug: str,
    intent: OrderIntent,
    run_id: str,
) -> None:
    client_order_id = make_client_order_id(persona_slug, intent.ticker, intent.signal_ts.date())

    trade_id = trade_repo.insert_pending_trade(
        persona_id=persona_id,
        ticker=intent.ticker,
        side=intent.side,
        qty=intent.qty,
        signal_ts=intent.signal_ts,
        run_id=run_id,
        signal_payload=intent.payload,
        client_order_id=client_order_id,
    )
    if trade_id is None:
        # Some earlier run already inserted this (persona_id, ticker,
        # signal_ts) row — never submit a second broker order for it.
        return

    broker_order = broker.submit_market_order(
        symbol=intent.ticker,
        side=intent.side,
        qty=intent.qty,
        client_order_id=client_order_id,
    )
    trade_repo.attach_broker_order_id(trade_id, broker_order.broker_order_id)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _apply_order_state(trade: Any, order: Any, trade_repo: TradeRepo) -> None:
    status = order.status
    if status == _FILLED:
        trade_repo.update_trade_status(
            trade.id,
            status="filled",
            filled_at=_utcnow(),
            fill_price=order.filled_avg_price,
        )
    elif status == _REJECTED:
        trade_repo.update_trade_status(trade.id, status="rejected")
    elif status in _CANCELED_LIKE:
        if order.filled_qty and order.filled_qty > 0:
            payload = dict(trade.signal_payload)
            payload["original_qty"] = trade.qty
            trade_repo.update_trade_status(
                trade.id,
                status="filled",
                filled_at=_utcnow(),
                fill_price=order.filled_avg_price,
                qty=order.filled_qty,
                signal_payload=payload,
            )
        else:
            trade_repo.update_trade_status(trade.id, status="canceled")
    # Anything else (new/accepted/pending_new/still-open partial fill) means
    # the order is still live — leave the trade row as 'pending'.


def reconcile_open_orders(
    broker: Broker, trade_repo: TradeRepo, today_session_date: date
) -> None:
    """Resolve every `pending` trade row against the broker's view of reality.

    Runs at the start of every `main()` invocation, before any new signals
    are evaluated for the day.
    """
    for trade in trade_repo.get_pending_trades():
        if trade.broker_order_id:
            order = broker.get_order_by_client_id(trade.client_order_id)
            if order is None:
                continue
            _apply_order_state(trade, order, trade_repo)
            continue

        order = broker.get_order_by_client_id(trade.client_order_id)
        if order is not None:
            trade_repo.attach_broker_order_id(trade.id, order.broker_order_id)
            _apply_order_state(trade, order, trade_repo)
            continue

        if trade.signal_ts.date() == today_session_date:
            broker_order = broker.submit_market_order(
                symbol=trade.ticker,
                side=trade.side,
                qty=trade.qty,
                client_order_id=trade.client_order_id,
            )
            trade_repo.attach_broker_order_id(trade.id, broker_order.broker_order_id)
        else:
            payload = dict(trade.signal_payload)
            payload["cancel_reason"] = "never_submitted_stale"
            trade_repo.update_trade_status(trade.id, status="canceled", signal_payload=payload)
