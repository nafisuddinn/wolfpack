"""plan_order truth table + the "never write rationale*" safety assertion."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from wolfpack_worker.execution import (
    FixedNotionalSizer,
    execute_intent,
    plan_order,
)
from wolfpack_worker.strategies.base import TargetPosition

from fakes import FakeBroker, InMemoryTradeRepo

SIGNAL_TS = datetime(2026, 1, 30, 21, 0, tzinfo=timezone.utc)
SIZER = FixedNotionalSizer(notional_per_position=3000.0)


def make_target(exposure: float) -> TargetPosition:
    return TargetPosition(
        ticker="SPY",
        target_exposure=exposure,
        signal_ts=SIGNAL_TS,
        payload={"last_close": 100.0},
    )


def test_flat_to_long_produces_buy_order():
    target = make_target(1.0)
    intent = plan_order(target, current_qty=0.0, has_open_order=False, last_close=100.0, sizer=SIZER)

    assert intent is not None
    assert intent.side == "buy"
    assert intent.qty == 30  # floor(3000 / 100)
    assert intent.ticker == "SPY"


def test_long_to_flat_produces_sell_of_full_held_qty():
    target = make_target(0.0)
    intent = plan_order(target, current_qty=17.0, has_open_order=False, last_close=100.0, sizer=SIZER)

    assert intent is not None
    assert intent.side == "sell"
    assert intent.qty == 17


def test_long_to_long_is_a_no_op_even_if_desired_qty_differs():
    target = make_target(1.0)
    # last_close moved, so desired qty (30) differs from what's held (10) —
    # v1 does not rebalance within a state.
    intent = plan_order(target, current_qty=10.0, has_open_order=False, last_close=100.0, sizer=SIZER)

    assert intent is None


def test_flat_to_flat_is_a_no_op():
    target = make_target(0.0)
    intent = plan_order(target, current_qty=0.0, has_open_order=False, last_close=100.0, sizer=SIZER)

    assert intent is None


def test_qty_rounding_to_zero_means_no_buy():
    target = make_target(1.0)
    intent = plan_order(
        target, current_qty=0.0, has_open_order=False, last_close=1_000_000.0, sizer=SIZER
    )

    assert intent is None


def test_open_order_means_skip_even_if_state_would_otherwise_change():
    target = make_target(1.0)
    intent = plan_order(target, current_qty=0.0, has_open_order=True, last_close=100.0, sizer=SIZER)

    assert intent is None


def test_no_insert_or_update_dict_ever_contains_a_rationale_key():
    broker = FakeBroker()
    trade_repo = InMemoryTradeRepo()
    persona_id = trade_repo.get_persona_id("trend-follower")

    target = make_target(1.0)
    intent = plan_order(target, current_qty=0.0, has_open_order=False, last_close=100.0, sizer=SIZER)
    assert intent is not None

    execute_intent(
        broker=broker,
        trade_repo=trade_repo,
        persona_id=persona_id,
        persona_slug="trend-follower",
        intent=intent,
        run_id="local-test",
    )

    for record in [*trade_repo.inserts, *trade_repo.updates]:
        assert not any(key.startswith("rationale") for key in record), record
