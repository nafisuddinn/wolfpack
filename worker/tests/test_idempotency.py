"""Idempotency: exactly one broker submission per (persona, ticker, signal_ts),
even across re-runs, crashes, and non-trading-day re-invocations.
"""

from __future__ import annotations

import pandas as pd
import pytest

from wolfpack_worker.daily_trades import run_persona
from wolfpack_worker.execution import FixedNotionalSizer
from wolfpack_worker.strategies.trend_follower import LOOKBACK_BARS, TrendFollower
from wolfpack_worker.universe import UNIVERSE

from fakes import BrokerCrashError, FakeBroker, InMemoryPriceStore, InMemoryTradeRepo

TICKER = "SPY"
SIZER = FixedNotionalSizer(notional_per_position=3000.0)


def make_bars(closes: list[float]) -> pd.DataFrame:
    idx = pd.bdate_range(start="2026-01-01", periods=len(closes), tz="UTC")
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1_000_000] * len(closes),
        },
        index=idx,
    )


def _setup(closes: list[float]):
    bars = make_bars(closes)
    price_store = InMemoryPriceStore()
    price_store.upsert_bars(TICKER, "1Day", bars)
    trade_repo = InMemoryTradeRepo()
    return bars, price_store, trade_repo


def test_running_twice_with_same_inputs_produces_exactly_one_submission():
    closes = [100.0 + i for i in range(LOOKBACK_BARS)]  # rising -> long signal
    bars, price_store, trade_repo = _setup(closes)
    broker = FakeBroker()
    as_of = bars.index[-1].to_pydatetime()
    strategy = TrendFollower()

    for _ in range(2):
        run_persona(
            strategy=strategy,
            broker=broker,
            price_store=price_store,
            trade_repo=trade_repo,
            as_of=as_of,
            universe=[TICKER],
            sizer=SIZER,
            run_id="local-test",
            dry_run=False,
        )

    assert len(broker.submissions) == 1


def test_crash_after_accept_then_rerun_attaches_via_reconciliation_no_duplicate():
    closes = [100.0 + i for i in range(LOOKBACK_BARS)]
    bars, price_store, trade_repo = _setup(closes)
    broker = FakeBroker(crash_after_accept=True)
    as_of = bars.index[-1].to_pydatetime()
    strategy = TrendFollower()

    with pytest.raises(BrokerCrashError):
        run_persona(
            strategy=strategy,
            broker=broker,
            price_store=price_store,
            trade_repo=trade_repo,
            as_of=as_of,
            universe=[TICKER],
            sizer=SIZER,
            run_id="local-test",
            dry_run=False,
        )

    assert len(broker.submissions) == 1
    # The broker accepted the order, but the trade row never got a
    # broker_order_id attached before the crash.
    pending = trade_repo.get_pending_trades()
    assert len(pending) == 1
    assert pending[0].broker_order_id is None

    # Simulate a re-run: broker no longer crashes; reconciliation should
    # find the already-accepted order by client_order_id and attach it
    # rather than submitting a second order.
    broker.crash_after_accept = False
    run_persona(
        strategy=strategy,
        broker=broker,
        price_store=price_store,
        trade_repo=trade_repo,
        as_of=as_of,
        universe=[TICKER],
        sizer=SIZER,
        run_id="local-test-2",
        dry_run=False,
    )

    assert len(broker.submissions) == 1
    pending_after = trade_repo.get_pending_trades()
    assert len(pending_after) == 1
    assert pending_after[0].broker_order_id is not None


def test_position_drift_against_broker_is_warned_not_auto_corrected(caplog):
    """Position is planned from the persona's own filled trades, never read
    back from the shared Alpaca account directly (2026-09-24 Decision Log) —
    a mismatch is only logged, never silently "corrected" by trusting the
    broker's (possibly other-persona-blended) view instead.
    """
    closes = [100.0 + i for i in range(LOOKBACK_BARS)]
    bars, price_store, trade_repo = _setup(closes)
    # The shared broker account reports a position for this ticker (e.g. a
    # different persona's holding), but this persona has no filled trades.
    broker = FakeBroker(positions={TICKER: 5.0})
    as_of = bars.index[-1].to_pydatetime()
    strategy = TrendFollower()

    import logging

    with caplog.at_level(logging.WARNING):
        run_persona(
            strategy=strategy,
            broker=broker,
            price_store=price_store,
            trade_repo=trade_repo,
            as_of=as_of,
            universe=[TICKER],
            sizer=SIZER,
            run_id="local-test",
            dry_run=False,
        )

    assert any("position drift" in record.message for record in caplog.records)
    # Planning still used the persona's own (zero) position, not the
    # broker's 5.0 — so it still placed a flat->long buy, not a no-op.
    assert len(broker.submissions) == 1
    assert broker.submissions[0].side == "buy"


def test_weekend_as_of_produces_no_new_orders():
    closes = [100.0 + i for i in range(LOOKBACK_BARS)]
    bars, price_store, trade_repo = _setup(closes)
    broker = FakeBroker()
    friday_as_of = bars.index[-1].to_pydatetime()
    strategy = TrendFollower()

    run_persona(
        strategy=strategy,
        broker=broker,
        price_store=price_store,
        trade_repo=trade_repo,
        as_of=friday_as_of,
        universe=[TICKER],
        sizer=SIZER,
        run_id="local-friday",
        dry_run=False,
    )
    assert len(broker.submissions) == 1

    # A weekend as_of, with no new price bar available since Friday's close
    # — the strategy re-derives the same signal_ts (Friday's bar), so the
    # idempotent insert is a no-op and no new order is placed.
    saturday_as_of = friday_as_of + pd.Timedelta(days=1)
    run_persona(
        strategy=strategy,
        broker=broker,
        price_store=price_store,
        trade_repo=trade_repo,
        as_of=saturday_as_of,
        universe=[TICKER],
        sizer=SIZER,
        run_id="local-saturday",
        dry_run=False,
    )

    assert len(broker.submissions) == 1
