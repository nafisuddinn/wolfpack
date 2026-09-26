"""In-memory fakes for Broker / PriceStore / TradeRepo used across tests.

No real network calls, no real DB, no real Alpaca API hit — ever — from the
test suite. These fakes implement the same protocols as the real
Alpaca/Supabase-backed implementations in broker.py / store.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

from wolfpack_worker.broker import BrokerOrder, Session


class BrokerCrashError(RuntimeError):
    """Simulates a broker/network failure after an order was accepted."""


@dataclass
class FakeBroker:
    """A Broker double that records submissions and lets tests simulate:

    - normal fills
    - open orders that should block re-planning
    - a crash *after* the order was accepted broker-side (to exercise
      reconciliation's "attach via client_order_id" path)
    """

    positions: dict[str, float] = field(default_factory=dict)
    open_order_tickers: set[str] = field(default_factory=set)
    sessions: list[Session] = field(default_factory=list)

    submissions: list[BrokerOrder] = field(default_factory=list, init=False)
    _orders_by_client_id: dict[str, BrokerOrder] = field(default_factory=dict, init=False)
    crash_on_submit: bool = False
    crash_after_accept: bool = False
    _next_broker_order_id: int = field(default=1, init=False)

    def submit_market_order(
        self, *, symbol: str, side: str, qty: float, client_order_id: str
    ) -> BrokerOrder:
        if self.crash_on_submit:
            raise BrokerCrashError("simulated network failure before acceptance")

        broker_order_id = f"broker-order-{self._next_broker_order_id}"
        self._next_broker_order_id += 1
        order = BrokerOrder(
            client_order_id=client_order_id,
            broker_order_id=broker_order_id,
            symbol=symbol,
            side=side,
            qty=qty,
            status="accepted",
        )
        self._orders_by_client_id[client_order_id] = order
        self.submissions.append(order)

        if self.crash_after_accept:
            # The broker accepted the order (it now exists server-side and
            # is findable via get_order_by_client_id), but the process
            # crashes before it can record the broker_order_id anywhere.
            raise BrokerCrashError("simulated crash after broker accepted the order")

        return order

    def get_order_by_client_id(self, client_order_id: str) -> Optional[BrokerOrder]:
        return self._orders_by_client_id.get(client_order_id)

    def get_open_orders(self, symbol: Optional[str] = None) -> list[BrokerOrder]:
        orders = [
            o
            for ticker in self.open_order_tickers
            if symbol is None or ticker == symbol
            for o in [
                BrokerOrder(
                    client_order_id=f"open-{ticker}",
                    broker_order_id=f"open-broker-{ticker}",
                    symbol=ticker,
                    side="buy",
                    qty=1,
                    status="new",
                )
            ]
        ]
        return orders

    def get_position_qty(self, symbol: str) -> float:
        return self.positions.get(symbol, 0.0)

    def get_calendar(self, start: date, end: date) -> list[Session]:
        return [s for s in self.sessions if start <= s.date <= end]

    def fill(self, client_order_id: str, fill_price: float) -> None:
        order = self._orders_by_client_id[client_order_id]
        self._orders_by_client_id[client_order_id] = BrokerOrder(
            client_order_id=order.client_order_id,
            broker_order_id=order.broker_order_id,
            symbol=order.symbol,
            side=order.side,
            qty=order.qty,
            status="filled",
            filled_qty=order.qty,
            filled_avg_price=fill_price,
        )


@dataclass
class InMemoryPriceStore:
    bars: dict[str, "object"] = field(default_factory=dict)

    def get_bars(self, ticker: str, as_of: datetime, limit: int):
        df = self.bars.get(ticker)
        if df is None:
            import pandas as pd

            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        truncated = df.loc[df.index <= as_of]
        return truncated.iloc[-limit:] if limit else truncated

    def upsert_bars(self, ticker: str, timeframe: str, bars) -> None:
        self.bars[ticker] = bars

    def latest_bar_ts(self, ticker: str, timeframe: str):
        df = self.bars.get(ticker)
        if df is None or len(df) == 0:
            return None
        return df.index.max()

    def has_any_bars(self, ticker: str, timeframe: str) -> bool:
        df = self.bars.get(ticker)
        return df is not None and len(df) > 0


@dataclass
class PendingTrade:
    id: str
    persona_id: str
    ticker: str
    side: str
    qty: float
    signal_ts: datetime
    client_order_id: Optional[str]
    broker_order_id: Optional[str]
    signal_payload: dict
    status: str = "pending"


@dataclass
class InMemoryTradeRepo:
    persona_ids: dict[str, str] = field(default_factory=dict)
    trades: dict[str, PendingTrade] = field(default_factory=dict)
    inserts: list[dict] = field(default_factory=list, init=False)
    updates: list[dict] = field(default_factory=list, init=False)
    _by_key: dict[tuple, str] = field(default_factory=dict, init=False)
    _next_id: int = field(default=1, init=False)

    def get_persona_id(self, slug: str) -> str:
        return self.persona_ids.setdefault(slug, f"persona-{slug}")

    def get_position_qty(self, persona_id: str, ticker: str) -> float:
        qty = 0.0
        for trade in self.trades.values():
            if trade.persona_id != persona_id or trade.ticker != ticker:
                continue
            if trade.status != "filled":
                continue
            qty += trade.qty if trade.side == "buy" else -trade.qty
        return qty

    def insert_pending_trade(
        self,
        *,
        persona_id: str,
        ticker: str,
        side: str,
        qty: float,
        signal_ts: datetime,
        run_id: str,
        signal_payload: dict,
        client_order_id: str,
    ) -> Optional[str]:
        insert_dict = {
            "persona_id": persona_id,
            "ticker": ticker,
            "side": side,
            "qty": qty,
            "signal_ts": signal_ts,
            "run_id": run_id,
            "signal_payload": signal_payload,
            "client_order_id": client_order_id,
            "broker_order_id": None,
            "status": "pending",
        }
        assert not any(k.startswith("rationale") for k in insert_dict), (
            "worker must never write rationale* fields"
        )
        self.inserts.append(insert_dict)

        key = (persona_id, ticker, signal_ts)
        if key in self._by_key:
            return None  # simulates on_conflict ignore_duplicates=True

        trade_id = f"trade-{self._next_id}"
        self._next_id += 1
        self._by_key[key] = trade_id
        self.trades[trade_id] = PendingTrade(
            id=trade_id,
            persona_id=persona_id,
            ticker=ticker,
            side=side,
            qty=qty,
            signal_ts=signal_ts,
            client_order_id=client_order_id,
            broker_order_id=None,
            signal_payload=signal_payload,
        )
        return trade_id

    def attach_broker_order_id(self, trade_id: str, broker_order_id: str) -> None:
        update = {"id": trade_id, "broker_order_id": broker_order_id}
        assert not any(k.startswith("rationale") for k in update)
        self.updates.append(update)
        self.trades[trade_id].broker_order_id = broker_order_id

    def get_pending_trades(self) -> list[PendingTrade]:
        return [t for t in self.trades.values() if t.status == "pending"]

    def update_trade_status(self, trade_id: str, **fields) -> None:
        update = {"id": trade_id, **fields}
        assert not any(k.startswith("rationale") for k in update), (
            "worker must never write rationale* fields"
        )
        self.updates.append(update)
        trade = self.trades[trade_id]
        for key, value in fields.items():
            setattr(trade, key, value) if hasattr(trade, key) else None
        if "status" in fields:
            trade.status = fields["status"]

    def get_open_client_order_ids(self, persona_id: str) -> set[str]:
        return {
            t.client_order_id
            for t in self.trades.values()
            if t.persona_id == persona_id and t.status == "pending" and t.client_order_id
        }
