"""Broker abstraction + the one real implementation (Alpaca, paper only).

`Broker` is a Protocol so `execution.py` and `daily_trades.py` never import
`alpaca-py` directly and tests never need real network access — they use
`tests/fakes.py::FakeBroker` instead.

SAFETY-CRITICAL: `make_trading_client` is the second of the worker's
paper-trading backstops (config.py's `assert_paper_trading_endpoint` is the
first). It re-asserts the paper endpoint here, at the one call site that
actually constructs an Alpaca `TradingClient`, and always passes the
*literal* `paper=True` — never a variable — so no config value or code path
can ever cause this to construct a live-trading client.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal, Optional, Protocol

from wolfpack_worker.config import WorkerConfig, assert_paper_trading_endpoint


@dataclass(frozen=True)
class Session:
    """One trading-calendar session (from Alpaca's market calendar)."""

    date: date
    open: datetime
    close: datetime


@dataclass(frozen=True)
class BrokerOrder:
    client_order_id: str
    broker_order_id: Optional[str]
    symbol: str
    side: Literal["buy", "sell"]
    qty: float
    status: str
    filled_qty: Optional[float] = None
    filled_avg_price: Optional[float] = None


class Broker(Protocol):
    def submit_market_order(
        self, *, symbol: str, side: Literal["buy", "sell"], qty: float, client_order_id: str
    ) -> BrokerOrder: ...

    def get_order_by_client_id(self, client_order_id: str) -> Optional[BrokerOrder]: ...

    def get_open_orders(self, symbol: Optional[str] = None) -> list[BrokerOrder]: ...

    def get_position_qty(self, symbol: str) -> float: ...

    def get_calendar(self, start: date, end: date) -> list[Session]: ...


def make_trading_client(config: WorkerConfig):
    """Construct an alpaca-py `TradingClient` hard-wired to paper trading.

    Never returns a client capable of touching a live account: re-asserts
    the paper endpoint (defense in depth beyond `load_config`), then passes
    the literal `paper=True` to alpaca-py — this must never become a
    variable derived from config, env, or any other runtime value.
    """
    assert_paper_trading_endpoint(config.alpaca_base_url)

    from alpaca.trading.client import TradingClient

    return TradingClient(
        api_key=config.alpaca_api_key,
        secret_key=config.alpaca_secret_key,
        paper=True,
    )


class AlpacaPaperBroker:
    """`Broker` implementation backed by alpaca-py's `TradingClient`.

    Not exercised against the real Alpaca API in the test suite (no network
    calls in tests) — only `make_trading_client`'s paper-only guarantee is
    tested directly. This class is a thin adapter converting alpaca-py's
    models to/from this module's `BrokerOrder`/`Session`.
    """

    def __init__(self, config: WorkerConfig) -> None:
        self._client = make_trading_client(config)

    def submit_market_order(
        self, *, symbol: str, side: Literal["buy", "sell"], qty: float, client_order_id: str
    ) -> BrokerOrder:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
        request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.DAY,
            client_order_id=client_order_id,
        )
        order = self._client.submit_order(order_data=request)
        return _order_to_broker_order(order)

    def get_order_by_client_id(self, client_order_id: str) -> Optional[BrokerOrder]:
        from alpaca.common.exceptions import APIError

        try:
            order = self._client.get_order_by_client_id(client_order_id)
        except APIError as exc:
            if exc.status_code == 404:
                return None
            raise
        return _order_to_broker_order(order)

    def get_open_orders(self, symbol: Optional[str] = None) -> list[BrokerOrder]:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        request = GetOrdersRequest(
            status=QueryOrderStatus.OPEN,
            symbols=[symbol] if symbol else None,
        )
        orders = self._client.get_orders(filter=request)
        return [_order_to_broker_order(o) for o in orders]

    def get_position_qty(self, symbol: str) -> float:
        from alpaca.common.exceptions import APIError

        try:
            position = self._client.get_open_position(symbol)
        except APIError as exc:
            if exc.status_code == 404:
                return 0.0
            raise
        return float(position.qty)

    def get_calendar(self, start: date, end: date) -> list[Session]:
        from alpaca.trading.requests import GetCalendarRequest

        request = GetCalendarRequest(start=start, end=end)
        calendar = self._client.get_calendar(filters=request)
        return [Session(date=c.date, open=c.open, close=c.close) for c in calendar]


def _order_to_broker_order(order) -> BrokerOrder:
    # alpaca-py 0.44.0's `Order.side` field is typed `Optional[OrderSide]` and
    # pydantic always parses the API's raw string into that enum member — so
    # a single `== OrderSide.BUY` comparison is sufficient (confirmed against
    # the installed alpaca.trading.models.Order source; no string/enum
    # inconsistency to defend against here).
    from alpaca.trading.enums import OrderSide

    side = "buy" if order.side == OrderSide.BUY else "sell"
    filled_qty = float(order.filled_qty) if order.filled_qty is not None else None
    filled_avg_price = (
        float(order.filled_avg_price) if getattr(order, "filled_avg_price", None) is not None else None
    )
    status = getattr(order.status, "value", str(order.status))
    return BrokerOrder(
        client_order_id=order.client_order_id,
        broker_order_id=str(order.id) if order.id is not None else None,
        symbol=order.symbol,
        side=side,
        qty=float(order.qty) if order.qty is not None else 0.0,
        status=status,
        filled_qty=filled_qty,
        filled_avg_price=filled_avg_price,
    )
