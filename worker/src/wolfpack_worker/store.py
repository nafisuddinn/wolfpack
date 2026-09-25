"""PriceStore / TradeRepo protocols + the Supabase-backed implementations.

Protocols exist so `execution.py`/`daily_trades.py` never import the
`supabase` client directly and tests use `tests/fakes.py`'s in-memory
doubles instead of a real database.

Neither `SupabasePriceStore` nor `SupabaseTradeRepo` ever includes a
`rationale`, `rationale_written_at`, or `rationale_author` key in any
insert/update — those columns are write-once via the `set_trade_rationale`
RPC, owned by a separate, least-privileged step.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Optional, Protocol

import pandas as pd
from supabase import Client

_PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]


class PriceStore(Protocol):
    def get_bars(self, ticker: str, as_of: datetime, limit: int) -> pd.DataFrame: ...

    def upsert_bars(self, ticker: str, timeframe: str, bars: pd.DataFrame) -> None: ...

    def latest_bar_ts(self, ticker: str, timeframe: str) -> Optional[datetime]: ...

    def has_any_bars(self, ticker: str, timeframe: str) -> bool: ...


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


class TradeRepo(Protocol):
    def get_persona_id(self, slug: str) -> str: ...

    def get_position_qty(self, persona_id: str, ticker: str) -> float:
        """This persona's own position, derived from its filled trades

        (buys minus sells) — the source of truth this worker plans orders
        against, per the 2026-09-24 Decision Log entry: a persona's position
        is never read back from the shared Alpaca paper account directly,
        since multiple personas trade through one account. Callers should
        additionally do a warn-only sanity check against the broker's actual
        position (see daily_trades.run_persona) to catch drift, without
        hard-enforcing consistency.
        """
        ...

    def insert_pending_trade(
        self,
        *,
        persona_id: str,
        ticker: str,
        side: str,
        qty: float,
        signal_ts: datetime,
        run_id: str,
        signal_payload: Mapping[str, Any],
        client_order_id: str,
    ) -> Optional[str]: ...

    def attach_broker_order_id(self, trade_id: str, broker_order_id: str) -> None: ...

    def get_pending_trades(self) -> list[PendingTrade]: ...

    def update_trade_status(self, trade_id: str, **fields: Any) -> None: ...


class SupabasePriceStore:
    """Backed by the `prices` table (see supabase/migrations init schema)."""

    def __init__(self, client: Client) -> None:
        self._client = client

    def get_bars(self, ticker: str, as_of: datetime, limit: int) -> pd.DataFrame:
        response = (
            self._client.table("prices")
            .select("ts,open,high,low,close,volume")
            .eq("ticker", ticker)
            .eq("timeframe", "1Day")
            .lte("ts", as_of.isoformat())
            .order("ts", desc=True)
            .limit(limit)
            .execute()
        )
        rows = list(reversed(response.data or []))
        if not rows:
            return pd.DataFrame(columns=_PRICE_COLUMNS)
        df = pd.DataFrame(rows)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = df.set_index("ts").sort_index()
        return df[_PRICE_COLUMNS]

    def upsert_bars(self, ticker: str, timeframe: str, bars: pd.DataFrame) -> None:
        if bars.empty:
            return
        records = []
        for ts, row in bars.iterrows():
            records.append(
                {
                    "ticker": ticker,
                    "timeframe": timeframe,
                    "ts": ts.isoformat(),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": int(row["volume"]) if pd.notna(row["volume"]) else None,
                }
            )
        self._client.table("prices").upsert(records, on_conflict="ticker,timeframe,ts").execute()

    def latest_bar_ts(self, ticker: str, timeframe: str) -> Optional[datetime]:
        response = (
            self._client.table("prices")
            .select("ts")
            .eq("ticker", ticker)
            .eq("timeframe", timeframe)
            .order("ts", desc=True)
            .limit(1)
            .execute()
        )
        if not response.data:
            return None
        return pd.Timestamp(response.data[0]["ts"]).to_pydatetime()

    def has_any_bars(self, ticker: str, timeframe: str) -> bool:
        return self.latest_bar_ts(ticker, timeframe) is not None


class SupabaseTradeRepo:
    """Backed by the `trades`/`personas` tables. Never touches `rationale*`."""

    def __init__(self, client: Client) -> None:
        self._client = client
        self._persona_id_cache: dict[str, str] = {}

    def get_persona_id(self, slug: str) -> str:
        if slug in self._persona_id_cache:
            return self._persona_id_cache[slug]
        response = self._client.table("personas").select("id").eq("slug", slug).single().execute()
        persona_id = response.data["id"]
        self._persona_id_cache[slug] = persona_id
        return persona_id

    def get_position_qty(self, persona_id: str, ticker: str) -> float:
        response = (
            self._client.table("trades")
            .select("side,qty")
            .eq("persona_id", persona_id)
            .eq("ticker", ticker)
            .eq("status", "filled")
            .execute()
        )
        qty = 0.0
        for row in response.data or []:
            qty += float(row["qty"]) if row["side"] == "buy" else -float(row["qty"])
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
        signal_payload: Mapping[str, Any],
        client_order_id: str,
    ) -> Optional[str]:
        record = {
            "persona_id": persona_id,
            "ticker": ticker,
            "side": side,
            "qty": qty,
            "signal_ts": signal_ts.isoformat(),
            "run_id": run_id,
            "signal_payload": dict(signal_payload),
            "client_order_id": client_order_id,
            "status": "pending",
        }
        response = (
            self._client.table("trades")
            .upsert(
                record,
                on_conflict="persona_id,ticker,signal_ts",
                ignore_duplicates=True,
            )
            .execute()
        )
        if not response.data:
            return None
        return response.data[0]["id"]

    def attach_broker_order_id(self, trade_id: str, broker_order_id: str) -> None:
        self._client.table("trades").update({"broker_order_id": broker_order_id}).eq(
            "id", trade_id
        ).execute()

    def get_pending_trades(self) -> list[PendingTrade]:
        response = self._client.table("trades").select("*").eq("status", "pending").execute()
        trades = []
        for row in response.data or []:
            trades.append(
                PendingTrade(
                    id=row["id"],
                    persona_id=row["persona_id"],
                    ticker=row["ticker"],
                    side=row["side"],
                    qty=row["qty"],
                    signal_ts=pd.Timestamp(row["signal_ts"]).to_pydatetime(),
                    client_order_id=row.get("client_order_id"),
                    broker_order_id=row.get("broker_order_id"),
                    signal_payload=row.get("signal_payload") or {},
                    status=row["status"],
                )
            )
        return trades

    def update_trade_status(self, trade_id: str, **fields: Any) -> None:
        assert not any(k.startswith("rationale") for k in fields), (
            "SupabaseTradeRepo must never write rationale* fields"
        )
        update = dict(fields)
        if "filled_at" in update and isinstance(update["filled_at"], datetime):
            update["filled_at"] = update["filled_at"].isoformat()
        self._client.table("trades").update(update).eq("id", trade_id).execute()
