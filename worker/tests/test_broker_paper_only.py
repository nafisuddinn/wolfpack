"""Paper-trading safety for the one call site that constructs a real
alpaca-py TradingClient.

Per project rule: this file must never contain the literal live-endpoint
URL string anywhere — a PreToolUse hook blocks files referencing it. Tests
instead assert equality against `PAPER_BASE_URL` and use unmistakably-fake
non-paper URLs to prove the rejection path works.
"""

from __future__ import annotations

import inspect

import pytest

import wolfpack_worker.broker as broker_module
from wolfpack_worker.broker import make_trading_client
from wolfpack_worker.config import (
    PAPER_BASE_URL,
    NonPaperTradingEndpointError,
    WorkerConfig,
)


def _config(base_url: str) -> WorkerConfig:
    return WorkerConfig(
        alpaca_api_key="test-key",
        alpaca_secret_key="test-secret",
        alpaca_base_url=base_url,
        supabase_url="https://example.supabase.co",
        supabase_service_role_key="test",
    )


def test_make_trading_client_constructs_client_for_paper_endpoint():
    client = make_trading_client(_config(PAPER_BASE_URL))

    from alpaca.trading.client import TradingClient

    assert isinstance(client, TradingClient)


def test_make_trading_client_source_hard_codes_literal_paper_true():
    """`paper=True` must be a source-level literal, never a variable derived
    from config/env — this is what makes it impossible for any config value
    to flip this call site to a live-trading client.
    """
    source = inspect.getsource(broker_module.make_trading_client)
    assert "paper=True" in source
    # Guard against someone "fixing" this by passing a variable that happens
    # to be named/valued True at runtime instead of the literal.
    assert "paper=config" not in source
    assert "paper=self" not in source


@pytest.mark.parametrize(
    "bad_url",
    [
        "https://not-the-paper-endpoint.example.com",
        "http://" + PAPER_BASE_URL.removeprefix("https://"),  # scheme downgrade
        "",
    ],
)
def test_make_trading_client_raises_for_any_non_paper_endpoint(bad_url: str) -> None:
    with pytest.raises(NonPaperTradingEndpointError):
        make_trading_client(_config(bad_url))
