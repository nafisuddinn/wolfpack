"""Tests for the paper-trading-endpoint safety assertion in config.py.

This is the most safety-critical test in the worker: it proves the third
backstop (beyond the PreToolUse hook and the RUNBOOK.md note) actually works
— that a non-paper ALPACA_BASE_URL causes a hard failure, not a silent
pass-through.
"""

from __future__ import annotations

import pytest

from wolfpack_worker.config import (
    PAPER_BASE_URL,
    NonPaperTradingEndpointError,
    assert_paper_trading_endpoint,
    load_config,
)


def test_paper_endpoint_passes() -> None:
    # Should not raise.
    assert_paper_trading_endpoint(PAPER_BASE_URL)


@pytest.mark.parametrize(
    "bad_url",
    [
        "https://api.alpaca.markets",  # live trading endpoint
        "https://paper-api.alpaca.markets.evil.com",  # lookalike
        "http://paper-api.alpaca.markets",  # http instead of https
        "",
        "not-a-url",
    ],
)
def test_non_paper_endpoint_raises(bad_url: str) -> None:
    with pytest.raises(NonPaperTradingEndpointError):
        assert_paper_trading_endpoint(bad_url)


def test_load_config_raises_on_live_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPACA_BASE_URL", "https://api.alpaca.markets")
    monkeypatch.setenv("ALPACA_API_KEY", "test")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "test")

    with pytest.raises(NonPaperTradingEndpointError):
        load_config(env_file=None)


def test_load_config_succeeds_on_paper_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPACA_BASE_URL", PAPER_BASE_URL)
    monkeypatch.setenv("ALPACA_API_KEY", "test")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "test")

    config = load_config(env_file=None)
    assert config.alpaca_base_url == PAPER_BASE_URL


def test_load_config_exits_nonzero_as_a_script(monkeypatch: pytest.MonkeyPatch) -> None:
    """Proves the `python -m wolfpack_worker.config` CLI entrypoint actually
    exits non-zero (not just that the Python exception is raised) on a
    non-paper endpoint — this is what CI would observe.
    """
    import subprocess
    import sys

    env = {
        "ALPACA_BASE_URL": "https://api.alpaca.markets",
        "ALPACA_API_KEY": "test",
        "ALPACA_SECRET_KEY": "test",
        "SUPABASE_URL": "https://example.supabase.co",
        "SUPABASE_SERVICE_ROLE_KEY": "test",
        "PATH": "/usr/bin:/bin",
    }
    result = subprocess.run(
        [sys.executable, "-m", "wolfpack_worker.config"],
        env=env,
        cwd=None,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "FATAL" in result.stderr
