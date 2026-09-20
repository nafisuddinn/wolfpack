"""Environment/config loading for the WolfPack worker.

SAFETY-CRITICAL: this module is the third of three independent backstops
against ever placing a real-money trade (alongside the `.claude/settings.json`
PreToolUse hook, which only covers Claude's own tool calls, and the RUNBOOK.md
note, which is only a human-readable reminder). This module hard-asserts, at
import/load time, that ALPACA_BASE_URL points at Alpaca's paper-trading
endpoint and exits non-zero (raises) otherwise — a misconfigured secret in CI
or a local `.env` must be caught here, deterministically, every run.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

from dotenv import load_dotenv

PAPER_BASE_URL = "https://paper-api.alpaca.markets"


class NonPaperTradingEndpointError(RuntimeError):
    """Raised when ALPACA_BASE_URL is not Alpaca's paper-trading endpoint.

    This must never be caught and swallowed anywhere in this codebase —
    paper trading only, never real money, no exceptions.
    """


@dataclass(frozen=True)
class WorkerConfig:
    alpaca_api_key: str
    alpaca_secret_key: str
    alpaca_base_url: str
    supabase_url: str
    supabase_service_role_key: str
    news_api_key: str | None = None


def assert_paper_trading_endpoint(alpaca_base_url: str) -> None:
    """Hard-fail if `alpaca_base_url` is not Alpaca's paper-trading endpoint.

    Raises NonPaperTradingEndpointError (which callers should let propagate —
    this process must exit non-zero) if the URL does not exactly match
    PAPER_BASE_URL.
    """
    if alpaca_base_url != PAPER_BASE_URL:
        raise NonPaperTradingEndpointError(
            "Refusing to run: ALPACA_BASE_URL is not the paper-trading "
            f"endpoint. Got {alpaca_base_url!r}, expected {PAPER_BASE_URL!r}. "
            "This worker must NEVER place trades against a live/real-money "
            "brokerage endpoint."
        )


def load_config(env_file: str | None = ".env") -> WorkerConfig:
    """Load worker configuration from the environment.

    Calls `assert_paper_trading_endpoint` before returning — any caller that
    obtains a WorkerConfig from this function is guaranteed to be paper-only,
    or the process will already have raised/exited.
    """
    if env_file:
        load_dotenv(env_file, override=False)

    alpaca_base_url = os.environ.get("ALPACA_BASE_URL", "")
    assert_paper_trading_endpoint(alpaca_base_url)

    return WorkerConfig(
        alpaca_api_key=os.environ.get("ALPACA_API_KEY", ""),
        alpaca_secret_key=os.environ.get("ALPACA_SECRET_KEY", ""),
        alpaca_base_url=alpaca_base_url,
        supabase_url=os.environ.get("SUPABASE_URL", ""),
        supabase_service_role_key=os.environ.get("SUPABASE_SERVICE_ROLE_KEY", ""),
        news_api_key=os.environ.get("NEWS_API_KEY") or None,
    )


if __name__ == "__main__":
    # Allows `python -m wolfpack_worker.config` as a standalone sanity check
    # (e.g. in CI, before any trading code runs) that exits non-zero on a
    # misconfigured endpoint.
    try:
        load_config()
    except NonPaperTradingEndpointError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        sys.exit(1)
    print("Config OK: ALPACA_BASE_URL is the paper-trading endpoint.")
