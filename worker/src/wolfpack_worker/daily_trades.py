"""Entrypoint stub for the daily mechanical trade-execution job.

Invoked by .github/workflows/daily-trades.yml's mechanical step as:
    uv run --project worker -m wolfpack_worker.daily_trades

This is intentionally a stub for this backlog item (scaffolding + schema
only) — actual persona strategy logic (Trend Follower, Contrarian, etc.) is
a separate, later backlog item. What this stub DOES do for real: load config
(which hard-asserts the paper-trading endpoint, see config.py) and confirm it
can construct a service-role DB client, so the wiring is genuinely exercised
end-to-end rather than being a silent no-op.
"""

from __future__ import annotations

from wolfpack_worker.config import load_config
from wolfpack_worker.db import get_client


def main() -> None:
    config = load_config()
    get_client(config)
    print(
        "wolfpack_worker.daily_trades: config loaded, paper-trading endpoint "
        "verified, DB client constructed. Persona strategy logic is not yet "
        "implemented (separate backlog item) — no trades were placed."
    )


if __name__ == "__main__":
    main()
