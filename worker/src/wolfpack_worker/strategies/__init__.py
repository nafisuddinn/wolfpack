"""Registry mapping a persona's `strategies.slug` to a strategy factory.

Adding Persona #2 (Contrarian), #3 (Analyst), or Scout is meant to be: write
a new `strategies/<name>.py` module implementing the `Strategy` protocol from
`strategies/base.py`, then add one line here. Nothing in `execution.py` or
`daily_trades.py` should need to change.
"""

from __future__ import annotations

from typing import Callable, Dict

from wolfpack_worker.strategies.base import Strategy
from wolfpack_worker.strategies.trend_follower import TrendFollower

REGISTRY: Dict[str, Callable[[], Strategy]] = {
    "trend-follower": TrendFollower,
}
