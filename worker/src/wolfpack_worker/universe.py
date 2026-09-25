"""Shared trading-universe and sizing constants for all mechanical personas.

Kept in one place so every persona's strategy, price-ingestion, and
execution code agrees on what tickers exist and how much capital each
position gets — deviating per-persona would make cross-persona feed
comparisons meaningless.
"""

from __future__ import annotations

UNIVERSE: tuple[str, ...] = ("SPY", "QQQ", "AAPL", "JPM", "XOM")

# Preferred market-data feed for Alpaca's historical bars endpoint. "sip" is
# the consolidated tape; some accounts/plans don't have SIP entitlement, in
# which case market_data.refresh_prices falls back to "iex" and logs it.
DATA_FEED = "sip"

# Split-adjusted, NOT dividend-adjusted. This is a known limitation: for
# dividend-paying tickers (e.g. XOM, JPM), raw close-to-close returns used by
# these strategies will slightly understate total return around ex-dividend
# dates. Acceptable for v1 mechanical personas; revisit if a persona's edge
# claims start to hinge on precise total-return accuracy.
ADJUSTMENT = "split"

# NOTE: `FixedNotionalSizer` floors to whole shares, so any ticker priced
# above this notional amount would floor to zero shares and silently produce
# no trade. Watch for this if the universe ever expands to include a
# $3,000+ stock.
NOTIONAL_PER_POSITION = 3000.0
