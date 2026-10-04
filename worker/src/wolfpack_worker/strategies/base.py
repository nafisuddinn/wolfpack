"""Shared strategy contracts: the shape every persona's strategy plugs into.

This module is deliberately domain-agnostic about *how* a strategy decides a
target exposure — it only enforces the one non-negotiable rule that applies
to every persona: no lookahead. `StrategyContext` cannot be constructed with
future information, so a strategy that only ever reads from `ctx.bars`,
`ctx.news` and `ctx.as_of` structurally cannot cheat.

News (The Scout) gets the same treatment as bars, and one more check,
because a news leak is easier to miss: an article published after the
close, explaining why the stock moved that day, would leak the outcome into
the input. So a context refuses (LookaheadError) any article with
created_at > as_of, any naive timestamp, any market session closing after
as_of, and any article first seen by WolfPack after the run's news cutoff
(a late arrival is never used for a decision it was not available for).
Strategies that need news declare `requires_news = True`; the orchestrator
builds a `NewsSnapshot` only for them, truncated with `truncate_news`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Optional, Protocol

import pandas as pd

from wolfpack_worker.broker import Session


class LookaheadError(ValueError):
    """Raised when a StrategyContext would let a strategy see future data.

    This must never be caught and swallowed — per CLAUDE.md's no-lookahead
    rule, any code path that could compute a signal using information from
    after `as_of` must hard-fail, not silently truncate.
    """


NEWS_STATUSES = ("ok", "failed", "stale")


def _is_aware(ts: Any) -> bool:
    return getattr(ts, "tzinfo", None) is not None and ts.tzinfo.utcoffset(ts) is not None


def _empty_articles() -> pd.DataFrame:
    from wolfpack_worker.news import articles_frame

    return articles_frame([])


@dataclass(frozen=True)
class NewsSnapshot:
    """The point-in-time news a news-using strategy may see for one decision.

    articles: rows of the news store (news.ARTICLE_COLUMNS), already
        truncated to created_at <= as_of and first_seen_at <= cutoff.
        Includes headline text, which strategies may SCORE but must never
        copy into a payload, log line or rationale (licensing: headlines
        stay private).
    sessions: market calendar sessions, ascending; the last one closes at
        as_of and the first one only bounds the earliest window.
    cutoff: the run's news cutoff (time the daily refresh finished).
    status: "ok", or "failed" / "stale" (then no articles; the strategy
        must return [] and log `reason`).
    late_arrivals: live-ingested articles in the lookback first seen after
        the decision their window belonged to (reported, never used
        retroactively). excluded_after_cutoff: rows dropped by the cutoff.
    """

    articles: pd.DataFrame
    sessions: tuple[Session, ...]
    cutoff: Optional[datetime]
    status: str
    reason: Optional[str] = None
    late_arrivals: int = 0
    excluded_after_cutoff: int = 0

    def __post_init__(self) -> None:
        if self.status not in NEWS_STATUSES:
            raise ValueError(f"NewsSnapshot.status must be one of {NEWS_STATUSES}, got {self.status!r}")
        if self.status != "ok" and len(self.articles):
            raise ValueError("an unavailable NewsSnapshot must not carry articles")

    @classmethod
    def unavailable(cls, status: str, reason: str) -> "NewsSnapshot":
        return cls(articles=_empty_articles(), sessions=(), cutoff=None, status=status, reason=reason)


def _check_news(news: NewsSnapshot, as_of: datetime) -> None:
    if news.cutoff is not None and not _is_aware(news.cutoff):
        raise LookaheadError("NewsSnapshot.cutoff must be timezone-aware.")
    for s in news.sessions:
        if not (_is_aware(s.close) and _is_aware(s.open)):
            raise LookaheadError(f"market session {s.date} has a naive timezone-less open/close.")
        if s.close > as_of:
            raise LookaheadError(
                f"market session {s.date} closes at {s.close!r}, after as_of ({as_of!r}): the news "
                "calendar would describe a session that has not finished."
            )
    arts = news.articles
    if arts is None or len(arts) == 0:
        return
    for col in ("created_at", "first_seen_at"):
        if not isinstance(arts[col].dtype, pd.DatetimeTZDtype):
            raise LookaheadError(f"news column {col!r} must hold timezone-aware timestamps.")
    latest = arts["created_at"].max()
    if latest > as_of:
        raise LookaheadError(
            f"News contains an article published at {latest!r}, after as_of ({as_of!r}). An article "
            "published after the close can describe that day's move; refusing to let it into a decision."
        )
    if news.cutoff is not None:
        seen = arts["first_seen_at"].max()
        if seen > news.cutoff:
            raise LookaheadError(
                f"News contains an article first seen at {seen!r}, after this run's news cutoff "
                f"({news.cutoff!r}); late arrivals are never used retroactively."
            )


@dataclass(frozen=True)
class StrategyContext:
    """Everything a `Strategy.evaluate()` call is allowed to see.

    `as_of` must be tz-aware. `bars` maps ticker -> a DataFrame with a
    tz-aware, ascending DatetimeIndex and columns open/high/low/close/volume.
    `news` is None unless the strategy `requires_news`.
    Construction raises `LookaheadError` if `as_of` is naive, if any bar
    in any ticker's DataFrame has a timestamp after `as_of`, or if `news`
    breaks any of the checks in this module's docstring.
    """

    as_of: datetime
    universe: tuple[str, ...]
    bars: Mapping[str, pd.DataFrame]
    news: Optional[NewsSnapshot] = field(default=None)

    def __post_init__(self) -> None:
        if self.as_of.tzinfo is None or self.as_of.tzinfo.utcoffset(self.as_of) is None:
            raise LookaheadError(
                "StrategyContext.as_of must be timezone-aware — a naive "
                "datetime is ambiguous and unsafe for lookahead checks."
            )
        if self.news is not None:
            _check_news(self.news, self.as_of)
        for ticker, df in self.bars.items():
            if df is None or len(df) == 0:
                continue
            last_ts = df.index.max()
            if last_ts > self.as_of:
                raise LookaheadError(
                    f"Bar data for {ticker!r} contains a timestamp "
                    f"({last_ts!r}) after as_of ({self.as_of!r}) — refusing "
                    "to construct a StrategyContext that could leak future "
                    "information into a strategy."
                )


@dataclass(frozen=True)
class TargetPosition:
    """A strategy's desired exposure for one ticker as of one signal time."""

    ticker: str
    target_exposure: float
    signal_ts: datetime
    payload: Mapping[str, Any]


class Strategy(Protocol):
    """A persona's signal logic: price history in, target exposures out.

    `evaluate()` returns one `TargetPosition` per ticker it has an opinion
    on. Omitting a ticker from the returned list means "no opinion, hold
    current position" — this covers both insufficient history (not enough
    bars yet to compute a signal at all, e.g. Trend Follower's original use)
    and an indeterminate state given a full lookback window (e.g. Contrarian
    finding no decisive bar anywhere in its window). Either way, the
    orchestrator takes no action for that ticker; it does not mean "flat".
    """

    slug: str
    version: str
    lookback_bars: int
    # Optional (read with getattr, default False / 0): a strategy that sets
    # requires_news = True gets ctx.news covering news_lookback_sessions
    # feature windows; every other strategy gets ctx.news = None.
    # requires_news: bool
    # news_lookback_sessions: int

    def evaluate(self, ctx: StrategyContext) -> list[TargetPosition]: ...


def truncate_bars(
    bars: Mapping[str, pd.DataFrame], as_of: datetime
) -> dict[str, pd.DataFrame]:
    """Drop any bar rows with a timestamp after `as_of`, per ticker.

    This is the orchestrator's job, not `StrategyContext`'s: a `PriceStore`
    may legitimately hold data past `as_of` (e.g. the DB simply has more
    history than a given run needs), and the orchestrator is expected to
    truncate before ever constructing a `StrategyContext`. `StrategyContext`
    itself stays strict and raises on anything left over after truncation —
    defense in depth, not a substitute for this step.
    """

    truncated: dict[str, pd.DataFrame] = {}
    for ticker, df in bars.items():
        if df is None or len(df) == 0:
            truncated[ticker] = df
            continue
        truncated[ticker] = df.loc[df.index <= as_of]
    return truncated


def truncate_news(
    articles: pd.DataFrame, as_of: datetime, cutoff: Optional[datetime]
) -> tuple[pd.DataFrame, int]:
    """Keep articles with created_at <= as_of and first_seen_at <= cutoff.

    Returns (kept, number dropped by the cutoff). The orchestrator's job,
    like truncate_bars: a news store legitimately holds articles published
    after the close (they belong to the next session's window) and articles
    first seen later; StrategyContext refuses anything left over.
    """
    if articles is None or len(articles) == 0:
        return articles, 0
    kept = articles.loc[articles["created_at"] <= as_of]
    if cutoff is None:
        return kept.reset_index(drop=True), 0
    late = kept["first_seen_at"] > cutoff
    return kept.loc[~late].reset_index(drop=True), int(late.sum())
