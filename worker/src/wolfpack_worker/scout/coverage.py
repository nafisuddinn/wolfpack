"""M1: The Scout's news coverage / timing / revision report.

Uses ONLY the headlines' timestamps, symbols and sources plus the market
calendar: no prices, no labels, no scores-vs-returns. It is produced before
the Scout's trial is registered, so nothing here can have steered the
registered recipe toward the holdout (design section 12, M1: no label
correlations looked at before registration).

What it measures (all reported in MODEL_CARD_SCOUT.md by the documentarian):
* coverage: headlines per ticker per year, share of sessions with >= 1
  headline (has_news_1 = 1), deduplicated market headlines per year;
* timing: where created_at falls relative to the session (pre-open,
  in-session, after the close, non-session days); after-close and
  overnight headlines roll into the NEXT session's window;
* revisions (residual leak, design section 2): the stored headline is the
  vendor's current text, so a headline revised after the decision it feeds
  could carry later information. Reported: share with updated_at >
  created_at, > created_at + 1 min, and > the close of its own window
  (revised after the decision time) and > that close + 1 day;
* late arrivals (live rows only): first_seen_at - created_at quantiles.
  Backfilled rows have no real first-seen time, so vendor archive backfill
  is not measurable historically (only going forward, from live ingests);
* gaps: calendar months with zero universe headlines (a backfill gap would
  silently look like "no news", so training refuses if any exist).
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from wolfpack_worker.broker import Session
from wolfpack_worker.scout.windows import MARKET_TZ, assign_windows, session_closes


def _q(x: pd.Series) -> dict[str, float]:
    if x.empty:
        return {}
    qs = x.quantile([0.5, 0.9, 0.99])
    return {"p50": float(qs.iloc[0]), "p90": float(qs.iloc[1]), "p99": float(qs.iloc[2]), "max": float(x.max())}


def windows_for(articles: pd.DataFrame, sessions: Sequence[Session]) -> np.ndarray:
    return assign_windows(pd.DatetimeIndex(articles["created_at"]), sessions) if len(articles) else np.array([], int)


def revised_after_decision(articles: pd.DataFrame, sessions: Sequence[Session]) -> np.ndarray:
    """Per article: vendor_updated_at > close of its own window (False if
    the article has no window or no updated time)."""
    w = windows_for(articles, sessions)
    closes = session_closes(sessions)
    out = np.zeros(len(articles), dtype=bool)
    ok = w >= 1
    if ok.any():
        upd = pd.to_datetime(articles["vendor_updated_at"], utc=True)
        own = pd.Series(pd.NaT, index=articles.index, dtype="datetime64[ns, UTC]")
        own[ok] = closes[w[ok]]
        out = (upd > own).fillna(False).to_numpy(dtype=bool) & ok
    return out


def revision_flags(
    articles: pd.DataFrame, sessions: Sequence[Session], universe: Sequence[str]
) -> dict[str, np.ndarray]:
    """{ticker: bool per session}: does W_t hold a headline for the ticker
    that the vendor revised after close_t? Used to split holdout metrics
    (reported, not gated): a model leaning on post-decision revisions would
    look better on flagged rows."""
    m = len(sessions)
    w = windows_for(articles, sessions)
    rev = revised_after_decision(articles, sessions)
    syms = articles["symbols"].tolist() if len(articles) else []
    out = {}
    for t in universe:
        mask = rev & np.array([t in s for s in syms], dtype=bool) & (w >= 1)
        flags = np.zeros(m, dtype=bool)
        flags[w[mask]] = True
        out[t] = flags
    return out


def month_gaps(articles: pd.DataFrame, sessions: Sequence[Session], universe: Sequence[str]) -> list[str]:
    """Calendar months (between the first and last session) with zero
    headlines mentioning any universe ticker."""
    if not sessions:
        return []
    uni = set(universe)
    has = articles["symbols"].map(lambda s: bool(uni.intersection(s))) if len(articles) else pd.Series([], dtype=bool)
    months = set(pd.DatetimeIndex(articles.loc[has, "created_at"]).tz_convert(MARKET_TZ).strftime("%Y-%m")) \
        if len(articles) else set()
    want = pd.period_range(pd.Timestamp(sessions[0].date), pd.Timestamp(sessions[-1].date), freq="M").strftime("%Y-%m")
    return [m for m in want if m not in months]


def _phase(articles: pd.DataFrame, sessions: Sequence[Session]) -> pd.Series:
    by_date = {s.date: s for s in sessions}
    local = pd.DatetimeIndex(articles["created_at"]).tz_convert(MARKET_TZ)
    created = pd.DatetimeIndex(articles["created_at"])
    out = []
    for k, d in enumerate(local.date):
        s = by_date.get(d)
        if s is None:
            out.append("non_session_day")
        elif created[k] < s.open:
            out.append("pre_open")
        elif created[k] <= s.close:
            out.append("in_session")
        else:
            out.append("after_close")
    return pd.Series(out, index=articles.index)


def coverage_report(
    articles: pd.DataFrame, sessions: Sequence[Session], universe: Sequence[str]
) -> dict[str, Any]:
    if not len(articles):
        return {"n_articles": 0}
    uni = set(universe)
    arts = articles.loc[articles["symbols"].map(lambda s: bool(uni.intersection(s)))].reset_index(drop=True)
    w = windows_for(arts, sessions)
    in_cal = w >= 1
    arts = arts.loc[in_cal].reset_index(drop=True)
    w = w[in_cal]
    years = pd.DatetimeIndex(arts["created_at"]).tz_convert(MARKET_TZ).year
    session_year = pd.Series([s.date.year for s in sessions])
    m = len(sessions)

    per_ticker: dict[str, Any] = {}
    for t in universe:
        mask = arts["symbols"].map(lambda s, t=t: t in s).to_numpy()
        counts = np.bincount(w[mask], minlength=m)[1:]  # sessions 1..m-1
        yrs = session_year.iloc[1:].to_numpy()
        per_year = {}
        for y in sorted(set(yrs)):
            sel = yrs == y
            per_year[str(y)] = {
                "headlines": int(counts[sel].sum()),
                "sessions": int(sel.sum()),
                "share_sessions_with_news": float((counts[sel] > 0).mean()),
                "mean_per_session": float(counts[sel].mean()),
            }
        per_ticker[t] = {
            "headlines": int(mask.sum()),
            "share_sessions_with_news": float((counts > 0).mean()) if len(counts) else 0.0,
            "per_year": per_year,
        }

    created = pd.DatetimeIndex(arts["created_at"])
    upd = pd.to_datetime(arts["vendor_updated_at"], utc=True)
    lag = (upd - pd.Series(created, index=arts.index))
    closes = session_closes(sessions)
    own_close = pd.Series(closes[w], index=arts.index)
    rev_after = (upd > own_close).fillna(False)
    rev_after_1d = (upd > own_close + timedelta(days=1)).fillna(False)
    live = arts.loc[arts["ingest_mode"] == "live"]
    latency_min = ((live["first_seen_at"] - live["created_at"]).dt.total_seconds() / 60.0) if len(live) else pd.Series([], dtype=float)
    phase = _phase(arts, sessions)
    n_syms = arts["symbols"].map(len)

    return {
        "n_articles_universe": int(len(arts)),
        "created_at_first": created.min().isoformat(),
        "created_at_last": created.max().isoformat(),
        "sessions": {"first": str(sessions[0].date), "last": str(sessions[-1].date), "n": m},
        "sources": {str(k): int(v) for k, v in arts["source"].value_counts().items()},
        "per_ticker": per_ticker,
        "market_dedup_per_year": {str(y): int(c) for y, c in pd.Series(years).value_counts().sort_index().items()},
        "timing_share": {str(k): float(v) for k, v in phase.value_counts(normalize=True).sort_index().items()},
        "revisions": {
            "share_updated_after_created": float((lag > timedelta(0)).mean()),
            "share_updated_more_than_1min_after_created": float((lag > timedelta(minutes=1)).mean()),
            "share_revised_after_own_decision_close": float(rev_after.mean()),
            "share_revised_more_than_1day_after_own_decision_close": float(rev_after_1d.mean()),
            "update_lag_minutes_quantiles": _q(lag.dt.total_seconds() / 60.0),
        },
        "symbols_per_article": {"median": float(n_syms.median()), "share_more_than_10": float((n_syms > 10).mean())},
        "live_latency_minutes": {"n_live": int(len(live)), **_q(latency_min)},
        "vendor_archive_backfill": "not measurable from backfilled rows (no real first-seen time); measured going "
                                   "forward from live ingests (live_latency_minutes)",
        "month_gaps": month_gaps(articles, sessions, universe),
        "labels_or_prices_used": False,
    }
