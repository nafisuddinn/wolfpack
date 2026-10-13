"""The Scout's feature spec `scout_v1`: headline sentiment only, no price features.

One function for training AND inference. Inputs are the market calendar
(sessions), the stored articles (created_at, symbols, headline; a `score`
column is used if present, else VADER is applied to the headline), and the
universe. For ticker k and session t, with W_t = (close_{t-1}, close_t] on
created_at (scout/windows.py), n_t = number of k's headlines in W_t and
S_t = sum of their VADER compound scores:

 1. s_mean_1      S_t / n_t (0 if n_t = 0)
 2. has_news_1    1 if n_t > 0 else 0
 3. n_surprise    log1p(n_t) - mean(log1p(n_j), j = t-20..t-1)
 4. s_mean_5      count-weighted mean over W_{t-4..t}: sum S / sum n (0 if no headlines)
 5. s_change      s_mean_5 - count-weighted mean over W_{t-24..t-5} (0 if none there)
 6. mkt_s_mean_1  mean score over the DEDUPLICATED headlines of all universe
                  tickers in W_t (an article tagged AAPL and SPY counts once; 0 if none)

Stationarity (CLAUDE.md: changes, not raw levels): discussion volume only
ever enters as a change (n_surprise; a raw count is never a feature, and a
steady doubling of coverage leaves every feature unchanged, tested). The
sentiment terms are per-window flows bounded in [-1, 1] (like a return, not
like a price level), and s_change is their change against the previous
month. The design (docs/design/scout-design.md, CONFIRMED) fixes this set.

Lookahead discipline: a row for session t only reads headlines with
created_at <= close_t (W_t's right edge), and the label (dataset.py,
y_t = 1[ln(O_{t+2}/O_{t+1}) > 0]) starts at the next open. Tested: exact
truncation invariance, an after-close canary, and bit-identical values
between a full-history build and the daily 26-session lookback (all
rolling sums are fixed-length, term-by-term, in a fixed order; per-session
sums accumulate articles sorted by (created_at, id)).

Warmup: 25 defined windows (t-24..t), i.e. 26 calendar sessions including
the one whose close bounds W_{t-24}. Earlier rows are NaN (dropped in
training; the strategy needs the full lookback).

MODEL-RISK LIMITATION: little or no predictive signal is expected from
these features (in-session news is priced in; ETF headlines are sparse;
VADER misreads finance language). See MODEL_CARD_SCOUT.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from wolfpack_worker.broker import Session
from wolfpack_worker.scout.sentiment import VADER_PACKAGE_VERSION, score_headlines
from wolfpack_worker.scout.windows import assign_windows, session_closes

SCOUT_FEATURE_NAMES: tuple[str, ...] = (
    "s_mean_1",
    "has_news_1",
    "n_surprise",
    "s_mean_5",
    "s_change",
    "mkt_s_mean_1",
)
N_SURPRISE_LOOKBACK = 20
SHORT_WINDOWS = 5
LONG_WINDOWS = 20  # W_{t-24..t-5}
WARMUP_SESSIONS = SHORT_WINDOWS + LONG_WINDOWS  # 25 windows: t-24..t


def lookback_sessions_needed() -> int:
    """Feature windows the strategy needs (calendar sessions = this + 1)."""
    return WARMUP_SESSIONS


def _window_sum(x: np.ndarray, w: int) -> np.ndarray:
    """out[i] = x[i-w+1] + ... + x[i], accumulated oldest-first, term by term
    (never a BLAS/pairwise reduction whose order could depend on array
    length). NaN where the window is incomplete or contains NaN."""
    n = x.shape[0]
    out = np.full(n, np.nan)
    if n < w:
        return out
    acc = np.zeros(n - w + 1)
    for j in range(w):
        acc = acc + x[j : n - w + 1 + j]
    out[w - 1 :] = acc
    return out


def _shift(x: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=float)
    if k < x.shape[0]:
        out[k:] = x[: x.shape[0] - k]
    return out


def _ratio_or_zero(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.zeros_like(num, dtype=float)
    pos = den > 0
    out[pos] = num[pos] / den[pos]
    out[np.isnan(num) | np.isnan(den)] = np.nan
    return out


def scored(articles: pd.DataFrame) -> pd.DataFrame:
    """Articles sorted by (created_at, id) with a float `score` column
    (VADER on the headline unless already present)."""
    df = articles.sort_values(["created_at", "id"], kind="mergesort").reset_index(drop=True)
    if "score" not in df.columns:
        df = df.assign(score=score_headlines(df["headline"].tolist()) if len(df) else np.array([], dtype=float))
    return df


def session_counts(
    articles: pd.DataFrame, sessions: Sequence[Session], universe: Sequence[str]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray, pd.DataFrame]:
    """Per-session (n, S) per ticker and for the deduplicated market set.
    Index 0 (no defined window) is NaN. Also returns the scored articles
    with their `window` column."""
    m = len(sessions)
    session_closes(sessions)  # validates aware + sorted
    df = scored(articles)
    w = assign_windows(pd.DatetimeIndex(df["created_at"]), sessions) if len(df) else np.array([], dtype=np.int64)
    df = df.assign(window=w)
    valid = w >= 1
    score = df["score"].to_numpy(dtype=float) if len(df) else np.array([], dtype=float)
    syms = df["symbols"].tolist() if len(df) else []
    uni = set(universe)

    def agg(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = np.bincount(w[mask], minlength=m).astype(float)[:m]
        s = np.bincount(w[mask], weights=score[mask], minlength=m).astype(float)[:m]
        n[0] = np.nan
        s[0] = np.nan
        return n, s

    n_by, s_by = {}, {}
    for ticker in universe:
        mask = valid & np.array([ticker in s for s in syms], dtype=bool)
        n_by[ticker], s_by[ticker] = agg(mask)
    mkt_mask = valid & np.array([bool(uni.intersection(s)) for s in syms], dtype=bool)
    mkt_n, mkt_s = agg(mkt_mask)
    return n_by, s_by, mkt_n, mkt_s, df


def build_scout_features(
    articles: pd.DataFrame, sessions: Sequence[Session], universe: Sequence[str]
) -> dict[str, pd.DataFrame]:
    """{ticker: frame indexed by session date, SCOUT_FEATURE_NAMES columns}."""
    n_by, s_by, mkt_n, mkt_s, _ = session_counts(articles, sessions, universe)
    dates = [s.date for s in sessions]
    m = len(sessions)
    mkt_mean = _ratio_or_zero(mkt_s, mkt_n)
    out: dict[str, pd.DataFrame] = {}
    for ticker in universe:
        n, S = n_by[ticker], s_by[ticker]
        s_mean_1 = _ratio_or_zero(S, n)
        has_news = np.where(np.isnan(n), np.nan, (n > 0).astype(float))
        logn = np.log1p(n)
        base = _shift(_window_sum(logn, N_SURPRISE_LOOKBACK), 1) / N_SURPRISE_LOOKBACK
        n_surprise = logn - base
        s_mean_5 = _ratio_or_zero(_window_sum(S, SHORT_WINDOWS), _window_sum(n, SHORT_WINDOWS))
        old_mean = _ratio_or_zero(_shift(_window_sum(S, LONG_WINDOWS), SHORT_WINDOWS),
                                  _shift(_window_sum(n, LONG_WINDOWS), SHORT_WINDOWS))
        s_change = s_mean_5 - old_mean
        frame = pd.DataFrame(
            {
                "s_mean_1": s_mean_1,
                "has_news_1": has_news,
                "n_surprise": n_surprise,
                "s_mean_5": s_mean_5,
                "s_change": s_change,
                "mkt_s_mean_1": mkt_mean,
            },
            index=pd.Index(dates, name="session_date"),
            columns=list(SCOUT_FEATURE_NAMES),
        )
        frame.iloc[: min(WARMUP_SESSIONS, m)] = np.nan
        out[ticker] = frame
    return out


# ---------------------------------------------------------------------------
# Feature-spec registry (APPEND-ONLY, same rules as analyst/features.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoutFeatureSpec:
    version: str
    names: tuple[str, ...]
    build_fn: Callable[[pd.DataFrame, Sequence[Session], Sequence[str]], dict[str, pd.DataFrame]]
    warmup_sessions: int
    scorer: str


SCOUT_FEATURE_SPECS: Mapping[str, ScoutFeatureSpec] = MappingProxyType(
    {
        "scout_v1": ScoutFeatureSpec(
            version="scout_v1",
            names=SCOUT_FEATURE_NAMES,
            build_fn=build_scout_features,
            warmup_sessions=WARMUP_SESSIONS,
            scorer=f"vaderSentiment=={VADER_PACKAGE_VERSION} compound, headline only",
        ),
    }
)


def get_scout_feature_spec(version: str) -> ScoutFeatureSpec:
    try:
        return SCOUT_FEATURE_SPECS[version]
    except KeyError:
        raise KeyError(f"unknown Scout feature_spec_version {version!r}; known: {sorted(SCOUT_FEATURE_SPECS)}") from None
