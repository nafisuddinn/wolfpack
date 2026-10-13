"""VADER headline scoring for The Scout (pinned vaderSentiment==3.3.2, MIT).

Score = VADER's `compound` in [-1, 1] for the headline text only (no body).
Deterministic, no network, no model download, no GPU, so it runs in the
daily cron. Scores are computed, never stored (design section 5).

Chosen over FinBERT (torch ~1 GB in a daily cron) and LLM scoring (a model
that has read later coverage of an event is hidden lookahead, and it would
break the no-LLM mechanical path). Known limitation, stated plainly: VADER is
a general-purpose social-media lexicon and misreads finance language
("short", "beat", "cut", "bull"/"bear" in tickers' context), so scores are
noisy. That is one reason little or no signal is expected.

NEUTRAL_BAND is VADER's documented neutral threshold (compound within
+/-0.05 is "neutral"), used as-is by the rule fallback; it is not tuned.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import numpy as np

VADER_PACKAGE_VERSION = "3.3.2"
NEUTRAL_BAND = 0.05


@lru_cache(maxsize=1)
def _analyzer():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    return SentimentIntensityAnalyzer()


def vader_compound(text: str) -> float:
    if not text or not str(text).strip():
        return 0.0
    return float(_analyzer().polarity_scores(str(text))["compound"])


def score_headlines(headlines: Iterable[str]) -> np.ndarray:
    return np.array([vader_compound(h) for h in headlines], dtype=float)
