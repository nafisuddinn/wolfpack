"""Persona #4: The Scout — headline-sentiment classifier, gate-or-fallback.

Design: docs/design/scout-design.md (CONFIRMED 2026-10-04).

Each daily run, after the close of session t (`ctx.as_of` = close_t), The
Scout scores the Alpaca/Benzinga headlines in its point-in-time snapshot
(`ctx.news`: created_at in the last 25 windows W = (close_{s-1}, close_s],
created_at <= close_t, first seen by WolfPack no later than this run's news
cutoff) with VADER, builds feature spec scout_v1 (scout/features.py, the
same function training used), and then:

* mode "model": only if a champion exists in worker/models/scout/champion/,
  which only the alphagate gate can write (and `load_champion` re-checks
  its PROMOTE record): long if p_up > 0.5, else flat.
* mode "rule_fallback" (no champion: the trial was rejected, or has not
  run): the untuned VADER neutral band on today's mean headline tone,
  s_mean_1 > +0.05 long, < -0.05 flat, otherwise (including no headlines)
  no opinion = hold. The band is VADER's documented neutral threshold, not
  fitted to anything.

Stale or failed news (refresh failed, history not backfilled, calendar
mismatch): return [] and log the reason. A broken champion (present but
failing integrity checks, or a PROMOTE record with no champion files) is a
loud error, never a silent fallback to the rule.

Public payload rule (Decision Log 2026-10-04): `signal_payload` is public
(anon-readable, shown in the feed and used for rationales). It carries
scores, counts, source names and article ids ONLY. Headline text and URLs
stay in the private `news_articles` table and never appear in the payload,
a log line, or rationale text (Benzinga redistribution terms are unclear).

MODEL-RISK LIMITATION: no trading edge is claimed. Headlines for five
large, heavily covered US tickers are mostly priced in during the session
they appear, ETF headlines are sparse, and VADER misreads finance language.
The rule fallback is no evidence of edge either. Every payload carries the
gate result and the base-rate comparison so no rationale can present a
call as more than it is.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from wolfpack_worker.analyst.gate_log import PROMOTE, read_gate_log
from wolfpack_worker.analyst.metrics import THRESHOLD
from wolfpack_worker.analyst.model_io import MANIFEST_FILENAME, Champion, ModelIntegrityError, load_champion
from wolfpack_worker.analyst.paths import PersonaPaths
from wolfpack_worker.scout.features import WARMUP_SESSIONS, get_scout_feature_spec, scored
from wolfpack_worker.scout.paths import SCOUT_PATHS
from wolfpack_worker.scout.sentiment import NEUTRAL_BAND
from wolfpack_worker.scout.windows import assign_windows
from wolfpack_worker.strategies.base import LookaheadError, StrategyContext, TargetPosition

logger = logging.getLogger(__name__)

MARKET_TZ = ZoneInfo("America/New_York")
TOP_ARTICLES = 3
DEFAULT_SPEC = "scout_v1"
STOCKTWITS_STATUS = "deferred"  # ingestion client (C3) deferred, not a model input (Decision Log 2026-10-04)

LIMITATIONS = (
    "No trading edge is claimed. The Scout scores public news headlines with VADER, a general-purpose "
    "sentiment lexicon that misreads finance language, for five large, heavily covered US tickers whose news is "
    "largely priced in during the session it appears. Its model trades only if it beat the base rate on the "
    "promotion gate; otherwise it follows an untuned neutral-band rule (mode rule_fallback), which is no "
    "evidence of edge either. Compare the holdout and base-rate numbers in this payload. Paper trading only; a "
    "process demonstration, not investment advice."
)

__all__ = ["LIMITATIONS", "Scout", "exposure_from_p_up", "rule_exposure"]


def exposure_from_p_up(p_up: float) -> float:
    return 1.0 if p_up > THRESHOLD else 0.0


def rule_exposure(s_mean_1: float, band: float = NEUTRAL_BAND) -> Optional[float]:
    """1.0 above +band, 0.0 below -band, None (hold) otherwise."""
    if s_mean_1 > band:
        return 1.0
    if s_mean_1 < -band:
        return 0.0
    return None


def _f(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _latest_trial(records) -> Optional[Mapping[str, Any]]:
    trials = [r for r in records if (r.get("context") or {}).get("kind") == "trial"]
    return trials[-1] if trials else None


@dataclass
class Scout:
    slug: str = "the-scout"
    version: str = "scout/v1"
    lookback_bars: int = 5
    requires_news: bool = True
    news_lookback_sessions: int = WARMUP_SESSIONS
    paths: PersonaPaths = field(default=SCOUT_PATHS)

    # -- gate state --------------------------------------------------------------

    def _champion(self, records) -> Optional[Champion]:
        if (self.paths.champion_dir / MANIFEST_FILENAME).exists():
            return load_champion(paths=self.paths)  # strict: integrity + PROMOTE record
        if any(r.get("decision") == PROMOTE for r in records):
            raise ModelIntegrityError(
                "The Scout's gate log has a PROMOTE record but the champion files are missing: refusing to fall "
                "back to the rule silently. Restore worker/models/scout/champion/ from git."
            )
        return None

    def _gate_fields(self, champ: Optional[Champion], records) -> dict[str, Any]:
        if champ is not None:
            m = champ.manifest
            tm = m["test_metrics"]
            return {
                "mode": "model",
                "model_rejected": False,
                "gate_status": "promoted",
                "gate_record_id": m["gate_record_ids"][-1],
                "model_version": m["model_version"],
                "trained_through": m["trained_through"],
                "holdout_accuracy": _f(tm.get("accuracy")),
                "baseline_accuracy": _f(tm.get("baseline_accuracy")),
                "holdout_logloss": _f(tm.get("logloss")),
                "baseline_logloss": _f(tm.get("baseline_logloss")),
                "beats_baseline_logloss": bool(tm.get("beats_baseline_logloss")),
            }
        last = _latest_trial(records)
        if last is None:
            return {"mode": "rule_fallback", "model_rejected": False, "gate_status": "not_yet_gated",
                    "gate_record_id": None, "model_version": None, "trained_through": None,
                    "holdout_accuracy": None, "baseline_accuracy": None, "holdout_logloss": None,
                    "baseline_logloss": None, "beats_baseline_logloss": None}
        ch = last.get("challenger_score") or {}
        details = ch.get("details") or {}
        ll, base = _f(ch.get("value")), _f((last.get("champion_score") or {}).get("value"))
        if base is None:
            base = _f(details.get("baseline_logloss"))
        return {
            "mode": "rule_fallback",
            "model_rejected": last.get("decision") != PROMOTE,
            "gate_status": f"rejected ({last.get('reason_code')})",
            "gate_record_id": last.get("record_id"),
            "model_version": None,
            "trained_through": None,
            "holdout_accuracy": _f(details.get("accuracy")),
            "baseline_accuracy": None,
            "holdout_logloss": ll,
            "baseline_logloss": base,
            "beats_baseline_logloss": (ll < base) if (ll is not None and base is not None) else None,
        }

    # -- evaluate ------------------------------------------------------------------

    def evaluate(self, ctx: StrategyContext) -> list[TargetPosition]:
        news = ctx.news
        if news is None or news.status != "ok":
            why = "no news snapshot" if news is None else f"news {news.status}: {news.reason}"
            logger.warning("scout.evaluate: %s; skipping this session (no trades).", why,
                           extra={"reason": "news_unavailable"})
            return []
        if len(news.sessions) < self.news_lookback_sessions + 1 or news.sessions[-1].close != ctx.as_of:
            logger.warning("scout.evaluate: news calendar does not cover the lookback ending at as_of; skipping.",
                           extra={"reason": "news_calendar"})
            return []

        records = read_gate_log(self.paths.gate_log_path)
        champ = self._champion(records)
        gate_fields = self._gate_fields(champ, records)
        if champ is not None:
            trained_through = pd.Timestamp(champ.manifest["trained_through"])
            if trained_through.tzinfo is None:
                raise ModelIntegrityError("manifest trained_through must be timezone-aware")
            if trained_through >= pd.Timestamp(ctx.as_of):
                raise LookaheadError(
                    f"The Scout's model was trained on labels through {trained_through.isoformat()}, not strictly "
                    f"before as_of {ctx.as_of.isoformat()}."
                )
        spec = champ.spec if champ is not None else get_scout_feature_spec(DEFAULT_SPEC)
        names = list(spec.names)

        sessions = news.sessions
        arts = scored(news.articles)  # VADER once; the same scores the feature builder uses
        feats = spec.build_fn(arts, sessions, ctx.universe)
        t_pos = len(sessions) - 1
        t_date = sessions[-1].date
        w = assign_windows(pd.DatetimeIndex(arts["created_at"]), sessions) if len(arts) else np.array([], int)
        today = arts.loc[w == t_pos]

        rows: list[tuple[str, pd.DataFrame, dict[str, float]]] = []
        for ticker in ctx.universe:
            df = ctx.bars.get(ticker)
            if df is None or len(df) == 0:
                continue
            if pd.Timestamp(df.index[-1]).tz_convert(MARKET_TZ).date() != t_date:
                logger.info("scout.evaluate: %s has no bar for session %s; holding.", ticker, t_date,
                            extra={"reason": "stale_bar", "ticker": ticker})
                continue
            x = feats[ticker].loc[t_date]
            if x.isna().any():
                logger.info("scout.evaluate: incomplete news lookback for %s; holding.", ticker,
                            extra={"reason": "nan_feature", "ticker": ticker})
                continue
            rows.append((ticker, df, {n: float(x[n]) for n in names}))
        if not rows:
            return []

        p_up_all = None
        if champ is not None:
            import xgboost as xgb

            X = np.array([[f[n] for n in names] for _, _, f in rows], dtype=float)
            p_up_all = champ.booster.predict(xgb.DMatrix(X, feature_names=names))

        targets: list[TargetPosition] = []
        for i, (ticker, df, f) in enumerate(rows):
            mine = today.loc[today["symbols"].map(lambda s, t=ticker: t in s)]
            order = sorted(range(len(mine)), key=lambda j: (-abs(float(mine["score"].iloc[j])),
                                                              int(mine["id"].iloc[j])))
            top = [{"id": int(mine["id"].iloc[j]), "score": float(mine["score"].iloc[j]),
                    "source": str(mine["source"].iloc[j])} for j in order[:TOP_ARTICLES]]
            if p_up_all is not None:
                p_up = float(p_up_all[i])
                exposure: Optional[float] = exposure_from_p_up(p_up)
                rule_value = None
            else:
                p_up = None
                rule_value = f["s_mean_1"]
                exposure = rule_exposure(rule_value)
                if exposure is None:
                    logger.info("scout.evaluate: %s tone %.3f inside the neutral band (or no headlines); holding.",
                                ticker, rule_value, extra={"reason": "neutral_band", "ticker": ticker})
                    continue
            bar_ts = df.index[-1]
            payload = {
                "strategy": "news_sentiment_gate_or_fallback",
                "version": self.version,
                **gate_fields,
                "p_up": p_up,
                "threshold": THRESHOLD,
                "rule_value": rule_value,
                "rule_band": NEUTRAL_BAND,
                "feature_spec_version": spec.version,
                "features": f,
                "news_cutoff": news.cutoff.isoformat() if news.cutoff else None,
                "n_articles": int(len(mine)),
                "article_ids": [int(a) for a in mine["id"].tolist()],
                "top_articles": top,
                "late_arrivals": int(news.late_arrivals),
                "excluded_after_cutoff": int(news.excluded_after_cutoff),
                "stocktwits": STOCKTWITS_STATUS,
                "last_close": float(df["close"].iloc[-1]),
                "bar_ts": bar_ts.isoformat(),
                "as_of": ctx.as_of.isoformat(),
                "limitations": LIMITATIONS,
            }
            targets.append(TargetPosition(ticker=ticker, target_exposure=float(exposure), signal_ts=bar_ts,
                                          payload=payload))
        return targets
