"""Persona #3: The Analyst — an XGBoost next-session direction classifier.

Long-only (v1): long when the committed champion model's P(up) for the next
open-to-open session is > 0.5, flat otherwise (ties flat). One pooled model
across the 5-ticker universe; features are log returns / log ratios only
(see analyst/features.py). The model is trained offline and becomes the
committed champion in `worker/models/analyst/champion/` only through the
alphagate promotion gate (`python -m wolfpack_worker.analyst.retrain`);
`load_champion` refuses a champion without a PROMOTE record. This strategy
never trains. The feature builder is the one named by the champion
manifest's `feature_spec_version`, so inference always uses the exact
feature function the model was trained on.

MODEL-RISK LIMITATION: no real trading edge is claimed. Daily-bar direction
on large, liquid US tickers is close to a coin flip for any model built from
public price data, and this one is no exception — its holdout accuracy and
log loss are carried in every signal payload next to the naive baselines
precisely so no rationale can present a prediction as more than it is.

Lookahead guards, beyond StrategyContext's own:
- the champion's `trained_through` (last bar any training label read) must
  be strictly before `ctx.as_of`, else LookaheadError;
- features for bar t use only bars <= t (tested for exact truncation
  invariance);
- a ticker whose latest bar is not the context's latest session, or whose
  latest feature row has any NaN (e.g. SPY's bar missing that day), is left
  out (= hold), never traded on stale or partial inputs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from wolfpack_worker.analyst.metrics import THRESHOLD
from wolfpack_worker.analyst.model_io import (
    DEFAULT_CHAMPION_DIR,
    ModelIntegrityError,
    load_champion,
)
from wolfpack_worker.strategies.base import LookaheadError, StrategyContext, TargetPosition

logger = logging.getLogger(__name__)

LOOKBACK_BARS = 80
TOP_CONTRIBUTIONS = 4

LIMITATIONS = (
    "No trading edge is claimed. This is a gradient-boosted classifier on "
    "public daily OHLCV data for five large, liquid US tickers; next-session "
    "direction on such data is close to a coin flip. Compare holdout_accuracy "
    "and holdout_logloss with the naive baselines in this payload: the model "
    "only 'beats baseline' if its log loss is lower. Paper trading only; a "
    "process demonstration, not investment advice."
)

__all__ = ["Analyst", "LIMITATIONS", "LOOKBACK_BARS", "THRESHOLD", "exposure_from_p_up"]


def exposure_from_p_up(p_up: float) -> float:
    """Long (1.0) iff p_up > THRESHOLD; exact ties go flat."""
    return 1.0 if p_up > THRESHOLD else 0.0


def _iso(ts) -> str:
    return ts.isoformat() if hasattr(ts, "isoformat") else str(ts)


@dataclass
class Analyst:
    slug: str = "the-analyst"
    version: str = "analyst/v1"
    lookback_bars: int = field(default=LOOKBACK_BARS)
    model_dir: Path = field(default=DEFAULT_CHAMPION_DIR)

    def evaluate(self, ctx: StrategyContext) -> list[TargetPosition]:
        champ = load_champion(self.model_dir)
        manifest = champ.manifest
        spec = champ.spec
        feature_names = spec.names

        trained_through = pd.Timestamp(manifest["trained_through"])
        if trained_through.tzinfo is None:
            raise ModelIntegrityError("manifest trained_through must be timezone-aware")
        if trained_through >= pd.Timestamp(ctx.as_of):
            raise LookaheadError(
                f"The Analyst's model was trained on labels through "
                f"{trained_through.isoformat()}, which is not strictly before "
                f"as_of {ctx.as_of.isoformat()} — using it would leak future "
                "information into this decision."
            )

        window = {
            t: df.iloc[-self.lookback_bars :]
            for t, df in ctx.bars.items()
            if df is not None and len(df) > 0
        }
        if not window:
            return []
        feats = spec.build_fn(window)
        latest_session = max(df.index[-1] for df in window.values())

        tickers: list[str] = []
        rows: list[np.ndarray] = []
        for ticker in ctx.universe:
            df = window.get(ticker)
            f = feats.get(ticker)
            if df is None or f is None or len(df) < spec.warmup_bars:
                continue  # insufficient history -> hold
            if df.index[-1] != latest_session:
                logger.info(
                    "analyst.evaluate: %s latest bar %s is not the latest session %s — holding.",
                    ticker, df.index[-1], latest_session,
                    extra={"reason": "stale_bar", "ticker": ticker},
                )
                continue
            x = f.iloc[-1]
            if x.isna().any():
                logger.info(
                    "analyst.evaluate: NaN feature(s) %s for %s — holding.",
                    list(x.index[x.isna()]), ticker,
                    extra={"reason": "nan_feature", "ticker": ticker},
                )
                continue
            tickers.append(ticker)
            rows.append(x.to_numpy(dtype=float))

        if not rows:
            return []

        dmat = xgb.DMatrix(np.vstack(rows), feature_names=list(feature_names))
        p_up_all = champ.booster.predict(dmat)
        # Per-prediction feature contributions (TreeSHAP, built into xgboost),
        # in log-odds units; last column is the bias (base log-odds).
        contribs_all = champ.booster.predict(dmat, pred_contribs=True)

        metrics = manifest["test_metrics"]
        targets: list[TargetPosition] = []
        for i, ticker in enumerate(tickers):
            df = window[ticker]
            bar_ts = df.index[-1]
            p_up = float(p_up_all[i])
            contribs = contribs_all[i]
            feat_values = {name: float(rows[i][j]) for j, name in enumerate(feature_names)}
            order = sorted(range(len(feature_names)), key=lambda j: (-abs(float(contribs[j])), j))
            top = [
                {
                    "feature": feature_names[j],
                    "value": feat_values[feature_names[j]],
                    "contribution": float(contribs[j]),
                }
                for j in order[:TOP_CONTRIBUTIONS]
            ]
            exposure = exposure_from_p_up(p_up)
            payload = {
                "strategy": "xgb_direction_classifier",
                "version": self.version,
                "model_version": manifest["model_version"],
                "mlflow_run_id": manifest["mlflow_run_id"],
                "trained_through": manifest["trained_through"],
                "p_up": p_up,
                "threshold": THRESHOLD,
                "regime": "model_bullish" if exposure > 0 else "model_bearish",
                "features": feat_values,
                "top_contributions": top,
                "base_log_odds": float(contribs[-1]),
                "holdout_accuracy": metrics["accuracy"],
                "baseline_accuracy": metrics["baseline_accuracy"],
                "holdout_logloss": metrics["logloss"],
                "baseline_logloss": metrics["baseline_logloss"],
                "beats_baseline_logloss": bool(metrics["beats_baseline_logloss"]),
                "last_close": float(df["close"].iloc[-1]),
                "bar_ts": _iso(bar_ts),
                "as_of": ctx.as_of.isoformat(),
                "bars_used": int(len(df)),
                "limitations": LIMITATIONS,
            }
            targets.append(
                TargetPosition(
                    ticker=ticker,
                    target_exposure=exposure,
                    signal_ts=bar_ts,
                    payload=payload,
                )
            )
        return targets
