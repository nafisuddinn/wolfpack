"""Exploration for The Analyst: walk-forward metrics on data strictly BEFORE
the gate holdout. Needs no registration and is not a trial, because it never
sees a session the promotion gate will score.

The boundary is the start of the holdout the next gate call would use right
now (`gating.shared_cutoff` over the explored recipe's and the champion's
datasets). Every row explore touches has `ts < boundary` AND
`label_end_ts < boundary` (its label reads no holdout bar); a final guard
re-checks every fold and refuses (HoldoutContaminationError) rather than
report anything that crossed it. Results go to the MLflow experiment
`the-analyst-dev`, never `the-analyst`.

MODEL-RISK LIMITATION: exploration numbers are research notes, not
evidence of edge; only the gate's paired test on unseen data counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd

from wolfpack_worker.analyst.dataset import walk_forward_folds
from wolfpack_worker.analyst.features import get_feature_spec
from wolfpack_worker.analyst.gating import matured, shared_cutoff, truncate_to
from wolfpack_worker.analyst.metrics import classification_metrics
from wolfpack_worker.analyst.recipe import Recipe
from wolfpack_worker.analyst.train import _dmatrix, _walk_forward_summary, fit_model, prepare_dataset

DEV_EXPERIMENT_NAME = "the-analyst-dev"


class HoldoutContaminationError(RuntimeError):
    """Exploration would touch (or did touch) the gate holdout."""


@dataclass
class ExploreResult:
    recipe: Recipe
    boundary: pd.Timestamp
    end: pd.Timestamp
    folds: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, float] = field(default_factory=dict)


def explore_boundary(
    bars: Mapping[str, pd.DataFrame], recipe: Recipe, champion_recipe: Recipe, as_of
) -> pd.Timestamp:
    """Start of the holdout a gate call would use now for this recipe vs the champion."""
    bars = truncate_to(bars, as_of)
    pairs = [
        (recipe.feature_spec_version, matured(prepare_dataset(bars, recipe), as_of)),
        (champion_recipe.feature_spec_version, matured(prepare_dataset(bars, champion_recipe), as_of)),
    ]
    return shared_cutoff(pairs)


def run_explore(
    bars: Mapping[str, pd.DataFrame],
    recipe: Recipe,
    *,
    boundary: pd.Timestamp,
    until: pd.Timestamp | None = None,
) -> ExploreResult:
    boundary = pd.Timestamp(boundary)
    end = boundary if until is None else pd.Timestamp(until)
    if end > boundary:
        raise HoldoutContaminationError(
            f"explore end {end} is after the gate holdout start {boundary}; exploration must stay "
            "strictly before the holdout"
        )
    names = get_feature_spec(recipe.feature_spec_version).names
    ds = prepare_dataset(bars, recipe)
    ds = ds.loc[(ds["ts"] < end) & (ds["label_end_ts"] < end)].reset_index(drop=True)

    folds: list[dict[str, Any]] = []
    for fold in walk_forward_folds(ds):
        for part in (fold.train, fold.test):
            if len(part) and (part["ts"].max() >= boundary or part["label_end_ts"].max() >= boundary):
                raise HoldoutContaminationError(
                    f"fold {fold.name} uses rows at/after the gate holdout start {boundary}; refusing"
                )
        up = float(fold.train["y"].mean())
        booster = fit_model(fold.train, recipe)
        p = booster.predict(_dmatrix(fold.test, names, with_label=False))
        m = classification_metrics(fold.test["y"].to_numpy(), p, up)
        folds.append(
            {
                "name": fold.name,
                "train_start": fold.train["ts"].min().isoformat(),
                "trained_through": fold.train["label_end_ts"].max().isoformat(),
                "test_start": fold.test_start.isoformat(),
                "test_end": fold.test["ts"].max().isoformat(),
                "max_label_end": fold.test["label_end_ts"].max().isoformat(),
                "n_train": int(len(fold.train)),
                **m,
            }
        )
    return ExploreResult(recipe=recipe, boundary=boundary, end=end, folds=folds,
                         summary=_walk_forward_summary(folds) if folds else {})


def log_explore_to_mlflow(result: ExploreResult, *, tracking_uri: str, artifact_root, extra: Mapping[str, Any]) -> str:
    """One parent run + one nested run per fold in `the-analyst-dev`."""
    from pathlib import Path

    import mlflow

    artifact_root = Path(artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    exp = client.get_experiment_by_name(DEV_EXPERIMENT_NAME)
    exp_id = exp.experiment_id if exp else client.create_experiment(
        DEV_EXPERIMENT_NAME, artifact_location=artifact_root.as_uri()
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_id=exp_id)
    with mlflow.start_run(experiment_id=exp_id, run_name=f"explore-{result.recipe.recipe_id}") as run:
        mlflow.set_tags({
            "persona": "the-analyst",
            "run_kind": "explore",
            "recipe_id": result.recipe.recipe_id,
            "model_risk": "Exploration only, before the gate holdout. Not evidence of edge.",
        })
        mlflow.log_params({
            "recipe_json": result.recipe.canonical_json(),
            "gate_holdout_start": result.boundary.isoformat(),
            "explore_end": result.end.isoformat(),
            **{k: str(v) for k, v in extra.items()},
        })
        if result.summary:
            mlflow.log_metrics({k: float(v) for k, v in result.summary.items() if np.isfinite(v)})
        for f in result.folds:
            with mlflow.start_run(experiment_id=exp_id, run_name=f"wf_{f['name']}", nested=True):
                mlflow.log_params({k: str(f[k]) for k in ("name", "train_start", "trained_through",
                                                         "test_start", "test_end", "n_train")})
                mlflow.log_metrics({k: float(f[k]) for k in ("accuracy", "auc", "logloss", "brier",
                                                            "baseline_accuracy", "baseline_logloss", "n")
                                    if np.isfinite(float(f[k]))})
        return run.info.run_id
