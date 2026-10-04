"""Train The Scout's model + the reported-not-gated diagnostics.

Pure functions of a prepared dataset (scout.dataset.build_scout_dataset):
no network, no MLflow (except `log_to_mlflow`), deterministic.

Split (shared with The Analyst, analyst.dataset): test = the trailing 252
labeled sessions; train = everything before, minus a 2-session embargo, and
only rows whose label ended before the test window (so `trained_through` <
test start). Strictly chronological, never shuffled. The returned model is
the one evaluated on the holdout; it is never refit on it.

Model: pooled XGBoost across the 5 tickers with the untuned Analyst-v1
settings from the recipe (no tuning, no early stopping, nothing selected on
the holdout), single-threaded for bit-reproducibility.

Reported, NOT gated (design section 6): yearly walk-forward folds (a fold
may overlap the gate holdout; reporting only), holdout accuracy split by
has_news_1, the untuned rule baseline's accuracy and pre-cost return, the
2020 Mar-Apr stress window, and a split by "this row's window holds a
headline the vendor revised after the decision" (the residual revision
leak). Returns are ISOLATED: equal-weight long/flat per ticker on the
label's open-to-open log return, pre-cost, plus a net line at an
illustrative flat cost. The project's full pipeline (vol-targeted sizing,
its transaction-cost assumption, DSR) does not exist yet (Week 3 backlog),
so no full-pipeline number can be reported and none is implied.

MODEL-RISK LIMITATION: little or no signal is expected. Only the gate's
paired test on the holdout decides whether the model trades at all.
"""

from __future__ import annotations

import json
import logging
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import xgboost as xgb

from wolfpack_worker.analyst.dataset import (
    EMBARGO_SESSIONS,
    TEST_SESSIONS,
    split_at,
    trailing_test_start,
    walk_forward_folds,
)
from wolfpack_worker.analyst.metrics import THRESHOLD, classification_metrics, long_flat_backtest
from wolfpack_worker.analyst.recipe import LABELS, Recipe
from wolfpack_worker.analyst.train import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_TRACKING_URI,
    ILLUSTRATIVE_COST_BPS_PER_SIDE,
    TRAIN_NTHREAD,
    TrainingResult,
    _feature_matrix_sha256,
    _git_info,
    _importance,
    _roundtrip,
    _walk_forward_summary,
)
from wolfpack_worker.scout.features import get_scout_feature_spec
from wolfpack_worker.scout.sentiment import NEUTRAL_BAND

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "the-scout"
STRESS_WINDOW = ("2020-03-01", "2020-04-30")  # COVID crash and rebound (design section 6)


def feature_names(recipe: Recipe) -> tuple[str, ...]:
    return tuple(get_scout_feature_spec(recipe.feature_spec_version).names)


def _dmatrix(df: pd.DataFrame, names: Sequence[str], with_label: bool = True) -> xgb.DMatrix:
    x = df[list(names)].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float) if with_label else None
    return xgb.DMatrix(x, label=y, feature_names=list(names))


def fit_model(train: pd.DataFrame, recipe: Recipe) -> xgb.Booster:
    names = feature_names(recipe)
    params = {**recipe.xgb_params, "nthread": TRAIN_NTHREAD}
    return xgb.train(params, _dmatrix(train, names), num_boost_round=recipe.num_boost_round)


def scout_model_version(model_bytes: bytes, today: date | None = None) -> str:
    import hashlib

    today = today or datetime.now(timezone.utc).date()
    return f"scout-{today:%Y%m%d}-{hashlib.sha256(model_bytes).hexdigest()[:8]}"


# ---------------------------------------------------------------------------
# Rule fallback (untuned VADER neutral band) and diagnostics
# ---------------------------------------------------------------------------


def rule_signal(s_mean_1: np.ndarray, band: float = NEUTRAL_BAND) -> np.ndarray:
    """1.0 long (> +band), 0.0 flat (< -band), NaN hold (inside the band or
    no news, since s_mean_1 = 0 then). The strategy's rule_fallback."""
    s = np.asarray(s_mean_1, dtype=float)
    out = np.full(s.shape, np.nan)
    out[s > band] = 1.0
    out[s < -band] = 0.0
    return out


def rule_backtest(rows: pd.DataFrame, cost_bps_per_side: float = ILLUSTRATIVE_COST_BPS_PER_SIDE) -> dict[str, Any]:
    """The rule on `rows` (ts, ticker, s_mean_1, y, fwd_logret), isolated.

    Accuracy is over rows where the rule takes a view (long = "up",
    flat = "not up"). Returns: per ticker, start flat, position follows the
    signal and holds inside the band; gross = sum(position * fwd_logret),
    net subtracts the illustrative cost per position change."""
    sig = rule_signal(rows["s_mean_1"].to_numpy())
    fired = ~np.isnan(sig)
    y = rows["y"].to_numpy(dtype=float)
    out: dict[str, Any] = {
        "band": NEUTRAL_BAND,
        "n_rows": int(len(rows)),
        "n_fired": int(fired.sum()),
        "coverage": float(fired.mean()) if len(rows) else float("nan"),
        "n_long_signals": int(np.sum(sig == 1.0)),
        "n_flat_signals": int(np.sum(sig == 0.0)),
        "accuracy_when_fired": float(np.mean(sig[fired] == y[fired])) if fired.any() else float("nan"),
        "up_rate_when_long_signal": float(np.mean(y[sig == 1.0])) if np.any(sig == 1.0) else float("nan"),
        "up_rate_when_flat_signal": float(np.mean(y[sig == 0.0])) if np.any(sig == 0.0) else float("nan"),
        "up_rate_all_rows": float(y.mean()) if len(y) else float("nan"),
    }
    cost = cost_bps_per_side / 1e4
    per = {}
    for ticker, g in rows.assign(_sig=sig).sort_values("ts", kind="mergesort").groupby("ticker", sort=True):
        pos = pd.Series(g["_sig"].to_numpy()).ffill().fillna(0.0).to_numpy()
        ret = g["fwd_logret"].to_numpy(dtype=float)
        prev = np.concatenate([[0.0], pos[:-1]])
        flips = int(np.sum(pos != prev))
        gross = float(np.sum(pos * ret))
        per[ticker] = {"fraction_long": float(pos.mean()), "flips": flips, "gross_logret": gross,
                       "net_logret": gross - flips * cost, "buy_hold_logret": float(np.sum(ret))}
    out["per_ticker"] = per
    for k in ("fraction_long", "flips", "gross_logret", "net_logret", "buy_hold_logret"):
        out[f"mean_{k}"] = float(np.mean([v[k] for v in per.values()])) if per else float("nan")
    out["cost_bps_per_side"] = float(cost_bps_per_side)
    out["label"] = "isolated rule baseline: pre-cost (gross); net uses a flat illustrative cost; not full-pipeline"
    return out


def _group_metrics(test: pd.DataFrame, p: np.ndarray, up: float, mask: np.ndarray) -> dict[str, Any]:
    if mask.sum() == 0:
        return {"n": 0}
    y = test["y"].to_numpy()[mask]
    if len(np.unique(y)) < 2:
        return {"n": int(mask.sum()), "accuracy": float(np.mean((p[mask] > THRESHOLD) == y))}
    return classification_metrics(y, p[mask], up)


def holdout_diagnostics(
    test: pd.DataFrame, p_test: np.ndarray, train_up_rate: float, revised_flag: Optional[np.ndarray] = None
) -> dict[str, Any]:
    has = test["has_news_1"].to_numpy() > 0.5
    out = {
        "by_has_news_1": {
            "has_news": _group_metrics(test, p_test, train_up_rate, has),
            "no_news": _group_metrics(test, p_test, train_up_rate, ~has),
            "share_rows_with_news": float(has.mean()),
        },
        "rule_baseline": rule_backtest(test),
    }
    if revised_flag is not None:
        rf = np.asarray(revised_flag, dtype=bool)
        out["by_revised_after_decision"] = {
            "revised": _group_metrics(test, p_test, train_up_rate, rf),
            "not_revised": _group_metrics(test, p_test, train_up_rate, ~rf),
            "share_rows_flagged": float(rf.mean()),
        }
    return out


def _stress(fold_test: pd.DataFrame, p: np.ndarray, up: float) -> Optional[dict[str, Any]]:
    lo, hi = (pd.Timestamp(x, tz="UTC") for x in STRESS_WINDOW)
    mask = ((fold_test["ts"] >= lo) & (fold_test["ts"] <= hi + pd.Timedelta(days=1))).to_numpy()
    if mask.sum() == 0:
        return None
    sub = fold_test.loc[mask]
    preds = sub[["ts", "ticker", "fwd_logret"]].assign(p_up=p[mask])
    return {
        "window": list(STRESS_WINDOW),
        "model_trained_through": None,  # filled by caller
        **_group_metrics(fold_test, p, up, mask),
        "model_backtest_isolated": long_flat_backtest(preds, ILLUSTRATIVE_COST_BPS_PER_SIDE),
        "rule_baseline": rule_backtest(sub),
    }


def run_scout_training(
    ds: pd.DataFrame,
    recipe: Recipe,
    *,
    test_start: pd.Timestamp | None = None,
    walk_forward: bool = True,
    revised_flag: Optional[pd.Series] = None,
) -> TrainingResult:
    """`ds` is a matured Scout dataset. `revised_flag` (optional, aligned
    with ds's index) marks rows whose window has a post-decision revision."""
    names = feature_names(recipe)
    if test_start is None:
        test_start = trailing_test_start(ds)
    split = split_at(ds, test_start)
    train, test = split.train, split.test
    up = float(train["y"].mean())

    model_bytes, booster = _roundtrip(fit_model(train, recipe))
    p_test = booster.predict(_dmatrix(test, names, with_label=False))
    test_metrics = classification_metrics(test["y"].to_numpy(), p_test, up)
    per_acc, per_ll = {}, {}
    for ticker in sorted(test["ticker"].unique()):
        mask = (test["ticker"] == ticker).to_numpy()
        m = classification_metrics(test["y"].to_numpy()[mask], p_test[mask], up)
        per_acc[ticker], per_ll[ticker] = m["accuracy"], m["logloss"]
    test_metrics["accuracy_per_ticker"] = per_acc
    test_metrics["logloss_per_ticker"] = per_ll

    preds = test[["ts", "ticker", "y", "fwd_logret", "label_end_ts", *names]].copy()
    preds["p_up"] = p_test
    preds["position"] = (p_test > THRESHOLD).astype(int)
    backtest = long_flat_backtest(preds, ILLUSTRATIVE_COST_BPS_PER_SIDE)

    rf = None if revised_flag is None else revised_flag.loc[test.index].to_numpy(dtype=bool)
    diagnostics = holdout_diagnostics(test, p_test, up, rf)

    wf_rows, stress = [], None
    for fold in (walk_forward_folds(ds, split.test_start) if walk_forward else ()):
        fold_up = float(fold.train["y"].mean())
        fp = fit_model(fold.train, recipe).predict(_dmatrix(fold.test, names, with_label=False))
        fm = classification_metrics(fold.test["y"].to_numpy(), fp, fold_up)
        has = fold.test["has_news_1"].to_numpy() > 0.5
        wf_rows.append({
            "name": fold.name,
            "train_start": fold.train["ts"].min().isoformat(),
            "train_end": fold.train["ts"].max().isoformat(),
            "trained_through": fold.train["label_end_ts"].max().isoformat(),
            "test_start": fold.test_start.isoformat(),
            "test_end": fold.test["ts"].max().isoformat(),
            "n_train": int(len(fold.train)),
            "share_rows_with_news": float(has.mean()),
            "rule_accuracy_when_fired": rule_backtest(fold.test)["accuracy_when_fired"],
            **fm,
        })
        if fold.name == "2020":
            stress = _stress(fold.test, fp, fold_up)
            if stress is not None:
                stress["model_trained_through"] = fold.train["label_end_ts"].max().isoformat()
    diagnostics["stress_window"] = stress

    return TrainingResult(
        recipe=recipe,
        feature_names=names,
        model_bytes=model_bytes,
        train=train,
        test=test,
        test_start=split.test_start,
        trained_through=split.trained_through,
        train_up_rate=up,
        test_metrics=test_metrics,
        test_predictions=preds.reset_index(drop=True),
        walk_forward=wf_rows,
        walk_forward_summary=_walk_forward_summary(wf_rows),
        backtest=backtest,
        feature_importance=_importance(booster, names),
        feature_matrix_sha256=_feature_matrix_sha256(ds, names),
        data_first_ts=pd.Timestamp(ds["ts"].min()),
        data_last_ts=pd.Timestamp(ds["label_end_ts"].max()),
        n_dataset_rows=int(len(ds)),
        extra={"diagnostics": diagnostics},
    )


# ---------------------------------------------------------------------------
# Manifest + MLflow (no headline text or URLs anywhere: public repo)
# ---------------------------------------------------------------------------

LIMITATIONS = (
    "No trading edge claimed. Headline-sentiment (VADER) model for five large US tickers; little or no "
    "signal expected. Deployed model == evaluated model (not refit on the holdout year)."
)


def build_scout_manifest(
    result: TrainingResult, run_id: str, *, model_version: str, gate_record_ids: Sequence[str],
    trial_number: int | None, promoted_by: str,
) -> dict[str, Any]:
    from wolfpack_worker.analyst.train import build_manifest

    m = build_manifest(result, run_id, model_version=model_version, gate_record_ids=gate_record_ids,
                       trial_number=trial_number, promoted_by=promoted_by)
    m["limitations"] = LIMITATIONS
    m["persona"] = "the-scout"
    return m


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def log_to_mlflow(
    result: TrainingResult,
    *,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    extra_params: Mapping[str, Any] | None = None,
    tags: Mapping[str, str] | None = None,
) -> str:
    """One run in MLflow experiment `the-scout` (+ nested walk-forward runs).
    Artifacts hold numbers, feature values and timestamps only: never a
    headline or URL (they are licensed; the repo is public)."""
    import mlflow

    artifact_root = Path(artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    exp_id = exp.experiment_id if exp is not None else client.create_experiment(
        EXPERIMENT_NAME, artifact_location=artifact_root.as_uri())
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_id=exp_id)
    r = result.recipe
    m = result.test_metrics
    diag = result.extra.get("diagnostics", {})
    params = {
        "recipe_id": r.recipe_id,
        "recipe_json": r.canonical_json(),
        "feature_names": json.dumps(list(result.feature_names)),
        "feature_spec_version": r.feature_spec_version,
        "scorer": get_scout_feature_spec(r.feature_spec_version).scorer,
        "label_definition": LABELS[r.label],
        "threshold": THRESHOLD,
        "num_boost_round": r.num_boost_round,
        **{f"xgb_{k}": v for k, v in r.xgb_params.items()},
        "early_stopping": "none",
        "nthread": TRAIN_NTHREAD,
        "train_start": result.train["ts"].min().isoformat(),
        "trained_through": result.trained_through.isoformat(),
        "test_start": result.test_start.isoformat(),
        "test_end": result.test["ts"].max().isoformat(),
        "test_sessions": TEST_SESSIONS,
        "embargo_sessions": EMBARGO_SESSIONS,
        "n_train_rows": len(result.train),
        "n_test_rows": len(result.test),
        "train_up_rate": round(result.train_up_rate, 6),
        "illustrative_cost_bps_per_side": ILLUSTRATIVE_COST_BPS_PER_SIDE,
        "xgboost_version": xgb.__version__,
        "feature_matrix_sha256": result.feature_matrix_sha256,
        **_git_info(),
        **dict(extra_params or {}),
    }
    metrics = {f"test_{k}": float(m[k]) for k in ("accuracy", "auc", "logloss", "brier", "baseline_accuracy",
                                                    "baseline_logloss", "n")}
    metrics["test_beats_baseline_logloss"] = float(m["beats_baseline_logloss"])
    metrics.update({k: float(v) for k, v in result.walk_forward_summary.items()})
    metrics["backtest_mean_gross_logret"] = float(result.backtest["mean_gross_logret"])
    metrics["backtest_mean_buy_hold_logret"] = float(result.backtest["mean_buy_hold_logret"])
    rb = diag.get("rule_baseline") or {}
    for k in ("accuracy_when_fired", "coverage", "mean_gross_logret"):
        if rb.get(k) is not None and np.isfinite(rb[k]):
            metrics[f"rule_{k}"] = float(rb[k])
    run_name = f"scout-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    with mlflow.start_run(experiment_id=exp_id, run_name=run_name) as run:
        mlflow.set_tags({"persona": "the-scout", "recipe_id": r.recipe_id,
                         "model_risk": "No trading edge claimed; compare against the base rate.",
                         **dict(tags or {})})
        mlflow.log_params({k: str(v) for k, v in params.items()})
        mlflow.log_metrics({k: v for k, v in metrics.items() if np.isfinite(v)})
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "model.json").write_bytes(result.model_bytes)
            (tmp / "feature_importance.json").write_text(json.dumps(result.feature_importance, indent=2))
            (tmp / "walk_forward.json").write_text(json.dumps(_jsonable(result.walk_forward), indent=2))
            (tmp / "diagnostics.json").write_text(json.dumps(_jsonable(diag), indent=2))
            (tmp / "backtest_isolated.json").write_text(json.dumps(_jsonable(result.backtest), indent=2))
            preds = result.test_predictions.copy()
            for col in ("ts", "label_end_ts"):
                preds[col] = preds[col].map(lambda t: t.isoformat())
            preds.to_csv(tmp / "test_predictions.csv", index=False)
            for name in ("model.json", "feature_importance.json", "walk_forward.json", "diagnostics.json",
                         "backtest_isolated.json", "test_predictions.csv"):
                mlflow.log_artifact(str(tmp / name))
        for f in result.walk_forward:
            with mlflow.start_run(experiment_id=exp_id, run_name=f"wf_{f['name']}", nested=True):
                mlflow.log_params({k: str(f[k]) for k in ("name", "train_start", "trained_through", "test_start",
                                                         "test_end", "n_train")})
                mlflow.log_metrics({k: float(f[k]) for k in ("accuracy", "auc", "logloss", "baseline_logloss",
                                                            "baseline_accuracy", "n", "share_rows_with_news")
                                    if f.get(k) is not None and np.isfinite(float(f[k]))})
        return run.info.run_id
