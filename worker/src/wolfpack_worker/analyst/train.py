"""Train The Analyst: backfill -> features/labels -> chronological split ->
fixed-setting XGBoost (a `Recipe`) -> holdout + walk-forward metrics vs
baselines -> local MLflow run.

    uv run --project worker --group train -m wolfpack_worker.analyst.train

The CLI only reproduces the frozen v1 recipe's report. It never writes a
champion: since the alphagate gate exists, the ONLY way a model becomes the
champion is `python -m wolfpack_worker.analyst.retrain ...`, which calls
`alphagate.gate()` and promotes only on a logged PROMOTE decision (and
`model_io.load_champion` refuses any champion without one). The old
`--promote` flag is retired. `run_training` here is the shared training
core that retrain.py calls.

Every run re-runs the full backfill first (so any split since the last run is
back-adjusted across all history), then reads `prices` back via paged
`get_history`, then refuses to train if any daily |log return| > 0.25.

Deployed model == evaluated model: the booster whose holdout metrics are
reported is byte-for-byte the one promoted. It is NOT refit on the holdout
window before deploying, so its `trained_through` is ~1 year behind — an
accepted cost of reporting an honest out-of-sample number for the exact
model that trades.

Walk-forward folds are reporting only (metrics, no model files).

MODEL-RISK LIMITATION: no real trading edge is claimed. The metrics logged
here exist to show how close to a coin flip this is, not to sell it.

"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import xgboost as xgb

from wolfpack_worker.analyst.dataset import (
    EMBARGO_SESSIONS,
    MAX_ABS_DAILY_LOG_RETURN,
    TEST_SESSIONS,
    assert_no_split_artifacts,
    build_dataset,
    split_at,
    trailing_test_start,
    walk_forward_folds,
)
from wolfpack_worker.analyst.features import get_feature_spec
from wolfpack_worker.analyst.metrics import THRESHOLD, classification_metrics, long_flat_backtest
from wolfpack_worker.analyst.recipe import LABELS, Recipe, load_v1_recipe
from wolfpack_worker.universe import ADJUSTMENT

logger = logging.getLogger(__name__)

# Model settings (XGBoost params, boosting rounds, feature spec, ...) live in
# a Recipe (recipe.py; v1 = worker/recipes/analyst/v1.toml). Fixed settings:
# no tuning, no early stopping (nothing is selected on the holdout, so the
# holdout stays a clean out-of-sample estimate).
# Single-threaded for bit-reproducibility across machines (multi-threaded
# histogram reductions can differ in float rounding by thread count). Only
# costs a second or two at this data size.
TRAIN_NTHREAD = 1
# Illustrative only, for the isolated backtest's "net" line. The project's
# real transaction-cost assumption (applied before any Sharpe/DSR) is a
# separate Week 3 backlog item and is not decided here.
ILLUSTRATIVE_COST_BPS_PER_SIDE = 5.0

_WORKER_ROOT = Path(__file__).resolve().parents[3]
MLFLOW_DIR = _WORKER_ROOT / "mlflow"
DEFAULT_TRACKING_URI = f"sqlite:///{MLFLOW_DIR / 'mlflow.db'}"
DEFAULT_ARTIFACT_ROOT = MLFLOW_DIR / "artifacts"
EXPERIMENT_NAME = "the-analyst"


@dataclass
class TrainingResult:
    recipe: Recipe
    feature_names: tuple[str, ...]
    model_bytes: bytes
    train: pd.DataFrame
    test: pd.DataFrame
    test_start: pd.Timestamp
    trained_through: pd.Timestamp
    train_up_rate: float
    test_metrics: dict[str, Any]
    test_predictions: pd.DataFrame
    walk_forward: list[dict[str, Any]]
    walk_forward_summary: dict[str, float]
    backtest: dict[str, Any]
    feature_importance: dict[str, dict[str, float]]
    feature_matrix_sha256: str
    data_first_ts: pd.Timestamp
    data_last_ts: pd.Timestamp
    n_dataset_rows: int
    extra: dict[str, Any] = field(default_factory=dict)


def _dmatrix(df: pd.DataFrame, names: Sequence[str], with_label: bool = True) -> xgb.DMatrix:
    x = df[list(names)].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float) if with_label else None
    return xgb.DMatrix(x, label=y, feature_names=list(names))


def fit_model(train: pd.DataFrame, recipe: Recipe) -> xgb.Booster:
    names = get_feature_spec(recipe.feature_spec_version).names
    params = {**recipe.xgb_params, "nthread": TRAIN_NTHREAD}
    return xgb.train(params, _dmatrix(train, names), num_boost_round=recipe.num_boost_round)


def bars_since(bars: Mapping[str, pd.DataFrame], since) -> dict[str, pd.DataFrame]:
    """Drop bars before the recipe's `train_since` (a recipe setting: it
    decides which early rows survive the feature warmup)."""
    start = pd.Timestamp(since).tz_localize("UTC") if pd.Timestamp(since).tzinfo is None else pd.Timestamp(since)
    return {t: df.loc[df.index >= start] for t, df in bars.items()}


def prepare_dataset(bars: Mapping[str, pd.DataFrame], recipe: Recipe) -> pd.DataFrame:
    """Split-artifact guard + the recipe's feature spec on bars from train_since."""
    assert_no_split_artifacts(bars)
    return build_dataset(bars_since(bars, recipe.train_since_date), recipe.feature_spec_version)


def _roundtrip(booster: xgb.Booster) -> tuple[bytes, xgb.Booster]:
    """Serialize to XGBoost JSON and reload — every reported metric is then
    computed from the exact bytes that get deployed."""
    model_bytes = bytes(booster.save_raw("json"))
    reloaded = xgb.Booster()
    reloaded.load_model(bytearray(model_bytes))
    return model_bytes, reloaded


def _feature_matrix_sha256(ds: pd.DataFrame, names: Sequence[str]) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(list(names)).encode())
    h.update(np.ascontiguousarray(ds[list(names)].to_numpy(dtype=np.float64)).tobytes())
    h.update(np.ascontiguousarray(ds["y"].to_numpy(dtype=np.int64)).tobytes())
    h.update(np.ascontiguousarray(ds["ts"].astype("int64").to_numpy()).tobytes())
    h.update("\n".join(ds["ticker"].tolist()).encode())
    return h.hexdigest()


def _importance(booster: xgb.Booster, names: Sequence[str]) -> dict[str, dict[str, float]]:
    out = {}
    for kind in ("gain", "weight"):
        score = booster.get_score(importance_type=kind)
        out[kind] = {name: float(score.get(name, 0.0)) for name in names}
    return out


def run_training(
    bars: Mapping[str, pd.DataFrame],
    recipe: Recipe | None = None,
    *,
    test_start: pd.Timestamp | None = None,
    walk_forward: bool = True,
) -> TrainingResult:
    """Pure training core (no network, no MLflow). Deterministic.

    `recipe` defaults to the frozen v1 recipe. `test_start` defaults to the
    first of the trailing TEST_SESSIONS sessions; the gate passes an explicit
    one so every model in a comparison shares one cutoff. The returned model
    is the one fit on the pre-cutoff rows and evaluated on the rest; it is
    never refit on the holdout.
    """
    recipe = recipe or load_v1_recipe()
    names = get_feature_spec(recipe.feature_spec_version).names
    ds = prepare_dataset(bars, recipe)
    if test_start is None:
        test_start = trailing_test_start(ds)
    split = split_at(ds, test_start)
    train, test = split.train, split.test
    train_up_rate = float(train["y"].mean())

    model_bytes, booster = _roundtrip(fit_model(train, recipe))
    p_test = booster.predict(_dmatrix(test, names, with_label=False))

    test_metrics = classification_metrics(test["y"].to_numpy(), p_test, train_up_rate)
    per_ticker_acc = {}
    per_ticker_ll = {}
    for ticker in sorted(test["ticker"].unique()):
        mask = (test["ticker"] == ticker).to_numpy()
        m = classification_metrics(test["y"].to_numpy()[mask], p_test[mask], train_up_rate)
        per_ticker_acc[ticker] = m["accuracy"]
        per_ticker_ll[ticker] = m["logloss"]
    test_metrics["accuracy_per_ticker"] = per_ticker_acc
    test_metrics["logloss_per_ticker"] = per_ticker_ll

    preds = test[["ts", "ticker", "y", "fwd_logret", "label_end_ts"]].copy()
    preds["p_up"] = p_test
    preds["position"] = (p_test > THRESHOLD).astype(int)
    backtest = long_flat_backtest(preds, ILLUSTRATIVE_COST_BPS_PER_SIDE)

    walk_forward_rows = []
    for fold in (walk_forward_folds(ds, split.test_start) if walk_forward else ()):
        fold_up = float(fold.train["y"].mean())
        fb = fit_model(fold.train, recipe)
        fp = fb.predict(_dmatrix(fold.test, names, with_label=False))
        fm = classification_metrics(fold.test["y"].to_numpy(), fp, fold_up)
        walk_forward_rows.append(
            {
                "name": fold.name,
                "train_start": fold.train["ts"].min().isoformat(),
                "train_end": fold.train["ts"].max().isoformat(),
                "trained_through": fold.train["label_end_ts"].max().isoformat(),
                "test_start": fold.test_start.isoformat(),
                "test_end": fold.test["ts"].max().isoformat(),
                "n_train": int(len(fold.train)),
                **fm,
            }
        )
    walk_forward_summary = _walk_forward_summary(walk_forward_rows)

    return TrainingResult(
        recipe=recipe,
        feature_names=tuple(names),
        model_bytes=model_bytes,
        train=train,
        test=test,
        test_start=split.test_start,
        trained_through=split.trained_through,
        train_up_rate=train_up_rate,
        test_metrics=test_metrics,
        test_predictions=preds.reset_index(drop=True),
        walk_forward=walk_forward_rows,
        walk_forward_summary=walk_forward_summary,
        backtest=backtest,
        feature_importance=_importance(booster, names),
        feature_matrix_sha256=_feature_matrix_sha256(ds, names),
        data_first_ts=pd.Timestamp(ds["ts"].min()),
        data_last_ts=pd.Timestamp(ds["label_end_ts"].max()),
        n_dataset_rows=int(len(ds)),
    )


def _walk_forward_summary(walk_forward: list[dict[str, Any]]) -> dict[str, float]:
    if not walk_forward:
        return {}
    accs = np.array([f["accuracy"] for f in walk_forward])
    edge = np.array([f["logloss"] - f["baseline_logloss"] for f in walk_forward])
    acc_edge = np.array([f["accuracy"] - f["baseline_accuracy"] for f in walk_forward])
    return {
        "walk_forward_accuracy_mean": float(accs.mean()),
        "walk_forward_accuracy_std": float(accs.std(ddof=1)) if len(accs) > 1 else 0.0,
        "walk_forward_accuracy_min": float(accs.min()),
        "walk_forward_accuracy_max": float(accs.max()),
        "walk_forward_accuracy_minus_baseline_mean": float(acc_edge.mean()),
        "walk_forward_logloss_minus_baseline_mean": float(edge.mean()),
        "walk_forward_logloss_minus_baseline_std": float(edge.std(ddof=1)) if len(edge) > 1 else 0.0,
        "walk_forward_folds": float(len(walk_forward)),
        "walk_forward_folds_beating_baseline_logloss": float(np.sum(edge < 0)),
    }


# ---------------------------------------------------------------------------
# MLflow (local sqlite + local artifact dir; free, no server)
# ---------------------------------------------------------------------------


def _git_info() -> dict[str, str]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_WORKER_ROOT, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--", "src"], cwd=_WORKER_ROOT, text=True
        ).strip()
        return {"git_commit": commit, "git_src_dirty": str(bool(dirty)).lower()}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": "unknown", "git_src_dirty": "unknown"}


def _params(result: TrainingResult, extra: Mapping[str, Any]) -> dict[str, str]:
    r = result.recipe
    p: dict[str, Any] = {
        "recipe_id": r.recipe_id,
        "recipe_json": r.canonical_json(),
        "feature_names": json.dumps(list(result.feature_names)),
        "feature_spec_version": r.feature_spec_version,
        "label_definition": LABELS[r.label],
        "threshold": THRESHOLD,
        "num_boost_round": r.num_boost_round,
        "early_stopping": "none",
        "nthread": TRAIN_NTHREAD,
        "seed": r.xgb_params.get("seed", "unset"),
        **{f"xgb_{k}": v for k, v in r.xgb_params.items()},
        "universe": json.dumps(list(r.universe)),
        "train_since": r.train_since,
        "pooled_model_ticker_id_feature": "none",
        "data_first_ts": result.data_first_ts.isoformat(),
        "data_last_label_end_ts": result.data_last_ts.isoformat(),
        "train_start": result.train["ts"].min().isoformat(),
        "train_end": result.train["ts"].max().isoformat(),
        "trained_through": result.trained_through.isoformat(),
        "test_start": result.test_start.isoformat(),
        "test_end": result.test["ts"].max().isoformat(),
        "test_sessions": TEST_SESSIONS,
        "embargo_sessions": EMBARGO_SESSIONS,
        "n_dataset_rows": result.n_dataset_rows,
        "n_train_rows": len(result.train),
        "n_test_rows": len(result.test),
        "train_up_rate": round(result.train_up_rate, 6),
        "test_up_rate": round(result.test_metrics["test_up_rate"], 6),
        "adjustment": ADJUSTMENT,
        "split_guard_max_abs_daily_logret": MAX_ABS_DAILY_LOG_RETURN,
        "illustrative_cost_bps_per_side": ILLUSTRATIVE_COST_BPS_PER_SIDE,
        "xgboost_version": xgb.__version__,
        "feature_matrix_sha256": result.feature_matrix_sha256,
        **_git_info(),
        **extra,
    }
    return {k: str(v) for k, v in p.items()}


def _metrics(result: TrainingResult) -> dict[str, float]:
    m = result.test_metrics
    out = {
        f"test_{k}": float(m[k])
        for k in (
            "accuracy", "auc", "logloss", "brier", "baseline_accuracy",
            "baseline_logloss", "baseline_brier", "test_up_rate", "n",
        )
    }
    out["test_beats_baseline_logloss"] = float(m["beats_baseline_logloss"])
    for t, v in m["accuracy_per_ticker"].items():
        out[f"test_accuracy_{t}"] = v
    for t, v in m["logloss_per_ticker"].items():
        out[f"test_logloss_{t}"] = v
    for f in result.walk_forward:
        for k in ("accuracy", "logloss", "baseline_accuracy", "baseline_logloss", "auc"):
            out[f"wf_{f['name']}_{k}"] = float(f[k])
    out.update(result.walk_forward_summary)
    bt = result.backtest
    for k in ("fraction_long", "flips", "gross_logret", "net_logret", "buy_hold_logret"):
        out[f"backtest_{k}"] = float(bt[f"mean_{k}"])
        for t, v in bt["per_ticker"].items():
            out[f"backtest_{k}_{t}"] = float(v[k])
    return out


def log_to_mlflow(
    result: TrainingResult,
    *,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    extra_params: Mapping[str, Any] | None = None,
    experiment_name: str = EXPERIMENT_NAME,
    tags: Mapping[str, str] | None = None,
) -> str:
    import mlflow

    artifact_root = Path(artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    exp = client.get_experiment_by_name(experiment_name)
    exp_id = (
        exp.experiment_id
        if exp is not None
        else client.create_experiment(experiment_name, artifact_location=artifact_root.as_uri())
    )
    mlflow.set_tracking_uri(tracking_uri)
    # Also pin the active experiment: nested start_run() calls otherwise land
    # in MLflow's "Default" experiment rather than the parent's.
    mlflow.set_experiment(experiment_id=exp_id)

    run_name = f"analyst-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    with mlflow.start_run(experiment_id=exp_id, run_name=run_name) as run:
        mlflow.set_tags({
            "persona": "the-analyst",
            "backtest_label": result.backtest["label"],
            "model_risk": "No trading edge claimed; compare against baselines.",
            "recipe_id": result.recipe.recipe_id,
            **dict(tags or {}),
        })
        mlflow.log_params(_params(result, extra_params or {}))
        mlflow.log_metrics(_metrics(result))

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "model.json").write_bytes(result.model_bytes)
            (tmp / "feature_importance.json").write_text(
                json.dumps(result.feature_importance, indent=2, sort_keys=True)
            )
            (tmp / "walk_forward.json").write_text(json.dumps(result.walk_forward, indent=2))
            (tmp / "backtest_isolated.json").write_text(json.dumps(result.backtest, indent=2))
            preds = result.test_predictions.copy()
            for col in ("ts", "label_end_ts"):
                preds[col] = preds[col].map(lambda t: t.isoformat())
            preds.to_csv(tmp / "test_predictions.csv", index=False)
            for name in (
                "model.json", "feature_importance.json", "test_predictions.csv",
                "walk_forward.json", "backtest_isolated.json",
            ):
                mlflow.log_artifact(str(tmp / name))

        for f in result.walk_forward:
            with mlflow.start_run(
                experiment_id=exp_id, run_name=f"wf_{f['name']}", nested=True
            ):
                mlflow.log_params({
                    k: str(f[k]) for k in (
                        "name", "train_start", "train_end", "trained_through",
                        "test_start", "test_end", "n_train", "train_up_rate",
                    )
                })
                mlflow.log_metrics({
                    k: float(f[k]) for k in (
                        "accuracy", "auc", "logloss", "brier", "baseline_accuracy",
                        "baseline_logloss", "n", "test_up_rate",
                    )
                })
        return run.info.run_id


# ---------------------------------------------------------------------------
# Champion
# ---------------------------------------------------------------------------


def model_version_for(model_bytes: bytes, today: date | None = None) -> str:
    """`analyst-<UTC date>-<first 8 of model sha256>`; the gate's challenger_id."""
    today = today or datetime.now(timezone.utc).date()
    return f"analyst-{today:%Y%m%d}-{hashlib.sha256(model_bytes).hexdigest()[:8]}"


def build_manifest(
    result: TrainingResult,
    run_id: str,
    *,
    model_version: str,
    gate_record_ids: Sequence[str],
    trial_number: int | None,
    promoted_by: str,
) -> dict[str, Any]:
    """Champion manifest for an evaluated model. Only the gate's promotion
    step (gating.promote_from_gate) writes one to the champion directory."""
    gain = result.feature_importance["gain"]
    top = dict(sorted(gain.items(), key=lambda kv: -kv[1])[:5])
    m = result.test_metrics
    r = result.recipe
    return {
        "model_version": model_version,
        "mlflow_run_id": run_id,
        "recipe_id": r.recipe_id,
        "recipe": r.to_dict(),
        "trial_number": trial_number,
        "gate_record_ids": list(gate_record_ids),
        "promoted_by": promoted_by,
        "trained_through": result.trained_through.isoformat(),
        "train_start": result.train["ts"].min().isoformat(),
        "test_start": result.test_start.isoformat(),
        "test_end": result.test["ts"].max().isoformat(),
        "feature_spec_version": r.feature_spec_version,
        "feature_names": list(result.feature_names),
        "label_definition": LABELS[r.label],
        "threshold": THRESHOLD,
        "xgboost_version": xgb.__version__,
        "model_params": {**r.xgb_params, "num_boost_round": r.num_boost_round},
        "test_metrics": {
            k: m[k]
            for k in (
                "n", "accuracy", "auc", "logloss", "brier", "baseline_accuracy",
                "baseline_logloss", "baseline_brier", "beats_baseline_logloss",
                "train_up_rate", "test_up_rate", "accuracy_per_ticker",
            )
        },
        "walk_forward_summary": result.walk_forward_summary,
        "top_importances_gain": top,
        "limitations": (
            "No trading edge claimed. Deployed model == evaluated model "
            "(not refit on the holdout year)."
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _summary(result: TrainingResult) -> str:
    m = result.test_metrics
    wf = result.walk_forward_summary
    lines = [
        f"train rows={len(result.train)} test rows={len(result.test)} "
        f"test={result.test_start.date()}..{result.test['ts'].max().date()} "
        f"trained_through={result.trained_through.date()}",
        f"holdout accuracy={m['accuracy']:.4f} (baseline {m['baseline_accuracy']:.4f})  "
        f"AUC={m['auc']:.4f}  logloss={m['logloss']:.5f} (baseline {m['baseline_logloss']:.5f})  "
        f"brier={m['brier']:.5f}  beats_baseline_logloss={m['beats_baseline_logloss']}",
        "per-ticker accuracy: "
        + ", ".join(f"{t}={v:.3f}" for t, v in m["accuracy_per_ticker"].items()),
    ]
    if wf:
        lines.append(
            f"walk-forward accuracy mean={wf['walk_forward_accuracy_mean']:.4f} "
            f"std={wf['walk_forward_accuracy_std']:.4f} "
            f"range=[{wf['walk_forward_accuracy_min']:.4f}, {wf['walk_forward_accuracy_max']:.4f}]; "
            f"folds beating baseline logloss: {int(wf['walk_forward_folds_beating_baseline_logloss'])}"
            f"/{int(wf['walk_forward_folds'])}"
        )
    for f in result.walk_forward:
        lines.append(
            f"  {f['name']:>13}: acc={f['accuracy']:.4f} base_acc={f['baseline_accuracy']:.4f} "
            f"ll={f['logloss']:.5f} base_ll={f['baseline_logloss']:.5f} auc={f['auc']:.4f} n={f['n']}"
        )
    bt = result.backtest
    lines.append(
        f"isolated long/flat (pre-cost, equal-weight mean per ticker): "
        f"fraction_long={bt['mean_fraction_long']:.3f} flips={bt['mean_flips']:.1f} "
        f"gross={bt['mean_gross_logret']:+.4f} net@{bt['cost_bps_per_side']:.0f}bps="
        f"{bt['mean_net_logret']:+.4f} buy&hold={bt['mean_buy_hold_logret']:+.4f} (log-return sums)"
    )
    return "\n".join(lines)


def load_price_history(
    since: date, *, backfill: bool = True
) -> tuple[dict[str, pd.DataFrame], datetime, dict[str, Any]]:
    """Read every universe ticker's daily bars from `prices` (Supabase).

    backfill=True first re-runs the full split-adjusted backfill (writes the
    `prices` table; same as the original training run) and uses the latest
    completed session as as_of. backfill=False is read-only: as_of = now,
    and `prices` only ever holds completed sessions.
    """
    from wolfpack_worker.config import load_config
    from wolfpack_worker.db import get_client
    from wolfpack_worker.store import SupabasePriceStore
    from wolfpack_worker.universe import UNIVERSE

    feeds: dict[str, Any] = {}
    if backfill:
        from wolfpack_worker.backfill import run_backfill

        feeds, as_of = run_backfill(since)
    else:
        as_of = datetime.now(timezone.utc)
    store = SupabasePriceStore(get_client(load_config()))
    start = datetime(since.year, since.month, since.day, tzinfo=timezone.utc)
    bars = {t: store.get_history(t, start, as_of) for t in UNIVERSE}
    for t, df in bars.items():
        logger.info("history: %s %d bars %s..%s", t, len(df), df.index.min(), df.index.max())
    return bars, as_of, feeds


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce The Analyst's v1 recipe report (local MLflow). Never writes a "
            "champion: promotion only happens through `wolfpack_worker.analyst.retrain`."
        )
    )
    args = parser.parse_args(argv)
    del args
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    recipe = load_v1_recipe()
    bars, as_of, feeds = load_price_history(recipe.train_since_date, backfill=True)
    result = run_training(bars, recipe)
    run_id = log_to_mlflow(
        result,
        extra_params={
            "data_feed": json.dumps(feeds, sort_keys=True),
            "data_as_of": as_of.isoformat(),
            "backfill_since": recipe.train_since,
            "bars_per_ticker": json.dumps({t: len(df) for t, df in bars.items()}, sort_keys=True),
        },
        tags={"run_kind": "reproduce_v1_report"},
    )
    print(f"MLflow run: {run_id}")
    print(_summary(result))
    print("Report only: no champion was written (promotion goes through the alphagate gate).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
