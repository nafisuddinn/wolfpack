"""Training pipeline for The Analyst (analyst/train.py + analyst/metrics.py).

The pure training core is exercised on synthetic data; MLflow logging is
smoke-tested against a throwaway sqlite store in tmp_path (never the
committed `worker/mlflow/` store).
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from analyst_helpers import make_universe_bars
from wolfpack_worker.analyst.dataset import SuspectPriceDataError
from wolfpack_worker.analyst.features import FEATURE_NAMES, FEATURE_SPEC_VERSION
from wolfpack_worker.analyst.metrics import (
    classification_metrics,
    long_flat_backtest,
)
from wolfpack_worker.analyst.model_io import ModelIntegrityError, load_champion, write_champion
from wolfpack_worker.analyst.recipe import load_v1_recipe
from wolfpack_worker.analyst.train import run_training


# ---------------------------------------------------------------------------
# Fixed model settings (no tuning, no early stopping)
# ---------------------------------------------------------------------------


def test_model_settings_are_the_fixed_design_values():
    # The settings moved from train.MODEL_PARAMS / NUM_BOOST_ROUND into the
    # frozen v1 recipe (worker/recipes/analyst/v1.toml); same values.
    recipe = load_v1_recipe()
    assert recipe.num_boost_round == 300
    assert dict(recipe.xgb_params) == {
        "objective": "binary:logistic",
        "max_depth": 3,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 20,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "seed": 42,
    }


# ---------------------------------------------------------------------------
# Metrics + baselines
# ---------------------------------------------------------------------------


def test_classification_metrics_and_baselines_known_values():
    y = np.array([1, 0, 1, 1])
    p = np.array([0.9, 0.2, 0.4, 0.6])
    m = classification_metrics(y, p, train_up_rate=0.55)
    assert m["accuracy"] == pytest.approx(0.75)
    expected_ll = -np.mean(np.log([0.9, 0.8, 0.4, 0.6]))
    assert m["logloss"] == pytest.approx(expected_ll)
    assert m["brier"] == pytest.approx(np.mean((p - y) ** 2))
    assert m["auc"] == pytest.approx(1.0)  # every positive scores above the negative
    # Baseline: always predict the training up-rate.
    assert m["baseline_accuracy"] == pytest.approx(0.75)  # up-rate > .5 -> always "up"
    expected_base_ll = -(3 * math.log(0.55) + math.log(0.45)) / 4
    assert m["baseline_logloss"] == pytest.approx(expected_base_ll)
    assert m["beats_baseline_logloss"] == (m["logloss"] < m["baseline_logloss"])
    assert m["n"] == 4


def test_baseline_accuracy_uses_down_class_when_train_up_rate_below_half():
    y = np.array([1, 0, 0])
    m = classification_metrics(y, np.array([0.5, 0.5, 0.5]), train_up_rate=0.4)
    assert m["baseline_accuracy"] == pytest.approx(2 / 3)
    # A constant 0.5 prediction is "down" by the tie rule -> 2/3 correct.
    assert m["accuracy"] == pytest.approx(2 / 3)


def test_long_flat_backtest_known_values():
    ts = pd.bdate_range("2025-01-01", periods=4, tz="UTC")
    df = pd.DataFrame(
        {
            "ts": list(ts) * 2,
            "ticker": ["A"] * 4 + ["B"] * 4,
            "p_up": [0.6, 0.6, 0.4, 0.7, 0.4, 0.4, 0.4, 0.4],
            "fwd_logret": [0.01, -0.02, 0.03, 0.01, 0.01, 0.01, 0.01, 0.01],
        }
    ).sort_values(["ts", "ticker"])
    bt = long_flat_backtest(df, cost_bps_per_side=10.0)
    a = bt["per_ticker"]["A"]
    assert a["fraction_long"] == pytest.approx(0.75)
    assert a["flips"] == 3  # enter(from flat) at t0, exit t2, enter t3
    assert a["gross_logret"] == pytest.approx(0.01 - 0.02 + 0.01)
    assert a["buy_hold_logret"] == pytest.approx(0.03)
    assert a["net_logret"] == pytest.approx(0.0 - 3 * 0.001)
    b = bt["per_ticker"]["B"]
    assert b["fraction_long"] == 0.0 and b["flips"] == 0 and b["gross_logret"] == 0.0
    assert bt["label"] == "isolated, pre-cost (gross); net uses a flat illustrative cost"
    assert bt["mean_gross_logret"] == pytest.approx((a["gross_logret"] + 0.0) / 2)


# ---------------------------------------------------------------------------
# End-to-end training core
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bars():
    return make_universe_bars(2800, seed=31)


@pytest.fixture(scope="module")
def result(bars):
    return run_training(bars)


def test_training_split_is_leak_free_and_deployed_model_is_evaluated_model(result):
    assert result.trained_through < result.test_start
    assert (result.train["label_end_ts"] < result.test_start).all()
    assert result.test["ts"].nunique() == 252
    # Deployed == evaluated: reloading the serialized bytes reproduces the
    # exact holdout predictions that the metrics were computed from.
    import xgboost as xgb

    booster = xgb.Booster()
    booster.load_model(bytearray(result.model_bytes))
    d = xgb.DMatrix(result.test[list(FEATURE_NAMES)].to_numpy(), feature_names=list(FEATURE_NAMES))
    np.testing.assert_array_equal(booster.predict(d), result.test_predictions["p_up"].to_numpy())


def test_training_reports_baselines_and_walk_forward(result):
    m = result.test_metrics
    for k in ("accuracy", "auc", "logloss", "brier", "baseline_accuracy", "baseline_logloss"):
        assert np.isfinite(m[k])
    assert set(m["accuracy_per_ticker"]) == {"SPY", "QQQ", "AAPL", "JPM", "XOM"}
    names = [f["name"] for f in result.walk_forward]
    assert names == [str(y) for y in range(2019, 2026)] + ["trailing_252"]
    assert "walk_forward_accuracy_mean" in result.walk_forward_summary
    assert "walk_forward_accuracy_std" in result.walk_forward_summary


def test_training_is_deterministic(bars, result):
    again = run_training(bars)
    assert again.model_bytes == result.model_bytes
    assert again.feature_matrix_sha256 == result.feature_matrix_sha256


def test_training_aborts_on_split_artifact(bars):
    bad = dict(bars)
    q = bad["QQQ"].copy()
    q.iloc[1000:, :4] = q.iloc[1000:, :4] * 2
    bad["QQQ"] = q
    with pytest.raises(SuspectPriceDataError):
        run_training(bad)


# ---------------------------------------------------------------------------
# MLflow logging + promotion (tmp store only)
# ---------------------------------------------------------------------------


def test_mlflow_logging_and_manifest(tmp_path, result):
    mlflow = pytest.importorskip("mlflow")
    from wolfpack_worker.analyst import train
    from wolfpack_worker.analyst.train import build_manifest, log_to_mlflow, model_version_for

    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    artifact_root = tmp_path / "artifacts"
    run_id = log_to_mlflow(
        result,
        tracking_uri=tracking_uri,
        artifact_root=artifact_root,
        extra_params={"data_feed": "sip"},
    )
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)
    p = run.data.params
    assert p["feature_spec_version"] == FEATURE_SPEC_VERSION
    assert p["recipe_id"] == load_v1_recipe().recipe_id
    assert run.data.tags["recipe_id"] == load_v1_recipe().recipe_id
    assert json.loads(p["feature_names"]) == list(FEATURE_NAMES)
    assert p["embargo_sessions"] == "2"
    assert p["feature_matrix_sha256"] == result.feature_matrix_sha256
    assert "label_definition" in p and "xgboost_version" in p and "git_commit" in p
    mets = run.data.metrics
    for k in ("test_accuracy", "test_auc", "test_logloss", "test_brier",
              "test_baseline_accuracy", "test_baseline_logloss",
              "walk_forward_accuracy_mean", "walk_forward_accuracy_std",
              "test_accuracy_SPY", "backtest_fraction_long"):
        assert k in mets, k
    arts = {a.path for a in client.list_artifacts(run_id)}
    assert {"model.json", "feature_importance.json", "test_predictions.csv"} <= arts
    children = client.search_runs(
        [run.info.experiment_id], filter_string=f"tags.mlflow.parentRunId = '{run_id}'"
    )
    assert len(children) == 8
    for c in children:
        assert client.list_artifacts(c.info.run_id) == []  # metrics only

    # `train.promote` / `--promote` are retired: the only promotion path is
    # the alphagate gate (gating.promote_from_gate). A manifest written
    # directly, without a gate PROMOTE record, must not load.
    assert not hasattr(train, "promote")
    with pytest.raises(SystemExit):
        train.main(["--promote"])
    version = model_version_for(result.model_bytes)
    manifest = build_manifest(result, run_id, model_version=version, gate_record_ids=["r1"],
                              trial_number=None, promoted_by="test")
    champ_dir = tmp_path / "champion"
    written = write_champion(champ_dir, result.model_bytes, manifest)
    for key in ("model_version", "mlflow_run_id", "trained_through", "feature_spec_version",
                "feature_names", "model_sha256", "xgboost_version", "test_metrics",
                "top_importances_gain", "recipe_id", "recipe", "trial_number", "gate_record_ids"):
        assert key in written, key
    assert written["mlflow_run_id"] == run_id
    assert written["recipe_id"] == load_v1_recipe().recipe_id
    assert written["model_version"].startswith("analyst-")
    assert written["model_version"].endswith(written["model_sha256"][:8])
    with pytest.raises(ModelIntegrityError):
        load_champion(champ_dir)
