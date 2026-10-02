"""explore (walk-forward strictly before the gate holdout) and monitor
(forward scoring of each champion after its promotion)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

pytest.importorskip("alphagate")
from alphagate import spend_alpha  # noqa: E402

from analyst_helpers import make_predictable_universe_bars, weak_recipe_dict  # noqa: E402
from wolfpack_worker.analyst import explore as explore_mod  # noqa: E402
from wolfpack_worker.analyst import forward, gating  # noqa: E402
from wolfpack_worker.analyst.dataset import Fold, build_dataset  # noqa: E402
from wolfpack_worker.analyst.gate_log import read_gate_log, read_jsonl  # noqa: E402
from wolfpack_worker.analyst.model_io import load_champion, write_champion  # noqa: E402
from wolfpack_worker.analyst.recipe import Recipe, load_v1_recipe  # noqa: E402
from wolfpack_worker.analyst.train import build_manifest, model_version_for, run_training  # noqa: E402


@pytest.fixture(scope="module")
def bars():
    return make_predictable_universe_bars(1100, seed=4)


# --- explore -------------------------------------------------------------------------


def test_explore_boundary_is_the_next_gate_holdout_start(bars):
    recipe = load_v1_recipe()
    ds = build_dataset(bars)
    as_of = bars["SPY"].index[-1] + pd.Timedelta(hours=20)
    b = explore_mod.explore_boundary(bars, recipe, recipe, as_of)
    assert b == gating.shared_cutoff([("v1", ds)])


def test_explore_never_touches_the_gate_holdout(bars):
    recipe = load_v1_recipe()
    as_of = bars["SPY"].index[-1] + pd.Timedelta(hours=20)
    boundary = explore_mod.explore_boundary(bars, recipe, recipe, as_of)
    res = explore_mod.run_explore(bars, recipe, boundary=boundary)
    assert res.folds, "expected at least one fold"
    for f in res.folds:
        assert pd.Timestamp(f["test_end"]) < boundary
        assert pd.Timestamp(f["max_label_end"]) < boundary  # labels read no holdout bar
        assert pd.Timestamp(f["trained_through"]) < pd.Timestamp(f["test_start"])
    assert res.boundary == boundary


def test_explore_refuses_an_end_inside_the_gate_holdout(bars):
    recipe = load_v1_recipe()
    as_of = bars["SPY"].index[-1] + pd.Timedelta(hours=20)
    boundary = explore_mod.explore_boundary(bars, recipe, recipe, as_of)
    with pytest.raises(explore_mod.HoldoutContaminationError):
        explore_mod.run_explore(bars, recipe, boundary=boundary, until=boundary + pd.Timedelta(days=1))
    # Ending earlier is fine.
    early = boundary - pd.Timedelta(days=200)
    res = explore_mod.run_explore(bars, recipe, boundary=boundary, until=early)
    assert all(pd.Timestamp(f["max_label_end"]) < early for f in res.folds)


def test_explore_guard_catches_a_contaminated_fold(bars, monkeypatch):
    """If fold construction ever regressed and yielded holdout rows, explore
    must refuse rather than report."""
    recipe = load_v1_recipe()
    as_of = bars["SPY"].index[-1] + pd.Timedelta(hours=20)
    boundary = explore_mod.explore_boundary(bars, recipe, recipe, as_of)
    full = build_dataset(bars)

    def leaky(ds, *a, **k):
        yield Fold(name="leaky", train=full.loc[full["ts"] < boundary - pd.Timedelta(days=30)],
                   test=full.loc[full["ts"] >= boundary], test_start=boundary)

    monkeypatch.setattr(explore_mod, "walk_forward_folds", leaky)
    with pytest.raises(explore_mod.HoldoutContaminationError):
        explore_mod.run_explore(bars, recipe, boundary=boundary)


# --- monitor -------------------------------------------------------------------------


@pytest.fixture()
def bootstrapped(tmp_path, bars):
    """Champion promoted (bootstrap) at bar 800; bars after that are 'forward'."""
    recipe = Recipe.from_dict(weak_recipe_dict())
    early = {t: df.iloc[:800] for t, df in bars.items()}
    res = run_training(early, recipe, walk_forward=False)
    m = build_manifest(res, "r", model_version=model_version_for(res.model_bytes), gate_record_ids=[],
                       trial_number=None, promoted_by="x")
    for k in ("recipe_id", "recipe", "trial_number", "gate_record_ids", "promoted_by"):
        m.pop(k)
    models = tmp_path / "models"
    write_champion(models / "champion", res.model_bytes, m)
    as_of = (early["SPY"].index[-1] + pd.Timedelta(hours=20)).to_pydatetime()
    gating.run_bootstrap(early, champion_dir=models / "champion", gate_log_path=models / "gate_log.jsonl",
                         as_of=as_of, recipe=recipe)
    return models


def _promoted_at(models):
    return pd.Timestamp(read_gate_log(models / "gate_log.jsonl")[0]["decided_at"])


def test_monitor_scores_only_sessions_after_promotion_with_matured_labels(bootstrapped, bars, monkeypatch):
    models = bootstrapped
    # Pretend the promotion decision happened right after bar 799's close.
    promoted = bars["SPY"].index[799] + pd.Timedelta(hours=20)
    monkeypatch.setattr(forward, "_promotion_time", lambda recs, v: promoted)
    as_of = bars["SPY"].index[999] + pd.Timedelta(hours=20)
    out = forward.run_monitor(bars, as_of=as_of.to_pydatetime(), champion_dir=models / "champion",
                              archive_dir=models / "archive", gate_log_path=models / "gate_log.jsonl",
                              forward_log_path=models / "forward_log.jsonl")
    (rec,) = out
    # Forward window: sessions 800.. whose label (2 bars later) has matured by bar 999.
    assert pd.Timestamp(rec["window_start"]) == bars["SPY"].index[800]
    assert pd.Timestamp(rec["window_end"]) == bars["SPY"].index[997]
    assert rec["n_sessions"] == 198
    assert rec["look_number"] == 1 and rec["alpha_look"] == pytest.approx(spend_alpha(0.05, 1))
    assert rec["edge_significant"] == (rec["p"] < rec["alpha_look"])
    assert rec["baseline_p_up"] == pytest.approx(
        load_champion(models / "champion", gate_log_path=models / "gate_log.jsonl").manifest["test_metrics"]["train_up_rate"]
    )
    lines = read_jsonl(models / "forward_log.jsonl")
    assert lines == [json.loads(json.dumps(rec))]


def test_monitor_short_window_is_not_a_look(bootstrapped, bars, monkeypatch):
    models = bootstrapped
    promoted = bars["SPY"].index[799] + pd.Timedelta(hours=20)
    monkeypatch.setattr(forward, "_promotion_time", lambda recs, v: promoted)
    as_of = bars["SPY"].index[900] + pd.Timedelta(hours=20)
    (rec,) = forward.run_monitor(bars, as_of=as_of.to_pydatetime(), champion_dir=models / "champion",
                                 archive_dir=models / "archive", gate_log_path=models / "gate_log.jsonl",
                                 forward_log_path=models / "forward_log.jsonl")
    assert rec["n_sessions"] < forward.MIN_EDGE_SESSIONS
    assert rec["look_number"] is None and rec["edge_significant"] is False


def test_monitor_counts_repeated_looks(bootstrapped, bars, monkeypatch):
    models = bootstrapped
    promoted = bars["SPY"].index[799] + pd.Timedelta(hours=20)
    monkeypatch.setattr(forward, "_promotion_time", lambda recs, v: promoted)
    kw = dict(champion_dir=models / "champion", archive_dir=models / "archive",
              gate_log_path=models / "gate_log.jsonl", forward_log_path=models / "forward_log.jsonl")
    for i, idx in enumerate((950, 1000, 1050), start=1):
        as_of = (bars["SPY"].index[idx] + pd.Timedelta(hours=20)).to_pydatetime()
        (rec,) = forward.run_monitor(bars, as_of=as_of, **kw)
        assert rec["look_number"] == i
        assert rec["alpha_look"] == pytest.approx(spend_alpha(0.05, i))


def test_monitor_with_no_forward_sessions_yet(bootstrapped, bars):
    models = bootstrapped
    as_of = (bars["SPY"].index[799] + pd.Timedelta(hours=20)).to_pydatetime()
    (rec,) = forward.run_monitor(
        {t: df.iloc[:800] for t, df in bars.items()}, as_of=as_of, champion_dir=models / "champion",
        archive_dir=models / "archive", gate_log_path=models / "gate_log.jsonl",
        forward_log_path=models / "forward_log.jsonl",
    )
    assert rec["n_sessions"] == 0 and rec["logloss"] is None and rec["edge_significant"] is False


def test_forward_records_are_strict_json(bootstrapped, bars, monkeypatch):
    models = bootstrapped
    promoted = bars["SPY"].index[799] + pd.Timedelta(hours=20)
    monkeypatch.setattr(forward, "_promotion_time", lambda recs, v: promoted)
    as_of = (bars["SPY"].index[803] + pd.Timedelta(hours=20)).to_pydatetime()
    forward.run_monitor(bars, as_of=as_of, champion_dir=models / "champion", archive_dir=models / "archive",
                        gate_log_path=models / "gate_log.jsonl", forward_log_path=models / "forward_log.jsonl")
    text = (models / "forward_log.jsonl").read_text()
    json.loads(text.splitlines()[-1], parse_constant=lambda c: pytest.fail(f"non-strict JSON constant {c}"))
