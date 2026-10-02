"""The Analyst's alphagate promotion gate (analyst/gating.py).

Covers: the shared holdout (rows valid under every spec, paired samples
aligned), gate chronology (embargo=0 is correct because the 2-session
embargo is already in the training rows; holdout.end is the latest LABEL
end), the per-session log-loss metric, and the bootstrap / refresh /
experiment flows end to end on synthetic data, with every decision written
to a real JsonlSink and the champion only ever written on PROMOTE.
"""

from __future__ import annotations

import math
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

alphagate = pytest.importorskip("alphagate")
from alphagate import Candidate, Decision, ListSink, LookaheadError, gate, spend_alpha  # noqa: E402

from analyst_helpers import (  # noqa: E402
    make_predictable_universe_bars,
    weak_recipe_dict,
)
from wolfpack_worker.analyst import gating  # noqa: E402
from wolfpack_worker.analyst.dataset import EMBARGO_SESSIONS, build_dataset, session_dates, split_at  # noqa: E402
from wolfpack_worker.analyst.features import FEATURE_NAMES  # noqa: E402
from wolfpack_worker.analyst.gate_log import read_gate_log  # noqa: E402
from wolfpack_worker.analyst.model_io import load_champion, read_champion_artifact, write_champion  # noqa: E402
from wolfpack_worker.analyst.recipe import Recipe, load_v1_recipe  # noqa: E402
from wolfpack_worker.analyst.registration import Registration  # noqa: E402
from wolfpack_worker.analyst.train import build_manifest, model_version_for, run_training  # noqa: E402

N_FULL = 900
N_EARLY = 650


@pytest.fixture(scope="module")
def full_bars():
    return make_predictable_universe_bars(N_FULL, seed=1)


@pytest.fixture(scope="module")
def early_bars(full_bars):
    return {t: df.iloc[:N_EARLY] for t, df in full_bars.items()}


@pytest.fixture(scope="module")
def weak_recipe():
    return Recipe.from_dict(weak_recipe_dict())


@pytest.fixture(scope="module")
def weak_result(early_bars, weak_recipe):
    return run_training(early_bars, weak_recipe, walk_forward=False)


def _as_of(bars):
    return (bars["SPY"].index[-1] + pd.Timedelta(hours=20)).to_pydatetime()


@pytest.fixture()
def layout(tmp_path):
    models = tmp_path / "models"
    return {
        "champion_dir": models / "champion",
        "gate_log_path": models / "gate_log.jsonl",
        "archive_dir": models / "archive",
    }


@pytest.fixture()
def champion(layout, weak_result, early_bars, weak_recipe):
    """A bootstrapped weak champion: artifact written, then the one-time
    bootstrap gate run on its own holdout (as for the real v1)."""
    version = model_version_for(weak_result.model_bytes)
    manifest = build_manifest(weak_result, "run-weak", model_version=version, gate_record_ids=[],
                              trial_number=None, promoted_by="pre-gate")
    for k in ("recipe_id", "recipe", "trial_number", "gate_record_ids", "promoted_by"):
        manifest.pop(k)  # the real v1 manifest predates these keys
    write_champion(layout["champion_dir"], weak_result.model_bytes, manifest)
    rec = gating.run_bootstrap(
        early_bars, champion_dir=layout["champion_dir"], gate_log_path=layout["gate_log_path"],
        as_of=_as_of(early_bars), recipe=weak_recipe,
    )
    return rec


# --- shared holdout -------------------------------------------------------------------


def test_shared_holdout_keeps_only_rows_valid_under_every_spec(full_bars):
    ds = build_dataset(full_bars)
    start = gating.shared_cutoff([("v1", ds)])
    # A second "spec" that is missing some rows: one AAPL row and one whole session.
    sessions = session_dates(ds)
    drop_session = sessions[-100]
    drop_row = (sessions[-50], "AAPL")
    other = ds.loc[(ds["ts"] != drop_session) & ~((ds["ts"] == drop_row[0]) & (ds["ticker"] == drop_row[1]))]
    other = other.rename(columns={"r_1": "alt_r_1"})
    names = {"v1": list(FEATURE_NAMES), "alt": ["alt_r_1", "r_5"]}
    h = gating.build_shared_holdout([("v1", ds), ("alt", other)], start, names=names)
    data = h.data
    keys = data.keys
    assert drop_session not in set(keys["ts"])
    assert not ((keys["ts"] == drop_row[0]) & (keys["ticker"] == drop_row[1])).any()
    assert len(data.sessions) == 251  # 252 minus the dropped session
    assert h.n_samples == 251 and len(h.timestamps) == 251
    # Rows are aligned: v1's r_1 column equals alt's alt_r_1 column row by row.
    np.testing.assert_array_equal(data.X["v1"][:, 0], data.X["alt"][:, 0])
    # Chronological order, (ts, ticker) sorted.
    assert keys["ts"].is_monotonic_increasing


def test_shared_holdout_refuses_label_disagreement(full_bars):
    ds = build_dataset(full_bars)
    start = gating.shared_cutoff([("v1", ds)])
    bad = ds.copy()
    bad.loc[bad.index[-1], "y"] = 1 - bad.loc[bad.index[-1], "y"]
    with pytest.raises(gating.HoldoutAlignmentError):
        gating.build_shared_holdout([("v1", ds), ("v1", bad)], start)


def test_shared_cutoff_uses_aligned_sessions(full_bars):
    ds = build_dataset(full_bars)
    sessions = session_dates(ds)
    assert gating.shared_cutoff([("v1", ds)]) == sessions[-252]
    trimmed = ds.loc[ds["ts"] != sessions[-1]]
    # Dropping the last session from one dataset moves the shared cutoff back one.
    assert gating.shared_cutoff([("v1", ds), ("v1", trimmed)]) == sessions[-253]


# --- chronology ------------------------------------------------------------------------


def test_trained_through_is_exactly_one_session_before_test_start(full_bars):
    ds = build_dataset(full_bars)
    sessions = session_dates(ds)
    c = sessions[-252]
    split = split_at(ds, c)
    pos = sessions.get_loc(c)
    assert split.trained_through == sessions[pos - 1]
    # ...because the last training ROW is EMBARGO_SESSIONS + 1 sessions back.
    assert split.train["ts"].max() == sessions[pos - 1 - EMBARGO_SESSIONS]


def test_holdout_end_is_latest_label_end_not_latest_row(full_bars):
    ds = build_dataset(full_bars)
    h = gating.build_shared_holdout([("v1", ds)], gating.shared_cutoff([("v1", ds)]))
    assert pd.Timestamp(h.end) == ds["label_end_ts"].max()
    assert pd.Timestamp(h.end) > ds["ts"].max()  # two bars later
    assert pd.Timestamp(h.start) == gating.shared_cutoff([("v1", ds)])


def _gate_with(h, trained_through, as_of):
    model = gating.ConstantScorer(0.5)
    return gate(
        challenger=Candidate(model=model, id="c", trained_through=trained_through),
        champion=None, metric=gating.LOGLOSS_METRIC, holdout=h, sink=ListSink(),
        embargo=gating.EMBARGO, as_of=as_of,
    )


def test_gate_embargo_zero_accepts_the_real_split_and_rejects_overlap(full_bars):
    ds = build_dataset(full_bars)
    c = gating.shared_cutoff([("v1", ds)])
    h = gating.build_shared_holdout([("v1", ds)], c)
    split = split_at(ds, c)
    as_of = _as_of(full_bars)
    assert gating.EMBARGO == timedelta(0)
    _gate_with(h, split.trained_through.to_pydatetime(), as_of)  # gap = 1 session: OK
    with pytest.raises(LookaheadError):
        _gate_with(h, c.to_pydatetime(), as_of)  # trained through the holdout start


def test_gate_refuses_unmatured_labels(full_bars):
    ds = build_dataset(full_bars)
    c = gating.shared_cutoff([("v1", ds)])
    h = gating.build_shared_holdout([("v1", ds)], c)
    split = split_at(ds, c)
    # as_of between the last row's bar and its label end: labels not yet known.
    as_of = (ds["ts"].max() + pd.Timedelta(hours=20)).to_pydatetime()
    with pytest.raises(LookaheadError, match="as_of"):
        _gate_with(h, split.trained_through.to_pydatetime(), as_of)


def test_matured_drops_rows_whose_label_ends_after_as_of(full_bars):
    ds = build_dataset(full_bars)
    as_of = ds["label_end_ts"].max() - pd.Timedelta(days=3)
    m = gating.matured(ds, as_of)
    assert (m["label_end_ts"] <= as_of).all() and len(m) < len(ds)


# --- metric ------------------------------------------------------------------------------


def test_per_session_logloss_known_values():
    ts = pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"], utc=True)
    keys = pd.DataFrame({"ts": ts, "ticker": ["A", "B", "A"], "y": [1, 0, 1],
                         "label_end_ts": ts + pd.Timedelta(days=2)})
    data = gating.SharedHoldout(keys=keys, X={}, sessions=tuple(pd.DatetimeIndex(ts.unique())),
                                session_codes=np.array([0, 0, 1]))
    from alphagate import Holdout

    h = Holdout(data=data, start=ts[0].to_pydatetime(), end=ts[-1].to_pydatetime())
    r = gating.per_session_logloss(gating.ConstantScorer(0.8), h)
    s0 = (-math.log(0.8) - math.log(0.2)) / 2
    s1 = -math.log(0.8)
    assert list(r.samples) == pytest.approx([s0, s1])
    assert r.value == pytest.approx((s0 + s1) / 2)  # mean over sessions, not rows
    assert r.details["row_mean_logloss"] == pytest.approx((2 * s0 + s1) / 3)
    assert r.details["n_sessions"] == 2 and r.details["n_rows"] == 3
    assert r.details["baseline_logloss"] == pytest.approx(r.value)  # constant 0.8 IS its baseline
    assert r.details["accuracy"] == pytest.approx(2 / 3)


# --- bootstrap ---------------------------------------------------------------------------


def test_bootstrap_logs_one_honest_no_incumbent_promotion(champion, layout, weak_result):
    rec = champion
    assert rec.decision is Decision.PROMOTE and rec.reason_code == "no_incumbent"
    assert rec.champion_id is None and rec.chronology_checked and rec.embargo_seconds == 0
    assert rec.context["kind"] == "bootstrap"
    assert "after the fact" in rec.context["note"]
    log = read_gate_log(layout["gate_log_path"])
    assert [r["record_id"] for r in log] == [rec.record_id]
    champ = load_champion(layout["champion_dir"], gate_log_path=layout["gate_log_path"])
    assert champ.manifest["gate_record_ids"] == [rec.record_id]
    assert champ.manifest["recipe_id"] == weak_result.recipe.recipe_id
    assert champ.manifest["trial_number"] is None
    assert (layout["champion_dir"] / "model.json").read_bytes() == weak_result.model_bytes
    # It scored the model on its OWN holdout.
    assert pd.Timestamp(rec.holdout_start) == weak_result.test_start
    assert rec.challenger_score.value == pytest.approx(weak_result.test_metrics["logloss"], abs=1e-12)


def test_bootstrap_refuses_once_any_record_exists(champion, layout, early_bars, weak_recipe):
    with pytest.raises(gating.BootstrapError, match="already"):
        gating.run_bootstrap(early_bars, champion_dir=layout["champion_dir"],
                             gate_log_path=layout["gate_log_path"], as_of=_as_of(early_bars),
                             recipe=weak_recipe)


def test_bootstrap_refuses_if_holdout_numbers_do_not_reproduce(layout, weak_result, early_bars, weak_recipe):
    version = model_version_for(weak_result.model_bytes)
    manifest = build_manifest(weak_result, "r", model_version=version, gate_record_ids=[],
                              trial_number=None, promoted_by="x")
    manifest["test_metrics"]["logloss"] += 1e-6
    write_champion(layout["champion_dir"], weak_result.model_bytes, manifest)
    with pytest.raises(gating.BootstrapError, match="reproduce"):
        gating.run_bootstrap(early_bars, champion_dir=layout["champion_dir"],
                             gate_log_path=layout["gate_log_path"], as_of=_as_of(early_bars),
                             recipe=weak_recipe)
    assert not layout["gate_log_path"].exists()  # nothing logged


# --- refresh -----------------------------------------------------------------------------


def test_refresh_does_nothing_when_cutoff_would_not_advance_20_sessions(champion, layout, early_bars, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("refresh must not train when there is nothing to do")

    monkeypatch.setattr(gating, "run_training", boom)
    before = layout["gate_log_path"].read_text()
    out = gating.run_refresh(early_bars, as_of=_as_of(early_bars), **layout)
    assert out.status == "skipped" and out.advance_sessions == 0
    assert "nothing to do" in out.message
    assert layout["gate_log_path"].read_text() == before


def test_refresh_gates_non_inferiority_against_the_deployed_champion(champion, layout, full_bars):
    old = load_champion(layout["champion_dir"], gate_log_path=layout["gate_log_path"]).manifest
    out = gating.run_refresh(full_bars, as_of=_as_of(full_bars), **layout)
    assert out.status in ("promoted", "rejected")
    assert out.advance_sessions >= 20
    rec = out.record
    assert rec.context["kind"] == "refresh"
    assert rec.champion_id == old["model_version"]  # the DEPLOYED model, not a refit
    assert rec.comparator_name == "paired_dm"
    p = rec.comparator_params
    assert (p["mode"], p["margin"], p["alpha"], p["hac_lags"]) == ("non_inferiority", 0.002, 0.05, 5)
    assert rec.challenger_trained_through < rec.holdout_start
    assert rec.champion_trained_through < rec.holdout_start
    assert rec.embargo_seconds == 0
    assert rec.holdout_n_samples == 252 and len(rec.challenger_score.samples) == 252
    assert rec.challenger_metadata["recipe_id"] == old["recipe_id"]  # same recipe
    log = read_gate_log(layout["gate_log_path"])
    assert log[-1]["record_id"] == rec.record_id  # logged whatever the decision
    # This stump refit (row/column subsampling) is 0.0022 worse on average,
    # past the 0.002 margin, so non-inferiority is not shown: REJECT, and the
    # deployed champion is untouched.
    assert out.status == "rejected" and rec.reason_code == "inferior"
    assert rec.comparator_stats["mean_diff"] < -0.002
    now = load_champion(layout["champion_dir"], gate_log_path=layout["gate_log_path"]).manifest
    assert now["model_version"] == old["model_version"]
    assert not layout["archive_dir"].exists()


def test_refresh_stable_refit_is_non_inferior_and_promoted(layout, full_bars):
    # A deterministic stump (no subsampling) refit with 30 more sessions makes
    # almost the same predictions, so non-inferiority is demonstrable.
    d = weak_recipe_dict()
    d["xgb_params"] = {**d["xgb_params"], "subsample": 1.0, "colsample_bytree": 1.0}
    recipe = Recipe.from_dict(d)
    early = {t: df.iloc[:870] for t, df in full_bars.items()}
    res = run_training(early, recipe, walk_forward=False)
    manifest = build_manifest(res, "r", model_version=model_version_for(res.model_bytes), gate_record_ids=[],
                              trial_number=None, promoted_by="x")
    for k in ("recipe_id", "recipe", "trial_number", "gate_record_ids", "promoted_by"):
        manifest.pop(k)
    write_champion(layout["champion_dir"], res.model_bytes, manifest)
    gating.run_bootstrap(early, champion_dir=layout["champion_dir"], gate_log_path=layout["gate_log_path"],
                         as_of=_as_of(early), recipe=recipe)
    old = load_champion(layout["champion_dir"], gate_log_path=layout["gate_log_path"]).manifest

    out = gating.run_refresh(full_bars, as_of=_as_of(full_bars), **layout)
    rec = out.record
    assert out.status == "promoted" and out.advance_sessions == 30
    assert rec.reason_code == "non_inferior"
    now = load_champion(layout["champion_dir"], gate_log_path=layout["gate_log_path"]).manifest
    assert now["model_version"] == rec.challenger_id != old["model_version"]
    assert now["gate_record_ids"] == [rec.record_id]
    assert now["recipe_id"] == old["recipe_id"] and now["promoted_by"] == "refresh"
    assert pd.Timestamp(now["test_start"]) == pd.Timestamp(rec.holdout_start)
    assert (layout["archive_dir"] / old["model_version"] / "model.json").is_file()
    # The archived champion still loads (it has its own PROMOTE record).
    load_champion(layout["archive_dir"] / old["model_version"], gate_log_path=layout["gate_log_path"])


# --- experiment ----------------------------------------------------------------------------


def _registration(recipe: Recipe, n: int = 1, hypothesis: str = "h") -> Registration:
    import datetime as dt

    return Registration(path=Path(f"/nonexistent/{n:03d}-x.toml"), trial_number=n,
                        registered=dt.date(2026, 10, 2), hypothesis=hypothesis, recipe=recipe,
                        abandoned=None)


def test_experiment_better_recipe_passes_both_gates_and_is_promoted(champion, layout, full_bars):
    old = load_champion(layout["champion_dir"], gate_log_path=layout["gate_log_path"]).manifest
    reg = _registration(load_v1_recipe(), n=2, hypothesis="Full v1 settings learn the AR signal.")
    out = gating.run_experiment(full_bars, reg, k=3, as_of=_as_of(full_bars), git_commit="abc", **layout)
    vs_champ, floor = out.records
    assert vs_champ.context["role"] == "vs_champion_refit" and floor.context["role"] == "floor_vs_base_rate"
    # Same holdout, same challenger, same cutoff for every model.
    assert vs_champ.holdout_fingerprint == floor.holdout_fingerprint
    assert vs_champ.challenger_id == floor.challenger_id
    assert vs_champ.challenger_trained_through == vs_champ.champion_trained_through  # refit at same C
    assert vs_champ.champion_id.startswith("refit-" + old["recipe_id"])
    assert vs_champ.champion_metadata["recipe_id"] == old["recipe_id"]
    # Multiple-testing level: alpha_k with k = number of registrations.
    assert vs_champ.context["k"] == 3 and vs_champ.context["trial_number"] == 2
    assert vs_champ.comparator_params["alpha"] == pytest.approx(spend_alpha(0.05, 3))
    assert vs_champ.context["alpha_k"] == pytest.approx(spend_alpha(0.05, 3))
    assert vs_champ.comparator_params["mode"] == "superiority" and vs_champ.comparator_params["hac_lags"] == 5
    assert floor.comparator_name == "margin" and floor.comparator_params["min_delta"] == 0.0
    edge = vs_champ.context["edge_vs_baseline"]
    assert vs_champ.context["edge_vs_baseline_significant"] == (edge["p"] < vs_champ.context["alpha_k"])
    # On data with real signal, the full recipe beats a 1-round stump.
    assert vs_champ.promoted and vs_champ.reason_code == "significant_improvement"
    assert floor.promoted
    assert out.promoted
    champ = load_champion(layout["champion_dir"], gate_log_path=layout["gate_log_path"])
    m = champ.manifest
    assert m["model_version"] == vs_champ.challenger_id
    assert m["gate_record_ids"] == [vs_champ.record_id, floor.record_id]
    assert m["trial_number"] == 2 and m["recipe_id"] == load_v1_recipe().recipe_id
    # The deployed artifact is the EVALUATED model (not refit).
    assert m["model_sha256"] == vs_champ.challenger_metadata["model_sha256"]
    assert (layout["archive_dir"] / old["model_version"] / "manifest.json").is_file()


def test_experiment_no_better_recipe_is_rejected_and_both_records_are_logged(champion, layout, full_bars):
    old = load_champion(layout["champion_dir"], gate_log_path=layout["gate_log_path"]).manifest
    # A different recipe (different recipe_id) that makes IDENTICAL predictions
    # to the champion's recipe: `verbosity` only changes logging. Its
    # per-session losses equal the refit's exactly, so there is nothing to
    # find. (Note: a stump with a different seed is NOT a valid "no better"
    # case on this synthetic data: it can split on the predictive feature and
    # be genuinely, consistently, a little better, which the test detects.)
    d = weak_recipe_dict()
    d["xgb_params"] = {**d["xgb_params"], "verbosity": 0}
    same_ish = Recipe.from_dict(d)
    assert same_ish.recipe_id != Recipe.from_dict(weak_recipe_dict()).recipe_id
    n_before = len(read_gate_log(layout["gate_log_path"]))
    out = gating.run_experiment(full_bars, _registration(same_ish), k=1, as_of=_as_of(full_bars),
                                git_commit="abc", **layout)
    vs_champ, floor = out.records
    assert not vs_champ.promoted and vs_champ.reason_code == "not_significant"
    assert vs_champ.comparator_stats["mean_diff"] == 0.0 and vs_champ.comparator_stats["p"] == 1.0
    assert not out.promoted
    log = read_gate_log(layout["gate_log_path"])
    assert len(log) == n_before + 2  # both always logged, including the loser
    assert {r["record_id"] for r in log[-2:]} == {vs_champ.record_id, floor.record_id}
    assert log[-2]["explanation"]  # why it lost, in words
    now = load_champion(layout["champion_dir"], gate_log_path=layout["gate_log_path"]).manifest
    assert now["model_version"] == old["model_version"]


def test_experiment_refuses_the_champions_own_recipe(champion, layout, full_bars, weak_recipe):
    with pytest.raises(ValueError, match="champion's own recipe"):
        gating.run_experiment(full_bars, _registration(weak_recipe), k=1, as_of=_as_of(full_bars),
                              git_commit="abc", **layout)


# --- promotion requires every gate call to PROMOTE ----------------------------------------


def test_promote_from_gate_requires_all_records_promote(champion, layout, weak_result):
    champ_m = read_champion_artifact(layout["champion_dir"]).manifest
    good = read_gate_log(layout["gate_log_path"])[0]

    class R:  # minimal GateRecord stand-in
        def __init__(self, decision, cid, sha, rid):
            self.decision, self.challenger_id, self.record_id = decision, cid, rid
            self.challenger_metadata = {"model_sha256": sha}

        @property
        def promoted(self):
            return self.decision is Decision.PROMOTE

    sha = champ_m["model_sha256"]
    v = champ_m["model_version"]
    manifest = {**champ_m, "gate_record_ids": ["a", "b"]}
    with pytest.raises(gating.PromotionError, match="REJECT"):
        gating.promote_from_gate(
            [R(Decision.PROMOTE, v, sha, "a"), R(Decision.REJECT, v, sha, "b")],
            model_bytes=weak_result.model_bytes, manifest=manifest, **layout,
        )
    with pytest.raises(gating.PromotionError, match="challenger_id"):
        gating.promote_from_gate(
            [R(Decision.PROMOTE, "someone-else", sha, "a")],
            model_bytes=weak_result.model_bytes, manifest={**champ_m, "gate_record_ids": ["a"]}, **layout,
        )
    with pytest.raises(gating.PromotionError, match="no gate records"):
        gating.promote_from_gate([], model_bytes=weak_result.model_bytes, manifest=champ_m, **layout)
    assert good["decision"] == "promote"
