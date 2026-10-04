"""The Scout's gate-or-fallback trial (scout/gating.py) + training diagnostics.

Synthetic data only. `signal_bars` plants a real relationship (the headline
tone of session t predicts the sign of ln(O_{t+2}/O_{t+1})) so a test can
watch a genuinely better model pass the gate; with tone independent of
returns the gate must reject and leave no champion. Real headlines have
nothing like the planted relationship.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("alphagate")
pytest.importorskip("sklearn")


from analyst_helpers import write_registration  # noqa: E402
from scout_helpers import SECRET, synthetic  # noqa: E402
from wolfpack_worker.analyst.gate_log import read_gate_log  # noqa: E402
from wolfpack_worker.analyst.model_io import load_champion  # noqa: E402
from wolfpack_worker.analyst.registration import RegistrationError, parse_registration  # noqa: E402
from wolfpack_worker.scout import gating as G  # noqa: E402
from wolfpack_worker.scout.dataset import build_scout_dataset  # noqa: E402
from wolfpack_worker.scout.paths import SCOUT_PATHS  # noqa: E402
from wolfpack_worker.scout.recipe import scout_v1_recipe_dict  # noqa: E402
from wolfpack_worker.scout.train import rule_backtest, rule_signal, run_scout_training  # noqa: E402

@pytest.fixture()
def scout_paths(tmp_path):
    return replace(SCOUT_PATHS, models_dir=tmp_path / "models" / "scout",
                   experiments_dir=tmp_path / "experiments" / "scout", model_card_path=tmp_path / "CARD.md")


def _register(paths, n=1, **over):
    d = scout_v1_recipe_dict()
    d["xgb_params"] = {**d["xgb_params"], **over}
    return parse_registration(write_registration(paths.experiments_dir, n, d, hypothesis="h"), paths=paths)


CFG = G.ScoutGateConfig(alpha_total=0.05, source="test.toml")


def _trial(paths, signal, seed=0, cfg=CFG, reg=None):
    sessions, bars, arts = synthetic(signal, seed)
    ds = build_scout_dataset(bars, arts, sessions)
    reg = reg or _register(paths)
    return G.run_scout_trial(ds, reg, as_of=sessions[-1].close, git_commit="test", paths=paths, gate_config=cfg), ds


def test_real_signal_promotes_and_the_champion_loads_through_the_backstop(scout_paths):
    out, _ = _trial(scout_paths, signal=True)
    rec = out.record
    assert out.promoted and rec.reason_code == "significant_improvement"
    assert rec.context["role"] == "vs_base_rate" and rec.context["kind"] == "trial"
    assert out.k == 1 and rec.context["alpha_k"] == pytest.approx(0.025) and rec.context["alpha_total"] == 0.05
    assert rec.comparator_params["alpha"] == pytest.approx(0.025)
    assert rec.comparator_params["margin"] == 0.0005 and rec.comparator_params["hac_lags"] == 5
    assert rec.comparator_params["mode"] == "superiority"
    assert rec.chronology_checked and rec.challenger_trained_through < rec.holdout_start
    assert rec.holdout_n_samples == 252
    champ = load_champion(paths=scout_paths)
    assert champ.manifest["model_version"].startswith("scout-") and champ.spec.version == "scout_v1"
    assert champ.manifest["gate_record_ids"] == [rec.record_id]


def test_no_signal_is_rejected_logged_and_leaves_no_champion(scout_paths):
    out, _ = _trial(scout_paths, signal=False, seed=3)
    assert not out.promoted and out.record.reason_code == "not_significant"
    assert not (scout_paths.champion_dir / "manifest.json").exists()
    (logged,) = read_gate_log(scout_paths.gate_log_path)
    assert logged["decision"] == "reject" and logged["record_id"] == out.record.record_id
    assert "rule_fallback" in logged["context"]["note"]


def test_a_registration_runs_once(scout_paths):
    reg = _register(scout_paths)
    _trial(scout_paths, signal=False, seed=3, reg=reg)
    with pytest.raises(RegistrationError, match="already been run"):
        _trial(scout_paths, signal=False, seed=3, reg=reg)


def test_second_trial_gets_k2_and_a_smaller_alpha(scout_paths):
    _trial(scout_paths, signal=False, seed=3)
    reg2 = _register(scout_paths, n=2, max_depth=2)
    out, _ = _trial(scout_paths, signal=False, seed=3, reg=reg2)
    assert out.k == 2 and out.alpha_k == pytest.approx(0.05 / 6)


def test_alpha_budget_cannot_change_after_a_trial(scout_paths):
    _trial(scout_paths, signal=False, seed=3)
    reg2 = _register(scout_paths, n=2, max_depth=2)
    with pytest.raises(G.GateConfigError, match="retroactive"):
        _trial(scout_paths, signal=False, seed=3, reg=reg2, cfg=G.ScoutGateConfig(0.015, "ledger.toml"))


def test_trial_against_an_existing_champion_is_refused(scout_paths):
    _trial(scout_paths, signal=True)
    reg2 = _register(scout_paths, n=2, max_depth=2)
    with pytest.raises(NotImplementedError, match="champion"):
        _trial(scout_paths, signal=True, reg=reg2)


def test_scout_trial_count_is_independent_of_the_analysts(scout_paths, tmp_path):
    from analyst_helpers import weak_recipe_dict

    write_registration(tmp_path / "experiments" / "analyst", 1, weak_recipe_dict())
    write_registration(tmp_path / "experiments" / "analyst", 2, weak_recipe_dict(seed=9))
    out, _ = _trial(scout_paths, signal=False, seed=3)
    assert out.k == 1


def test_committed_gate_config_is_the_confirmed_design_value():
    cfg = G.ScoutGateConfig.load()
    assert cfg.alpha_total == 0.05  # spend_alpha(0.05, 1) = 0.025 (design section 6)


def test_gate_config_validation(tmp_path):
    p = tmp_path / "g.toml"
    p.write_text("alpha_total = 1.5\n")
    with pytest.raises(G.GateConfigError):
        G.ScoutGateConfig.load(p)
    p.write_text("alpha_total = 0.05\nmargin = 0.1\n")
    with pytest.raises(G.GateConfigError, match="unknown"):
        G.ScoutGateConfig.load(p)


def test_no_headline_text_in_records_manifest_or_predictions(scout_paths):
    out, _ = _trial(scout_paths, signal=True)
    blobs = [scout_paths.gate_log_path.read_text(), (scout_paths.champion_dir / "manifest.json").read_text(),
             out.result.test_predictions.to_csv(), json.dumps(out.result.extra, default=str),
             json.dumps(out.result.walk_forward, default=str)]
    for b in blobs:
        assert SECRET not in b and "example.invalid" not in b


# --- training diagnostics ----------------------------------------------------------------------


def test_training_split_is_chronological_with_embargo():
    sessions, bars, arts = synthetic(signal=False, seed=1)
    ds = build_scout_dataset(bars, arts, sessions)
    res = run_scout_training(ds, _recipe(), walk_forward=False)
    assert res.train["ts"].max() < res.test["ts"].min()
    assert res.train["label_end_ts"].max() < res.test_start == res.test["ts"].min()
    assert res.trained_through == res.train["label_end_ts"].max()
    gap = pd.DatetimeIndex(ds["ts"].unique()).sort_values()
    pos = gap.searchsorted(res.test_start)
    assert res.train["ts"].max() < gap[pos - 2]  # 2-session embargo before the holdout
    assert res.test["ts"].nunique() == 252


def _recipe():
    from wolfpack_worker.scout.recipe import parse_scout_recipe

    return parse_scout_recipe(scout_v1_recipe_dict())


def test_diagnostics_are_reported():
    sessions, bars, arts = synthetic(signal=True, seed=2)
    ds = build_scout_dataset(bars, arts, sessions)
    flag = pd.Series(False, index=ds.index)
    res = run_scout_training(ds, _recipe(), walk_forward=False, revised_flag=flag)
    d = res.extra["diagnostics"]
    assert d["by_has_news_1"]["share_rows_with_news"] == 1.0
    assert d["rule_baseline"]["n_fired"] == len(res.test)
    assert d["by_revised_after_decision"]["share_rows_flagged"] == 0.0
    assert d["stress_window"] is None  # synthetic data has no 2020


def test_rule_signal_is_the_untuned_vader_band():
    s = np.array([0.06, 0.05, 0.0, -0.05, -0.051, 0.9])
    out = rule_signal(s)
    assert out[0] == 1.0 and np.isnan(out[1]) and np.isnan(out[2]) and np.isnan(out[3]) and out[4] == 0.0
    assert out[5] == 1.0


def test_rule_backtest_holds_inside_the_band():
    ts = pd.DatetimeIndex(pd.date_range("2025-01-01", periods=4, tz="UTC"))
    rows = pd.DataFrame({"ts": ts, "ticker": "AAPL", "s_mean_1": [0.5, 0.0, -0.5, 0.0], "y": [1, 1, 0, 0],
                         "fwd_logret": [0.01, 0.02, -0.03, 0.04]})
    bt = rule_backtest(rows, cost_bps_per_side=0.0)
    # positions: long, hold(long), flat, hold(flat)
    assert bt["per_ticker"]["AAPL"]["gross_logret"] == pytest.approx(0.03)
    assert bt["n_fired"] == 2 and bt["accuracy_when_fired"] == 1.0 and bt["per_ticker"]["AAPL"]["flips"] == 2


def test_render_shows_a_no_incumbent_trial_honestly(scout_paths):
    from wolfpack_worker.analyst import render_history as rh

    _trial(scout_paths, signal=False, seed=3)
    table = rh.render_table(read_gate_log(scout_paths.gate_log_path), [], n_registered=1, paths=scout_paths)
    row = next(line for line in table.splitlines() if "trial #1" in line)
    assert "n/a (no champion)" in row and "vs base rate" in row
    assert "REJECT (vs base rate: not_significant)" in row and "no detectable change" in row
    assert "n/a (not deployed)" in row
    assert "while no champion exists, vs the base rate" in table
    # The Analyst's legend is untouched.
    assert "while no champion exists" not in rh.render_table([], [], n_registered=0)
