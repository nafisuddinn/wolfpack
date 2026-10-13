"""PersonaPaths parameterization of the gate-governance modules (Decision Log
2026-10-04: one gate implementation for The Analyst and The Scout).

The Analyst's behaviour must be unchanged (its own tests are untouched and
still pass); these tests pin that the Analyst defaults are exactly the old
locations, and that a second persona really is isolated: its own
experiments directory and trial count k, its own recipe parser, its own
feature-spec registry, its own gate log and generated-table markers.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import xgboost as xgb

from analyst_helpers import append_jsonl, gate_record, write_registration
from wolfpack_worker.analyst import log_guard, render_history
from wolfpack_worker.analyst.forward import FORWARD_LOG_PATH
from wolfpack_worker.analyst.model_io import (
    DEFAULT_CHAMPION_DIR,
    GATE_LOG_PATH,
    MODELS_DIR,
    ModelIntegrityError,
    load_champion,
    sha256_bytes,
    write_champion,
)
from wolfpack_worker.analyst.paths import ANALYST_PATHS, REPO_ROOT, WORKER_ROOT, PersonaPaths
from wolfpack_worker.analyst.recipe import Recipe, RecipeError, load_v1_recipe
from wolfpack_worker.analyst.registration import (
    EXPERIMENTS_DIR,
    RegistrationError,
    list_registrations,
    parse_registration,
    trial_count,
)


def test_analyst_paths_are_the_original_locations():
    assert MODELS_DIR == WORKER_ROOT / "models" / "analyst"
    assert DEFAULT_CHAMPION_DIR == MODELS_DIR / "champion" == ANALYST_PATHS.champion_dir
    assert GATE_LOG_PATH == MODELS_DIR / "gate_log.jsonl" == ANALYST_PATHS.gate_log_path
    assert FORWARD_LOG_PATH == MODELS_DIR / "forward_log.jsonl" == ANALYST_PATHS.forward_log_path
    assert EXPERIMENTS_DIR == REPO_ROOT / "worker" / "experiments" / "analyst" == ANALYST_PATHS.experiments_dir
    assert render_history.MODEL_CARD_PATH == REPO_ROOT / "MODEL_CARD.md" == ANALYST_PATHS.model_card_path


def test_analyst_markers_are_byte_identical_to_the_committed_card():
    card = (REPO_ROOT / "MODEL_CARD.md").read_text()
    assert render_history.BEGIN == (
        "<!-- BEGIN GENERATED: analyst-gate-history. Written by "
        "worker/src/wolfpack_worker/analyst/render_history.py from gate_log.jsonl + forward_log.jsonl; "
        "do not edit by hand. -->"
    )
    assert render_history.END == "<!-- END GENERATED: analyst-gate-history -->"
    assert render_history.has_markers(card)


def _other_persona(tmp_path: Path, *, parser=None, lookup=None) -> PersonaPaths:
    return PersonaPaths(
        persona="other",
        display_name="The Other",
        models_dir=tmp_path / "models" / "other",
        experiments_dir=tmp_path / "experiments" / "other",
        model_card_path=tmp_path / "CARD.md",
        recipe_parser=parser or ANALYST_PATHS.recipe_parser,
        feature_spec_lookup=lookup or ANALYST_PATHS.feature_spec_lookup,
    )


def test_log_relpaths_are_repo_relative():
    assert ANALYST_PATHS.log_relpaths() == (
        "worker/models/analyst/gate_log.jsonl",
        "worker/models/analyst/forward_log.jsonl",
    )


def test_log_guard_covers_every_personas_logs():
    """log_guard keeps a literal list (stdlib-only for CI); it must not miss
    any gated persona's gate/forward log."""
    from wolfpack_worker.scout.paths import SCOUT_PATHS

    for paths in (ANALYST_PATHS, SCOUT_PATHS):
        for rel in paths.log_relpaths():
            assert rel in log_guard.LOG_PATHS, rel


def test_registration_uses_the_personas_parser(tmp_path):
    seen = []

    def parser(d):
        seen.append(dict(d))
        if d.get("feature_spec_version") != "v1":
            raise RecipeError("not mine")
        return Recipe.from_dict(d)

    other = _other_persona(tmp_path, parser=parser)
    p = write_registration(other.experiments_dir, 1, load_v1_recipe().to_dict() | {"num_boost_round": 7})
    reg = parse_registration(p, paths=other)
    assert reg.recipe.num_boost_round == 7 and len(seen) == 1

    bad = load_v1_recipe().to_dict() | {"feature_spec_version": "zzz"}
    p2 = write_registration(tmp_path / "bad", 1, bad)
    with pytest.raises(RegistrationError, match="not mine"):
        parse_registration(p2, paths=other)


def test_each_persona_has_its_own_trial_count(tmp_path):
    other = _other_persona(tmp_path)
    for n in (1, 2, 3):
        write_registration(other.experiments_dir, n, load_v1_recipe().to_dict() | {"num_boost_round": n})
    # Defaults resolve to the persona's own directory, not The Analyst's.
    assert len(list_registrations(paths=other)) == 3
    assert trial_count(None, [], paths=other) == 3
    assert trial_count(None, [], paths=ANALYST_PATHS) == trial_count(EXPERIMENTS_DIR, [])


def test_load_champion_uses_the_personas_spec_registry_and_files(tmp_path):
    from wolfpack_worker.analyst.features import FEATURE_NAMES

    lookups = []

    class Spec:
        version = "other_v1"
        names = FEATURE_NAMES

    def lookup(v):
        lookups.append(v)
        if v != "other_v1":
            raise KeyError(v)
        return Spec

    other = _other_persona(tmp_path, lookup=lookup)
    import numpy as np

    d = xgb.DMatrix(np.random.default_rng(0).normal(size=(50, len(FEATURE_NAMES))),
                    label=np.arange(50) % 2, feature_names=list(FEATURE_NAMES))
    model = bytes(xgb.train({"objective": "binary:logistic", "nthread": 1}, d, 2).save_raw("json"))
    manifest = {
        "model_version": "other-1", "mlflow_run_id": "x", "trained_through": "2020-01-01T00:00:00+00:00",
        "feature_spec_version": "other_v1", "feature_names": list(FEATURE_NAMES),
        "xgboost_version": xgb.__version__, "test_metrics": {}, "recipe_id": "r", "gate_record_ids": ["rec-1"],
    }
    write_champion(other.champion_dir, model, manifest)
    # No gate log yet: refused, and the message names the persona.
    with pytest.raises(ModelIntegrityError, match="gate log"):
        load_champion(paths=other)
    append_jsonl(other.gate_log_path, gate_record("other-1", sha256_bytes(model), record_id="rec-1"))
    champ = load_champion(paths=other)
    assert champ.spec is Spec and "other_v1" in lookups

    # The Analyst's registry does not know other_v1.
    with pytest.raises(ModelIntegrityError, match="not a registered feature spec"):
        load_champion(other.champion_dir, gate_log_path=other.gate_log_path)

    missing = replace(other, models_dir=tmp_path / "nothing")
    with pytest.raises(ModelIntegrityError, match="The Other's champion file is missing"):
        load_champion(paths=missing)


def test_render_markers_and_splice_are_per_persona(tmp_path):
    other = _other_persona(tmp_path)
    b, e = render_history.markers(other)
    assert "other-gate-history" in b and "other-gate-history" in e
    card = f"x\n{b}\nold\n{e}\ny\n"
    assert render_history.splice(card, "T", paths=other) == f"x\n{b}\nT\n{e}\ny\n"
    # The Analyst's markers are not the other persona's.
    with pytest.raises(ValueError, match="marker"):
        render_history.splice(card, "T")
    assert not render_history.has_markers(card)
