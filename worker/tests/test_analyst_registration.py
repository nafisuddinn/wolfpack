"""Experiment pre-registration: parsing, trial counting, and the refusals that
stop a recipe from being tried on the gate holdout without being counted.

The git checks run against a throwaway repo in tmp_path, never this one.
"""

from __future__ import annotations

import subprocess
from datetime import date
from pathlib import Path

import pytest

from analyst_helpers import append_jsonl, gate_record
from wolfpack_worker.analyst.recipe import V1_RECIPE_PATH, load_v1_recipe
from wolfpack_worker.analyst.registration import (
    EXPERIMENTS_DIR,
    REPO_ROOT,
    RegistrationError,
    CLEAN_PATHS,
    count_trials,
    file_sha256,
    list_registrations,
    logged_trials,
    parse_registration,
    preflight,
    trial_count,
)


def _recipe_block(**overrides) -> str:
    lines = []
    for line in V1_RECIPE_PATH.read_text().splitlines():
        if line.startswith("#"):
            continue
        if line.strip() == "[xgb_params]":
            line = "[recipe.xgb_params]"
        for k, v in overrides.items():
            if line.startswith(f"{k} ="):
                line = f"{k} = {v}"
        lines.append(line)
    return "[recipe]\n" + "\n".join(lines) + "\n"


def registration_text(n: int, *, hypothesis="Deeper trees capture interactions.", extra="", **overrides) -> str:
    return (
        f"trial_number = {n}\n"
        "registered = 2026-10-02\n"
        f'hypothesis = "{hypothesis}"\n'
        f"{extra}\n" + _recipe_block(**overrides)
    )


# --- parsing ------------------------------------------------------------------------


def test_parse_valid_registration(tmp_path):
    p = tmp_path / "001-deeper-trees.toml"
    p.write_text(registration_text(1, max_depth=4))
    reg = parse_registration(p)
    assert reg.trial_number == 1
    assert reg.registered == date(2026, 10, 2)
    assert reg.hypothesis == "Deeper trees capture interactions."
    assert reg.recipe.xgb_params["max_depth"] == 4
    assert reg.recipe.recipe_id != load_v1_recipe().recipe_id
    assert reg.abandoned is None


@pytest.mark.parametrize(
    "name,text,match",
    [
        ("001-x.toml", registration_text(2), "trial_number"),
        ("1-x.toml", registration_text(1), "file name"),
        ("001_x.toml", registration_text(1), "file name"),
        ("001-x.toml", registration_text(1, hypothesis=""), "hypothesis"),
        ("001-x.toml", 'trial_number = 1\nregistered = 2026-10-02\nhypothesis = "h"\n', "recipe"),
        ("001-x.toml", registration_text(1).replace("registered = 2026-10-02", 'registered = "soon"'), "registered"),
        ("001-x.toml", registration_text(1, extra="grid = [1, 2]"), "unknown"),
    ],
)
def test_parse_rejects_bad_registrations(tmp_path, name, text, match):
    p = tmp_path / name
    p.write_text(text)
    with pytest.raises(RegistrationError, match=match):
        parse_registration(p)


def test_multiline_hypothesis_rejected(tmp_path):
    p = tmp_path / "001-x.toml"
    p.write_text(registration_text(1).replace('hypothesis = "Deeper trees capture interactions."',
                                              'hypothesis = """two\nlines"""'))
    with pytest.raises(RegistrationError, match="one line"):
        parse_registration(p)


def test_abandoned_line_is_parsed(tmp_path):
    p = tmp_path / "001-x.toml"
    p.write_text(registration_text(1, extra='abandoned = "data bug found before running"'))
    assert parse_registration(p).abandoned == "data bug found before running"


# --- trial counting ------------------------------------------------------------------


def test_count_trials_counts_registration_files_not_runs(tmp_path):
    assert count_trials(tmp_path) == 0
    (tmp_path / "README.md").write_text("protocol")
    (tmp_path / "001-a.toml").write_text(registration_text(1, max_depth=4))
    (tmp_path / "002-b.toml").write_text(registration_text(2, max_depth=5, extra='abandoned = "x"'))
    assert count_trials(tmp_path) == 2  # abandoned still counts
    assert [r.trial_number for r in list_registrations(tmp_path)] == [1, 2]


def test_trial_numbers_must_be_contiguous_and_unique(tmp_path):
    (tmp_path / "001-a.toml").write_text(registration_text(1, max_depth=4))
    (tmp_path / "003-c.toml").write_text(registration_text(3, max_depth=5))
    with pytest.raises(RegistrationError, match="1..2"):
        count_trials(tmp_path)


def test_duplicate_recipe_across_registrations_rejected(tmp_path):
    (tmp_path / "001-a.toml").write_text(registration_text(1, max_depth=4))
    (tmp_path / "002-b.toml").write_text(registration_text(2, max_depth=4, hypothesis="Same again."))
    with pytest.raises(RegistrationError, match="same recipe"):
        count_trials(tmp_path)


def test_stray_toml_with_bad_name_is_an_error_not_ignored(tmp_path):
    (tmp_path / "001-a.toml").write_text(registration_text(1, max_depth=4))
    (tmp_path / "draft.toml").write_text(registration_text(2, max_depth=5))
    with pytest.raises(RegistrationError, match="file name"):
        count_trials(tmp_path)


# --- preflight against a throwaway git repo ------------------------------------------


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false",
         *args],
        cwd=repo, check=True, capture_output=True,
    )


@pytest.fixture()
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "worker/src/pkg").mkdir(parents=True)
    (root / "worker/src/pkg/mod.py").write_text("x = 1\n")
    (root / "worker/recipes/analyst").mkdir(parents=True)
    (root / "worker/recipes/analyst/v1.toml").write_text(V1_RECIPE_PATH.read_text())
    (root / "worker/pyproject.toml").write_text("[project]\nname='w'\n")
    (root / "worker/uv.lock").write_text("lock\n")
    exp = root / "worker/experiments/analyst"
    exp.mkdir(parents=True)
    (exp / "README.md").write_text("protocol\n")
    (root / "worker/models/analyst").mkdir(parents=True)
    (root / "worker/models/analyst/gate_log.jsonl").write_text("")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "init")
    return root


def _register(repo: Path, n: int = 1, commit: bool = True, **overrides) -> Path:
    p = repo / f"worker/experiments/analyst/{n:03d}-deeper.toml"
    p.write_text(registration_text(n, **(overrides or {"max_depth": 4})))
    if commit:
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", f"register trial {n}")
    return p


def _preflight(repo: Path, path: Path, records=()):
    return preflight(path, repo_root=repo, experiments_dir=repo / "worker/experiments/analyst",
                     gate_records=list(records))


def test_preflight_passes_on_committed_clean_unevaluated(repo):
    p = _register(repo)
    reg, k = _preflight(repo, p)
    assert reg.trial_number == 1 and k == 1


def test_preflight_refuses_uncommitted_registration(repo):
    p = _register(repo, commit=False)
    with pytest.raises(RegistrationError, match="not committed"):
        _preflight(repo, p)
    _git(repo, "add", "-A")  # staged but not committed is still not registered
    with pytest.raises(RegistrationError, match="not committed|uncommitted"):
        _preflight(repo, p)


def test_preflight_refuses_registration_edited_after_commit(repo):
    p = _register(repo)
    p.write_text(p.read_text().replace("max_depth = 4", "max_depth = 6"))
    with pytest.raises(RegistrationError, match="uncommitted"):
        _preflight(repo, p)


def test_preflight_refuses_dirty_src(repo):
    p = _register(repo)
    (repo / "worker/src/pkg/mod.py").write_text("x = 2\n")
    with pytest.raises(RegistrationError, match="worker/src"):
        _preflight(repo, p)


def test_preflight_refuses_untracked_file_in_src(repo):
    p = _register(repo)
    (repo / "worker/src/pkg/new.py").write_text("y = 1\n")
    with pytest.raises(RegistrationError, match="worker/src"):
        _preflight(repo, p)


def test_preflight_refuses_uncommitted_other_registration(repo):
    p = _register(repo, 1)
    (repo / "worker/experiments/analyst/002-other.toml").write_text(registration_text(2, max_depth=5))
    with pytest.raises(RegistrationError, match="experiments"):
        _preflight(repo, p)


def test_preflight_refuses_already_evaluated_recipe(repo):
    p = _register(repo)
    rid = parse_registration(p).recipe.recipe_id
    rec = gate_record("analyst-x", "0" * 64)
    rec["challenger_metadata"]["recipe_id"] = rid
    with pytest.raises(RegistrationError, match="already been evaluated"):
        _preflight(repo, p, [rec])


def test_preflight_refuses_v1_recipe_as_a_new_trial(repo):
    p = _register(repo, max_depth=3)  # == v1 recipe
    assert parse_registration(p).recipe == load_v1_recipe()
    rec = gate_record("analyst-20261001-7e2cc1bc", "0" * 64)
    rec["challenger_metadata"]["recipe_id"] = load_v1_recipe().recipe_id
    with pytest.raises(RegistrationError, match="already been evaluated"):
        _preflight(repo, p, [rec])


def test_preflight_refuses_abandoned(repo):
    p = _register(repo, extra='abandoned = "found a bug"', max_depth=4)
    with pytest.raises(RegistrationError, match="abandoned"):
        _preflight(repo, p)


def test_preflight_refuses_file_outside_experiments_dir(repo):
    p = repo / "worker/001-deeper.toml"
    p.write_text(registration_text(1, max_depth=4))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "x")
    with pytest.raises(RegistrationError, match="experiments"):
        _preflight(repo, p)


def test_k_counts_every_registration_file(repo):
    _register(repo, 1, max_depth=4)
    p2 = _register(repo, 2, max_depth=5)
    _, k = _preflight(repo, p2)
    assert k == 2


# --- the real repo's registrations -----------------------------------------------------


def test_real_experiments_dir_is_where_preflight_looks():
    assert EXPERIMENTS_DIR == REPO_ROOT / "worker" / "experiments" / "analyst"
    assert EXPERIMENTS_DIR.is_dir()
    assert (EXPERIMENTS_DIR / "README.md").is_file()


# --- trial-count integrity against the gate log (tester findings 1-3) --------------------


def _trial_record(path: Path, n: int, *, decision="reject", rid=None, sha=True, event=None):
    reg = parse_registration(path)
    ctx = {"trial_number": n, "registration_file": f"worker/experiments/analyst/{path.name}",
           "role": "vs_champion_refit", "event_id": event or f"trial{n:03d}-e"}
    if sha:
        ctx["registration_sha256"] = file_sha256(path)
    rec = gate_record(f"analyst-m{n}", "0" * 64, record_id=rid or f"r{n}", decision=decision,
                      kind="trial", comparator_name="paired_dm", context=ctx)
    rec["challenger_metadata"]["recipe_id"] = reg.recipe.recipe_id
    return rec


def test_edited_registration_cannot_be_rerun_after_it_was_logged(repo):
    """Tester repro: trial 1 ran and was REJECTED; editing 001 and committing it
    gave a new recipe_id and preflight passed with k still 1."""
    p = _register(repo, 1, max_depth=4)
    rec = _trial_record(p, 1)
    p.write_text(registration_text(1, max_depth=5))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "edit trial 1")
    with pytest.raises(RegistrationError, match="already been run"):
        _preflight(repo, p, [rec])


def test_logged_trial_is_refused_by_number_even_without_a_logged_hash(repo):
    p = _register(repo, 1, max_depth=4)
    rec = _trial_record(p, 1, sha=False)
    p.write_text(registration_text(1, max_depth=5))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "edit")
    with pytest.raises(RegistrationError, match="already been run"):
        _preflight(repo, p, [rec])


def test_registration_edited_after_its_run_blocks_every_later_trial(repo):
    p1 = _register(repo, 1, max_depth=4)
    rec = _trial_record(p1, 1)
    p1.write_text(registration_text(1, max_depth=5, hypothesis="Rewritten after the fact."))
    p2 = _register(repo, 2, max_depth=6)  # commits both
    with pytest.raises(RegistrationError, match="edited after it was run"):
        _preflight(repo, p2, [rec])


def test_deleting_a_logged_registration_is_refused(repo):
    """Tester repro: deleting the last registration dropped k although its
    gate record remained."""
    _register(repo, 1, max_depth=4)
    p2 = _register(repo, 2, max_depth=6)
    rec2 = _trial_record(p2, 2)
    p2.unlink()
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "delete trial 2")
    with pytest.raises(RegistrationError, match="missing"):
        _preflight(repo, repo / "worker/experiments/analyst/001-deeper.toml", [rec2])
    with pytest.raises(RegistrationError, match="missing"):
        trial_count(repo / "worker/experiments/analyst", [rec2])


def test_renaming_a_logged_registration_is_refused(repo):
    p1 = _register(repo, 1, max_depth=4)
    rec = _trial_record(p1, 1)
    p1.rename(p1.with_name("001-renamed.toml"))
    p2 = _register(repo, 2, max_depth=6)
    with pytest.raises(RegistrationError, match="renamed"):
        _preflight(repo, p2, [rec])


def test_same_trial_number_logged_twice_is_refused(repo):
    p1 = _register(repo, 1, max_depth=4)
    a = _trial_record(p1, 1, rid="a", event="e-a")
    b = _trial_record(p1, 1, rid="b", event="e-b")
    with pytest.raises(RegistrationError, match="more than once"):
        logged_trials([a, b])


def test_k_is_max_of_registration_files_and_logged_trials(repo):
    p1 = _register(repo, 1, max_depth=4)
    p2 = _register(repo, 2, max_depth=6)
    rec = _trial_record(p1, 1)
    exp = repo / "worker/experiments/analyst"
    assert trial_count(exp, [rec]) == 2  # 2 files, 1 logged
    reg, k = _preflight(repo, p2, [rec])
    assert (reg.trial_number, k) == (2, 2)


def test_preflight_refuses_uncommitted_gate_log_change(repo):
    """Tester repro: deleting a REJECT record from gate_log.jsonl (uncommitted)
    let an already-evaluated recipe run again."""
    assert "worker/models" in CLEAN_PATHS
    p = _register(repo, 1, max_depth=4)
    append_jsonl(repo / "worker/models/analyst/gate_log.jsonl", gate_record("x", "0" * 64, decision="reject"))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "log")
    (repo / "worker/models/analyst/gate_log.jsonl").write_text("")  # delete the record, uncommitted
    with pytest.raises(RegistrationError, match="worker/models"):
        _preflight(repo, p)
