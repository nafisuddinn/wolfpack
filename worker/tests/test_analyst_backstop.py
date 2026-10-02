"""Hard backstop: a champion only loads if alphagate PROMOTED it.

`model_io.load_champion` (which the daily cron calls) refuses any champion
whose `model_version` has no PROMOTE record in gate_log.jsonl, so a manifest
swapped in by hand (or by a script that skipped the gate) fails loudly
instead of trading. The check parses the JSONL directly: the daily cron does
not install alphagate.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import xgboost as xgb

from analyst_helpers import append_jsonl, gate_record, make_universe_bars, write_gated_champion
from wolfpack_worker.analyst.dataset import build_dataset
from wolfpack_worker.analyst.features import FEATURE_NAMES, FEATURE_SPEC_VERSION
from wolfpack_worker.analyst.model_io import (
    DEFAULT_CHAMPION_DIR,
    GATE_LOG_PATH,
    ModelIntegrityError,
    load_champion,
    sha256_bytes,
    write_champion,
)
from wolfpack_worker.analyst.recipe import load_v1_recipe


def _booster_bytes() -> bytes:
    bars = make_universe_bars(200, seed=3)
    ds = build_dataset(bars)
    d = xgb.DMatrix(ds[list(FEATURE_NAMES)].to_numpy(), label=ds["y"].to_numpy(),
                    feature_names=list(FEATURE_NAMES))
    b = xgb.train({"objective": "binary:logistic", "max_depth": 2, "seed": 0, "nthread": 1}, d, 5)
    return bytes(b.save_raw("json"))


def _manifest(version: str) -> dict:
    r = load_v1_recipe()
    return {
        "model_version": version,
        "mlflow_run_id": "run",
        "recipe_id": r.recipe_id,
        "recipe": r.to_dict(),
        "trial_number": None,
        "trained_through": pd.Timestamp("2024-06-03", tz="UTC").isoformat(),
        "feature_spec_version": FEATURE_SPEC_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "xgboost_version": xgb.__version__,
        "test_metrics": {"accuracy": 0.5, "logloss": 0.69, "baseline_accuracy": 0.5,
                         "baseline_logloss": 0.69, "beats_baseline_logloss": False},
    }


@pytest.fixture(scope="module")
def model_bytes():
    return _booster_bytes()


@pytest.fixture()
def gated(tmp_path, model_bytes):
    champ = tmp_path / "champion"
    write_gated_champion(champ, model_bytes, _manifest("analyst-20260101-aaaaaaaa"))
    return champ


def test_gated_champion_loads(gated):
    champ = load_champion(gated)
    assert champ.manifest["model_version"] == "analyst-20260101-aaaaaaaa"
    assert champ.spec.version == "v1"


def test_manifest_swapped_without_gate_record_fails_loudly(gated, model_bytes):
    # Same bytes, hand-edited version: no PROMOTE record names it.
    m = json.loads((gated / "manifest.json").read_text())
    m["model_version"] = "analyst-20260102-bbbbbbbb"
    (gated / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ModelIntegrityError, match="PROMOTE"):
        load_champion(gated)


def test_new_model_written_without_gate_fails(tmp_path, model_bytes):
    champ = tmp_path / "champion"
    manifest = {**_manifest("analyst-20260101-cccccccc"), "gate_record_ids": ["made-up"]}
    write_champion(champ, model_bytes, manifest)
    with pytest.raises(ModelIntegrityError, match="gate log"):
        load_champion(champ)  # no gate_log.jsonl at all
    (tmp_path / "gate_log.jsonl").write_text("")
    with pytest.raises(ModelIntegrityError, match="PROMOTE"):
        load_champion(champ)


def test_rejected_model_never_loads_even_with_a_promote_record(gated, model_bytes):
    append_jsonl(
        gated.parent / "gate_log.jsonl",
        gate_record("analyst-20260101-aaaaaaaa", sha256_bytes(model_bytes),
                    record_id="rec-0002", decision="reject"),
    )
    with pytest.raises(ModelIntegrityError, match="REJECT"):
        load_champion(gated)


def test_promote_record_for_different_bytes_fails(tmp_path, model_bytes):
    champ = tmp_path / "champion"
    manifest = {**_manifest("analyst-20260101-dddddddd"), "gate_record_ids": ["rec-x"]}
    write_champion(champ, model_bytes, manifest)
    append_jsonl(tmp_path / "gate_log.jsonl",
                 gate_record("analyst-20260101-dddddddd", "0" * 64, record_id="rec-x"))
    with pytest.raises(ModelIntegrityError, match="sha256"):
        load_champion(champ)


def test_untimed_promotion_is_not_accepted(tmp_path, model_bytes):
    champ = tmp_path / "champion"
    manifest = {**_manifest("analyst-20260101-eeeeeeee"), "gate_record_ids": ["rec-u"]}
    write_champion(champ, model_bytes, manifest)
    append_jsonl(tmp_path / "gate_log.jsonl",
                 gate_record("analyst-20260101-eeeeeeee", sha256_bytes(model_bytes),
                             record_id="rec-u", chronology_checked=False))
    with pytest.raises(ModelIntegrityError, match="chronology"):
        load_champion(champ)


def test_manifest_gate_record_ids_must_all_be_promotes_for_this_model(gated, model_bytes):
    m = json.loads((gated / "manifest.json").read_text())
    m["gate_record_ids"] = ["rec-0001", "rec-missing"]
    (gated / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ModelIntegrityError, match="rec-missing"):
        load_champion(gated)


def test_corrupt_gate_log_fails_loudly(gated):
    with (gated.parent / "gate_log.jsonl").open("a") as fh:
        fh.write("{not json\n")
    with pytest.raises(ModelIntegrityError, match="line 2"):
        load_champion(gated)


def test_missing_gate_keys_in_manifest(tmp_path, model_bytes):
    champ = tmp_path / "champion"
    write_champion(champ, model_bytes, _manifest("analyst-20260101-ffffffff"))  # no gate_record_ids
    with pytest.raises(ModelIntegrityError, match="gate_record_ids"):
        load_champion(champ)


def test_explicit_gate_log_path(tmp_path, model_bytes):
    champ = tmp_path / "somewhere" / "champion"
    write_gated_champion(champ, model_bytes, _manifest("analyst-20260101-12121212"))
    moved = tmp_path / "elsewhere.jsonl"
    (champ.parent / "gate_log.jsonl").rename(moved)
    with pytest.raises(ModelIntegrityError):
        load_champion(champ)
    assert load_champion(champ, gate_log_path=moved).manifest["model_version"].endswith("12121212")


def test_committed_champion_has_a_promote_record_in_the_committed_gate_log():
    assert GATE_LOG_PATH == DEFAULT_CHAMPION_DIR.parent / "gate_log.jsonl"
    champ = load_champion()
    lines = [json.loads(x) for x in GATE_LOG_PATH.read_text().splitlines() if x.strip()]
    ids = {r["record_id"] for r in lines if r["challenger_id"] == champ.manifest["model_version"]
           and r["decision"] == "promote"}
    assert set(champ.manifest["gate_record_ids"]) <= ids


def test_daily_path_works_without_alphagate_mlflow_or_sklearn():
    """The daily cron installs neither the `train` group nor alphagate. Simulate
    that by making those imports fail, then run the daily path's imports and
    the champion backstop. (xgboost imports sklearn opportunistically when it
    is installed and copes when it isn't, so "absent" is the honest test, not
    "not in sys.modules".)"""
    code = (
        "import sys, importlib.abc\n"
        "class Block(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in ('alphagate', 'mlflow', 'sklearn'):\n"
        "            raise ModuleNotFoundError(name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, Block())\n"
        "import wolfpack_worker.daily_trades, wolfpack_worker.strategies.analyst\n"
        "from wolfpack_worker.analyst.model_io import load_champion\n"
        "champ = load_champion()\n"
        "assert champ.manifest['gate_record_ids']\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).parent)


# --- tamper hardening (tester finding 4) ---------------------------------------------


def _forge(tmp_path, model_bytes, record: dict, version="analyst-20260101-0f0f0f0f"):
    champ = tmp_path / "champion"
    manifest = {**_manifest(version), "gate_record_ids": [record["record_id"]]}
    write_champion(champ, model_bytes, manifest)
    record = {**record, "challenger_id": version}
    record.setdefault("challenger_metadata", {})["model_sha256"] = sha256_bytes(model_bytes)
    append_jsonl(tmp_path / "gate_log.jsonl", record)
    return champ


def test_minimal_hand_written_promote_record_is_refused(tmp_path, model_bytes):
    """Tester repro: a 4-field PROMOTE line used to be accepted."""
    rec = {"record_id": "fake1", "decision": "promote", "chronology_checked": True}
    champ = _forge(tmp_path, model_bytes, rec)
    with pytest.raises(ModelIntegrityError, match="kind"):
        load_champion(champ, experiments_dir=tmp_path / "exp")


@pytest.mark.parametrize("kind", [None, "manual", "hotfix"])
def test_promote_with_unknown_kind_is_refused(tmp_path, model_bytes, kind):
    rec = gate_record("x", "x", record_id="k1")
    if kind is None:
        del rec["context"]["kind"]
    else:
        rec["context"]["kind"] = kind
    with pytest.raises(ModelIntegrityError, match="kind"):
        load_champion(_forge(tmp_path, model_bytes, rec), experiments_dir=tmp_path / "exp")


@pytest.mark.parametrize("score", [{}, {"value": None}, {"value": "NaN", "samples": [0.7]},
                                   {"value": 0.69, "samples": []}, None])
def test_promote_without_a_real_challenger_score_is_refused(tmp_path, model_bytes, score):
    rec = gate_record("x", "x", record_id="s1")
    rec["challenger_score"] = score
    with pytest.raises(ModelIntegrityError, match="challenger_score"):
        load_champion(_forge(tmp_path, model_bytes, rec), experiments_dir=tmp_path / "exp")


def test_promote_with_unrecognised_comparator_is_refused(tmp_path, model_bytes):
    rec = gate_record("x", "x", record_id="c1", comparator_name="always_yes")
    with pytest.raises(ModelIntegrityError, match="comparator"):
        load_champion(_forge(tmp_path, model_bytes, rec), experiments_dir=tmp_path / "exp")


def test_trial_promote_must_match_a_registration(tmp_path, model_bytes):
    from analyst_helpers import weak_recipe_dict, write_registration
    from wolfpack_worker.analyst.registration import file_sha256

    exp = tmp_path / "exp"
    rec = gate_record("x", "x", record_id="t1", kind="trial", comparator_name="paired_dm",
                      context={"trial_number": 1, "registration_file": "worker/experiments/analyst/001-x.toml"})
    champ = _forge(tmp_path, model_bytes, rec)
    with pytest.raises(ModelIntegrityError, match="registration"):
        load_champion(champ, experiments_dir=exp)  # no registration file at all
    reg = write_registration(exp, 1, weak_recipe_dict(), slug="other")
    with pytest.raises(ModelIntegrityError, match="registration"):
        load_champion(champ, experiments_dir=exp)  # wrong file name
    reg.rename(exp / "001-x.toml")
    assert load_champion(champ, experiments_dir=exp)  # matches

    # ...and if the log recorded the registration hash, the bytes must match.
    log = tmp_path / "gate_log.jsonl"
    line = json.loads(log.read_text())
    line["context"]["registration_sha256"] = file_sha256(exp / "001-x.toml")
    log.write_text(json.dumps(line) + "\n")
    assert load_champion(champ, experiments_dir=exp)
    (exp / "001-x.toml").write_text((exp / "001-x.toml").read_text().replace('"h"', '"edited"'))
    with pytest.raises(ModelIntegrityError, match="registration"):
        load_champion(champ, experiments_dir=exp)


def test_bootstrap_promote_must_be_the_first_record_in_the_log(tmp_path, model_bytes):
    append_jsonl(tmp_path / "gate_log.jsonl", gate_record("earlier", "0" * 64, record_id="z0", decision="reject"))
    rec = gate_record("x", "x", record_id="b2")  # kind=bootstrap, but not first
    with pytest.raises(ModelIntegrityError, match="bootstrap"):
        load_champion(_forge(tmp_path, model_bytes, rec), experiments_dir=tmp_path / "exp")
