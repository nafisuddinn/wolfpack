"""Champion model artifact: `model.json` (XGBoost native) + `manifest.json`.

The daily strategy loads the committed champion from
`worker/models/analyst/champion/`. A missing, modified, or
spec-incompatible file is a bug, not a reason to quietly hold — every check
here raises `ModelIntegrityError` so the daily run fails loudly.

Only `xgboost` is needed to load a champion (no scikit-learn / MLflow /
alphagate), so the daily cron doesn't need the `train` dependency group.

Promotion backstop: `load_champion` also refuses a champion unless
`gate_log.jsonl` (next to the champion directory) holds an alphagate PROMOTE
record for exactly this `model_version` and model sha256, and no REJECT
record for it (see gate_log.verify_promotion). `write_champion` is a plain
file writer; the only caller that writes the real champion is
`gating.promote_from_gate`, and a champion written any other way fails to
load here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import xgboost as xgb

from wolfpack_worker.analyst.features import FeatureSpec, get_feature_spec
from wolfpack_worker.analyst.gate_log import (
    GATE_LOG_FILENAME,
    GateLogError,
    read_gate_log,
    verify_promotion,
)

logger = logging.getLogger(__name__)

# worker/src/wolfpack_worker/analyst/model_io.py -> worker/
_WORKER_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = _WORKER_ROOT / "models" / "analyst"
DEFAULT_CHAMPION_DIR = MODELS_DIR / "champion"
GATE_LOG_PATH = MODELS_DIR / GATE_LOG_FILENAME
MODEL_FILENAME = "model.json"
MANIFEST_FILENAME = "manifest.json"

# Keys every champion artifact has (including the pre-gate v1 manifest).
ARTIFACT_MANIFEST_KEYS = (
    "model_version",
    "mlflow_run_id",
    "trained_through",
    "feature_spec_version",
    "feature_names",
    "model_sha256",
    "xgboost_version",
    "test_metrics",
)
# Keys a deployable (gated) champion must also have.
REQUIRED_MANIFEST_KEYS = ARTIFACT_MANIFEST_KEYS + ("recipe_id", "gate_record_ids")


class ModelIntegrityError(RuntimeError):
    """The champion artifact is missing, altered, or incompatible with the
    current feature spec. Never catch-and-hold; fix the artifact."""


@dataclass(frozen=True)
class Champion:
    booster: xgb.Booster
    manifest: dict[str, Any]
    spec: FeatureSpec


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def write_champion(champion_dir: Path, model_bytes: bytes, manifest: dict[str, Any]) -> dict[str, Any]:
    """Write model.json + manifest.json (manifest gets `model_sha256`)."""
    champion_dir = Path(champion_dir)
    manifest = dict(manifest)
    manifest["model_sha256"] = sha256_bytes(model_bytes)
    _atomic_write(champion_dir / MODEL_FILENAME, model_bytes)
    _atomic_write(
        champion_dir / MANIFEST_FILENAME,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
    )
    return manifest


def default_gate_log_path(champion_dir: Path) -> Path:
    return Path(champion_dir).parent / GATE_LOG_FILENAME


def _registration_index(experiments_dir: Path | None):
    def load():
        from wolfpack_worker.analyst.registration import (
            EXPERIMENTS_DIR,
            RegistrationError,
            file_sha256,
            list_registrations,
        )

        try:
            regs = list_registrations(Path(experiments_dir) if experiments_dir is not None else EXPERIMENTS_DIR)
        except RegistrationError as exc:
            raise GateLogError(f"experiment registrations are invalid: {exc}") from None
        return {r.trial_number: (r.path.name, file_sha256(r.path)) for r in regs}

    return load


def load_champion(
    champion_dir: Path = DEFAULT_CHAMPION_DIR,
    *,
    gate_log_path: Path | None = None,
    experiments_dir: Path | None = None,
) -> Champion:
    """Load + verify the champion, INCLUDING that the alphagate gate promoted it.

    `gate_log_path` defaults to `<champion_dir>/../gate_log.jsonl`;
    `experiments_dir` (where trial registrations live) defaults to
    worker/experiments/analyst.
    """
    champ = read_champion_artifact(champion_dir, required_keys=REQUIRED_MANIFEST_KEYS)
    path = Path(gate_log_path) if gate_log_path is not None else default_gate_log_path(champion_dir)
    if not path.is_file():
        raise ModelIntegrityError(
            f"gate log {path} is missing: a champion is only valid with the alphagate PROMOTE "
            "record that put it there."
        )
    m = champ.manifest
    try:
        verify_promotion(
            read_gate_log(path),
            model_version=m["model_version"],
            model_sha256=m["model_sha256"],
            gate_record_ids=list(m["gate_record_ids"]),
            registrations=_registration_index(experiments_dir),
        )
    except GateLogError as exc:
        raise ModelIntegrityError(f"champion {m['model_version']!r} failed the gate-log check: {exc}") from None
    return champ


def read_champion_artifact(
    champion_dir: Path = DEFAULT_CHAMPION_DIR, *, required_keys: tuple[str, ...] = ARTIFACT_MANIFEST_KEYS
) -> Champion:
    """File-integrity checks only (presence, sha256, feature spec, names).

    Does NOT check the gate log. Used by load_champion (which then does) and
    by the one-time bootstrap gate, which evaluates the pre-gate v1 artifact
    before any gate record exists. Anything that will trade must use
    load_champion.
    """
    champion_dir = Path(champion_dir)
    model_path = champion_dir / MODEL_FILENAME
    manifest_path = champion_dir / MANIFEST_FILENAME
    for p in (model_path, manifest_path):
        if not p.is_file():
            raise ModelIntegrityError(
                f"The Analyst's champion file is missing: {p}. The committed "
                "champion must exist; a missing file is a bug, not a 'hold'."
            )

    manifest = json.loads(manifest_path.read_text())
    missing = [k for k in required_keys if k not in manifest]
    if missing:
        raise ModelIntegrityError(f"manifest.json is missing keys: {missing}")

    model_bytes = model_path.read_bytes()
    actual = sha256_bytes(model_bytes)
    if actual != manifest["model_sha256"]:
        raise ModelIntegrityError(
            f"model.json sha256 mismatch: manifest says {manifest['model_sha256']}, "
            f"file is {actual}. The model file was altered or doesn't match its manifest."
        )
    try:
        spec = get_feature_spec(manifest["feature_spec_version"])
    except KeyError:
        raise ModelIntegrityError(
            f"feature_spec_version mismatch: model trained on "
            f"{manifest['feature_spec_version']!r}, which is not a registered feature spec."
        ) from None
    names = list(spec.names)
    if list(manifest["feature_names"]) != names:
        raise ModelIntegrityError(
            "feature_names mismatch (names or ORDER differ) between the model "
            f"manifest {manifest['feature_names']} and feature spec "
            f"{spec.version} {names}."
        )

    booster = xgb.Booster()
    booster.load_model(bytearray(model_bytes))
    if booster.feature_names is not None and list(booster.feature_names) != names:
        raise ModelIntegrityError(
            f"feature_names inside model.json {booster.feature_names} differ from "
            f"feature spec {spec.version} {names}."
        )
    if manifest["xgboost_version"] != xgb.__version__:
        logger.warning(
            "analyst.model_io: champion was written by xgboost %s, loading with %s.",
            manifest["xgboost_version"],
            xgb.__version__,
        )
    return Champion(booster=booster, manifest=manifest, spec=spec)
