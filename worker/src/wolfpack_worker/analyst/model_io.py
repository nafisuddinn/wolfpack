"""Champion model artifact: `model.json` (XGBoost native) + `manifest.json`.

The daily strategy loads the committed champion from
`worker/models/analyst/champion/`. A missing, modified, or
spec-incompatible file is a bug, not a reason to quietly hold — every check
here raises `ModelIntegrityError` so the daily run fails loudly.

Only `xgboost` is needed to load a champion (no scikit-learn / MLflow), so
the daily cron doesn't need the `train` dependency group.

Week 3 note: from then on, only the alphagate promotion step should call
`write_champion`; `train --promote` exists for the very first model only.
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

from wolfpack_worker.analyst.features import FEATURE_NAMES, FEATURE_SPEC_VERSION

logger = logging.getLogger(__name__)

# worker/src/wolfpack_worker/analyst/model_io.py -> worker/
_WORKER_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHAMPION_DIR = _WORKER_ROOT / "models" / "analyst" / "champion"
MODEL_FILENAME = "model.json"
MANIFEST_FILENAME = "manifest.json"

REQUIRED_MANIFEST_KEYS = (
    "model_version",
    "mlflow_run_id",
    "trained_through",
    "feature_spec_version",
    "feature_names",
    "model_sha256",
    "xgboost_version",
    "test_metrics",
)


class ModelIntegrityError(RuntimeError):
    """The champion artifact is missing, altered, or incompatible with the
    current feature spec. Never catch-and-hold; fix the artifact."""


@dataclass(frozen=True)
class Champion:
    booster: xgb.Booster
    manifest: dict[str, Any]


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


def load_champion(champion_dir: Path = DEFAULT_CHAMPION_DIR) -> Champion:
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
    missing = [k for k in REQUIRED_MANIFEST_KEYS if k not in manifest]
    if missing:
        raise ModelIntegrityError(f"manifest.json is missing keys: {missing}")

    model_bytes = model_path.read_bytes()
    actual = sha256_bytes(model_bytes)
    if actual != manifest["model_sha256"]:
        raise ModelIntegrityError(
            f"model.json sha256 mismatch: manifest says {manifest['model_sha256']}, "
            f"file is {actual}. The model file was altered or doesn't match its manifest."
        )
    if manifest["feature_spec_version"] != FEATURE_SPEC_VERSION:
        raise ModelIntegrityError(
            f"feature_spec_version mismatch: model trained on "
            f"{manifest['feature_spec_version']!r}, code builds {FEATURE_SPEC_VERSION!r}."
        )
    if list(manifest["feature_names"]) != list(FEATURE_NAMES):
        raise ModelIntegrityError(
            "feature_names mismatch (names or ORDER differ) between the model "
            f"manifest {manifest['feature_names']} and features.FEATURE_NAMES "
            f"{list(FEATURE_NAMES)}."
        )

    booster = xgb.Booster()
    booster.load_model(bytearray(model_bytes))
    if booster.feature_names is not None and list(booster.feature_names) != list(FEATURE_NAMES):
        raise ModelIntegrityError(
            f"feature_names inside model.json {booster.feature_names} differ from "
            f"FEATURE_NAMES {list(FEATURE_NAMES)}."
        )
    if manifest["xgboost_version"] != xgb.__version__:
        logger.warning(
            "analyst.model_io: champion was written by xgboost %s, loading with %s.",
            manifest["xgboost_version"],
            xgb.__version__,
        )
    return Champion(booster=booster, manifest=manifest)
