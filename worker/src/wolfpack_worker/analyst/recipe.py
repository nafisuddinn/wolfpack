"""The Analyst's training recipe: everything that determines a model besides
the data and the cutoff date.

A recipe is identified by `recipe_id` = the first 12 hex chars of the sha256
of its canonical JSON (sorted keys, no whitespace). Two recipes with the same
id are the same recipe; changing any field (a hyperparameter, the boosting
rounds, the feature spec, the training start date) gives a new id, which is a
new trial for the promotion gate's multiple-testing count. That is the point:
there is no way to "tweak v1" without it showing up as a different recipe.

Recipes live in TOML files: `worker/recipes/analyst/v1.toml` for the first
champion, and the `[recipe]` table of each pre-registered experiment in
`worker/experiments/analyst/NNN-<slug>.toml`.

MODEL-RISK LIMITATION: a recipe is a set of modelling choices, not evidence
of edge. No recipe so far has shown real predictive edge on public daily
price data (see MODEL_CARD.md).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from wolfpack_worker.analyst.dataset import LABEL_DEFINITION
from wolfpack_worker.analyst.features import FEATURE_SPECS
from wolfpack_worker.analyst.metrics import THRESHOLD
from wolfpack_worker.universe import UNIVERSE

_WORKER_ROOT = Path(__file__).resolve().parents[3]
RECIPES_DIR = _WORKER_ROOT / "recipes" / "analyst"
V1_RECIPE_PATH = RECIPES_DIR / "v1.toml"

# Supported label definitions. Only one exists; a new one is a new key.
LABELS: Mapping[str, str] = MappingProxyType({"next_open_to_open_sign_v1": LABEL_DEFINITION})

_KEYS = (
    "feature_spec_version",
    "label",
    "threshold",
    "num_boost_round",
    "universe",
    "train_since",
    "xgb_params",
)


class RecipeError(ValueError):
    """A recipe is malformed or uses a setting the pipeline doesn't support."""


@dataclass(frozen=True)
class Recipe:
    feature_spec_version: str
    label: str
    threshold: float
    num_boost_round: int
    universe: tuple[str, ...]
    train_since: str
    xgb_params: Mapping[str, Any]

    # -- construction / validation ------------------------------------------

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], *, feature_specs: Mapping[str, Any] | None = None) -> "Recipe":
        """`feature_specs` is the persona's spec registry (default: The
        Analyst's FEATURE_SPECS; The Scout passes SCOUT_FEATURE_SPECS). It
        only validates `feature_spec_version`; the recipe_id hash is the
        same function of the fields either way."""
        d = dict(d)
        feature_specs = FEATURE_SPECS if feature_specs is None else feature_specs
        unknown = sorted(set(d) - set(_KEYS))
        if unknown:
            raise RecipeError(f"unknown recipe key(s): {unknown}")
        missing = [k for k in _KEYS if k not in d]
        if missing:
            raise RecipeError(f"recipe is missing key(s): {missing}")

        fsv = d["feature_spec_version"]
        if fsv not in feature_specs:
            raise RecipeError(
                f"feature_spec_version {fsv!r} is not registered (known: {sorted(feature_specs)})"
            )
        if d["label"] not in LABELS:
            raise RecipeError(f"label {d['label']!r} is not supported (known: {sorted(LABELS)})")

        threshold = d["threshold"]
        # The strategy's long/flat rule and the reported accuracy use
        # metrics.THRESHOLD; a recipe that silently disagreed with it would
        # trade differently from how it was evaluated.
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or float(
            threshold
        ) != THRESHOLD:
            raise RecipeError(
                f"threshold must be {THRESHOLD} (the only value the strategy supports), got {threshold!r}"
            )

        nbr = d["num_boost_round"]
        if isinstance(nbr, bool) or not isinstance(nbr, int) or nbr < 1:
            raise RecipeError(f"num_boost_round must be a positive int, got {nbr!r}")

        universe = tuple(d["universe"])
        # The universe is shared by every persona (Decision Log 2026-09-24);
        # changing it is a project decision, not a recipe tweak.
        if universe != tuple(UNIVERSE):
            raise RecipeError(f"universe must be the shared universe {list(UNIVERSE)}, got {list(universe)}")

        since = d["train_since"]
        if isinstance(since, _dt.date) and not isinstance(since, _dt.datetime):
            since = since.isoformat()
        try:
            _dt.date.fromisoformat(str(since))
        except ValueError:
            raise RecipeError(f"train_since must be an ISO date (YYYY-MM-DD), got {since!r}") from None

        params = d["xgb_params"]
        if not isinstance(params, Mapping):
            raise RecipeError("xgb_params must be a table")
        if params.get("objective") != "binary:logistic":
            raise RecipeError(
                "xgb_params.objective must be 'binary:logistic' (the gate scores probabilities "
                f"with log loss), got {params.get('objective')!r}"
            )
        if "nthread" in params:
            raise RecipeError("xgb_params must not set nthread (the trainer pins it for reproducibility)")
        for k, v in params.items():
            if not isinstance(v, (str, int, float, bool)):
                raise RecipeError(f"xgb_params.{k} must be a scalar, got {type(v).__name__}")

        return cls(
            feature_spec_version=fsv,
            label=d["label"],
            threshold=float(threshold),
            num_boost_round=nbr,
            universe=universe,
            train_since=str(since),
            xgb_params=MappingProxyType(dict(params)),
        )

    # -- identity -------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_spec_version": self.feature_spec_version,
            "label": self.label,
            "threshold": self.threshold,
            "num_boost_round": self.num_boost_round,
            "universe": list(self.universe),
            "train_since": self.train_since,
            "xgb_params": dict(self.xgb_params),
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        )

    @property
    def recipe_id(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()[:12]

    @property
    def train_since_date(self) -> _dt.date:
        return _dt.date.fromisoformat(self.train_since)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Recipe) and self.canonical_json() == other.canonical_json()

    def __hash__(self) -> int:
        return hash(self.canonical_json())


def load_recipe_file(path: Path) -> Recipe:
    """A plain recipe TOML, or an experiment registration's `[recipe]` table."""
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    if "recipe" in data:
        data = data["recipe"]
    return Recipe.from_dict(data)


def load_v1_recipe() -> Recipe:
    return load_recipe_file(V1_RECIPE_PATH)
