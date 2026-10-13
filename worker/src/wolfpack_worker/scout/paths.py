"""The Scout's governance locations (see analyst/paths.py for PersonaPaths).

The Scout has its OWN trial budget: k counts the registrations in
worker/experiments/scout/ and the trials in worker/models/scout/gate_log.jsonl
only, never The Analyst's.
"""

from __future__ import annotations

from typing import Any, Mapping

from wolfpack_worker.analyst.paths import REPO_ROOT, WORKER_ROOT, PersonaPaths


def _scout_recipe(d: Mapping[str, Any]):
    from wolfpack_worker.scout.recipe import parse_scout_recipe

    return parse_scout_recipe(d)


def _scout_spec(version: str):
    from wolfpack_worker.scout.features import get_scout_feature_spec

    return get_scout_feature_spec(version)


SCOUT_PATHS = PersonaPaths(
    persona="scout",
    display_name="The Scout",
    models_dir=WORKER_ROOT / "models" / "scout",
    experiments_dir=WORKER_ROOT / "experiments" / "scout",
    model_card_path=REPO_ROOT / "MODEL_CARD_SCOUT.md",
    recipe_parser=_scout_recipe,
    feature_spec_lookup=_scout_spec,
)
