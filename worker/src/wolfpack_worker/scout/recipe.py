"""The Scout's recipes: The Analyst's `Recipe` (same keys, same recipe_id
hash, same validation: shared universe, the next-open-to-open label,
threshold 0.5, binary:logistic, no nthread) validated against the Scout's
feature-spec registry instead of the Analyst's.

Scout v1 (registration worker/experiments/scout/001-*.toml) is the
Analyst v1 recipe with feature_spec_version = "scout_v1": pooled XGBoost
across the 5 tickers, untuned Analyst-v1 hyperparameters (design section 1),
so no hyperparameter was chosen by looking at Scout data.

MODEL-RISK LIMITATION: a recipe is a set of modelling choices, not evidence
of edge.
"""

from __future__ import annotations

from typing import Any, Mapping

from wolfpack_worker.analyst.recipe import Recipe, RecipeError, load_v1_recipe
from wolfpack_worker.scout.features import SCOUT_FEATURE_SPECS

__all__ = ["Recipe", "RecipeError", "parse_scout_recipe", "scout_v1_recipe_dict"]


def parse_scout_recipe(d: Mapping[str, Any]) -> Recipe:
    return Recipe.from_dict(d, feature_specs=SCOUT_FEATURE_SPECS)


def scout_v1_recipe_dict() -> dict[str, Any]:
    """Analyst v1's settings with the Scout's feature spec (for registration)."""
    d = load_v1_recipe().to_dict()
    d["feature_spec_version"] = "scout_v1"
    return d
