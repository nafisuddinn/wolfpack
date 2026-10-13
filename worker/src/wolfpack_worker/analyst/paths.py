"""`PersonaPaths`: where one gated persona's governance files live.

The promotion-gate machinery (champion artifact + backstop in `model_io`,
`gate_log`, trial `registration`, the append-only `log_guard`, `forward`
monitoring, the generated `render_history` table, and `gating`'s promotion
step) was written for The Analyst. The Scout uses the same machinery rather
than a copy (Decision Log 2026-10-04: one gate implementation to audit), so
everything persona-specific is collected here:

* where the models, gate log, forward log, champion and archive live;
* where the pre-registered experiments live (each persona has its OWN trial
  budget: k counts that persona's registrations and logged trials only);
* which model card carries the generated gate-history table;
* how a registration's `[recipe]` table is parsed (each persona's recipes
  name that persona's feature specs), and how a champion manifest's
  `feature_spec_version` is resolved to a spec with `.version` / `.names`.

Every governance function defaults to `ANALYST_PATHS`, so existing Analyst
callers (and tests) behave exactly as before.

MODEL-RISK LIMITATION: nothing here is about model quality. A file layout
does not make any persona's model better than a coin flip; see MODEL_CARD.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

# worker/src/wolfpack_worker/analyst/paths.py -> worker/ and the repo root
WORKER_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = Path(__file__).resolve().parents[4]

GATE_LOG_FILENAME = "gate_log.jsonl"
FORWARD_LOG_FILENAME = "forward_log.jsonl"


@dataclass(frozen=True)
class PersonaPaths:
    """Per-persona locations and the two persona-specific parsers.

    persona: directory name under worker/models and worker/experiments
        ("analyst", "scout"); also the generated-table marker prefix.
    display_name: used in error messages ("The Analyst").
    recipe_parser: mapping -> recipe; must raise
        `wolfpack_worker.analyst.recipe.RecipeError` on a bad recipe.
    feature_spec_lookup: version -> spec with `.version` and `.names`; must
        raise KeyError for an unknown version.
    """

    persona: str
    display_name: str
    models_dir: Path
    experiments_dir: Path
    model_card_path: Path
    recipe_parser: Callable[[Mapping[str, Any]], Any]
    feature_spec_lookup: Callable[[str], Any]

    @property
    def champion_dir(self) -> Path:
        return self.models_dir / "champion"

    @property
    def archive_dir(self) -> Path:
        return self.models_dir / "archive"

    @property
    def gate_log_path(self) -> Path:
        return self.models_dir / GATE_LOG_FILENAME

    @property
    def forward_log_path(self) -> Path:
        return self.models_dir / FORWARD_LOG_FILENAME

    @property
    def history_marker(self) -> str:
        return f"{self.persona}-gate-history"

    def log_relpaths(self, repo_root: Path = REPO_ROOT) -> tuple[str, str]:
        """Repo-relative gate/forward log paths (what log_guard protects)."""
        root = Path(repo_root).resolve()
        return (
            self.gate_log_path.resolve().relative_to(root).as_posix(),
            self.forward_log_path.resolve().relative_to(root).as_posix(),
        )

    def experiments_relpath(self, repo_root: Path = REPO_ROOT) -> str:
        try:
            return Path(self.experiments_dir).resolve().relative_to(Path(repo_root).resolve()).as_posix()
        except ValueError:
            return Path(self.experiments_dir).as_posix()


def _analyst_recipe(d: Mapping[str, Any]):
    from wolfpack_worker.analyst.recipe import Recipe

    return Recipe.from_dict(d)


def _analyst_spec(version: str):
    from wolfpack_worker.analyst.features import get_feature_spec

    return get_feature_spec(version)


ANALYST_PATHS = PersonaPaths(
    persona="analyst",
    display_name="The Analyst",
    models_dir=WORKER_ROOT / "models" / "analyst",
    experiments_dir=WORKER_ROOT / "experiments" / "analyst",
    model_card_path=REPO_ROOT / "MODEL_CARD.md",
    recipe_parser=_analyst_recipe,
    feature_spec_lookup=_analyst_spec,
)
