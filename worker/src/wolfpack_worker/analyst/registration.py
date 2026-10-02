"""Pre-registration of The Analyst's experiments (challenger recipes).

Why: the gate's significance test is only honest if every recipe ever tried
on the gate holdout is counted. A recipe tried quietly, found wanting, and
never mentioned is a hidden trial that makes a later "significant" result
look stronger than it is. So a trial is the COMMITTED REGISTRATION FILE, not
the completed run: k (the trial count that sets alpha_k =
spend_alpha(0.05, k)) is the number of registration files, abandoned ones
included.

A registration is `worker/experiments/analyst/NNN-<slug>.toml`:

    trial_number = 1                      # == NNN
    registered = 2026-10-02               # TOML date
    hypothesis = "one line, written before the run"
    # abandoned = "why"                   # optional; still counts as a trial
    [recipe]                              # one full recipe; no grids
    feature_spec_version = "v1"
    ...
    [recipe.xgb_params]
    ...

`preflight` refuses to start an experiment unless: the file is committed in
HEAD and unchanged; worker/src, worker/recipes, worker/pyproject.toml,
worker/uv.lock and the experiments directory have no uncommitted changes
(so the code and the trial count that produced a result are exactly what's
in git); the file is not abandoned; and its recipe_id has never been scored
as a challenger before (per the gate log).
"""

from __future__ import annotations

import datetime as _dt
import re
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from wolfpack_worker.analyst.gate_log import evaluated_recipe_ids
from wolfpack_worker.analyst.recipe import Recipe, RecipeError

REPO_ROOT = Path(__file__).resolve().parents[4]
EXPERIMENTS_DIR = REPO_ROOT / "worker" / "experiments" / "analyst"
# Paths (relative to the repo root) that must be clean for a trial to run.
CLEAN_PATHS = ("worker/src", "worker/recipes", "worker/pyproject.toml", "worker/uv.lock")

_NAME_RE = re.compile(r"^(\d{3})-[a-z0-9]+(?:-[a-z0-9]+)*\.toml$")
_KEYS = {"trial_number", "registered", "hypothesis", "recipe", "abandoned"}


class RegistrationError(ValueError):
    """The registration is malformed, or the run would not be honestly counted."""


@dataclass(frozen=True)
class Registration:
    path: Path
    trial_number: int
    registered: _dt.date
    hypothesis: str
    recipe: Recipe
    abandoned: str | None


def parse_registration(path: Path) -> Registration:
    path = Path(path)
    m = _NAME_RE.match(path.name)
    if not m:
        raise RegistrationError(
            f"{path.name}: registration file name must be NNN-<slug>.toml "
            "(3-digit trial number, lowercase slug)"
        )
    try:
        data: Mapping[str, Any] = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise RegistrationError(f"{path.name}: invalid TOML ({exc})") from None
    unknown = sorted(set(data) - _KEYS)
    if unknown:
        raise RegistrationError(f"{path.name}: unknown key(s) {unknown} (one recipe per file; no grids)")

    n = data.get("trial_number")
    if isinstance(n, bool) or not isinstance(n, int) or n != int(m.group(1)):
        raise RegistrationError(
            f"{path.name}: trial_number {n!r} must be an int equal to the file's NNN ({int(m.group(1))})"
        )
    registered = data.get("registered")
    if not isinstance(registered, _dt.date) or isinstance(registered, _dt.datetime):
        raise RegistrationError(f"{path.name}: registered must be a TOML date (YYYY-MM-DD)")
    hyp = data.get("hypothesis")
    if not isinstance(hyp, str) or not hyp.strip():
        raise RegistrationError(f"{path.name}: hypothesis is required")
    if "\n" in hyp.strip():
        raise RegistrationError(f"{path.name}: hypothesis must be one line")
    if "recipe" not in data or not isinstance(data["recipe"], Mapping):
        raise RegistrationError(f"{path.name}: a [recipe] table is required")
    try:
        recipe = Recipe.from_dict(data["recipe"])
    except RecipeError as exc:
        raise RegistrationError(f"{path.name}: {exc}") from None
    abandoned = data.get("abandoned")
    if abandoned is not None and (not isinstance(abandoned, str) or not abandoned.strip()):
        raise RegistrationError(f"{path.name}: abandoned must be a non-empty reason string")
    return Registration(
        path=path,
        trial_number=n,
        registered=registered,
        hypothesis=hyp.strip(),
        recipe=recipe,
        abandoned=abandoned.strip() if abandoned else None,
    )


def list_registrations(experiments_dir: Path = EXPERIMENTS_DIR) -> list[Registration]:
    """Every registration, sorted by trial number, validated as a set:
    numbers are exactly 1..k and no two files register the same recipe.
    Any *.toml that isn't a valid registration is an error (not ignored)."""
    experiments_dir = Path(experiments_dir)
    if not experiments_dir.is_dir():
        return []
    regs = sorted(
        (parse_registration(p) for p in experiments_dir.glob("*.toml")),
        key=lambda r: r.trial_number,
    )
    numbers = [r.trial_number for r in regs]
    if numbers != list(range(1, len(regs) + 1)):
        raise RegistrationError(
            f"trial numbers in {experiments_dir} must be exactly 1..{len(regs)}, got {numbers}"
        )
    seen: dict[str, int] = {}
    for r in regs:
        rid = r.recipe.recipe_id
        if rid in seen:
            raise RegistrationError(
                f"trials {seen[rid]} and {r.trial_number} register the same recipe ({rid})"
            )
        seen[rid] = r.trial_number
    return regs


def count_trials(experiments_dir: Path = EXPERIMENTS_DIR) -> int:
    """k = number of registration files (abandoned included), NOT completed runs."""
    return len(list_registrations(experiments_dir))


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True)


def _porcelain(repo_root: Path, *paths: str) -> str:
    out = _git(repo_root, "status", "--porcelain", "--untracked-files=all", "--", *paths)
    if out.returncode != 0:
        raise RegistrationError(f"git status failed: {out.stderr.strip()}")
    return out.stdout.strip()


def git_head(repo_root: Path = REPO_ROOT) -> str:
    out = _git(repo_root, "rev-parse", "HEAD")
    if out.returncode != 0:
        raise RegistrationError(f"git rev-parse failed: {out.stderr.strip()}")
    return out.stdout.strip()


def preflight(
    path: Path,
    *,
    repo_root: Path = REPO_ROOT,
    experiments_dir: Path = EXPERIMENTS_DIR,
    gate_records: Sequence[Mapping[str, Any]],
) -> tuple[Registration, int]:
    """Validate that `path` may be run now as a counted trial. Returns (reg, k)."""
    path = Path(path).resolve()
    repo_root = Path(repo_root).resolve()
    experiments_dir = Path(experiments_dir).resolve()
    if path.parent != experiments_dir:
        raise RegistrationError(f"{path} is not in the experiments directory {experiments_dir}")
    rel = path.relative_to(repo_root).as_posix()
    exp_rel = experiments_dir.relative_to(repo_root).as_posix()

    if _git(repo_root, "cat-file", "-e", f"HEAD:{rel}").returncode != 0:
        raise RegistrationError(f"{rel} is not committed in HEAD: commit the registration first")
    if _porcelain(repo_root, rel):
        raise RegistrationError(f"{rel} has uncommitted changes: a registration can't change after it's committed")
    dirty = _porcelain(repo_root, *CLEAN_PATHS)
    if dirty:
        raise RegistrationError(
            f"uncommitted changes under {', '.join(CLEAN_PATHS)} (results must come from committed code):\n{dirty}"
        )
    dirty_exp = _porcelain(repo_root, exp_rel)
    if dirty_exp:
        raise RegistrationError(
            f"uncommitted changes in the experiments directory {exp_rel} (the trial count must be "
            f"exactly what's committed):\n{dirty_exp}"
        )

    reg = parse_registration(path)
    if reg.abandoned:
        raise RegistrationError(f"{rel} is marked abandoned ({reg.abandoned}); it counts as a trial but is never run")
    if reg.recipe.recipe_id in evaluated_recipe_ids(gate_records):
        raise RegistrationError(
            f"recipe {reg.recipe.recipe_id} has already been evaluated on a gate holdout; "
            "re-running it would be an uncounted second look"
        )
    k = count_trials(experiments_dir)
    if reg.trial_number > k:
        raise RegistrationError(f"trial_number {reg.trial_number} > number of registrations {k}")
    return reg, k
