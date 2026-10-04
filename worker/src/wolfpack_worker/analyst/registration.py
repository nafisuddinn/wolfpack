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

Each persona has its own experiments directory and trial count (its
`PersonaPaths`, analyst/paths.py); every function here defaults to The
Analyst's. The registration's `[recipe]` table is parsed by
`paths.recipe_parser`.

`preflight` refuses to start an experiment unless: the file is committed in
HEAD and unchanged; worker/src, worker/recipes, worker/models (the gate and
forward logs), worker/pyproject.toml, worker/uv.lock and the experiments
directory have no uncommitted changes (so the code, the logs, and the trial
count that produced a result are exactly what's in git); the file is not
abandoned; and `verify_runnable` passes.

The gate log is the second witness to the trial count. Every trial gate
record carries its trial_number, registration_file and registration_sha256,
and `check_log_consistency` requires each logged trial to still have its
registration, under the same file name, with the same bytes. So a trial that
ran can't be edited into a different recipe and re-run under the same number,
deleted to shrink k, or renamed. k = max(registration files, distinct trial
numbers in the log).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import re
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from wolfpack_worker.analyst.gate_log import evaluated_recipe_ids
from wolfpack_worker.analyst.paths import ANALYST_PATHS, REPO_ROOT, PersonaPaths
from wolfpack_worker.analyst.recipe import Recipe, RecipeError

EXPERIMENTS_DIR = ANALYST_PATHS.experiments_dir
# Paths (relative to the repo root) that must be clean for a trial to run.
CLEAN_PATHS = ("worker/src", "worker/recipes", "worker/models", "worker/pyproject.toml", "worker/uv.lock")

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
    recipe: Recipe  # (or another persona's recipe type, via paths.recipe_parser)
    abandoned: str | None


def _dir(experiments_dir: Path | None, paths: PersonaPaths) -> Path:
    return Path(experiments_dir) if experiments_dir is not None else paths.experiments_dir


def parse_registration(path: Path, *, paths: PersonaPaths = ANALYST_PATHS) -> Registration:
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
        recipe = paths.recipe_parser(data["recipe"])
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


def list_registrations(
    experiments_dir: Path | None = None, *, paths: PersonaPaths = ANALYST_PATHS
) -> list[Registration]:
    """Every registration, sorted by trial number, validated as a set:
    numbers are exactly 1..k and no two files register the same recipe.
    Any *.toml that isn't a valid registration is an error (not ignored).
    `experiments_dir` defaults to `paths.experiments_dir`."""
    experiments_dir = _dir(experiments_dir, paths)
    if not experiments_dir.is_dir():
        return []
    regs = sorted(
        (parse_registration(p, paths=paths) for p in experiments_dir.glob("*.toml")),
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


def count_trials(experiments_dir: Path | None = None, *, paths: PersonaPaths = ANALYST_PATHS) -> int:
    """Number of registration files (abandoned included), NOT completed runs.
    The trial count used for alpha_k is `trial_count`, which also consults
    the gate log."""
    return len(list_registrations(experiments_dir, paths=paths))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def logged_trials(gate_records: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    """trial_number -> {registration_file, registration_sha256, event_id,
    record_ids} for every trial in the gate log. A trial number that appears
    in more than one gate event was run more than once: refused."""
    out: dict[int, dict[str, Any]] = {}
    for r in gate_records:
        ctx = r.get("context") or {}
        if ctx.get("kind") != "trial":
            continue
        n = ctx.get("trial_number")
        if isinstance(n, bool) or not isinstance(n, int):
            raise RegistrationError(f"gate record {r.get('record_id')} is a trial without an int trial_number")
        info = out.get(n)
        if info is None:
            out[n] = {
                "registration_file": ctx.get("registration_file"),
                "registration_sha256": ctx.get("registration_sha256"),
                "event_id": ctx.get("event_id"),
                "record_ids": [r.get("record_id")],
            }
        elif info["event_id"] != ctx.get("event_id") or info["registration_file"] != ctx.get("registration_file"):
            raise RegistrationError(
                f"trial {n} appears in the gate log more than once (events {info['event_id']} and "
                f"{ctx.get('event_id')}): a trial number can only ever be run once"
            )
        else:
            info["record_ids"].append(r.get("record_id"))
    return out


def check_log_consistency(regs: Sequence[Registration], gate_records: Sequence[Mapping[str, Any]]) -> None:
    """Every logged trial must still have its registration: same number, same
    file name, same bytes (when the log recorded a hash)."""
    by_num = {r.trial_number: r for r in regs}
    for n, info in sorted(logged_trials(gate_records).items()):
        logged_name = Path(str(info["registration_file"] or "")).name
        reg = by_num.get(n)
        if reg is None:
            raise RegistrationError(
                f"trial {n} is in the gate log but its registration {logged_name or '(unknown)'} is missing "
                "(deleted or renumbered); registrations of run trials are permanent"
            )
        if reg.path.name != logged_name:
            raise RegistrationError(
                f"trial {n} was run as {logged_name} but is now {reg.path.name}: renamed registrations "
                "are refused"
            )
        sha = info["registration_sha256"]
        if sha and file_sha256(reg.path) != sha:
            raise RegistrationError(
                f"{reg.path.name} was edited after it was run (sha256 differs from the gate log); "
                "restore it from git"
            )


def trial_count(
    experiments_dir: Path | None,
    gate_records: Sequence[Mapping[str, Any]],
    *,
    paths: PersonaPaths = ANALYST_PATHS,
) -> int:
    """k = max(registration files, distinct trial numbers in the gate log),
    after checking the two agree. Abandoned registrations count. Pass the
    SAME persona's directory and gate log: each persona has its own k."""
    regs = list_registrations(experiments_dir, paths=paths)
    check_log_consistency(regs, gate_records)
    return max(len(regs), len(logged_trials(gate_records)))


def verify_runnable(
    path: Path,
    *,
    experiments_dir: Path,
    gate_records: Sequence[Mapping[str, Any]],
    paths: PersonaPaths = ANALYST_PATHS,
) -> tuple[Registration, int]:
    """Everything preflight checks except git: may this registration be run
    now as a counted trial? Returns (registration, k). gating.run_experiment
    calls this itself, so it never trusts a caller-supplied k."""
    path = Path(path).resolve()
    experiments_dir = Path(experiments_dir).resolve()
    if path.parent != experiments_dir:
        raise RegistrationError(f"{path} is not in the experiments directory {experiments_dir}")
    reg = parse_registration(path, paths=paths)
    if reg.abandoned:
        raise RegistrationError(f"{path.name} is marked abandoned ({reg.abandoned}); it counts as a trial but is never run")
    logged = logged_trials(gate_records)
    if reg.trial_number in logged:
        raise RegistrationError(
            f"trial {reg.trial_number} has already been run (gate records {logged[reg.trial_number]['record_ids']}); "
            "a registration runs once. A new idea is a new registration with the next number"
        )
    if path.name in {Path(str(i["registration_file"] or "")).name for i in logged.values()}:
        raise RegistrationError(f"{path.name} has already been run")
    if reg.recipe.recipe_id in evaluated_recipe_ids(gate_records):
        raise RegistrationError(
            f"recipe {reg.recipe.recipe_id} has already been evaluated on a gate holdout; "
            "re-running it would be an uncounted second look"
        )
    k = trial_count(experiments_dir, gate_records, paths=paths)
    if reg.trial_number > k:
        raise RegistrationError(f"trial_number {reg.trial_number} > number of trials {k}")
    return reg, k


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
    experiments_dir: Path | None = None,
    gate_records: Sequence[Mapping[str, Any]],
    paths: PersonaPaths = ANALYST_PATHS,
) -> tuple[Registration, int]:
    """Validate that `path` may be run now as a counted trial. Returns (reg, k)."""
    path = Path(path).resolve()
    repo_root = Path(repo_root).resolve()
    experiments_dir = _dir(experiments_dir, paths).resolve()
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

    return verify_runnable(path, experiments_dir=experiments_dir, gate_records=gate_records, paths=paths)
