"""CI guard: every gated persona's gate and forward logs are append-only.

    python -m wolfpack_worker.analyst.log_guard --base <git ref>

Fails (exit 1) if, for any persona's gate_log.jsonl or forward_log.jsonl
(worker/models/<persona>/, one pair per PersonaPaths: The Analyst and The
Scout), the copy at `--base` is not a byte-prefix of the working
tree's copy, i.e. if any already-logged decision was deleted, edited, or
reordered (so a rejected challenger can't be quietly removed and its recipe
re-run). Appending new lines is the only allowed change.

* File absent at base: OK (first-ever commit of the log).
* File present at base but deleted now: violation.
* `--base` empty or all zeros (a push that created a branch, or a manual
  run): skipped with a message, exit 0.
* An unknown base ref is an error, never a silent pass.

Standard library only, so CI can run it without the `train` group. That is
why LOG_PATHS is a literal list rather than read from PersonaPaths; a test
(tests/test_persona_paths.py) fails if it misses any persona's logs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[4]
LOG_PATHS = (
    "worker/models/analyst/gate_log.jsonl",
    "worker/models/analyst/forward_log.jsonl",
    "worker/models/scout/gate_log.jsonl",
    "worker/models/scout/forward_log.jsonl",
)


class LogGuardError(RuntimeError):
    """The base ref could not be read (bad ref, git failure)."""


def append_only_violation(base: Optional[bytes], head: Optional[bytes]) -> Optional[str]:
    """None if `head` only appends whole lines to `base`; else a reason."""
    if base is None or base == b"":
        return None
    if head is None:
        return "the file was deleted (it existed on the base branch)"
    if not base.endswith(b"\n"):
        return "the base copy does not end with a newline, so anything after it would rewrite its last line"
    if head.startswith(base):
        return None
    base_lines = base.splitlines(keepends=True)
    head_lines = head.splitlines(keepends=True)
    for i, line in enumerate(base_lines, 1):
        if i > len(head_lines):
            return f"line {i} onwards was removed (base has {len(base_lines)} lines, now {len(head_lines)})"
        if head_lines[i - 1] != line:
            return f"line {i} was changed, removed, or reordered; logged lines may only be appended to"
    return "the base copy is not a prefix of the current copy"


def read_at_ref(repo: Path, ref: str, relpath: str) -> Optional[bytes]:
    """Bytes of `relpath` at `ref`; None if the path doesn't exist there."""
    ok = subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
                        cwd=repo, capture_output=True)
    if ok.returncode != 0:
        raise LogGuardError(f"base ref {ref!r} is not a commit in {repo} (fetch-depth: 0 needed in CI?)")
    exists = subprocess.run(["git", "cat-file", "-e", f"{ref}:{relpath}"], cwd=repo, capture_output=True)
    if exists.returncode != 0:
        return None
    out = subprocess.run(["git", "show", f"{ref}:{relpath}"], cwd=repo, capture_output=True)
    if out.returncode != 0:
        raise LogGuardError(f"git show {ref}:{relpath} failed: {out.stderr.decode(errors='replace').strip()}")
    return out.stdout


def check(repo: Path, base_ref: str, paths: Sequence[str] = LOG_PATHS) -> list[str]:
    """Violations (one string per bad file) of the working tree vs `base_ref`."""
    repo = Path(repo)
    problems = []
    for rel in paths:
        base = read_at_ref(repo, base_ref, rel)
        p = repo / rel
        head = p.read_bytes() if p.exists() else None
        v = append_only_violation(base, head)
        if v:
            problems.append(f"{rel}: {v}")
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="", help="git ref of the base (e.g. origin/main, or a push's before-SHA)")
    parser.add_argument("--repo", default=str(REPO_ROOT))
    args = parser.parse_args(argv)
    base = args.base.strip()
    if not base or set(base) == {"0"}:
        print("log_guard: no base ref (manual run or new-branch push); append-only check skipped.")
        return 0
    problems = check(Path(args.repo), base)
    if problems:
        print("log_guard: gate/forward logs must be append-only vs " + base + ":")
        for p in problems:
            print("  " + p)
        return 1
    print(f"log_guard: gate/forward logs only appended to vs {base}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
