"""Append-only check for gate_log.jsonl / forward_log.jsonl (log_guard.py).

CI fails a PR if the base branch's copy of either log is not a byte-prefix of
the PR's copy, so a logged rejection can't be deleted, edited, or reordered
after the fact.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from wolfpack_worker.analyst import log_guard as lg

R1 = b'{"record_id": "r1", "decision": "reject"}\n'
R2 = b'{"record_id": "r2", "decision": "promote"}\n'
R3 = b'{"record_id": "r3", "decision": "reject"}\n'


def test_appended_record_is_ok():
    assert lg.append_only_violation(R1 + R2, R1 + R2 + R3) is None


def test_unchanged_is_ok():
    assert lg.append_only_violation(R1 + R2, R1 + R2) is None


def test_absent_on_base_is_ok():
    assert lg.append_only_violation(None, R1) is None
    assert lg.append_only_violation(None, None) is None
    assert lg.append_only_violation(b"", R1) is None


def test_deleted_record_fails():
    v = lg.append_only_violation(R1 + R2 + R3, R1 + R3)
    assert v and "line 2" in v


def test_edited_record_fails():
    edited = R2.replace(b"promote", b"PROMOTE")
    v = lg.append_only_violation(R1 + R2, R1 + edited + R3)
    assert v and "line 2" in v


def test_reordered_records_fail():
    v = lg.append_only_violation(R1 + R2, R2 + R1)
    assert v and "line 1" in v


def test_truncated_or_deleted_file_fails():
    assert lg.append_only_violation(R1 + R2, R1)
    assert "deleted" in lg.append_only_violation(R1, None)


def test_append_must_start_on_a_new_line():
    # Base's last line lacks its newline; "appending" would rewrite that line.
    assert lg.append_only_violation(R1.rstrip(b"\n"), R1 + R2)


# --- against real git refs ------------------------------------------------------------


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false", *args],
        cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.strip()


@pytest.fixture()
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "worker/models/analyst").mkdir(parents=True)
    (root / "README").write_text("x\n")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "base")
    return root


def _write_commit(repo: Path, name: str, data: bytes) -> None:
    (repo / "worker/models/analyst" / name).write_bytes(data)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", name)


def test_check_against_git_base_ref(repo):
    base_no_logs = _git(repo, "rev-parse", "HEAD")
    _write_commit(repo, "gate_log.jsonl", R1)
    # First-ever commit of the file: no base copy -> OK.
    assert lg.check(repo, base_no_logs) == []
    base = _git(repo, "rev-parse", "HEAD")
    _write_commit(repo, "gate_log.jsonl", R1 + R2)
    _write_commit(repo, "forward_log.jsonl", b'{"n": 1}\n')
    assert lg.check(repo, base) == []
    base2 = _git(repo, "rev-parse", "HEAD")
    _write_commit(repo, "gate_log.jsonl", R2)  # r1 deleted
    problems = lg.check(repo, base2)
    assert len(problems) == 1 and "gate_log.jsonl" in problems[0]


def test_cli_skips_without_a_base_and_fails_on_violation(repo, capsys):
    assert lg.main(["--repo", str(repo), "--base", ""]) == 0
    assert lg.main(["--repo", str(repo), "--base", "0" * 40]) == 0  # new-branch push
    assert "skipped" in capsys.readouterr().out
    _write_commit(repo, "gate_log.jsonl", R1 + R2)
    base = _git(repo, "rev-parse", "HEAD")
    assert lg.main(["--repo", str(repo), "--base", base]) == 0
    _write_commit(repo, "gate_log.jsonl", R2 + R1)
    assert lg.main(["--repo", str(repo), "--base", base]) == 1


def test_unknown_base_ref_is_an_error_not_a_pass(repo):
    with pytest.raises(lg.LogGuardError):
        lg.check(repo, "no-such-ref")


def test_ci_workflow_runs_the_guard_with_full_history():
    root = Path(__file__).resolve().parents[2]
    text = (root / ".github/workflows/worker-tests.yml").read_text()
    assert "fetch-depth: 0" in text
    assert "wolfpack_worker.analyst.log_guard" in text
