"""CI checks over The Scout's COMMITTED registrations, gate log and card.

Mirror of The Analyst's checks in test_analyst_render.py, on SCOUT_PATHS.
"""

from __future__ import annotations

import pytest

from wolfpack_worker.analyst import render_history as rh
from wolfpack_worker.analyst.gate_log import read_gate_log
from wolfpack_worker.analyst.model_io import MANIFEST_FILENAME, load_champion
from wolfpack_worker.analyst.paths import REPO_ROOT
from wolfpack_worker.analyst.registration import list_registrations, trial_count
from wolfpack_worker.scout.paths import SCOUT_PATHS

P = SCOUT_PATHS


def test_scout_registrations_agree_with_the_scout_gate_log():
    assert trial_count(None, read_gate_log(P.gate_log_path), paths=P) >= 0


def test_every_registered_scout_trial_has_a_gate_record_or_is_abandoned():
    logged = {r["context"].get("registration_file") for r in read_gate_log(P.gate_log_path)
              if r.get("context", {}).get("kind") == "trial"}
    missing = [reg.path.resolve().relative_to(REPO_ROOT).as_posix() for reg in list_registrations(paths=P)
               if reg.abandoned is None
               and reg.path.resolve().relative_to(REPO_ROOT).as_posix() not in logged]
    assert not missing, f"registered Scout trial(s) with no gate record and no `abandoned` line: {missing}"


def test_every_scout_trial_record_points_at_a_registration_and_logs_its_alpha():
    regs = {r.path.resolve().relative_to(REPO_ROOT).as_posix() for r in list_registrations(paths=P)}
    for r in read_gate_log(P.gate_log_path):
        ctx = r.get("context", {})
        if ctx.get("kind") == "trial":
            assert ctx["registration_file"] in regs, r["record_id"]
            assert ctx["alpha_total"] > 0 and ctx["alpha_k"] == pytest.approx(
                ctx["alpha_total"] / (ctx["k"] * (ctx["k"] + 1)))


def test_scout_champion_exists_iff_the_log_promoted_one():
    promoted = any(r.get("decision") == "promote" for r in read_gate_log(P.gate_log_path))
    has_champion = (P.champion_dir / MANIFEST_FILENAME).exists()
    assert promoted == has_champion
    if has_champion:
        load_champion(paths=P)


def test_no_headline_text_or_urls_in_committed_scout_files():
    """Benzinga headlines and URLs are licensed and must never be committed."""
    roots = [P.models_dir, P.experiments_dir, REPO_ROOT / "worker" / "reports" / "scout"]
    for root in roots:
        if not root.exists():
            continue
        for f in root.rglob("*"):
            if f.is_file():
                text = f.read_text(encoding="utf-8", errors="replace").lower()
                assert "benzinga.com/" not in text and "http://" not in text and "https://" not in text, f


def test_scout_model_card_history_matches_the_committed_logs():
    card = P.model_card_path
    if not (card.is_file() and rh.has_markers(card.read_text(encoding="utf-8"), P)):
        pytest.skip("MODEL_CARD_SCOUT.md has no scout-gate-history markers yet (documentarian adds them)")
    assert rh.check(paths=P), "run `python -m wolfpack_worker.scout.retrain render`"
