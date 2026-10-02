"""MODEL_CARD.md's generated gate-history table (render_history) and the CI
checks over the committed logs: every registered trial has a gate record (or
an explicit `abandoned` line), and the model card matches the logs."""

from __future__ import annotations

import pytest

from wolfpack_worker.analyst import render_history as rh
from wolfpack_worker.analyst.gate_log import read_gate_log
from wolfpack_worker.analyst.model_io import GATE_LOG_PATH
from wolfpack_worker.analyst.registration import EXPERIMENTS_DIR, REPO_ROOT, list_registrations


def _rec(rid, kind, *, event, decision="promote", reason="no_incumbent", role=None, challenger="analyst-x",
         champion=None, chal_ll=0.69, champ_ll=None, base_ll=0.692, stats=None, params=None, ctx=None,
         comparator="margin"):
    return {
        "record_id": rid,
        "decided_at": "2026-10-05T12:00:00+00:00",
        "decision": decision,
        "reason_code": reason,
        "challenger_id": challenger,
        "champion_id": champion,
        "holdout_start": "2025-10-01T04:00:00+00:00",
        "holdout_end": "2026-10-01T04:00:00+00:00",
        "holdout_n_samples": 252,
        "challenger_score": {"value": chal_ll, "samples": None, "details": {"baseline_logloss": base_ll}},
        "champion_score": None if champ_ll is None else {"value": champ_ll, "samples": None, "details": {}},
        "comparator_name": comparator,
        "comparator_params": params or {},
        "comparator_stats": stats or {},
        "challenger_metadata": {"recipe_id": "aaaaaaaaaaaa"},
        "context": {"kind": kind, "event_id": event, **({"role": role} if role else {}), **(ctx or {})},
    }


def _trial(n, *, p, alpha, floor=True, edge_sig=False, challenger="analyst-t"):
    ctx = {"trial_number": n, "k": 2, "alpha_k": alpha, "recipe_id": "bbbbbbbbbbbb",
           "hypothesis": "Deeper | trees.", "edge_vs_baseline": {"p": 0.2}, "edge_vs_baseline_significant": edge_sig}
    sig = p < alpha
    r1 = _rec("r1", "trial", event=f"e{n}", role="vs_champion_refit", challenger=challenger,
              champion="refit-x", decision="promote" if sig else "reject",
              reason="significant_improvement" if sig else "not_significant", chal_ll=0.680, champ_ll=0.690,
              stats={"t": 2.5, "p": p}, params={"alpha": alpha, "mode": "superiority"}, comparator="paired_dm",
              ctx=ctx)
    r2 = _rec("r2", "trial", event=f"e{n}", role="floor_vs_base_rate", challenger=challenger,
              champion="base-rate", decision="promote" if floor else "reject",
              reason="improved" if floor else "not_improved", chal_ll=0.680, champ_ll=0.692, ctx=ctx)
    return [r1, r2]


def _fwd(model, n, sig, ll=0.68, base=0.69):
    return {"model_version": model, "n_sessions": n, "logloss": ll, "baseline_logloss": base,
            "edge_significant": sig, "p": 0.001 if sig else 0.4}


def _row(text, needle):
    return next(line for line in text.splitlines() if line.startswith("| 20") and needle in line)


def test_bootstrap_row_and_header():
    recs = [_rec("b", "bootstrap", event="e0", challenger="analyst-v1", chal_ll=0.6963, base_ll=0.6902,
                 ctx={"recipe_id": "a8e2709b0d8e"})]
    out = rh.render_table(recs, [], n_registered=0)
    assert "0 trials registered, 0 promoted." in out
    row = _row(out, "bootstrap")
    assert "0.6963" in row and "0.6902" in row and "PROMOTE (no_incumbent)" in row
    assert "no comparison (bootstrap)" in row
    assert "| no (not gated) |" in row  # 0.6963 is worse than the 0.6902 base rate
    assert "improved" not in row and "edge" not in row.split("|")[-2]


def test_significant_trial_reads_improved():
    out = rh.render_table(_trial(1, p=0.001, alpha=0.025), [], n_registered=2)
    row = _row(out, "trial #1")
    assert "improved" in row and "PROMOTE" in row
    assert "1 promoted" in out and "2 trials registered" in out
    assert "Deeper \\| trees." in row  # pipes escaped inside a table cell


def test_not_significant_trial_reads_no_detectable_change_even_if_point_estimate_better():
    out = rh.render_table(_trial(1, p=0.03, alpha=0.025), [], n_registered=1)
    row = _row(out, "trial #1")
    assert "no detectable change" in row and "improved" not in row
    assert "base-rate floor: passed" in row
    assert "REJECT" in row and "0 promoted" in out


def test_trial_promotes_only_if_floor_also_passes():
    out = rh.render_table(_trial(1, p=0.001, alpha=0.025, floor=False), [], n_registered=1)
    row = _row(out, "trial #1")
    assert "REJECT" in row and "0 promoted" in out
    assert "| no |" in row  # floor column


def test_refresh_promotion_is_never_called_an_improvement():
    rec = _rec("f", "refresh", event="e5", challenger="analyst-r", champion="analyst-v1", reason="non_inferior",
               champ_ll=0.6965, comparator="paired_dm", stats={"t": 2.0, "p": 0.02},
               params={"alpha": 0.05, "mode": "non_inferiority", "margin": 0.002})
    row = _row(rh.render_table([rec], [], n_registered=0), "refresh")
    assert "no detectable change" in row and "PROMOTE (non_inferior)" in row


def test_edge_requires_126_sessions_and_significance():
    recs = _trial(1, p=0.001, alpha=0.025)
    short = rh.render_table(recs, [_fwd("analyst-t", 100, True)], n_registered=1)
    assert "edge" not in _row(short, "trial #1").split("|")[-2]
    insignificant = rh.render_table(recs, [_fwd("analyst-t", 200, False)], n_registered=1)
    assert "edge" not in _row(insignificant, "trial #1").split("|")[-2]
    real = rh.render_table(recs, [_fwd("analyst-t", 100, False), _fwd("analyst-t", 200, True)], n_registered=1)
    assert "edge" in _row(real, "trial #1").split("|")[-2]
    assert "200 sessions" in _row(real, "trial #1")  # latest forward record is shown


def test_rejected_challenger_shows_not_deployed_forward():
    out = rh.render_table(_trial(1, p=0.5, alpha=0.025), [], n_registered=1)
    assert "not deployed" in _row(out, "trial #1")


def test_splice_between_markers():
    card = f"intro\n{rh.BEGIN}\nold\n{rh.END}\noutro\n"
    new = rh.splice(card, "TABLE")
    assert new == f"intro\n{rh.BEGIN}\nTABLE\n{rh.END}\noutro\n"
    with pytest.raises(ValueError, match="marker"):
        rh.splice("no markers here", "TABLE")


# --- CI checks over the committed repository state --------------------------------------


def test_every_registered_trial_has_a_gate_record_or_is_abandoned():
    records = read_gate_log(GATE_LOG_PATH)
    logged = {
        r["context"].get("registration_file")
        for r in records
        if r.get("context", {}).get("kind") == "trial"
    }
    missing = []
    for reg in list_registrations(EXPERIMENTS_DIR):
        rel = reg.path.resolve().relative_to(REPO_ROOT).as_posix()
        if reg.abandoned is None and rel not in logged:
            missing.append(rel)
    assert not missing, (
        f"registered trial(s) with no gate record and no `abandoned` line: {missing}. "
        "Run them (retrain experiment <file>) or mark them abandoned; never delete a registration."
    )


def test_every_trial_gate_record_points_at_a_registration_file():
    regs = {r.path.resolve().relative_to(REPO_ROOT).as_posix() for r in list_registrations(EXPERIMENTS_DIR)}
    for r in read_gate_log(GATE_LOG_PATH):
        if r.get("context", {}).get("kind") == "trial":
            assert r["context"]["registration_file"] in regs, r["record_id"]


def test_model_card_history_matches_the_committed_logs():
    assert rh.check(), (
        "MODEL_CARD.md's generated gate-history section is stale: run "
        "`uv run --project worker -m wolfpack_worker.analyst.render_history`"
    )
