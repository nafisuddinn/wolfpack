"""Generate MODEL_CARD.md's gate-history table from the committed logs.

    uv run --project worker -m wolfpack_worker.analyst.render_history          # rewrite the section
    uv run --project worker -m wolfpack_worker.analyst.render_history --check  # exit 1 if stale

Every number in the table comes from worker/models/analyst/gate_log.jsonl
(alphagate decision records) and forward_log.jsonl (monitor). Nothing is
typed by hand, and the section between the BEGIN/END markers is overwritten
on every run. A test fails CI if the committed card doesn't match the logs.

Wording rules (so the table can't oversell):
  * "improved" only when a trial's paired test vs the champion's recipe was
    significant at its alpha_k (reason significant_improvement);
  * "edge" only when the model's latest forward record covers >= 126
    sessions AND is significantly better than the base rate at its
    alpha-spent look level;
  * otherwise "no detectable change" (a refresh is at most non-inferior, so
    it never reads "improved"); the bootstrap row reads "no comparison".

Reads plain JSON only (no alphagate import).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from wolfpack_worker.analyst.forward import MIN_EDGE_SESSIONS
from wolfpack_worker.analyst.gate_log import FORWARD_LOG_FILENAME, PROMOTE, read_gate_log
from wolfpack_worker.analyst.model_io import GATE_LOG_PATH, MODELS_DIR
from wolfpack_worker.analyst.registration import EXPERIMENTS_DIR, REPO_ROOT, trial_count

MODEL_CARD_PATH = REPO_ROOT / "MODEL_CARD.md"
FORWARD_LOG_PATH = MODELS_DIR / FORWARD_LOG_FILENAME
BEGIN = (
    "<!-- BEGIN GENERATED: analyst-gate-history. Written by "
    "worker/src/wolfpack_worker/analyst/render_history.py from gate_log.jsonl + forward_log.jsonl; "
    "do not edit by hand. -->"
)
END = "<!-- END GENERATED: analyst-gate-history -->"

COLUMNS = (
    "Date (UTC)",
    "Kind",
    "Recipe / hypothesis",
    "Holdout",
    "Log loss: challenger / champion / base rate",
    "Paired test vs champion",
    "Beats base rate (point estimate)?",
    "Significantly better than base rate?",
    "Decision (reason)",
    "Forward since promotion: log loss vs base rate",
    "Reading",
)


def _cell(x: Any) -> str:
    return str(x).replace("|", "\\|").replace("\n", " ")


def _f(x: Any, nd: int = 4) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "n/a"


def _g(x: Any) -> str:
    try:
        return f"{float(x):.3g}"
    except (TypeError, ValueError):
        return "n/a"


def _events(records: Sequence[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    order: list[str] = []
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for r in records:
        eid = (r.get("context") or {}).get("event_id") or r["record_id"]
        if eid not in groups:
            order.append(eid)
            groups[eid] = []
        groups[eid].append(r)
    return [groups[e] for e in order]


def _latest_forward(forward: Sequence[Mapping[str, Any]], model: str) -> Mapping[str, Any] | None:
    mine = [f for f in forward if f.get("model_version") == model]
    return mine[-1] if mine else None


def _forward_cell(promoted: bool, model: str, forward: Sequence[Mapping[str, Any]]) -> tuple[str, bool]:
    if not promoted:
        return "n/a (not deployed)", False
    f = _latest_forward(forward, model)
    if f is None:
        return "not monitored yet", False
    n = int(f.get("n_sessions") or 0)
    if n == 0:
        return "no matured sessions yet", False
    has_edge = n >= MIN_EDGE_SESSIONS and f.get("edge_significant") is True
    return f"{_f(f.get('logloss'))} vs {_f(f.get('baseline_logloss'))} over {n} sessions", has_edge


def _holdout(r: Mapping[str, Any]) -> str:
    s, e = str(r.get("holdout_start") or "")[:10], str(r.get("holdout_end") or "")[:10]
    return f"{s} to {e} ({r.get('holdout_n_samples')} sessions; end = last label bar)"


def _row(event: Sequence[Mapping[str, Any]], forward: Sequence[Mapping[str, Any]]) -> tuple[list[str], bool, str]:
    first = event[0]
    ctx = first.get("context") or {}
    kind = ctx.get("kind", "unknown")
    date = str(first.get("decided_at", ""))[:10]
    chal = first["challenger_score"]
    base_ll = (chal.get("details") or {}).get("baseline_logloss")
    recipe_id = ctx.get("recipe_id") or (first.get("challenger_metadata") or {}).get("recipe_id")

    if kind == "trial":
        main = next((r for r in event if (r.get("context") or {}).get("role") == "vs_champion_refit"), first)
        floor = next((r for r in event if (r.get("context") or {}).get("role") == "floor_vs_base_rate"), None)
        mctx = main.get("context") or {}
        promoted = all(r.get("decision") == PROMOTE for r in event) and floor is not None
        kind_s = f"trial #{mctx.get('trial_number')} (k={mctx.get('k')})"
        recipe_s = f"`{recipe_id}`: {mctx.get('hypothesis', '')}"
        champ_ll = _f((main.get("champion_score") or {}).get("value"))
        st, pa = main.get("comparator_stats") or {}, main.get("comparator_params") or {}
        test_s = f"t={_f(st.get('t'), 2)}, p={_g(st.get('p'))} vs alpha_k={_g(pa.get('alpha'))} (superiority, vs champion recipe refit)"
        floor_s = "n/a" if floor is None else ("yes" if floor.get("decision") == PROMOTE else "no")
        edge = mctx.get("edge_vs_baseline") or {}
        sig_s = f"{'yes' if mctx.get('edge_vs_baseline_significant') else 'no'} (p={_g(edge.get('p'))})"
        decision = "PROMOTE" if promoted else "REJECT"
        # The floor uses MarginComparator, whose reason codes are
        # "improved"/"not_improved"; shown as passed/failed so the word
        # "improved" only ever means a significant win over the champion.
        floor_txt = "n/a" if floor is None else (
            "passed" if floor.get("decision") == PROMOTE else f"failed ({floor.get('reason_code')})"
        )
        decision_s = f"{decision} (vs champion: {main.get('reason_code')}; base-rate floor: {floor_txt})"
        reading = "improved" if main.get("reason_code") == "significant_improvement" else "no detectable change"
    elif kind == "refresh":
        main = next((r for r in event if (r.get("context") or {}).get("role") == "vs_deployed_champion"), first)
        guard = next((r for r in event if (r.get("context") or {}).get("role") == "anchor_guard"), None)
        promoted = all(r.get("decision") == PROMOTE for r in event)
        kind_s = "refresh"
        recipe_s = f"`{recipe_id}` (same recipe, later cutoff)"
        champ_ll = _f((main.get("champion_score") or {}).get("value")) + " (deployed)"
        st, pa = main.get("comparator_stats") or {}, main.get("comparator_params") or {}
        test_s = (
            f"t={_f(st.get('t'), 2)}, p={_g(st.get('p'))} vs alpha={_g(pa.get('alpha'))} "
            f"(non-inferiority, margin {pa.get('margin')})"
        )
        sig_s = "n/a"
        if guard is None:
            guard_txt = "n/a"
        else:
            gs = guard.get("comparator_stats") or {}
            guard_txt = (
                f"{'passed' if guard.get('decision') == PROMOTE else 'failed'} "
                f"(excess {_f(gs.get('gap'))} vs max {_f(gs.get('max_gap'))})"
            )
        floor_s = f"anchor guard: {guard_txt}"
        reasons = "; ".join(f"{(r.get('context') or {}).get('role', '?')}: {r.get('reason_code')}" for r in event)
        decision_s = f"{'PROMOTE' if promoted else 'REJECT'} ({reasons})"
        reading = "no detectable change"
    else:
        promoted = first.get("decision") == PROMOTE
        kind_s = kind
        recipe_s = f"`{recipe_id}`" + (" (v1)" if kind == "bootstrap" else "")
        champ_ll = "n/a"
        test_s, floor_s, sig_s = "n/a (no incumbent)", "n/a", "n/a"
        decision_s = f"{'PROMOTE' if promoted else 'REJECT'} ({first.get('reason_code')})"
        reading = "no comparison (bootstrap)" if kind == "bootstrap" else "no detectable change"

    if floor_s == "n/a":
        # Not gated on the base rate, but both numbers are known: say so.
        try:
            beats = float(chal.get("value")) < float(base_ll)
            floor_s = f"{'yes' if beats else 'no'} (not gated)"
        except (TypeError, ValueError):
            pass
    fwd_s, has_edge = _forward_cell(promoted, first["challenger_id"], forward)
    if has_edge:
        reading += "; edge"
    cells = [
        date, kind_s, recipe_s, _holdout(first),
        f"{_f(chal.get('value'))} / {champ_ll} / {_f(base_ll)}",
        test_s, floor_s, sig_s, decision_s, fwd_s, reading,
    ]
    return [_cell(c) for c in cells], promoted, kind


def render_table(
    records: Sequence[Mapping[str, Any]], forward: Sequence[Mapping[str, Any]], *, n_registered: int
) -> str:
    rows, promoted_trials = [], 0
    for event in _events(records):
        cells, promoted, kind = _row(event, forward)
        if kind == "trial" and promoted:
            promoted_trials += 1
        rows.append("| " + " | ".join(cells) + " |")
    lines = [
        f"**{n_registered} trials registered, {promoted_trials} promoted.** "
        "(A trial is a committed registration in `worker/experiments/analyst/`, counted whether or not it ran. "
        "Refreshes and the one-time bootstrap are not trials.)",
        "",
        "Log loss is the per-session mean across the 5 tickers (lower is better). \"Base rate\" = always "
        "predicting the training up-rate. \"Reading\" uses fixed words: *improved* only if the trial's paired "
        "test vs the champion's recipe was significant at its alpha_k; *edge* only if the model's forward "
        f"record covers at least {MIN_EDGE_SESSIONS} sessions and is significantly better than the base rate; "
        "otherwise *no detectable change*.",
        "",
        "| " + " | ".join(COLUMNS) + " |",
        "|" + "---|" * len(COLUMNS),
        *rows,
    ]
    if not rows:
        lines.append("| " + " | ".join(["(no gate decisions yet)"] + [""] * (len(COLUMNS) - 1)) + " |")
    return "\n".join(lines)


def splice(card: str, table: str) -> str:
    if card.count(BEGIN) != 1 or card.count(END) != 1 or card.index(BEGIN) > card.index(END):
        raise ValueError("MODEL_CARD.md must contain exactly one BEGIN and one END marker, in order")
    head, rest = card.split(BEGIN)
    _, tail = rest.split(END)
    return f"{head}{BEGIN}\n{table}\n{END}{tail}"


def current_table(
    gate_log_path: Path = GATE_LOG_PATH,
    forward_log_path: Path = FORWARD_LOG_PATH,
    experiments_dir: Path = EXPERIMENTS_DIR,
) -> str:
    records = read_gate_log(gate_log_path)
    return render_table(records, read_gate_log(forward_log_path), n_registered=trial_count(experiments_dir, records))


def write(model_card_path: Path = MODEL_CARD_PATH) -> bool:
    """Rewrite the generated section. Returns True if the file changed."""
    old = Path(model_card_path).read_text(encoding="utf-8")
    new = splice(old, current_table())
    if new != old:
        Path(model_card_path).write_text(new, encoding="utf-8")
    return new != old


def check(model_card_path: Path = MODEL_CARD_PATH) -> bool:
    old = Path(model_card_path).read_text(encoding="utf-8")
    return splice(old, current_table()) == old


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="exit 1 if MODEL_CARD.md is stale; write nothing")
    args = parser.parse_args(argv)
    if args.check:
        ok = check()
        print("MODEL_CARD.md gate history is up to date." if ok else "MODEL_CARD.md gate history is STALE.")
        return 0 if ok else 1
    changed = write()
    print("MODEL_CARD.md gate history " + ("updated." if changed else "already up to date."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
