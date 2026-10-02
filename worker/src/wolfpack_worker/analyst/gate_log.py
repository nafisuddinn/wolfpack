"""Read The Analyst's promotion-gate log (`gate_log.jsonl`) as plain JSON.

alphagate's `JsonlSink` appends one `GateRecord.to_json()` line per gate()
call, promotions and rejections alike. This module reads those lines WITHOUT
importing alphagate, because the daily trade cron (which must verify that
its champion was promoted by the gate) does not install the `train`
dependency group. Only the handful of fields read here are relied on.

Nothing in here can create a promotion: it only reads.
"""

from __future__ import annotations

import json
import math
import numbers
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

GATE_LOG_FILENAME = "gate_log.jsonl"
FORWARD_LOG_FILENAME = "forward_log.jsonl"

PROMOTE = "promote"
REJECT = "reject"

# Gate events WolfPack produces (gating.py), and the comparators it uses.
VALID_KINDS = frozenset({"bootstrap", "refresh", "trial"})
RECOGNISED_COMPARATORS = frozenset({"margin", "paired_dm", "anchored_gap"})

# trial_number -> (registration file name, sha256 of its bytes)
RegistrationIndex = Mapping[int, tuple[str, str]]


class GateLogError(ValueError):
    """The gate log is unreadable, or does not support the claimed promotion."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Every non-blank line as a dict. A malformed line raises (never skipped):
    a silently dropped REJECT line would weaken the backstop."""
    path = Path(path)
    out: list[dict[str, Any]] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GateLogError(f"{path}: line {i} is not valid JSON ({exc})") from None
        if not isinstance(rec, dict):
            raise GateLogError(f"{path}: line {i} is not a JSON object")
        out.append(rec)
    return out


def read_gate_log(path: Path) -> list[dict[str, Any]]:
    """All records, in file (= decision) order; [] if the file doesn't exist."""
    path = Path(path)
    return read_jsonl(path) if path.exists() else []


def records_for(records: Iterable[Mapping[str, Any]], challenger_id: str) -> list[Mapping[str, Any]]:
    return [r for r in records if r.get("challenger_id") == challenger_id]


def verify_promotion(
    records: Sequence[Mapping[str, Any]],
    *,
    model_version: str,
    model_sha256: str,
    gate_record_ids: Sequence[str],
    registrations: Optional[Callable[[], RegistrationIndex]] = None,
) -> list[Mapping[str, Any]]:
    """Raise GateLogError unless the log shows this exact model was promoted.

    Requires:
      * at least one PROMOTE record whose challenger_id == model_version;
      * NO reject record for that challenger_id (a model that failed any
        gate call, e.g. passed the champion test but failed the baseline
        floor, must never be deployed);
      * every PROMOTE record for it was chronology-checked and names the
        same model bytes (challenger_metadata.model_sha256);
      * every id in the manifest's gate_record_ids is one of those records;
      * each PROMOTE record looks like a real gate decision: context.kind is
        bootstrap/refresh/trial, a finite challenger_score with per-session
        samples, a recognised comparator; a bootstrap record is the first
        line of the log; a trial record's trial_number matches a registration
        file (same name, and same bytes if the log recorded a hash).
    These are consistency checks against accidents and casual hand edits,
    not cryptographic proof: someone writing a fully plausible record by hand
    can still forge one. Returns the PROMOTE records.
    """
    if not gate_record_ids:
        raise GateLogError("manifest gate_record_ids is empty")
    mine = records_for(records, model_version)
    rejects = [r for r in mine if r.get("decision") == REJECT]
    if rejects:
        raise GateLogError(
            f"gate log has REJECT record(s) {[r.get('record_id') for r in rejects]} for "
            f"{model_version!r}: a model that failed any gate call is never deployed"
        )
    promotes = [r for r in mine if r.get("decision") == PROMOTE]
    if not promotes:
        raise GateLogError(
            f"no PROMOTE record for {model_version!r} in the gate log: this champion was not "
            "promoted by the alphagate gate (was the manifest swapped or written by hand?)"
        )
    for r in promotes:
        _check_shape(r, records, model_version, registrations)
        if r.get("chronology_checked") is not True:
            raise GateLogError(
                f"PROMOTE record {r.get('record_id')} for {model_version!r} was not "
                "chronology-checked (allow_untimed); not acceptable for a champion"
            )
        sha = (r.get("challenger_metadata") or {}).get("model_sha256")
        if sha != model_sha256:
            raise GateLogError(
                f"PROMOTE record {r.get('record_id')} names model sha256 {sha!r}, but the "
                f"champion file is {model_sha256!r}: the promoted model is not this file"
            )
    promote_ids = {r.get("record_id") for r in promotes}
    unknown = [rid for rid in gate_record_ids if rid not in promote_ids]
    if unknown:
        raise GateLogError(
            f"manifest gate_record_ids {unknown} are not PROMOTE records for {model_version!r}"
        )
    return promotes


def _is_finite_number(x: Any) -> bool:
    return isinstance(x, numbers.Real) and not isinstance(x, bool) and math.isfinite(float(x))


def _check_shape(
    r: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    model_version: str,
    registrations: Optional[Callable[[], RegistrationIndex]],
) -> None:
    rid = r.get("record_id")
    ctx = r.get("context") or {}
    kind = ctx.get("kind")
    if kind not in VALID_KINDS:
        raise GateLogError(f"PROMOTE record {rid} has context.kind {kind!r}; expected one of {sorted(VALID_KINDS)}")
    score = r.get("challenger_score")
    if not (
        isinstance(score, Mapping)
        and _is_finite_number(score.get("value"))
        and isinstance(score.get("samples"), list)
        and score["samples"]
        and all(_is_finite_number(x) for x in score["samples"])
    ):
        raise GateLogError(f"PROMOTE record {rid} has no real challenger_score (finite value + per-session samples)")
    if r.get("comparator_name") not in RECOGNISED_COMPARATORS:
        raise GateLogError(
            f"PROMOTE record {rid} used comparator {r.get('comparator_name')!r}; expected one of "
            f"{sorted(RECOGNISED_COMPARATORS)}"
        )
    if kind == "bootstrap" and (not records or records[0].get("record_id") != rid):
        raise GateLogError(f"bootstrap PROMOTE record {rid} is not the first record of the gate log")
    if kind == "trial":
        n = ctx.get("trial_number")
        index = registrations() if registrations is not None else {}
        if n not in index:
            raise GateLogError(f"trial PROMOTE record {rid}: no registration file for trial_number {n!r}")
        name, sha = index[n]
        logged_name = Path(str(ctx.get("registration_file") or "")).name
        if name != logged_name:
            raise GateLogError(
                f"trial PROMOTE record {rid}: registration for trial {n} is {name}, but the record names {logged_name!r}"
            )
        logged_sha = ctx.get("registration_sha256")
        if logged_sha and logged_sha != sha:
            raise GateLogError(f"trial PROMOTE record {rid}: registration {name} was edited after it was run")


def evaluated_recipe_ids(records: Iterable[Mapping[str, Any]]) -> set[str]:
    """recipe_ids that have already been scored as a challenger on a gate holdout."""
    out = set()
    for r in records:
        rid = (r.get("challenger_metadata") or {}).get("recipe_id")
        if rid:
            out.add(rid)
    return out
