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
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

GATE_LOG_FILENAME = "gate_log.jsonl"
FORWARD_LOG_FILENAME = "forward_log.jsonl"

PROMOTE = "promote"
REJECT = "reject"


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
) -> list[Mapping[str, Any]]:
    """Raise GateLogError unless the log shows this exact model was promoted.

    Requires:
      * at least one PROMOTE record whose challenger_id == model_version;
      * NO reject record for that challenger_id (a model that failed any
        gate call, e.g. passed the champion test but failed the baseline
        floor, must never be deployed);
      * every PROMOTE record for it was chronology-checked and names the
        same model bytes (challenger_metadata.model_sha256);
      * every id in the manifest's gate_record_ids is one of those records.
    Returns the PROMOTE records.
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


def evaluated_recipe_ids(records: Iterable[Mapping[str, Any]]) -> set[str]:
    """recipe_ids that have already been scored as a challenger on a gate holdout."""
    out = set()
    for r in records:
        rid = (r.get("challenger_metadata") or {}).get("recipe_id")
        if rid:
            out.add(rid)
    return out
