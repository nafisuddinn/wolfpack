"""The Scout's promotion gate (alphagate), gate-or-fallback.

Design section 6 + Decision Log 2026-10-04 ("gate-or-fallback"): The
Scout's model trades only if a pre-registered trial passes the gate;
otherwise no champion exists and the strategy runs its untuned rule
(`rule_fallback`). Unlike The Analyst's bootstrap, nothing is promoted
"anyway".

While The Scout has no champion, a trial is ONE alphagate `gate()` call:

    challenger = the registered recipe, trained on the pre-holdout rows
    champion   = the constant base-rate predictor (training up-rate)
    holdout    = the trailing 252 labeled sessions (per-session mean log
                 loss across the 5 tickers; paired by session)
    comparator = PairedComparator(alpha = spend_alpha(alpha_total, k),
                 mode = superiority, margin = 0.0005 log loss, 5 Newey-West
                 lags): one-sided Diebold-Mariano; promote only if the
                 challenger beats the base rate by MORE than the margin,
                 significantly.

k = the Scout's own trial count (registrations in worker/experiments/scout/
and trials in worker/models/scout/gate_log.jsonl; never The Analyst's).
alpha_total is read from worker/recipes/scout/gate.toml (0.05 per the
confirmed design, so trial 1 is tested at 0.025; the unapproved signal
roadmap would make it 0.015 -> 0.0075, see that file). Every record logs
alpha_total and alpha_k, and the gate refuses to run if a logged Scout
trial used a different alpha_total.

PROMOTE -> `gating.promote_from_gate` writes worker/models/scout/champion/
(re-verified through load_champion with SCOUT_PATHS, the same check the
daily cron uses). REJECT -> logged with its reason, no champion, rule mode.
Trials against an existing Scout champion are not designed yet and are
refused (that needs an architect note, like The Analyst's refit protocol).

Chronology: training rows are embargoed (2 sessions; every training label
ends before the holdout starts), so the gate's embargo is 0; holdout.end is
the latest label end, which must be <= as_of.

MODEL-RISK LIMITATION: a PROMOTE would mean "measurably better than always
predicting the training up-rate on one year of data", not trading edge. A
REJECT is the expected outcome.
"""

from __future__ import annotations

import logging
import math
import tomllib
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd
from alphagate import Candidate, GateRecord, PairedComparator, RecordSink, gate, spend_alpha

from wolfpack_worker.analyst.gate_log import PROMOTE, read_gate_log
from wolfpack_worker.analyst.gating import (
    EMBARGO,
    HAC_LAGS,
    LOGLOSS_METRIC,
    TRIAL_SUPERIORITY_MARGIN,
    BoosterScorer,
    ConstantScorer,
    _booster,
    _event_id,
    _registration_file,
    _sink,
    _utc,
    build_shared_holdout,
    matured,
    promote_from_gate,
    shared_cutoff,
)
from wolfpack_worker.analyst.model_io import MANIFEST_FILENAME, ModelIntegrityError, sha256_bytes
from wolfpack_worker.analyst.paths import WORKER_ROOT, PersonaPaths
from wolfpack_worker.analyst.registration import Registration, RegistrationError, file_sha256, verify_runnable
from wolfpack_worker.analyst.train import TrainingResult
from wolfpack_worker.scout.paths import SCOUT_PATHS
from wolfpack_worker.scout.train import (
    build_scout_manifest,
    feature_names,
    run_scout_training,
    scout_model_version,
)

logger = logging.getLogger(__name__)

GATE_CONFIG_PATH = WORKER_ROOT / "recipes" / "scout" / "gate.toml"
ROLE = "vs_base_rate"


class GateConfigError(ValueError):
    """The Scout's gate config is missing/invalid, or changed after a trial ran."""


@dataclass(frozen=True)
class ScoutGateConfig:
    alpha_total: float
    source: str

    @classmethod
    def load(cls, path: Path = GATE_CONFIG_PATH) -> "ScoutGateConfig":
        try:
            data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise GateConfigError(f"cannot read the Scout gate config {path}: {exc}") from None
        unknown = sorted(set(data) - {"alpha_total"})
        if unknown:
            raise GateConfigError(f"{path}: unknown key(s) {unknown}")
        a = data.get("alpha_total")
        if isinstance(a, bool) or not isinstance(a, (int, float)) or not (0.0 < float(a) < 1.0):
            raise GateConfigError(f"{path}: alpha_total must be a number in (0, 1), got {a!r}")
        return cls(alpha_total=float(a), source=Path(path).name)


def check_alpha_consistency(cfg: ScoutGateConfig, records) -> None:
    """Refuse if any logged Scout trial ran under a different alpha_total:
    alpha spending is only valid if the budget is fixed before trial 1."""
    for r in records:
        ctx = r.get("context") or {}
        if ctx.get("kind") != "trial":
            continue
        logged = ctx.get("alpha_total")
        if logged is None or not math.isclose(float(logged), cfg.alpha_total, rel_tol=0, abs_tol=1e-15):
            raise GateConfigError(
                f"Scout trial {ctx.get('trial_number')} (record {r.get('record_id')}) ran with alpha_total="
                f"{logged!r}, but the gate config now says {cfg.alpha_total!r}. Changing the alpha budget after "
                "a trial has run is retroactive; it needs an explicit, logged project decision, not a config edit."
            )


@dataclass
class ScoutTrialOutcome:
    record: GateRecord
    promoted: bool
    result: TrainingResult
    k: int
    alpha_k: float
    manifest: Optional[dict[str, Any]] = None
    extra: dict[str, Any] = field(default_factory=dict)


def run_scout_trial(
    ds: pd.DataFrame,
    registration: Registration,
    *,
    as_of: datetime,
    git_commit: str,
    paths: PersonaPaths = SCOUT_PATHS,
    gate_config: Optional[ScoutGateConfig] = None,
    revised_flag: Optional[pd.Series] = None,
    today: date | None = None,
    sink: Optional[RecordSink] = None,
    run_logger=None,
    extra_context: Optional[Mapping[str, Any]] = None,
) -> ScoutTrialOutcome:
    """Run one registered Scout trial on a prepared (unmatured is fine)
    dataset. Re-checks the registration against the Scout's experiments
    directory and gate log itself and derives k itself (never trusts the
    caller); the git checks are registration.preflight (the CLI runs it)."""
    records = read_gate_log(paths.gate_log_path)
    on_disk, k = verify_runnable(registration.path, experiments_dir=paths.experiments_dir,
                                 gate_records=records, paths=paths)
    fields_ = lambda r: (r.trial_number, r.registered, r.hypothesis, r.recipe, r.abandoned)  # noqa: E731
    if Path(on_disk.path).resolve() != Path(registration.path).resolve() or fields_(on_disk) != fields_(registration):
        raise RegistrationError(f"the Registration passed in does not match {Path(registration.path).name} on disk")
    if (paths.champion_dir / MANIFEST_FILENAME).exists():
        raise NotImplementedError(
            "The Scout already has a champion: a trial against an incumbent needs its own protocol (champion "
            "recipe refit at a matched cutoff, as for The Analyst), which is not designed for The Scout yet."
        )
    if any(r.get("decision") == PROMOTE for r in records):
        raise ModelIntegrityError("the Scout gate log has a PROMOTE record but no champion directory exists")
    cfg = gate_config or ScoutGateConfig.load()
    check_alpha_consistency(cfg, records)
    alpha_k = spend_alpha(cfg.alpha_total, k)

    recipe = registration.recipe
    spec = recipe.feature_spec_version
    names = feature_names(recipe)
    ds = matured(ds, as_of)
    cutoff = shared_cutoff([(spec, ds)])
    result = run_scout_training(ds, recipe, test_start=cutoff, revised_flag=revised_flag)
    holdout = build_shared_holdout([(spec, ds)], cutoff, names={spec: names})
    if pd.Timestamp(holdout.start) != cutoff:
        raise RuntimeError("holdout does not start at the cutoff")

    version = scout_model_version(result.model_bytes, today)
    challenger = Candidate(
        model=BoosterScorer(_booster(result.model_bytes), spec, result.train_up_rate, names),
        id=version,
        trained_through=_utc(result.trained_through),
        metadata={
            "model_sha256": sha256_bytes(result.model_bytes),
            "recipe_id": recipe.recipe_id,
            "train_up_rate": result.train_up_rate,
            "n_train_rows": int(len(result.train)),
            "role": "challenger",
            "trial_number": registration.trial_number,
        },
    )
    baseline = Candidate(
        model=ConstantScorer(result.train_up_rate),
        id=f"base-rate-{cutoff:%Y%m%d}",
        trained_through=_utc(result.trained_through),
        metadata={"role": "constant_base_rate", "p_up": result.train_up_rate},
    )
    event = _event_id(f"scout-trial{registration.trial_number:03d}", as_of)
    run_id = (
        run_logger.log(result, tags={"event_id": event, "event_kind": "trial",
                                     "trial_number": str(registration.trial_number)},
                       params={"model_version": version, "hypothesis": registration.hypothesis})
        if run_logger else None
    )
    context = {
        "kind": "trial",
        "role": ROLE,
        "persona": "the-scout",
        "event_id": event,
        "trial_number": registration.trial_number,
        "k": k,
        "alpha_total": cfg.alpha_total,
        "alpha_k": alpha_k,
        "alpha_source": f"worker/recipes/scout/{cfg.source}",
        "recipe_id": recipe.recipe_id,
        "champion_recipe_id": None,
        "deployed_champion": None,
        "hypothesis": registration.hypothesis,
        "registration_file": _registration_file(registration),
        "registration_sha256": file_sha256(registration.path),
        "registered": registration.registered.isoformat(),
        "git_commit": git_commit,
        "cutoff": cutoff.isoformat(),
        "mlflow_run_id": run_id,
        "note": (
            "No Scout champion exists: one-sided paired DM superiority vs the constant base-rate predictor "
            f"(margin {TRIAL_SUPERIORITY_MARGIN}). REJECT -> no champion, the strategy runs rule_fallback. "
            "alpha_total from the confirmed Scout design; pending reconciliation with the unapproved "
            "signal-roadmap alpha ledger (0.015)."
        ),
        **dict(extra_context or {}),
    }
    record = gate(
        challenger=challenger,
        champion=baseline,
        metric=LOGLOSS_METRIC,
        holdout=holdout,
        sink=_sink(sink, paths.gate_log_path),
        comparator=PairedComparator(alpha=alpha_k, mode="superiority", margin=TRIAL_SUPERIORITY_MARGIN,
                                    hac_lags=HAC_LAGS),
        embargo=EMBARGO,
        as_of=as_of,
        context=context,
    )
    if run_logger and run_id:
        run_logger.tag(run_id, {"gate_record_id": record.record_id,
                                "gate_decision": "promote" if record.promoted else "reject"})
    if not record.promoted:
        return ScoutTrialOutcome(record, False, result, k, alpha_k)
    manifest = build_scout_manifest(result, run_id or "none", model_version=version,
                                    gate_record_ids=[record.record_id], trial_number=registration.trial_number,
                                    promoted_by=f"trial {registration.trial_number}")
    written = promote_from_gate(
        [record], model_bytes=result.model_bytes, manifest=manifest, champion_dir=paths.champion_dir,
        gate_log_path=paths.gate_log_path, archive_dir=paths.archive_dir, experiments_dir=paths.experiments_dir,
        paths=paths,
    )
    return ScoutTrialOutcome(record, True, result, k, alpha_k, written)
