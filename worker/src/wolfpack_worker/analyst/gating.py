"""The Analyst's champion/challenger promotion gate, built on alphagate.

This is the ONLY code path that writes the committed champion
(`promote_from_gate`), and it only does so after every alphagate `gate()`
call for that model returned PROMOTE and was logged to gate_log.jsonl.
`model_io.load_champion` independently re-checks that log, so a champion
written any other way fails to load in the daily cron.

Three kinds of gate event (all logged, promotions and rejections alike):

* bootstrap (one-time): `gate(challenger=v1, champion=None)` on v1's own
  holdout, so the pre-gate v1 champion has an honest NO_INCUMBENT record.
* refresh: the champion's recipe refit with a later cutoff (only if the
  cutoff advances >= 20 sessions). Not a new trial (same recipe). Two gate
  calls, both must PROMOTE: (1) vs the deployed champion on the new trailing
  252-session holdout, one-sided Diebold-Mariano NON-INFERIORITY test,
  margin 0.002 log loss, alpha 0.05; (2) an anchored no-drift guard
  (AnchoredGapComparator): the challenger's excess log loss over the base
  rate must be <= the excess recorded when this recipe was FIRST promoted
  (bootstrap or trial record) + ANCHOR_TOLERANCE (0.002). Step (1) alone
  would let a chain of individually non-inferior refreshes drift down; (2)
  caps the cumulative drift. (A plain base-rate floor would block every v1
  refresh, since v1 itself is worse than the base rate.)
* trial: a pre-registered new recipe (registration.py). Two gate calls, both
  must PROMOTE: (1) vs the champion's recipe REFIT at the same cutoff, on the
  same holdout, one-sided DM SUPERIORITY with margin
  TRIAL_SUPERIORITY_MARGIN (0.0005 log loss, so a negligible-but-consistent
  gain can't promote) at alpha_k = spend_alpha(0.05, k), k derived inside
  run_experiment from the registrations and the gate log, 5 Newey-West
  lags; (2) vs the constant
  base-rate predictor (training up-rate) with MarginComparator(0), a
  point-estimate floor. The paired test vs the base rate (at alpha_k) is
  also computed and recorded, but not required.

Metric (lower is better): per-session mean log loss across the tickers.
`MetricResult.value` = mean over the holdout sessions; `samples` = the
per-session values in session order (what the paired test uses); details =
accuracy, AUC, Brier, base-rate log loss. Holdout rows are the rows valid
under every feature spec involved, so the paired samples line up.

Chronology: training rows are already embargoed (dataset.split_at: last
training row is EMBARGO_SESSIONS+1 sessions before the cutoff, and every
training label ends before it), so `trained_through` (= last training label
end) is exactly one session before the holdout start and the gate is called
with embargo=0. `holdout.end` is the latest LABEL end in the holdout, so the
`holdout.end <= as_of` check covers the bars the labels actually read.

MODEL-RISK LIMITATION: passing this gate means "measurably better than the
incumbent / the base rate on one year of daily data", not "has trading
edge". No Analyst model has shown real predictive edge on public daily price
data; see MODEL_CARD.md for the observed failure mode (confidently wrong,
rather than falling back to the base rate, in high-volatility years such as
2020 and 2022).
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence

import numpy as np
import pandas as pd
import xgboost as xgb
from alphagate import (
    Candidate,
    GateRecord,
    Holdout,
    JsonlSink,
    MarginComparator,
    MetricResult,
    PairedComparator,
    RecordSink,
    Verdict,
    gate,
    metric,
    paired_test,
    spend_alpha,
)

from wolfpack_worker.analyst.dataset import EMBARGO_SESSIONS, TEST_SESSIONS, session_dates
from wolfpack_worker.analyst.features import get_feature_spec
from wolfpack_worker.analyst.gate_log import GateLogError, read_gate_log, verify_promotion
from wolfpack_worker.analyst.metrics import predict_label
from wolfpack_worker.analyst.paths import ANALYST_PATHS, PersonaPaths
from wolfpack_worker.analyst.model_io import (
    DEFAULT_CHAMPION_DIR,
    GATE_LOG_PATH,
    MANIFEST_FILENAME,
    MODEL_FILENAME,
    MODELS_DIR,
    Champion,
    load_champion,
    read_champion_artifact,
    registration_index,
    sha256_bytes,
    write_champion,
)
from wolfpack_worker.analyst.recipe import Recipe, load_v1_recipe
from wolfpack_worker.analyst.registration import (
    EXPERIMENTS_DIR,
    REPO_ROOT,
    Registration,
    RegistrationError,
    file_sha256,
    verify_runnable,
)
from wolfpack_worker.analyst.train import (
    TrainingResult,
    build_manifest,
    model_version_for,
    prepare_dataset,
    run_training,
)

logger = logging.getLogger(__name__)

ARCHIVE_DIR = ANALYST_PATHS.archive_dir
METRIC_NAME = "per_session_mean_logloss"
ALPHA_TOTAL = 0.05
HAC_LAGS = 5
REFRESH_ALPHA = 0.05
REFRESH_MARGIN = 0.002
REFRESH_MIN_ADVANCE_SESSIONS = 20
# Trial superiority test: the challenger must beat the champion's recipe by
# MORE than this (mean per-session log loss), significantly. Below ~0.0005 a
# gain is economically negligible for a direction classifier; without a
# margin, a tiny but very consistent gain (e.g. 1e-5 every session) is
# "significant" because the paired differences have almost no variance.
TRIAL_SUPERIORITY_MARGIN = 0.0005
# Refresh no-drift guard: max (excess log loss over base rate) =
# anchor excess + ANCHOR_TOLERANCE. Same size as the refresh
# non-inferiority margin: total cumulative degradation vs the level the
# recipe was first gated at is capped at one margin.
ANCHOR_TOLERANCE = 0.002
# The embargo is already applied to the training rows (see module docstring).
EMBARGO = timedelta(0)
_LOGLOSS_EPS = 1e-15

BOOTSTRAP_NOTE = (
    "Bootstrapped after the fact: v1 (analyst-20261001-7e2cc1bc) was trained and deployed on "
    "2026-09-30 before the alphagate gate existed. This record re-scores that same, unchanged "
    "artifact on its own original holdout so the champion has an honest NO_INCUMBENT gate "
    "record. No comparison was made and no edge is implied: on this holdout v1 is worse than "
    "the base rate (see challenger_score.details.baseline_logloss)."
)


class HoldoutAlignmentError(ValueError):
    """Two feature specs disagree about the labels of a shared holdout row."""


class BootstrapError(RuntimeError):
    """The one-time bootstrap gate can't run (already done, or v1 doesn't reproduce)."""


class PromotionError(RuntimeError):
    """Refusing to write a champion that the gate log does not support."""


# ---------------------------------------------------------------------------
# Shared holdout
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SharedHoldout:
    """The holdout payload alphagate passes (opaquely) to the metric.

    keys: ts, ticker, y, label_end_ts per row, sorted by (ts, ticker).
    X: feature matrix per feature-spec version, rows aligned with `keys`.
    sessions: the distinct holdout sessions, ascending.
    session_codes: per row, the index of its session in `sessions`.
    """

    keys: pd.DataFrame
    X: Mapping[str, np.ndarray]
    sessions: tuple
    session_codes: np.ndarray


def _utc(ts) -> datetime:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        raise ValueError(f"timestamp {t} must be timezone-aware")
    return t.tz_convert("UTC").to_pydatetime()


def matured(ds: pd.DataFrame, as_of) -> pd.DataFrame:
    """Rows whose label is fully known at `as_of` (label_end_ts <= as_of)."""
    return ds.loc[ds["label_end_ts"] <= pd.Timestamp(as_of)]


def truncate_to(bars: Mapping[str, pd.DataFrame], as_of) -> dict[str, pd.DataFrame]:
    as_of = pd.Timestamp(as_of)
    return {t: df.loc[df.index <= as_of] for t, df in bars.items()}


def _indexed(ds: pd.DataFrame) -> pd.DataFrame:
    return ds.set_index(["ts", "ticker"]).sort_index()


def _common_keys(pairs: Sequence[tuple[str, pd.DataFrame]], start=None, end=None) -> pd.MultiIndex:
    common: Optional[pd.MultiIndex] = None
    for _, ds in pairs:
        sub = ds
        if start is not None:
            sub = sub.loc[sub["ts"] >= pd.Timestamp(start)]
        if end is not None:
            sub = sub.loc[sub["ts"] <= pd.Timestamp(end)]
        idx = pd.MultiIndex.from_frame(sub[["ts", "ticker"]])
        common = idx if common is None else common.intersection(idx)
    assert common is not None
    return common.sort_values()


def shared_cutoff(pairs: Sequence[tuple[str, pd.DataFrame]], test_sessions: int = TEST_SESSIONS) -> pd.Timestamp:
    """First of the last `test_sessions` sessions that have rows valid under
    every (spec, dataset) pair."""
    common = _common_keys(pairs)
    sessions = pd.DatetimeIndex(common.get_level_values("ts").unique()).sort_values()
    if len(sessions) < test_sessions + EMBARGO_SESSIONS + 1:
        raise ValueError(
            f"need more than {test_sessions + EMBARGO_SESSIONS} shared labeled sessions; have {len(sessions)}"
        )
    return pd.Timestamp(sessions[-test_sessions])


def build_shared_holdout(
    pairs: Sequence[tuple[str, pd.DataFrame]],
    start,
    end=None,
    *,
    names: Mapping[str, Sequence[str]] | None = None,
) -> Holdout:
    """Holdout of rows with start <= ts (<= end) valid under EVERY pair.

    `pairs` are (feature spec version, dataset). The same spec may appear
    more than once (e.g. two recipes with different train_since); its
    features must then agree on the shared rows. Labels must agree across
    all pairs (they come from the same bars), else HoldoutAlignmentError.
    """
    if not pairs:
        raise ValueError("build_shared_holdout: no datasets")
    common = _common_keys(pairs, start, end)
    if len(common) == 0:
        raise ValueError("build_shared_holdout: no rows shared by every dataset")

    ref: Optional[pd.DataFrame] = None
    X: dict[str, np.ndarray] = {}
    for spec, ds in pairs:
        cols = list(names[spec]) if names and spec in names else list(get_feature_spec(spec).names)
        sub = _indexed(ds).loc[common]
        if ref is None:
            ref = sub[["y", "label_end_ts"]]
        elif not (
            np.array_equal(sub["y"].to_numpy(), ref["y"].to_numpy())
            and np.array_equal(sub["label_end_ts"].to_numpy(), ref["label_end_ts"].to_numpy())
        ):
            raise HoldoutAlignmentError(
                f"labels for spec {spec!r} disagree with the other datasets on the shared holdout rows"
            )
        x = sub[cols].to_numpy(dtype=float)
        if spec in X and not np.array_equal(X[spec], x):
            raise HoldoutAlignmentError(f"two datasets for spec {spec!r} disagree on shared rows")
        X[spec] = x

    assert ref is not None
    keys = ref.reset_index()
    keys["y"] = keys["y"].astype(int)
    sessions = pd.DatetimeIndex(keys["ts"].unique()).sort_values()
    codes = sessions.searchsorted(keys["ts"].to_numpy())

    h = hashlib.sha256()
    h.update(keys["ts"].astype("int64").to_numpy().tobytes())
    h.update("\n".join(keys["ticker"]).encode())
    h.update(keys["y"].to_numpy(dtype=np.int64).tobytes())
    h.update(keys["label_end_ts"].astype("int64").to_numpy().tobytes())
    for spec in sorted(X):
        h.update(spec.encode())
        h.update(np.ascontiguousarray(X[spec]).tobytes())

    data = SharedHoldout(keys=keys, X=X, sessions=tuple(sessions), session_codes=codes)
    return Holdout(
        data=data,
        start=_utc(sessions[0]),
        end=_utc(keys["label_end_ts"].max()),
        timestamps=[_utc(s) for s in sessions],
        fingerprint="sha256:" + h.hexdigest(),
        n_samples=len(sessions),
    )


# ---------------------------------------------------------------------------
# Models as scorers, and the metric
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoosterScorer:
    booster: xgb.Booster
    spec_version: str
    train_up_rate: float
    # Another persona's spec (not in The Analyst's registry) passes its
    # feature names explicitly; None = look up an Analyst spec.
    feature_names: Optional[tuple[str, ...]] = None

    def predict(self, data: SharedHoldout) -> np.ndarray:
        names = (
            list(self.feature_names)
            if self.feature_names is not None
            else list(get_feature_spec(self.spec_version).names)
        )
        return self.booster.predict(xgb.DMatrix(data.X[self.spec_version], feature_names=names))


@dataclass(frozen=True)
class ConstantScorer:
    """Predicts the same P(up) for every row: the base-rate baseline."""

    p: float

    @property
    def train_up_rate(self) -> float:
        return self.p

    def predict(self, data: SharedHoldout) -> np.ndarray:
        return np.full(len(data.keys), self.p, dtype=float)


def _per_session_mean(values: np.ndarray, codes: np.ndarray, n_sessions: int) -> np.ndarray:
    sums = np.bincount(codes, weights=values, minlength=n_sessions)
    counts = np.bincount(codes, minlength=n_sessions)
    return sums / counts


def _row_logloss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _LOGLOSS_EPS, 1 - _LOGLOSS_EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def per_session_logloss(model, holdout: Holdout) -> MetricResult:
    data: SharedHoldout = holdout.data
    y = data.keys["y"].to_numpy(dtype=float)
    p = np.asarray(model.predict(data), dtype=float)
    n_sessions = len(data.sessions)
    ll = _row_logloss(y, p)
    samples = _per_session_mean(ll, data.session_codes, n_sessions)
    base = _per_session_mean(
        _row_logloss(y, np.full_like(y, float(model.train_up_rate))), data.session_codes, n_sessions
    )
    if len(np.unique(y)) == 2 and len(np.unique(p)) > 1:
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(y, p))
    else:
        auc = 0.5 if len(np.unique(y)) == 2 else float("nan")
    return MetricResult(
        value=float(samples.mean()),
        samples=[float(s) for s in samples],
        details={
            "accuracy": float(np.mean(predict_label(p) == y)),
            "auc": auc,
            "brier": float(np.mean((p - y) ** 2)),
            "baseline_logloss": float(base.mean()),
            "train_up_rate": float(model.train_up_rate),
            "row_mean_logloss": float(ll.mean()),
            "n_rows": int(len(y)),
            "n_sessions": int(n_sessions),
        },
    )


LOGLOSS_METRIC = metric(METRIC_NAME, per_session_logloss, higher_is_better=False)


def _booster(model_bytes: bytes) -> xgb.Booster:
    b = xgb.Booster()
    b.load_model(bytearray(model_bytes))
    return b


def _training_candidate(result: TrainingResult, model_id: str, metadata: Mapping[str, Any]) -> Candidate:
    return Candidate(
        model=BoosterScorer(_booster(result.model_bytes), result.recipe.feature_spec_version, result.train_up_rate),
        id=model_id,
        trained_through=_utc(result.trained_through),
        metadata={
            "model_sha256": sha256_bytes(result.model_bytes),
            "recipe_id": result.recipe.recipe_id,
            "train_up_rate": result.train_up_rate,
            "n_train_rows": int(len(result.train)),
            **metadata,
        },
    )


def _deployed_candidate(champ: Champion, role: str) -> Candidate:
    m = champ.manifest
    return Candidate(
        model=BoosterScorer(champ.booster, m["feature_spec_version"], float(m["test_metrics"]["train_up_rate"])),
        id=m["model_version"],
        trained_through=_utc(m["trained_through"]),
        metadata={"model_sha256": m["model_sha256"], "recipe_id": m.get("recipe_id"), "role": role},
    )


def _manifest_recipe(manifest: Mapping[str, Any]) -> Recipe:
    recipe = Recipe.from_dict(manifest["recipe"])
    if recipe.recipe_id != manifest["recipe_id"]:
        raise PromotionError(
            f"champion manifest recipe hashes to {recipe.recipe_id}, but recipe_id says {manifest['recipe_id']}"
        )
    return recipe


def _event_id(kind: str, as_of: datetime) -> str:
    return f"{kind}-{pd.Timestamp(as_of).tz_convert('UTC'):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"


def _sink(sink: Optional[RecordSink], gate_log_path: Path) -> RecordSink:
    if sink is not None:
        return sink
    Path(gate_log_path).parent.mkdir(parents=True, exist_ok=True)
    return JsonlSink(gate_log_path)


class RunLogger(Protocol):
    """Optional MLflow hook (retrain.py provides the real one)."""

    def log(self, result: TrainingResult, *, tags: Mapping[str, str], params: Mapping[str, Any]) -> str: ...

    def tag(self, run_id: str, tags: Mapping[str, str]) -> None: ...


# ---------------------------------------------------------------------------
# Comparators WolfPack uses
# ---------------------------------------------------------------------------


def trial_comparator(alpha_k: float) -> PairedComparator:
    return PairedComparator(alpha=alpha_k, mode="superiority", margin=TRIAL_SUPERIORITY_MARGIN, hac_lags=HAC_LAGS)


WITHIN_ANCHOR = "within_anchor"
DRIFTED_FROM_ANCHOR = "drifted_from_anchor"


class AnchoredGapComparator:
    """Refresh no-drift guard (WolfPack-side alphagate Comparator).

    "champion" here is the constant base-rate predictor, scored on the same
    holdout. gap = challenger log loss - base-rate log loss (> 0 means worse
    than the base rate). Promote iff gap <= anchor_gap + tolerance, where
    anchor_gap is the gap recorded when this recipe was first promoted. A
    point-estimate rule, not a significance test.
    """

    name = "anchored_gap"

    def __init__(self, anchor_gap: float, tolerance: float = ANCHOR_TOLERANCE, anchor_record_id: str | None = None):
        if not (np.isfinite(anchor_gap) and np.isfinite(tolerance) and tolerance >= 0):
            raise ValueError("anchor_gap must be finite and tolerance finite and >= 0")
        self.anchor_gap = float(anchor_gap)
        self.tolerance = float(tolerance)
        self.anchor_record_id = anchor_record_id

    def params(self) -> Mapping[str, Any]:
        return {"anchor_gap": self.anchor_gap, "tolerance": self.tolerance,
                "anchor_record_id": self.anchor_record_id}

    def compare(self, champion: MetricResult, challenger: MetricResult, *, higher_is_better: bool) -> Verdict:
        if higher_is_better:
            raise ValueError("AnchoredGapComparator is for a lower-is-better loss")
        gap = float(challenger.value) - float(champion.value)
        max_gap = self.anchor_gap + self.tolerance
        stats = {"gap": gap, "anchor_gap": self.anchor_gap, "tolerance": self.tolerance, "max_gap": max_gap,
                 "challenger": float(challenger.value), "base_rate": float(champion.value)}
        if gap <= max_gap:
            return Verdict(True, WITHIN_ANCHOR,
                           f"excess log loss over the base rate {gap!r} <= anchor {self.anchor_gap!r} + "
                           f"tolerance {self.tolerance!r}", stats)
        return Verdict(False, DRIFTED_FROM_ANCHOR,
                       f"excess log loss over the base rate {gap!r} > anchor {self.anchor_gap!r} + tolerance "
                       f"{self.tolerance!r}: this recipe has drifted below the level it was first promoted at; "
                       "incumbent retained", stats)


def refresh_anchor(records: Sequence[Mapping[str, Any]], recipe_id: str) -> dict[str, Any]:
    """The gap (challenger log loss - its base-rate log loss) in the FIRST
    PROMOTE record of kind bootstrap/trial for this recipe. Refreshes never
    create an anchor, so a chain of refreshes can't move it."""
    for r in records:
        ctx = r.get("context") or {}
        if (
            r.get("decision") == "promote"
            and ctx.get("kind") in ("bootstrap", "trial")
            and (r.get("challenger_metadata") or {}).get("recipe_id") == recipe_id
        ):
            score = r["challenger_score"]
            gap = float(score["value"]) - float(score["details"]["baseline_logloss"])
            return {"record_id": r.get("record_id"), "gap": gap, "kind": ctx.get("kind")}
    raise PromotionError(f"no bootstrap/trial PROMOTE record for recipe {recipe_id}: no anchor for a refresh")


# ---------------------------------------------------------------------------
# Promotion (the only champion writer)
# ---------------------------------------------------------------------------


def promote_from_gate(
    records: Sequence[GateRecord],
    *,
    model_bytes: bytes,
    manifest: Mapping[str, Any],
    champion_dir: Path = DEFAULT_CHAMPION_DIR,
    gate_log_path: Path = GATE_LOG_PATH,
    archive_dir: Path = ARCHIVE_DIR,
    experiments_dir: Path = EXPERIMENTS_DIR,
    paths: PersonaPaths = ANALYST_PATHS,
) -> dict[str, Any]:
    """Write `model_bytes` as the champion iff EVERY record is a PROMOTE of it.

    Archives the outgoing champion under archive_dir/<model_version>/ first
    (so `monitor` can keep scoring it), then re-loads the new champion
    through load_champion, i.e. through the same gate-log check the daily
    cron uses. `paths` selects the persona's registration parser and
    feature-spec registry for those checks (the directories are the explicit
    arguments, so a caller passes all of them for a non-Analyst persona).
    """
    if not records:
        raise PromotionError("no gate records: nothing authorises this promotion")
    version = manifest["model_version"]
    sha = sha256_bytes(model_bytes)
    for r in records:
        if not r.promoted:
            raise PromotionError(f"gate record {r.record_id} is a REJECT; every gate call must PROMOTE")
        if r.challenger_id != version:
            raise PromotionError(f"gate record {r.record_id} challenger_id {r.challenger_id!r} != {version!r}")
        if (r.challenger_metadata or {}).get("model_sha256") != sha:
            raise PromotionError(f"gate record {r.record_id} was for different model bytes")
    ids = [r.record_id for r in records]
    if list(manifest.get("gate_record_ids", [])) != ids:
        raise PromotionError(f"manifest gate_record_ids must be exactly {ids}")
    try:
        verify_promotion(read_gate_log(gate_log_path), model_version=version, model_sha256=sha, gate_record_ids=ids,
                         registrations=registration_index(experiments_dir, paths=paths))
    except GateLogError as exc:
        raise PromotionError(f"the persisted gate log does not support this promotion: {exc}") from None

    champion_dir = Path(champion_dir)
    if (champion_dir / MANIFEST_FILENAME).is_file():
        old = json.loads((champion_dir / MANIFEST_FILENAME).read_text())
        if old.get("model_version") != version:
            dest = Path(archive_dir) / old["model_version"]
            dest.mkdir(parents=True, exist_ok=True)
            for name in (MODEL_FILENAME, MANIFEST_FILENAME):
                shutil.copy2(champion_dir / name, dest / name)
    written = write_champion(champion_dir, model_bytes, dict(manifest))
    load_champion(champion_dir, gate_log_path=gate_log_path, experiments_dir=experiments_dir,
                  paths=paths)  # daily-cron check
    return written


def promotion_authorised(records: Sequence[GateRecord]) -> bool:
    """A trial promotes only if EVERY gate call promoted."""
    return bool(records) and all(r.promoted for r in records)


# ---------------------------------------------------------------------------
# Bootstrap (one-time)
# ---------------------------------------------------------------------------


def run_bootstrap(
    bars: Mapping[str, pd.DataFrame],
    *,
    champion_dir: Path = DEFAULT_CHAMPION_DIR,
    gate_log_path: Path = GATE_LOG_PATH,
    as_of: datetime,
    recipe: Recipe | None = None,
    sink: Optional[RecordSink] = None,
) -> GateRecord:
    """Give the pre-gate champion an honest NO_INCUMBENT record.

    Re-scores the unchanged artifact on its own original holdout (the
    manifest's test_start..test_end), refuses unless that reproduces the
    manifest's holdout row count and log loss (1e-9), and refuses if the
    gate log already has any record. Then gate(champion=None) and rewrite the
    manifest (same model bytes) with recipe_id + gate_record_ids.
    """
    if read_gate_log(gate_log_path):
        raise BootstrapError(f"{gate_log_path} already has records; the bootstrap is one-time only")
    recipe = recipe or load_v1_recipe()
    art = read_champion_artifact(champion_dir)
    m = art.manifest
    if m["feature_spec_version"] != recipe.feature_spec_version or m.get("model_params") != {
        **recipe.xgb_params,
        "num_boost_round": recipe.num_boost_round,
    }:
        raise BootstrapError("champion manifest settings do not match the bootstrap recipe")

    bars = truncate_to(bars, as_of)
    ds = matured(prepare_dataset(bars, recipe), as_of)
    start, end = pd.Timestamp(m["test_start"]), pd.Timestamp(m["test_end"])
    holdout = build_shared_holdout([(recipe.feature_spec_version, ds)], start, end)
    if pd.Timestamp(holdout.start) != start:
        raise BootstrapError(f"holdout starts {holdout.start}, manifest test_start is {start}")
    tm = m["test_metrics"]
    scorer = BoosterScorer(art.booster, recipe.feature_spec_version, float(tm["train_up_rate"]))
    pre = per_session_logloss(scorer, holdout)
    if pre.details["n_rows"] != tm["n"] or abs(pre.details["row_mean_logloss"] - tm["logloss"]) > 1e-9:
        raise BootstrapError(
            f"v1's holdout does not reproduce from current data: {pre.details['n_rows']} rows / log loss "
            f"{pre.details['row_mean_logloss']!r} vs manifest {tm['n']} / {tm['logloss']!r}. "
            "Refusing to log a record that doesn't describe the deployed model."
        )

    challenger = Candidate(
        model=scorer,
        id=m["model_version"],
        trained_through=_utc(m["trained_through"]),
        metadata={
            "model_sha256": m["model_sha256"],
            "recipe_id": recipe.recipe_id,
            "mlflow_run_id": m["mlflow_run_id"],
            "role": "bootstrap",
        },
    )
    record = gate(
        challenger=challenger,
        champion=None,
        metric=LOGLOSS_METRIC,
        holdout=holdout,
        sink=_sink(sink, gate_log_path),
        embargo=EMBARGO,
        as_of=as_of,
        context={
            "kind": "bootstrap",
            "event_id": _event_id("bootstrap", as_of),
            "recipe_id": recipe.recipe_id,
            "trial_number": None,
            "mlflow_run_id": m["mlflow_run_id"],
            "note": BOOTSTRAP_NOTE,
        },
    )
    manifest = {
        **{k: v for k, v in m.items() if k != "model_sha256"},
        "recipe_id": recipe.recipe_id,
        "recipe": recipe.to_dict(),
        "trial_number": None,
        "gate_record_ids": [record.record_id],
        "promoted_by": "bootstrap",
    }
    model_bytes = (Path(champion_dir) / MODEL_FILENAME).read_bytes()
    promote_from_gate(
        [record], model_bytes=model_bytes, manifest=manifest, champion_dir=champion_dir,
        gate_log_path=gate_log_path, archive_dir=Path(champion_dir).parent / "archive",
    )
    return record


# ---------------------------------------------------------------------------
# Refresh (same recipe, later cutoff; not a trial)
# ---------------------------------------------------------------------------


@dataclass
class RefreshOutcome:
    status: str  # "skipped" | "promoted" | "rejected"
    advance_sessions: int
    message: str
    records: tuple[GateRecord, ...] = ()
    manifest: Optional[dict[str, Any]] = None

    @property
    def record(self) -> Optional[GateRecord]:
        """The non-inferiority record vs the deployed champion (first call)."""
        return self.records[0] if self.records else None


def cutoff_advance(ds: pd.DataFrame, old_cutoff, new_cutoff) -> int:
    """Sessions s in the data with old_cutoff < s <= new_cutoff."""
    s = session_dates(ds)
    return int(((s > pd.Timestamp(old_cutoff)) & (s <= pd.Timestamp(new_cutoff))).sum())


def run_refresh(
    bars: Mapping[str, pd.DataFrame],
    *,
    as_of: datetime,
    champion_dir: Path = DEFAULT_CHAMPION_DIR,
    gate_log_path: Path = GATE_LOG_PATH,
    archive_dir: Path = ARCHIVE_DIR,
    today: date | None = None,
    sink: Optional[RecordSink] = None,
    run_logger: Optional[RunLogger] = None,
    min_advance: int = REFRESH_MIN_ADVANCE_SESSIONS,
    experiments_dir: Path = EXPERIMENTS_DIR,
) -> RefreshOutcome:
    champ = load_champion(champion_dir, gate_log_path=gate_log_path, experiments_dir=experiments_dir)
    m = champ.manifest
    recipe = _manifest_recipe(m)
    bars = truncate_to(bars, as_of)
    ds = matured(prepare_dataset(bars, recipe), as_of)
    spec = recipe.feature_spec_version
    new_cutoff = shared_cutoff([(spec, ds)])
    advance = cutoff_advance(ds, m["test_start"], new_cutoff)
    if advance < min_advance:
        return RefreshOutcome(
            status="skipped",
            advance_sessions=advance,
            message=(
                f"cutoff would advance only {advance} session(s) (from {pd.Timestamp(m['test_start']).date()} "
                f"to {new_cutoff.date()}), fewer than {min_advance}; nothing to do"
            ),
        )

    anchor = refresh_anchor(read_gate_log(gate_log_path), recipe.recipe_id)
    result = run_training(bars, recipe, test_start=new_cutoff)
    version = model_version_for(result.model_bytes, today)
    if version == m["model_version"]:
        raise PromotionError("refresh produced byte-identical model to the champion")
    holdout = build_shared_holdout([(spec, ds)], new_cutoff)
    challenger = _training_candidate(result, version, {"role": "refresh_challenger"})
    champion = _deployed_candidate(champ, "deployed_champion")
    baseline = Candidate(
        model=ConstantScorer(result.train_up_rate),
        id=f"base-rate-{new_cutoff:%Y%m%d}",
        trained_through=_utc(result.trained_through),
        metadata={"role": "constant_base_rate", "p_up": result.train_up_rate},
    )
    event = _event_id("refresh", as_of)
    run_id = (
        run_logger.log(result, tags={"event_id": event, "event_kind": "refresh"},
                       params={"model_version": version})
        if run_logger
        else None
    )
    common = {
        "kind": "refresh",
        "event_id": event,
        "recipe_id": recipe.recipe_id,
        "trial_number": m.get("trial_number"),
        "previous_test_start": m["test_start"],
        "cutoff_advance_sessions": advance,
        "deployed_champion": m["model_version"],
        "mlflow_run_id": run_id,
    }
    sink = _sink(sink, gate_log_path)
    rec1 = gate(
        challenger=challenger,
        champion=champion,
        metric=LOGLOSS_METRIC,
        holdout=holdout,
        sink=sink,
        comparator=PairedComparator(
            alpha=REFRESH_ALPHA, mode="non_inferiority", margin=REFRESH_MARGIN, hac_lags=HAC_LAGS
        ),
        embargo=EMBARGO,
        as_of=as_of,
        context={**common, "role": "vs_deployed_champion",
                 "note": "Same recipe, later cutoff: not a new trial. Non-inferiority vs the deployed champion."},
    )
    rec2 = gate(
        challenger=challenger,
        champion=baseline,
        metric=LOGLOSS_METRIC,
        holdout=holdout,
        sink=sink,
        comparator=AnchoredGapComparator(anchor["gap"], ANCHOR_TOLERANCE, anchor["record_id"]),
        embargo=EMBARGO,
        as_of=as_of,
        context={**common, "role": "anchor_guard", "anchor_record_id": anchor["record_id"],
                 "note": "No-drift guard: excess log loss over the base rate may not exceed the recipe's "
                         "first-promotion excess + tolerance. Point estimate, not a significance test."},
    )
    records = (rec1, rec2)
    if run_logger and run_id:
        run_logger.tag(run_id, {
            "gate_record_id": rec1.record_id,
            "gate_record_id_anchor_guard": rec2.record_id,
            "gate_decision": "promote" if promotion_authorised(records) else "reject",
        })
    if not promotion_authorised(records):
        reasons = f"{rec1.reason_code}; anchor guard: {rec2.reason_code}"
        return RefreshOutcome("rejected", advance, f"refresh rejected ({reasons})", records)
    manifest = build_manifest(
        result, run_id or "none", model_version=version, gate_record_ids=[rec1.record_id, rec2.record_id],
        trial_number=m.get("trial_number"), promoted_by="refresh",
    )
    written = promote_from_gate(
        list(records), model_bytes=result.model_bytes, manifest=manifest, champion_dir=champion_dir,
        gate_log_path=gate_log_path, archive_dir=archive_dir, experiments_dir=experiments_dir,
    )
    return RefreshOutcome("promoted", advance, f"refresh promoted {version} ({rec1.reason_code})", records, written)


# ---------------------------------------------------------------------------
# Experiment (a pre-registered trial)
# ---------------------------------------------------------------------------


@dataclass
class ExperimentOutcome:
    records: tuple[GateRecord, GateRecord]
    promoted: bool
    manifest: Optional[dict[str, Any]] = None
    edge_vs_baseline: dict[str, Any] = field(default_factory=dict)


def _registration_file(reg: Registration) -> str:
    try:
        return Path(reg.path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return Path(reg.path).name


def run_experiment(
    bars: Mapping[str, pd.DataFrame],
    registration: Registration,
    *,
    as_of: datetime,
    git_commit: str,
    champion_dir: Path = DEFAULT_CHAMPION_DIR,
    gate_log_path: Path = GATE_LOG_PATH,
    archive_dir: Path = ARCHIVE_DIR,
    experiments_dir: Path = EXPERIMENTS_DIR,
    today: date | None = None,
    sink: Optional[RecordSink] = None,
    run_logger: Optional[RunLogger] = None,
) -> ExperimentOutcome:
    """Run one registered trial.

    Does not trust its caller: re-checks the registration against
    `experiments_dir` and the gate log (registration.verify_runnable: never
    run before, recipe never evaluated, every logged trial's registration
    intact) and derives k itself. The git checks (committed, clean tree) are
    registration.preflight, which the CLI runs first.
    """
    on_disk, k = verify_runnable(registration.path, experiments_dir=experiments_dir,
                                 gate_records=read_gate_log(gate_log_path))
    fields = lambda r: (r.trial_number, r.registered, r.hypothesis, r.recipe, r.abandoned)  # noqa: E731
    if Path(on_disk.path).resolve() != Path(registration.path).resolve() or fields(on_disk) != fields(registration):
        raise RegistrationError(
            f"the Registration passed in does not match {Path(registration.path).name} on disk"
        )
    registration_sha = file_sha256(registration.path)
    champ = load_champion(champion_dir, gate_log_path=gate_log_path, experiments_dir=experiments_dir)
    m = champ.manifest
    champ_recipe = _manifest_recipe(m)
    chal_recipe = registration.recipe
    if chal_recipe == champ_recipe:
        raise ValueError("the challenger is the champion's own recipe; that is a refresh, not a trial")

    bars = truncate_to(bars, as_of)
    pairs = [
        (chal_recipe.feature_spec_version, matured(prepare_dataset(bars, chal_recipe), as_of)),
        (champ_recipe.feature_spec_version, matured(prepare_dataset(bars, champ_recipe), as_of)),
    ]
    cutoff = shared_cutoff(pairs)
    chal = run_training(bars, chal_recipe, test_start=cutoff)
    refit = run_training(bars, champ_recipe, test_start=cutoff, walk_forward=False)
    holdout = build_shared_holdout(pairs, cutoff)
    if pd.Timestamp(holdout.start) != cutoff:
        raise HoldoutAlignmentError("shared holdout does not start at the shared cutoff")

    alpha_k = spend_alpha(ALPHA_TOTAL, k)
    version = model_version_for(chal.model_bytes, today)
    challenger = _training_candidate(chal, version, {"role": "challenger", "trial_number": registration.trial_number})
    refit_c = _training_candidate(
        refit,
        f"refit-{champ_recipe.recipe_id}-{cutoff:%Y%m%d}",
        {"role": "champion_recipe_refit", "deployed_champion": m["model_version"]},
    )
    baseline = Candidate(
        model=ConstantScorer(chal.train_up_rate),
        id=f"base-rate-{cutoff:%Y%m%d}",
        trained_through=_utc(chal.trained_through),
        metadata={"role": "constant_base_rate", "p_up": chal.train_up_rate},
    )

    # Reported, not required: is the challenger significantly better than the
    # base rate at the same alpha_k? (Improvement = base loss - challenger loss.)
    s_chal = LOGLOSS_METRIC(challenger.model, holdout)
    s_base = LOGLOSS_METRIC(baseline.model, holdout)
    et = paired_test(list(s_base.samples), list(s_chal.samples), HAC_LAGS)
    edge = {"mean_diff": et.mean_diff, "se": et.se, "t": et.t, "p": et.p, "n": et.n, "lags": et.lags,
            "alpha": alpha_k}
    edge_significant = bool(et.p < alpha_k)

    event = _event_id(f"trial{registration.trial_number:03d}", as_of)
    run_id = (
        run_logger.log(chal, tags={"event_id": event, "event_kind": "trial",
                                   "trial_number": str(registration.trial_number)},
                       params={"model_version": version, "hypothesis": registration.hypothesis})
        if run_logger
        else None
    )
    common = {
        "kind": "trial",
        "event_id": event,
        "trial_number": registration.trial_number,
        "k": k,
        "alpha_total": ALPHA_TOTAL,
        "alpha_k": alpha_k,
        "recipe_id": chal_recipe.recipe_id,
        "champion_recipe_id": champ_recipe.recipe_id,
        "deployed_champion": m["model_version"],
        "hypothesis": registration.hypothesis,
        "registration_file": _registration_file(registration),
        "registration_sha256": registration_sha,
        "registered": registration.registered.isoformat(),
        "git_commit": git_commit,
        "cutoff": cutoff.isoformat(),
        "mlflow_run_id": run_id,
    }
    sink = _sink(sink, gate_log_path)
    rec1 = gate(
        challenger=challenger,
        champion=refit_c,
        metric=LOGLOSS_METRIC,
        holdout=holdout,
        sink=sink,
        comparator=trial_comparator(alpha_k),
        embargo=EMBARGO,
        as_of=as_of,
        context={**common, "role": "vs_champion_refit", "edge_vs_baseline": edge,
                 "edge_vs_baseline_significant": edge_significant},
    )
    rec2 = gate(
        challenger=challenger,
        champion=baseline,
        metric=LOGLOSS_METRIC,
        holdout=holdout,
        sink=sink,
        comparator=MarginComparator(0.0),
        embargo=EMBARGO,
        as_of=as_of,
        context={**common, "role": "floor_vs_base_rate",
                 "note": "Point-estimate floor (challenger log loss < base-rate log loss), not a significance test."},
    )
    for r, v in ((rec1, s_chal.value), (rec2, s_chal.value)):
        if r.challenger_score.value != v:  # metric must be a pure function
            raise RuntimeError("challenger score changed between scorings; metric is not deterministic")
    if run_logger and run_id:
        run_logger.tag(run_id, {
            "gate_record_id_vs_champion": rec1.record_id,
            "gate_record_id_floor": rec2.record_id,
            "gate_decision": "promote" if promotion_authorised([rec1, rec2]) else "reject",
        })

    if not promotion_authorised([rec1, rec2]):
        return ExperimentOutcome((rec1, rec2), False, None, edge)
    manifest = build_manifest(
        chal, run_id or "none", model_version=version, gate_record_ids=[rec1.record_id, rec2.record_id],
        trial_number=registration.trial_number, promoted_by=f"trial {registration.trial_number}",
    )
    written = promote_from_gate(
        [rec1, rec2], model_bytes=chal.model_bytes, manifest=manifest, champion_dir=champion_dir,
        gate_log_path=gate_log_path, archive_dir=archive_dir, experiments_dir=experiments_dir,
    )
    return ExperimentOutcome((rec1, rec2), True, written, edge)
