"""Forward monitoring: how has each champion done since it was promoted?

For every champion (the current one, plus archived ones in
models/analyst/archive/), score it on the sessions AFTER its promotion
decision (and, if it has been replaced, up to the next promotion), using only
rows whose labels have matured by `as_of`. Compare with the constant
base-rate predictor at the champion's training up-rate, using the same
per-session log loss and one-sided paired Diebold-Mariano test as the gate.
Each run appends one line per champion to forward_log.jsonl (cumulative
since promotion, so the latest line per model is its current record).

"Edge" is only claimed (edge_significant=True) when the window has at least
MIN_EDGE_SESSIONS (126, ~6 months) sessions AND the test rejects at an
alpha-spent level: the j-th such look at the same model is tested at
spend_alpha(0.05, j), because checking a growing record every week and
stopping at the first good-looking p-value would otherwise inflate false
positives. Windows shorter than 126 sessions are reported but are not looks.

Promotion time is the gate record's `decided_at`; the champion actually
starts trading once that commit is merged and the next daily run picks it up,
so the first forward session can be slightly early. That's accepted.

MODEL-RISK LIMITATION: a short forward record is noise. So far no Analyst
champion has shown edge; see MODEL_CARD.md.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
from alphagate import paired_test, spend_alpha

from wolfpack_worker.analyst.gate_log import PROMOTE, read_gate_log
from wolfpack_worker.analyst.gating import (
    ALPHA_TOTAL,
    ARCHIVE_DIR,
    HAC_LAGS,
    BoosterScorer,
    ConstantScorer,
    build_shared_holdout,
    matured,
    per_session_logloss,
    truncate_to,
)
from wolfpack_worker.analyst.model_io import DEFAULT_CHAMPION_DIR, GATE_LOG_PATH, MODELS_DIR, Champion, load_champion
from wolfpack_worker.analyst.recipe import Recipe
from wolfpack_worker.analyst.registration import EXPERIMENTS_DIR
from wolfpack_worker.analyst.train import prepare_dataset

FORWARD_LOG_PATH = MODELS_DIR / "forward_log.jsonl"
MIN_EDGE_SESSIONS = 126


def _promotion_time(records: Sequence[Mapping[str, Any]], model_version: str) -> pd.Timestamp:
    times = [
        pd.Timestamp(r["decided_at"])
        for r in records
        if r.get("challenger_id") == model_version and r.get("decision") == PROMOTE
    ]
    if not times:
        raise ValueError(f"no PROMOTE record for {model_version}")
    return min(times)


def _champions(champion_dir: Path, archive_dir: Path, gate_log_path: Path, experiments_dir: Path) -> list[Champion]:
    champs = [load_champion(champion_dir, gate_log_path=gate_log_path, experiments_dir=experiments_dir)]
    if Path(archive_dir).is_dir():
        for d in sorted(Path(archive_dir).iterdir()):
            if d.is_dir():
                champs.append(load_champion(d, gate_log_path=gate_log_path, experiments_dir=experiments_dir))
    return champs


def _strict(x: Any) -> Any:
    if isinstance(x, float) and not math.isfinite(x):
        return "NaN" if math.isnan(x) else ("Infinity" if x > 0 else "-Infinity")
    return x


def _append(path: Path, rec: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({k: _strict(v) for k, v in rec.items()}, allow_nan=False)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def run_monitor(
    bars: Mapping[str, pd.DataFrame],
    *,
    as_of: datetime,
    champion_dir: Path = DEFAULT_CHAMPION_DIR,
    archive_dir: Path = ARCHIVE_DIR,
    gate_log_path: Path = GATE_LOG_PATH,
    forward_log_path: Path = FORWARD_LOG_PATH,
    experiments_dir: Path = EXPERIMENTS_DIR,
) -> list[dict[str, Any]]:
    records = read_gate_log(gate_log_path)
    prior = read_gate_log(forward_log_path)
    champs = _champions(champion_dir, archive_dir, gate_log_path, experiments_dir)
    promoted = {c.manifest["model_version"]: _promotion_time(records, c.manifest["model_version"]) for c in champs}
    bars = truncate_to(bars, as_of)
    out: list[dict[str, Any]] = []
    for champ in sorted(champs, key=lambda c: promoted[c.manifest["model_version"]]):
        m = champ.manifest
        version = m["model_version"]
        start = promoted[version]
        later = [t for t in promoted.values() if t > start]
        end = min(later) if later else None
        recipe = Recipe.from_dict(m["recipe"])
        ds = matured(prepare_dataset(bars, recipe), as_of)
        fwd = ds.loc[ds["ts"] > start]
        if end is not None:
            fwd = fwd.loc[fwd["ts"] <= end]
        p_up = float(m["test_metrics"]["train_up_rate"])
        rec: dict[str, Any] = {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "as_of": pd.Timestamp(as_of).isoformat(),
            "model_version": version,
            "recipe_id": m.get("recipe_id"),
            "promoted_at": start.isoformat(),
            "replaced_at": end.isoformat() if end is not None else None,
            "baseline_p_up": p_up,
            "window_start": None, "window_end": None, "n_sessions": 0, "n_rows": 0,
            "logloss": None, "baseline_logloss": None, "accuracy": None,
            "mean_diff": None, "se": None, "t": None, "p": None, "lags": None,
            "look_number": None, "alpha_look": None, "edge_significant": False,
            "metric": "per_session_mean_logloss (lower is better); diff = baseline - model",
        }
        if len(fwd):
            h = build_shared_holdout([(recipe.feature_spec_version, fwd)], fwd["ts"].min())
            s_m = per_session_logloss(BoosterScorer(champ.booster, recipe.feature_spec_version, p_up), h)
            s_b = per_session_logloss(ConstantScorer(p_up), h)
            n = int(s_m.details["n_sessions"])
            rec.update({
                "window_start": pd.Timestamp(h.data.sessions[0]).isoformat(),
                "window_end": pd.Timestamp(h.data.sessions[-1]).isoformat(),
                "n_sessions": n,
                "n_rows": int(s_m.details["n_rows"]),
                "logloss": s_m.value,
                "baseline_logloss": s_b.value,
                "accuracy": s_m.details["accuracy"],
            })
            if n >= 2:
                t = paired_test(list(s_b.samples), list(s_m.samples), HAC_LAGS)
                rec.update({"mean_diff": t.mean_diff, "se": t.se, "t": t.t, "p": t.p, "lags": t.lags})
                if n >= MIN_EDGE_SESSIONS:
                    looks = sum(
                        1 for r in prior
                        if r.get("model_version") == version and (r.get("n_sessions") or 0) >= MIN_EDGE_SESSIONS
                    )
                    look = looks + 1
                    alpha = spend_alpha(ALPHA_TOTAL, look)
                    rec.update({"look_number": look, "alpha_look": alpha, "edge_significant": bool(t.p < alpha)})
        _append(forward_log_path, rec)
        prior.append(rec)
        out.append(rec)
    return out
