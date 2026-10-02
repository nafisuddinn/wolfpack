"""The Analyst's retrain + alphagate promotion gate CLI.

    uv run --project worker --group train -m wolfpack_worker.analyst.retrain refresh
    uv run --project worker --group train -m wolfpack_worker.analyst.retrain experiment worker/experiments/analyst/NNN-<slug>.toml
    uv run --project worker --group train -m wolfpack_worker.analyst.retrain explore <recipe-or-registration.toml>
    uv run --project worker --group train -m wolfpack_worker.analyst.retrain monitor
    uv run --project worker --group train -m wolfpack_worker.analyst.retrain bootstrap   # one-time only

Add --no-backfill to any subcommand to read `prices` as-is (read-only; as_of =
now) instead of re-running the split-adjusted backfill first.

* refresh: champion's recipe, later cutoff; does nothing unless the cutoff
  would advance >= 20 sessions. Not a trial. Gated (non-inferiority) against
  the deployed champion.
* experiment: one pre-registered trial (see registration.py); refuses unless
  the registration is committed and unchanged, the code is clean, and the
  recipe was never evaluated. Two gate calls, both must PROMOTE.
* explore: walk-forward strictly before the gate holdout; MLflow
  `the-analyst-dev`; never touches the gate holdout; not a trial.
* monitor: forward record of each champion since its promotion.
* bootstrap: give the pre-gate v1 champion its honest NO_INCUMBENT record.

Every gate decision is appended to worker/models/analyst/gate_log.jsonl
(promotions AND rejections, with the reason), MLflow runs are tagged with the
record ids, and MODEL_CARD.md's generated history table is re-rendered.
Commit those files afterwards. Nothing here pushes, merges, or trades.

MODEL-RISK LIMITATION: a promotion means "measurably better than the
incumbent / the base rate on the holdout", never "has trading edge".
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from wolfpack_worker.analyst import render_history
from wolfpack_worker.analyst.gate_log import read_gate_log
from wolfpack_worker.analyst.model_io import DEFAULT_CHAMPION_DIR, GATE_LOG_PATH, load_champion
from wolfpack_worker.analyst.recipe import Recipe, load_recipe_file, load_v1_recipe
from wolfpack_worker.analyst.train import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_TRACKING_URI,
    TrainingResult,
    load_price_history,
    log_to_mlflow,
)

logger = logging.getLogger(__name__)


class MlflowRunLogger:
    """gating.RunLogger backed by the committed local MLflow store."""

    def __init__(self, tracking_uri: str = DEFAULT_TRACKING_URI, artifact_root: Path = DEFAULT_ARTIFACT_ROOT):
        self.tracking_uri = tracking_uri
        self.artifact_root = artifact_root

    def log(self, result: TrainingResult, *, tags: Mapping[str, str], params: Mapping[str, Any]) -> str:
        return log_to_mlflow(result, tracking_uri=self.tracking_uri, artifact_root=self.artifact_root,
                             extra_params=params, tags=tags)

    def tag(self, run_id: str, tags: Mapping[str, str]) -> None:
        import mlflow

        client = mlflow.MlflowClient(tracking_uri=self.tracking_uri)
        for k, v in tags.items():
            client.set_tag(run_id, k, v)


def _champion_recipe() -> Recipe:
    m = load_champion(DEFAULT_CHAMPION_DIR).manifest
    return Recipe.from_dict(m["recipe"])


def _since(*recipes: Recipe) -> date:
    return min(r.train_since_date for r in recipes)


def _render() -> None:
    changed = render_history.write()
    print("MODEL_CARD.md gate history " + ("re-rendered." if changed else "unchanged."))


def cmd_bootstrap(args) -> int:
    from wolfpack_worker.analyst.gating import run_bootstrap

    recipe = load_v1_recipe()
    bars, as_of, _ = load_price_history(recipe.train_since_date, backfill=args.backfill)
    rec = run_bootstrap(bars, as_of=as_of, recipe=recipe)
    run_id = rec.challenger_metadata.get("mlflow_run_id")
    if run_id:
        MlflowRunLogger().tag(run_id, {"gate_record_id_bootstrap": rec.record_id})
    print(f"bootstrap: {rec.decision.value.upper()} ({rec.reason_code}) {rec.challenger_id} "
          f"record {rec.record_id}; holdout log loss {rec.challenger_score.value:.6f} "
          f"vs base rate {rec.challenger_score.details['baseline_logloss']:.6f}")
    _render()
    return 0


def cmd_refresh(args) -> int:
    from wolfpack_worker.analyst.gating import run_refresh

    recipe = _champion_recipe()
    bars, as_of, _ = load_price_history(recipe.train_since_date, backfill=args.backfill)
    out = run_refresh(bars, as_of=as_of, run_logger=MlflowRunLogger())
    print(f"refresh: {out.status}: {out.message}")
    if out.record is not None:
        r = out.record
        print(f"  record {r.record_id}: {r.decision.value.upper()} ({r.reason_code}) {r.explanation}")
        _render()
    return 0


def cmd_experiment(args) -> int:
    from wolfpack_worker.analyst.gating import run_experiment
    from wolfpack_worker.analyst.registration import git_head, preflight

    reg, k = preflight(Path(args.file), gate_records=read_gate_log(GATE_LOG_PATH))
    champ_recipe = _champion_recipe()
    print(f"trial #{reg.trial_number} of k={k} registered: {reg.hypothesis}")
    bars, as_of, _ = load_price_history(_since(reg.recipe, champ_recipe), backfill=args.backfill)
    out = run_experiment(bars, reg, k=k, as_of=as_of, git_commit=git_head(), run_logger=MlflowRunLogger())
    for r in out.records:
        print(f"  {r.context['role']}: {r.decision.value.upper()} ({r.reason_code}) record {r.record_id}")
        print(f"    {r.explanation}")
    print(f"  edge vs base rate (reported, not required): p={out.edge_vs_baseline.get('p')!r} "
          f"at alpha_k={out.edge_vs_baseline.get('alpha')!r}")
    print("PROMOTED " + out.manifest["model_version"] if out.promoted else "not promoted; champion unchanged")
    _render()
    return 0


def cmd_explore(args) -> int:
    from wolfpack_worker.analyst.explore import explore_boundary, log_explore_to_mlflow, run_explore

    recipe = load_recipe_file(Path(args.file))
    champ_recipe = _champion_recipe()
    bars, as_of, _ = load_price_history(_since(recipe, champ_recipe), backfill=args.backfill)
    boundary = explore_boundary(bars, recipe, champ_recipe, as_of)
    until = pd.Timestamp(args.until, tz="UTC") if args.until else None
    res = run_explore(bars, recipe, boundary=boundary, until=until)
    run_id = log_explore_to_mlflow(res, tracking_uri=DEFAULT_TRACKING_URI, artifact_root=DEFAULT_ARTIFACT_ROOT,
                                   extra={"source_file": args.file, "as_of": pd.Timestamp(as_of).isoformat()})
    print(f"explore {recipe.recipe_id}: gate holdout starts {boundary.date()}; explored strictly before "
          f"{res.end.date()}; MLflow the-analyst-dev run {run_id}")
    for f in res.folds:
        print(f"  {f['name']:>13}: ll={f['logloss']:.5f} base_ll={f['baseline_logloss']:.5f} "
              f"acc={f['accuracy']:.4f} n={f['n']}")
    return 0


def cmd_monitor(args) -> int:
    from wolfpack_worker.analyst.forward import run_monitor

    recipe = _champion_recipe()
    bars, as_of, _ = load_price_history(recipe.train_since_date, backfill=args.backfill)
    for rec in run_monitor(bars, as_of=as_of):
        print(f"monitor {rec['model_version']}: n={rec['n_sessions']} sessions, log loss {rec['logloss']} vs "
              f"base rate {rec['baseline_logloss']}, p={rec['p']}, edge_significant={rec['edge_significant']}")
    _render()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--no-backfill", dest="backfill", action="store_false",
                        help="read `prices` as-is (read-only) instead of re-running the backfill first")
    sub.add_parser("bootstrap", parents=[common]).set_defaults(fn=cmd_bootstrap)
    sub.add_parser("refresh", parents=[common]).set_defaults(fn=cmd_refresh)
    p = sub.add_parser("experiment", parents=[common])
    p.add_argument("file")
    p.set_defaults(fn=cmd_experiment)
    p = sub.add_parser("explore", parents=[common])
    p.add_argument("file")
    p.add_argument("--until", type=date.fromisoformat, default=None,
                   help="explore only before this date (must not be after the gate holdout start)")
    p.set_defaults(fn=cmd_explore)
    sub.add_parser("monitor", parents=[common]).set_defaults(fn=cmd_monitor)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
