"""The Scout's CLI: news coverage report, gated trials, forward monitoring.

    uv run --project worker --group train -m wolfpack_worker.scout.retrain report  [--store local]
    uv run --project worker --group train -m wolfpack_worker.scout.retrain experiment worker/experiments/scout/NNN-<slug>.toml [--store local] [--no-backfill]
    uv run --project worker --group train -m wolfpack_worker.scout.retrain monitor [--store local] [--no-backfill]

* report (M1): headline coverage / timing / revision report from the news
  store and the market calendar only (no prices, no labels), written to
  worker/reports/scout/news_coverage.json. Run it BEFORE registering a trial.
* experiment: one pre-registered trial (registration.py rules, with the
  Scout's own experiments directory and trial count). Refuses unless the
  registration is committed and unchanged and the code is clean. One gate
  call vs the base rate (scout/gating.py); PROMOTE writes the champion,
  REJECT leaves The Scout in rule_fallback. Appends to
  worker/models/scout/gate_log.jsonl, logs MLflow experiment `the-scout`.
* monitor: forward record of a Scout champion since promotion (nothing to
  do while there is none).

`--store supabase` (default) reads the private `news_articles` table;
`--store local` reads the gitignored cache written by
`news_backfill --store local`. Prices always come from Supabase `prices`
(`--no-backfill`: read-only, as for The Analyst). The market calendar comes
from Alpaca's paper-trading calendar endpoint (read-only).

Nothing here pushes, merges, or trades. Paper trading only.

MODEL-RISK LIMITATION: a promotion would mean "measurably better than the
base rate on the holdout", never "has trading edge"; REJECT is expected.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from wolfpack_worker.analyst.paths import WORKER_ROOT
from wolfpack_worker.scout.paths import SCOUT_PATHS
from wolfpack_worker.universe import UNIVERSE

logger = logging.getLogger(__name__)

REPORT_PATH = WORKER_ROOT / "reports" / "scout" / "news_coverage.json"
DEFAULT_SINCE = date(2016, 1, 4)


class MlflowRunLogger:
    def log(self, result, *, tags: Mapping[str, str], params: Mapping[str, Any]) -> str:
        from wolfpack_worker.scout.train import log_to_mlflow

        return log_to_mlflow(result, extra_params=params, tags=tags)

    def tag(self, run_id: str, tags: Mapping[str, str]) -> None:
        import mlflow

        from wolfpack_worker.analyst.train import DEFAULT_TRACKING_URI

        client = mlflow.MlflowClient(tracking_uri=DEFAULT_TRACKING_URI)
        for k, v in tags.items():
            client.set_tag(run_id, k, v)


def load_calendar(since: date, as_of: datetime):
    from wolfpack_worker.broker import AlpacaPaperBroker
    from wolfpack_worker.config import load_config

    broker = AlpacaPaperBroker(load_config())
    sessions = [s for s in broker.get_calendar(since, as_of.date()) if s.close <= as_of]
    if not sessions:
        raise RuntimeError("empty market calendar")
    return tuple(sessions)


def latest_close(now: datetime) -> datetime:
    from wolfpack_worker.backfill import resolve_as_of
    from wolfpack_worker.broker import AlpacaPaperBroker
    from wolfpack_worker.config import load_config

    return resolve_as_of(AlpacaPaperBroker(load_config()), now)


def load_articles(store_kind: str, cache: Path | None, start: datetime, end: datetime) -> pd.DataFrame:
    from wolfpack_worker.news_backfill import make_store

    return make_store(store_kind, cache).get_articles(list(UNIVERSE), start, end)


def cmd_report(args) -> int:
    from wolfpack_worker.scout.coverage import coverage_report

    as_of = latest_close(datetime.now(timezone.utc))
    sessions = load_calendar(args.since, as_of)
    arts = load_articles(args.store, args.cache, sessions[0].close - timedelta(days=1), as_of)
    rep = coverage_report(arts, sessions, UNIVERSE)
    rep = {"generated_at": datetime.now(timezone.utc).isoformat(), "as_of": as_of.isoformat(),
           "store": args.store, **rep}
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
    print(f"wrote {REPORT_PATH}")
    print(json.dumps({k: rep[k] for k in ("n_articles_universe", "timing_share", "revisions", "month_gaps")},
                     indent=2))
    for t, v in rep["per_ticker"].items():
        print(f"  {t}: {v['headlines']} headlines, sessions with news {v['share_sessions_with_news']:.3f}")
    return 0


def prepare_inputs(args, since: date):
    """Prices (Supabase), calendar (Alpaca), headlines (store), all <= as_of."""
    from wolfpack_worker.analyst.gating import truncate_to
    from wolfpack_worker.analyst.train import load_price_history
    from wolfpack_worker.scout.coverage import month_gaps

    bars, as_of, _ = load_price_history(since, backfill=args.backfill)
    sessions = load_calendar(since, as_of)
    arts = load_articles(args.store, args.cache, sessions[0].close - timedelta(days=1), as_of)
    arts = arts.loc[arts["created_at"] <= as_of].reset_index(drop=True)
    gaps = month_gaps(arts, sessions, UNIVERSE)
    if gaps:
        raise RuntimeError(f"news store has no universe headlines in month(s) {gaps}: backfill gap, refusing to "
                           "train (missing news would silently look like 'no news')")
    return truncate_to(bars, as_of), sessions, arts, as_of


def _revised_flag(ds: pd.DataFrame, arts: pd.DataFrame, sessions) -> pd.Series:
    from wolfpack_worker.scout.coverage import revision_flags
    from wolfpack_worker.scout.windows import bar_sessions

    flags = revision_flags(arts, sessions, UNIVERSE)
    pos = bar_sessions(pd.DatetimeIndex(ds["ts"]), sessions)
    return pd.Series([bool(flags[t][p]) for t, p in zip(ds["ticker"], pos)], index=ds.index)


def _render() -> None:
    from wolfpack_worker.analyst import render_history

    card = SCOUT_PATHS.model_card_path
    if card.is_file() and render_history.has_markers(card.read_text(encoding="utf-8"), SCOUT_PATHS):
        changed = render_history.write(paths=SCOUT_PATHS)
        print(f"{card.name} gate history " + ("re-rendered." if changed else "unchanged."))
    else:
        b, e = render_history.markers(SCOUT_PATHS)
        print(f"{card.name} has no generated gate-history section yet; add the markers\n  {b}\n  {e}\n"
              "and re-run `python -m wolfpack_worker.scout.retrain render` (documentarian).")


def cmd_experiment(args) -> int:
    from wolfpack_worker.analyst.gate_log import read_gate_log
    from wolfpack_worker.analyst.registration import git_head, preflight
    from wolfpack_worker.scout.dataset import build_scout_dataset
    from wolfpack_worker.scout.gating import run_scout_trial

    reg, k = preflight(Path(args.file), gate_records=read_gate_log(SCOUT_PATHS.gate_log_path), paths=SCOUT_PATHS)
    print(f"Scout trial #{reg.trial_number} of k={k} registered: {reg.hypothesis}")
    bars, sessions, arts, as_of = prepare_inputs(args, reg.recipe.train_since_date)
    ds = build_scout_dataset(bars, arts, sessions, reg.recipe.feature_spec_version)
    out = run_scout_trial(
        ds, reg, as_of=as_of, git_commit=git_head(), revised_flag=_revised_flag(ds, arts, sessions),
        run_logger=MlflowRunLogger(),
        extra_context={"news_store": args.store, "n_articles": int(len(arts)),
                       "news_created_at_last": arts["created_at"].max().isoformat()},
    )
    r = out.record
    m = out.result.test_metrics
    print(f"  {r.decision.value.upper()} ({r.reason_code}) record {r.record_id}")
    print(f"    {r.explanation}")
    print(f"  holdout {pd.Timestamp(r.holdout_start).date()}..{pd.Timestamp(r.holdout_end).date()}: "
          f"accuracy {m['accuracy']:.4f} (baseline {m['baseline_accuracy']:.4f}), AUC {m['auc']:.4f}, "
          f"log loss {m['logloss']:.5f} (base rate {m['baseline_logloss']:.5f})")
    print(json.dumps(out.result.extra.get("diagnostics", {}), indent=1, default=str)[:4000])
    print("PROMOTED " + out.manifest["model_version"] if out.promoted
          else "not promoted: no Scout champion; The Scout trades rule_fallback")
    _render()
    return 0


def cmd_monitor(args) -> int:
    from wolfpack_worker.analyst.forward import run_monitor
    from wolfpack_worker.analyst.model_io import MANIFEST_FILENAME
    from wolfpack_worker.scout.dataset import build_scout_dataset

    if not (SCOUT_PATHS.champion_dir / MANIFEST_FILENAME).exists():
        print("monitor: The Scout has no champion (rule_fallback); nothing to monitor.")
        return 0
    bars, sessions, arts, as_of = prepare_inputs(args, DEFAULT_SINCE)
    recs = run_monitor(bars, as_of=as_of, paths=SCOUT_PATHS,
                       prepare=lambda b, recipe: build_scout_dataset(b, arts, sessions, recipe.feature_spec_version))
    for rec in recs:
        print(f"monitor {rec['model_version']}: n={rec['n_sessions']} log loss {rec['logloss']} vs base rate "
              f"{rec['baseline_logloss']}, p={rec['p']}, edge_significant={rec['edge_significant']}")
    _render()
    return 0


def cmd_render(args) -> int:
    _render()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--store", choices=("supabase", "local"), default="supabase")
    common.add_argument("--cache", type=Path, default=None)
    common.add_argument("--no-backfill", dest="backfill", action="store_false",
                        help="read `prices` as-is (read-only) instead of re-running the price backfill first")
    p = sub.add_parser("report", parents=[common])
    p.add_argument("--since", type=date.fromisoformat, default=DEFAULT_SINCE)
    p.set_defaults(fn=cmd_report)
    p = sub.add_parser("experiment", parents=[common])
    p.add_argument("file")
    p.set_defaults(fn=cmd_experiment)
    sub.add_parser("monitor", parents=[common]).set_defaults(fn=cmd_monitor)
    sub.add_parser("render", parents=[common]).set_defaults(fn=cmd_render)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
