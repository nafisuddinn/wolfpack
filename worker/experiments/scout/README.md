# The Scout: pre-registered experiments

Same rules as `worker/experiments/analyst/README.md` (the code is shared,
parameterized by `SCOUT_PATHS`), with The Scout's OWN trial budget:

- Each file `NNN-<slug>.toml` here is one **trial**: `trial_number`,
  `registered`, a one-line `hypothesis`, and a full `[recipe]` table whose
  `feature_spec_version` is a Scout spec (`scout_v1`). One recipe per file.
- Commit the file BEFORE it is run, then run
  `uv run --project worker --group train -m wolfpack_worker.scout.retrain experiment worker/experiments/scout/NNN-<slug>.toml`.
  The runner refuses if the file isn't committed and unchanged, if the code,
  recipes, models, pyproject or lockfile have uncommitted changes, or if the
  registration or recipe was ever evaluated before. A registration runs once;
  never edit, rename or delete one that has a gate record.
- **k = max(number of files here, distinct trials in
  `worker/models/scout/gate_log.jsonl`)**, counting only The Scout's trials.
  Trial k is tested at `alpha_k = alpha_total / (k (k + 1))`, with
  `alpha_total` from `worker/recipes/scout/gate.toml` (0.05 per the confirmed
  design, so trial 1 is tested at 0.025; pending reconciliation with the
  unapproved signal-roadmap alpha ledger, which would make it 0.015 -> 0.0075).
- While The Scout has no champion, a trial is one gate call against the
  constant base-rate predictor: one-sided paired Diebold-Mariano superiority,
  margin 0.0005 log loss, 5 Newey-West lags, trailing 252-session holdout.
  PROMOTE writes `worker/models/scout/champion/`; REJECT is logged with its
  reason and The Scout keeps trading its untuned `rule_fallback`.
- The M1 news coverage report (`... scout.retrain report`, written to
  `worker/reports/scout/news_coverage.json`) uses no prices or labels and is
  produced before registering, so it cannot steer a recipe toward the holdout.

No edge is claimed either way. Little or no signal is expected from
headline sentiment on five heavily covered tickers.
