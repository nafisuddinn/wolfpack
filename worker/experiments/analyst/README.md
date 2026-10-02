# The Analyst: pre-registered experiments

Each file `NNN-<slug>.toml` here is one **trial**: one challenger recipe for
The Analyst, registered (committed) *before* it is run against the
promotion gate's holdout. The format and the rules are in
`worker/src/wolfpack_worker/analyst/registration.py`; in short:

- One recipe per file, no grids. Fields: `trial_number` (= NNN),
  `registered` (date), `hypothesis` (one line, written before the run), and a
  full `[recipe]` table (same keys as `worker/recipes/analyst/v1.toml`).
- Commit the file, then run
  `uv run --project worker --group train -m wolfpack_worker.analyst.retrain experiment worker/experiments/analyst/NNN-<slug>.toml`.
  The runner refuses if the file isn't committed and unchanged, if
  `worker/src` (or recipes / pyproject / lockfile / this directory) has
  uncommitted changes, or if the recipe was ever evaluated before.
- **The trial count k is the number of files here**, not the number of runs.
  Every trial is tested at `alpha_k = 0.05 / (k (k + 1))`, so the chance of a
  false promotion across all trials ever run stays below 5%. Never delete a
  file; if a trial is dropped before running, add `abandoned = "<reason>"`
  (it still counts).
- A CI test fails if any file here has neither a gate record nor an
  `abandoned` line.

Exploration (walk-forward on data strictly before the gate holdout, MLflow
experiment `the-analyst-dev`) needs no registration:
`... -m wolfpack_worker.analyst.retrain explore <recipe.toml>`.
