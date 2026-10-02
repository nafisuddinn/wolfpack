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
- **A registration runs once.** Once a trial has a gate record, its file is
  permanent: never edit, rename, or delete it. Each trial gate record stores
  the trial number, the file name and the file's sha256, and the runner (and
  the daily job's champion check, for a promoted trial) refuses if any logged
  trial's registration is missing, renamed, or changed. A new idea after a
  rejection is a new file with the next number.
- **The trial count k = max(number of files here, number of distinct trials
  in the gate log)**, not the number of runs. Every trial is tested at
  `alpha_k = 0.05 / (k (k + 1))` with a superiority margin of 0.0005 log
  loss, so the chance of a false promotion across all trials ever run stays
  (nominally) below 5%. If a trial is dropped before running, add
  `abandoned = "<reason>"` (it still counts).
- The runner also refuses if `worker/models/` (the gate and forward logs) has
  uncommitted changes, so a logged rejection can't be quietly removed.
- A CI test (`.github/workflows/worker-tests.yml`) fails if any file here has
  neither a gate record nor an `abandoned` line, or disagrees with the log.

Exploration (walk-forward on data strictly before the gate holdout, MLflow
experiment `the-analyst-dev`) needs no registration:
`... -m wolfpack_worker.analyst.retrain explore <recipe.toml>`.
