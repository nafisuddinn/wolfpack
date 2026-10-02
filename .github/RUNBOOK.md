# WolfPack — Automated Operations Runbook

This file is read by scheduled GitHub Actions workflows. It describes only the
operational procedures those workflows run — not how this project is built.

## Daily rationale procedure

1. Read today's raw trades from the database (already written by the
   mechanical trade-execution step — this step does not place trades).
2. For each trade, write a first-person, plain-language rationale in that
   persona's voice:
   - The Analyst: base the rationale on that trade's feature importances.
   - The Pack: base the rationale on recent trust-weight changes.
3. Write each rationale back to the database, attached to its trade.
4. Commit the result.

## Weekly retrain / alphagate procedure

The mechanical step (not yet wired into `weekly-retrain.yml`; that waits for
two clean manual refresh runs) is
`uv run --project worker --group train -m wolfpack_worker.analyst.retrain refresh`.
It, and only it, trains, runs the `alphagate` gate, appends the decision
(promote OR reject, with the reason) to
`worker/models/analyst/gate_log.jsonl`, writes the champion only on PROMOTE,
tags the MLflow run, and regenerates the gate-history table in
`MODEL_CARD.md`. New recipes are never tried here: they go through committed
registrations in `worker/experiments/analyst/` and
`retrain experiment <file>`, run by a human.

The Claude step that follows writes prose only:

1. Read the newest record(s) in `worker/models/analyst/gate_log.jsonl`
   (decision, `reason_code`, `explanation`, scores, `comparator_stats`).
2. Write a short plain-language note of what was decided and why, including
   why a rejected challenger lost. Never describe a promotion as an
   improvement or as edge unless the generated table's "Reading" column says
   so.
3. Never edit, create, or delete anything under `worker/models/analyst/`
   (champion files, `gate_log.jsonl`, `forward_log.jsonl`), never hand-edit
   the generated table between the `analyst-gate-history` markers in
   `MODEL_CARD.md`, and never re-run the gate. The daily job refuses to load
   any champion without a matching PROMOTE record, so a hand-written champion
   would stop The Analyst from trading.
4. Commit the prose changes.

## Non-negotiables

These apply to every trade, rationale, and write-up produced by these
procedures:

- Paper trading only. Never real money, ever.
- When configuring repo secrets, `ALPACA_BASE_URL` must always be set to
  `https://paper-api.alpaca.markets` — the PreToolUse safety hook only covers
  Claude's own tool calls, not this CI secret's value, so a misconfigured
  secret here would not be caught automatically.
- `SUPABASE_SERVICE_ROLE_KEY` is a required repo secret for the mechanical
  (non-LLM) trade-execution and retrain steps only — it must never be passed
  to a Claude/prose-writing step's `env`. Those steps stay scoped to
  `SUPABASE_URL`/`SUPABASE_ANON_KEY`, least-privilege, per the 2026-09-19
  Decision Log entry.
- Log returns, not raw price, for any ML feature.
- Every trade needs a logged rationale. No silent trades.
- State model limitations plainly. Never imply real trading edge that
  hasn't been demonstrated.
