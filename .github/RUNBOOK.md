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

1. Retrain The Analyst on the past week of price data, using only data
   available as of the retrain date — strictly chronological, never
   including same-day/close-inclusive data in features (already run by the
   mechanical retrain step — this step does not train models).
2. Run `alphagate` to compare the newly trained model against the current
   champion model on held-out data.
3. Log the promote/reject decision to MLflow and to `MODEL_CARD.md`,
   including the reason if rejected.
4. If promoted, update the live model reference. If rejected, leave the
   current champion in place.
5. Commit all changes.

## Non-negotiables

These apply to every trade, rationale, and write-up produced by these
procedures:

- Paper trading only. Never real money, ever.
- When configuring repo secrets, `ALPACA_BASE_URL` must always be set to
  `https://paper-api.alpaca.markets` — the PreToolUse safety hook only covers
  Claude's own tool calls, not this CI secret's value, so a misconfigured
  secret here would not be caught automatically.
- Log returns, not raw price, for any ML feature.
- Every trade needs a logged rationale. No silent trades.
- State model limitations plainly. Never imply real trading edge that
  hasn't been demonstrated.
