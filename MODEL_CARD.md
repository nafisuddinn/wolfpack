# The Analyst — Model Card

*Maintained by Nafis Uddin, as part of [WolfPack](https://github.com/nafisuddinn/wolfpack).*

*Update this as training happens, not once at the end — report what actually happened, including the parts that don't flatter the model.*

## Why this exists

The Analyst is WolfPack's ML-driven persona: a gradient-boosted classifier predicting next-day price direction from engineered features. It exists to demonstrate a disciplined, honest ML development process — not to claim real trading edge. See the non-claim in `PRD-and-backlog.md` section 6a.

## What's in this repo

- `worker/src/wolfpack_worker/analyst/train.py` — the exact training script (`python -m wolfpack_worker.analyst.train`, report only: it never writes a champion; promotion goes through `python -m wolfpack_worker.analyst.retrain` and the alphagate gate); logs holdout metrics, baselines, walk-forward folds and the isolated backtest
- `worker/src/wolfpack_worker/analyst/features.py` — feature engineering (`FEATURE_SPEC_VERSION = "v1"`: 12 log-return / log-ratio features incl. volatility, RSI on log returns, MA spread, volume ratio, SPY returns)
- `worker/src/wolfpack_worker/analyst/dataset.py` — label, chronological split + 2-session embargo, walk-forward folds, split-artifact data guard
- `worker/src/wolfpack_worker/analyst/metrics.py` — holdout metrics vs base-rate baselines, isolated long/flat backtest
- `worker/src/wolfpack_worker/analyst/gating.py` — the promotion gate: two protocols (trial vs champion's recipe, same-recipe refresh non-inferiority), superiority margin, refresh anchor guard, alpha spending
- `worker/src/wolfpack_worker/analyst/registration.py` — trial pre-registration format and checks; `worker/experiments/analyst/` holds the committed registrations (none yet)
- `worker/src/wolfpack_worker/analyst/gate_log.py` — append-only `gate_log.jsonl` writer/reader and the `load_champion` backstop (refuses a champion with no matching PROMOTE record)
- `worker/src/wolfpack_worker/analyst/retrain.py` — CLI: `experiment <registration>` and `refresh`; the only path that can write a champion
- `worker/src/wolfpack_worker/analyst/explore.py` — unregistered exploration on data strictly before the gate holdout (MLflow experiment `the-analyst-dev`)
- `worker/src/wolfpack_worker/analyst/forward.py` — `forward_log.jsonl`: out-of-sample record of a promoted champion's live predictions
- `worker/src/wolfpack_worker/analyst/render_history.py` — generates the gate-history table below from the logs
- `worker/src/wolfpack_worker/analyst/log_guard.py` — CI check that the gate and forward logs are append-only vs the base branch
- `worker/recipes/analyst/v1.toml` — v1's recipe; `worker/models/analyst/gate_log.jsonl` — the gate decisions
- `worker/src/wolfpack_worker/strategies/analyst.py` — daily inference strategy (loads the committed champion)
- `worker/models/analyst/champion/` — committed champion `model.json` + `manifest.json`
- `worker/mlflow/mlflow.db` + `worker/mlflow/artifacts/` — MLflow tracking data (local sqlite, committed) — full experiment history

Reproduce with (from the repo root; needs Alpaca + Supabase service-role credentials in `.env`):
```bash
uv run --project worker --group train -m wolfpack_worker.analyst.train
```

Current champion's `model_version` is `analyst-20261001-7e2cc1bc` — the embedded date is UTC (training ran locally on 2026-09-30); this is a naming quirk of the versioning scheme, not a discrepancy in when the model was actually trained.

## Benchmark

*Fill in as each version trains. Report both the isolated backtest number and the number as realized through the full pipeline — they will differ.*

DSR isn't computed yet — vol-sizing and the Deflated Sharpe Ratio are both Week 3 backlog items, so "through full pipeline" below means "after this strategy's own vol-target sizing and txn costs are wired into the signal," which hasn't happened yet either. Numbers below are what v1 actually produced: holdout classification metrics plus an isolated gross (pre-cost) long/flat backtest on the same holdout window.

| Version | Holdout accuracy (vs. train-up-rate baseline) | Holdout log loss (vs. baseline) | Holdout AUC | Walk-forward accuracy (8 yearly folds) | Isolated backtest, gross log-return (vs. buy-and-hold) | DSR / full-pipeline (vol-sizing + txn costs) | Promoted by `alphagate`? |
|---|---|---|---|---|---|---|---|
| v1 (`analyst-20261001-7e2cc1bc`, trained 2026-09-30) | 0.519 vs 0.539 baseline (worse) | 0.6963 vs 0.6902 baseline (worse — higher is worse) | 0.485 (below 0.5 = random) | mean 0.531 ± 0.028; only 2/8 folds (2021, 2024) beat the log-loss baseline, 2024 essentially a tie | +0.162 vs +0.201 buy-and-hold (model underperforms) | not computed — Week 3 items (vol-sizing, DSR) not built yet | Yes, as the bootstrap champion with no incumbent (gate record `932e980ab5dc4e46a91a7d012d719fa0`, 2026-10-02, no comparison made) despite not beating baseline; see Decision Log 2026-09-30 / 2026-10-02 and Training History |

Per-ticker holdout accuracy: QQQ 0.548, JPM 0.536, SPY 0.516, AAPL 0.504, XOM 0.492.

## Stress test

*One specific historical high-volatility window. Report the result plainly — this is meant to be an honest check, not a flattering one.*

- **Window tested**: the two worst folds of the 8-year walk-forward validation, which happen to be 2020 and 2022 — both real high-volatility/drawdown years (COVID crash, 2022 rate-hike selloff).
- **Result**: 2022 — accuracy 0.480 (worse than a coin flip), log loss 0.728 vs. 0.700 baseline. 2020 — log loss 0.706 vs. 0.690 baseline. Both years the model's log loss got worse relative to baseline than its 0.531 average across all 8 folds, i.e. it degrades specifically in high-volatility years rather than holding steady.
- **Interpretation**: the model does not fall back toward the base rate when volatility spikes — it stays confidently wrong instead of hedging, which is the worse of the two failure modes (a model that just got less sure would be less harmful than one that stayed sure and wrong). This does not tell us how the model behaves in a real-time, as-yet-unseen crash, only how it behaved on two specific past ones already in its holdout/fold windows.

## Training history

*Pass-by-pass, including retrains that made things worse. Log rejected `alphagate` challengers here too, not just the promoted champion.*

| Date | Change | Result | Promoted? | Why |
|---|---|---|---|---|
| 2026-09-30 | Initial training — `analyst-20261001-7e2cc1bc`, first XGBoost champion, no prior model to compare against | Holdout accuracy 0.519 vs. 0.539 baseline, log loss 0.6963 vs. 0.6902 baseline (worse), AUC 0.485. Walk-forward mean accuracy 0.531 ± 0.028 across 8 yearly folds, only 2021 and 2024 beat the log-loss baseline (2024 a near-tie). Isolated gross backtest +0.162 log-return vs. +0.201 buy-and-hold. Worst folds 2020 and 2022 (see Stress test above) | Yes — promoted as `alphagate` champion despite not beating its own baseline | No existing champion to compare against; this backlog item is "ship an initial XGBoost model," not "ship a model with demonstrated edge." The `alphagate` champion/challenger gate now exists (built 2026-10-02) and is the only path by which a challenger can replace v1; v1 itself was never judged by it against an incumbent. See Decision Log 2026-09-30. |
| 2026-10-02 | Bootstrap gate record `932e980ab5dc4e46a91a7d012d719fa0` for v1 (recipe `a8e2709b0d8e`), written after the fact; re-scores the unchanged artifact on its own holdout (2025-09-26 to 2026-09-30, 252 sessions, 1260 rows) | Log loss 0.6963 vs. 0.6902 base rate (worse), accuracy 0.519, AUC 0.485. No comparison made. | Yes (`no_incumbent`) | The deployed champion needs a matching PROMOTE record or `load_champion` refuses it. This is bookkeeping, not evidence of edge. No experiments have been run through the gate yet. |

## Promotion-gate history (generated)

Generated from `worker/models/analyst/gate_log.jsonl` and `forward_log.jsonl` by `uv run --project worker -m wolfpack_worker.analyst.render_history`. Do not edit the table by hand; a test fails if it drifts from the logs.

<!-- BEGIN GENERATED: analyst-gate-history. Written by worker/src/wolfpack_worker/analyst/render_history.py from gate_log.jsonl + forward_log.jsonl; do not edit by hand. -->
**0 trials registered, 0 promoted.** (A trial is a committed registration in `worker/experiments/analyst/`, counted whether or not it ran. Refreshes and the one-time bootstrap are not trials.)

Log loss is the per-session mean across the 5 tickers (lower is better). "Base rate" = always predicting the training up-rate. "Reading" uses fixed words: *improved* only if the trial's paired test vs the champion's recipe was significant at its alpha_k; *edge* only if the model's forward record covers at least 126 sessions and is significantly better than the base rate; otherwise *no detectable change*.

| Date (UTC) | Kind | Recipe / hypothesis | Holdout | Log loss: challenger / champion / base rate | Paired test vs champion | Beats base rate (point estimate)? | Significantly better than base rate? | Decision (reason) | Forward since promotion: log loss vs base rate | Reading |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-02 | bootstrap | `a8e2709b0d8e` (v1) | 2025-09-26 to 2026-09-30 (252 sessions; end = last label bar) | 0.6963 / n/a / 0.6902 | n/a (no incumbent) | no (not gated) | n/a | PROMOTE (no_incumbent) | not monitored yet | no comparison (bootstrap) |
<!-- END GENERATED: analyst-gate-history -->

## Known limitations

*Name the specific observed failure mode once you have one — not a generic disclaimer.*

- No meaningful real-world predictive edge on public daily price data — this is a process demonstration, not an alpha claim. Confirmed, not just disclaimed: holdout accuracy (0.519) is below the trivial "always predict training up-rate" baseline (0.539), log loss is worse than that same baseline (0.6963 vs. 0.6902), and holdout AUC is 0.485 — below the 0.5 that pure chance would produce.
- **Observed failure mode**: during the two highest-volatility/drawdown years in the 8-fold walk-forward (2020, 2022), the model doesn't degrade toward the base rate the way a well-calibrated model should — it stays confidently mis-calibrated, producing a worse log loss gap vs. baseline than its already-weak average. See Stress test above.
- **Expectation, not a result**: the architect's honest read is that a realistic ceiling for this kind of model on public daily prices is holdout AUC around 0.51-0.53 and accuracy within about 1 point of the base rate; 60%+ accuracy should be treated as a likely leak, not a success. Nothing here promises any gate-approved model will reach even that.
- **Gate limitations** (all recorded in Decision Log 2026-10-02):
  - The Diebold-Mariano test is mildly liberal at n=252 (size ~5.8% iid, ~6.9% at autocorrelation 0.3, ~9.5% at 0.6 vs. nominal 5%), so the 5% lifetime false-promotion bound is nominal only.
  - Holdouts overlap and are reused across trials, which weakens the formal multiple-testing guarantee.
  - The refresh anchor guard is a point estimate on one noisy 252-session window; a hard-regime year may reject a legitimate refresh (it fails safe and keeps the champion).
  - Refresh non-inferiority power depends on how correlated the two models' losses are (~23% pass at true difference 0 with correlation 0.90, ~88% at 0.99). Real numbers will come from the first manual refreshes.
  - The gate log is a consistency check, not tamper-proof: `load_champion` and the append-only CI check catch accidents and casual edits, but a carefully field-copied forged record would still load.
  - Passing the gate means "not detectably worse" (refresh) or "detectably better than the champion's recipe on one year" (trial). It does not mean the model has edge; v1's own gap to the base rate (+0.0061 log loss) is the anchor refreshes are held to.
- Live champion's `trained_through` lags today by design (see Decision Log 2026-09-30) — the holdout numbers above describe exactly the model that is live-trading, not a model that was evaluated one way and swapped for another.

## License / reuse

Training code, eval harness, and MLflow tracking data are all in this repo — nothing depends on anything unreleased.
