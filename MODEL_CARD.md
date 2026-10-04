# The Analyst — Model Card

*Maintained by Nafis Uddin, as part of [WolfPack](https://github.com/nafisuddinn/wolfpack).*

*Update this as training happens, not once at the end — report what actually happened, including the parts that don't flatter the model.*

## Why this exists

The Analyst is WolfPack's ML-driven persona: a gradient-boosted classifier predicting next-day price direction from engineered features. It exists to demonstrate a disciplined, honest ML development process — not to claim real trading edge. See the non-claim in `PRD-and-backlog.md` section 6a.

## What's in this repo

- `worker/src/wolfpack_worker/analyst/train.py` — the exact training script (`python -m wolfpack_worker.analyst.train [--promote]`); logs holdout metrics, baselines, walk-forward folds and the isolated backtest
- `worker/src/wolfpack_worker/analyst/features.py` — feature engineering (`FEATURE_SPEC_VERSION = "v1"`: 12 log-return / log-ratio features incl. volatility, RSI on log returns, MA spread, volume ratio, SPY returns)
- `worker/src/wolfpack_worker/analyst/dataset.py` — label, chronological split + 2-session embargo, walk-forward folds, split-artifact data guard
- `worker/src/wolfpack_worker/analyst/metrics.py` — holdout metrics vs base-rate baselines, isolated long/flat backtest
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
| v1 (`analyst-20261001-7e2cc1bc`, trained 2026-09-30) | 0.519 vs 0.539 baseline (worse) | 0.6963 vs 0.6902 baseline (worse — higher is worse) | 0.485 (below 0.5 = random) | mean 0.531 ± 0.028; only 2/8 folds (2021, 2024) beat the log-loss baseline, 2024 essentially a tie | +0.162 vs +0.201 buy-and-hold (model underperforms) | not computed — Week 3 items (vol-sizing, DSR) not built yet | Yes — see Decision Log 2026-09-30 and Training History below for why, despite not beating baseline |

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
| 2026-09-30 | Initial training — `analyst-20261001-7e2cc1bc`, first XGBoost champion, no prior model to compare against | Holdout accuracy 0.519 vs. 0.539 baseline, log loss 0.6963 vs. 0.6902 baseline (worse), AUC 0.485. Walk-forward mean accuracy 0.531 ± 0.028 across 8 yearly folds, only 2021 and 2024 beat the log-loss baseline (2024 a near-tie). Isolated gross backtest +0.162 log-return vs. +0.201 buy-and-hold. Worst folds 2020 and 2022 (see Stress test above) | Yes — promoted as `alphagate` champion despite not beating its own baseline | No existing champion to compare against; this backlog item is "ship an initial XGBoost model," not "ship a model with demonstrated edge." `alphagate`'s champion/challenger gate (Week 3) is the mechanism meant to reject underperforming challengers going forward — having a champion in place now gives that gate something to compare future retrains against. See Decision Log 2026-09-30. |

## Known limitations

*Name the specific observed failure mode once you have one — not a generic disclaimer.*

- No meaningful real-world predictive edge on public daily price data — this is a process demonstration, not an alpha claim. Confirmed, not just disclaimed: holdout accuracy (0.519) is below the trivial "always predict training up-rate" baseline (0.539), log loss is worse than that same baseline (0.6963 vs. 0.6902), and holdout AUC is 0.485 — below the 0.5 that pure chance would produce.
- **Observed failure mode**: during the two highest-volatility/drawdown years in the 8-fold walk-forward (2020, 2022), the model doesn't degrade toward the base rate the way a well-calibrated model should — it stays confidently mis-calibrated, producing a worse log loss gap vs. baseline than its already-weak average. See Stress test above.
- Live champion's `trained_through` lags today by design (see Decision Log 2026-09-30) — the holdout numbers above describe exactly the model that is live-trading, not a model that was evaluated one way and swapped for another.

## License / reuse

Training code, eval harness, and MLflow tracking data are all in this repo — nothing depends on anything unreleased.
