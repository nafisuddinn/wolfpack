# The Scout — Model Card

*Maintained by Nafis Uddin, as part of [WolfPack](https://github.com/nafisuddinn/wolfpack).*

*Update this as trials happen, not once at the end. Report what actually happened, including the parts that don't flatter the model.*

## Status in one paragraph

Trial 001 (the only trial so far) was **rejected** by `alphagate` on 2026-10-04. There is **no Scout champion**. The Scout trades an untuned VADER rule, labelled `rule_fallback`, not a model. The model did not beat a constant base-rate predictor on the holdout (log loss 0.69383 vs 0.69023), and no edge is claimed. The numbers below are isolated: no volatility targeting, no Deflated Sharpe Ratio, and no full-pipeline transaction costs yet.

## Why this exists

The Scout is WolfPack's sentiment persona: gradient-boosted next-session direction from news-headline features. It exists to demonstrate a disciplined, honest ML process, not to claim trading edge. Little or no signal was expected going in (megacap news is largely priced in during the session, ETF headlines are sparse, VADER misreads finance language), and that is what was found. See the non-claim in `PRD-and-backlog.md` section 6a and the design in `docs/design/scout-design.md`.

## What's in this repo

- `worker/src/wolfpack_worker/scout/` — feature spec `scout_v1` (`features.py`), point-in-time windows (`windows.py`), dataset/labels (`dataset.py`), VADER scoring (`sentiment.py`), trial runner and CLI (`retrain.py`), gate wiring (`gating.py`), coverage report (`coverage.py`)
- `worker/experiments/scout/001-vader-headlines-v1.toml` — the pre-registered trial, committed before training
- `worker/models/scout/gate_log.jsonl` — the gate record (append-only)
- `worker/reports/scout/news_coverage.json` — news coverage report, written before registration from counts and timestamps only (no prices or labels)
- `worker/recipes/scout/gate.toml` — alpha budget
- MLflow run `9b6a0065cc104baaa3e00cbb9bdbd10a` (local file store) — diagnostics, walk-forward, isolated backtest artifacts

Reproduce a trial (a human runs this, never CI):
```bash
uv run --project worker --group train -m wolfpack_worker.scout.retrain experiment worker/experiments/scout/001-vader-headlines-v1.toml
```
Headline text and URLs are licensed (Benzinga via Alpaca) and are never committed; only scores, counts and ids appear in repo files and the public payload.

## Data and coverage

Source: Alpaca news (Benzinga-sourced), 2016-01-04 to 2026-10-02, 120,461 articles in the 5-ticker universe (SPY, QQQ, AAPL, JPM, XOM). No month gaps. StockTwits is **not** a model input (no usable history); its ingestion is deferred (Decision Log 2026-10-04). Reddit is not used.

| Ticker | Headlines | Share of sessions with at least one headline |
|---|---|---|
| SPY | 81,117 | 99.9% |
| AAPL | 31,898 | 99.3% |
| QQQ | 7,635 | 73.3% |
| JPM | 6,298 | 75.4% |
| XOM | 3,931 | 65.0% |

Coverage is uneven over time: QQQ had news on only 28% of sessions in 2016 and 35% in 2017; JPM on 58% in 2016 rising to 95% in 2026. Per-year detail is in `news_coverage.json`. Per-ticker counts overlap (an article tagged SPY and AAPL counts for both).

Timing of articles relative to the session: 61.8% in session, 26.3% pre-open, 8.3% after close, 3.6% on non-session days (rolled into the next session).

**Revisions**: Alpaca filters news by `updated_at`, not `created_at`, and the stored headline is the latest revision. 52.4% of articles have `updated_at` after `created_at`, but most of those are within about a second (p90 lag 1 second); 4.8% of all articles are updated more than a minute later. 0.4% of articles were revised after their own decision-time close, 0.18% by more than a day. This leak is measured, not eliminated: backtest features can see slightly more complete text than was available live. Vendor archive backfill cannot be measured from backfilled rows (no real first-seen time); live latency is unmeasured so far (0 live ingests).

## Features (`scout_v1`, headline sentiment only, no price features)

Per ticker and session t, window W_t = (close_{t-1}, close_t] on `created_at`. VADER compound score per headline (pinned `vaderSentiment`). Six features: `s_mean_1`, `has_news_1`, `n_surprise` (log1p count minus its trailing 20-session mean; a raw count is never a feature), `s_mean_5`, `s_change` (5-session mean minus the prior 20-session mean), `mkt_s_mean_1` (deduplicated universe-wide mean). Label: `y_t = 1[ln(O_{t+2}/O_{t+1}) > 0]` via the Analyst's `make_labels` (2-session embargo, trailing 252-session holdout). Model: untuned XGBoost, identical settings to The Analyst's v1 recipe (recipe `5f94c8619793`, 12,100 training rows, train up-rate 0.5462).

## Benchmark

All numbers isolated. Holdout 2025-09-26 to 2026-09-30, 252 sessions, 1,260 ticker-session rows, trained through 2025-09-25.

| Version | Directional accuracy (model vs base rate) | AUC | Log loss (model vs base rate) | Paired test vs base rate | DSR (isolated) | DSR (full pipeline: vol-sizing + txn costs) | Max drawdown | Promoted by `alphagate`? |
|---|---|---|---|---|---|---|---|---|
| trial 001 (`scout-20261004-6a384b7d`) | 0.5397 vs 0.5389 | 0.503 | 0.69383 vs 0.69023 (worse) | one-sided p = 0.9226 vs alpha 0.025; also rejected at 0.0075 | not computed | not computed (not built yet) | not computed | No, REJECT (`not_significant`) |

The accuracy "win" is 0.0008 on 1,260 rows, which is noise; log loss is worse. Brier 0.25028 vs 0.24854 (worse).

**What actually trades: the rule baseline** (`s_mean_1` > +0.05 long, < -0.05 flat, else hold; VADER's documented neutral band, untuned), holdout, isolated, mean over 5 tickers:

| | Log return, gross | Log return, net (5 bps/side, illustrative flat cost) |
|---|---|---|
| Rule baseline | +0.146 | +0.111 |
| Buy-and-hold | +0.201 | +0.201 (no trading cost applied) |

The rule fires on 67% of rows, is long about 64% of the time, and loses to buy-and-hold even before costs. Rule accuracy when it fires: 0.537 (up-rate 0.545 on long signals vs 0.477 on flat signals vs 0.539 unconditional). Net is a flat-cost estimate, not the full pipeline. For reference only (not traded): the rejected model's isolated backtest was +0.201 gross / +0.180 net, vs buy-and-hold +0.201.

## Walk-forward (8 folds, 2019 to 2025 plus the trailing-252 holdout)

The model beat the base rate on log loss in **1 of 8 folds**: 2025, by 0.00002 (0.68680 vs 0.68683), which is a tie. Every other fold, including the holdout, was worse. Fold accuracy ranged 0.488 to 0.552 against base rates of 0.492 to 0.563; fold AUC ranged 0.459 to 0.518 (2019, 2021 and 2024 below 0.5). Full table in the MLflow `walk_forward.json` artifact.

## Stress test

- **Window tested**: March to April 2020 (215 rows), model trained through 2019-12-31.
- **Result**: accuracy 0.484 vs 0.502 base rate, AUC 0.454, log loss 0.7092 vs 0.6986 (worse than base rate). Isolated model backtest: -0.174 gross / -0.181 net mean log return vs buy-and-hold -0.137, so it lost more than holding. The rule baseline over the same window: -0.006 gross / -0.010 net (it ended nearly flat, mostly by being out of the market 44% of the time, not by being right: accuracy when fired 0.520).
- **Interpretation**: the model is worse than guessing the base rate in a crash, with below-0.5 AUC. One window and one model, so it shows the model did not hold up here, not why. The rule's smaller loss is a flatness effect, not evidence of skill.

## News / no-news split (holdout)

95.1% of holdout rows had news in their window (1,198 rows); 62 did not.

| Split | n | Accuracy (model / base) | Log loss (model / base) |
|---|---|---|---|
| has news | 1,198 | 0.538 / 0.534 | 0.6944 / 0.6911 (worse) |
| no news | 62 | 0.581 / 0.629 | 0.6823 / 0.6735 (worse) |

Neither subset beats the base rate. The no-news subset is tiny and tells us little.

## Revision split (holdout, watch item)

Rows whose news was revised after decision time: n = 61 (4.8%). Model log loss 0.6753 vs base 0.6807 (better), accuracy 0.607 vs 0.590. This is the only subset where the model beats the base rate, and it is the subset where the stored headline may contain information published after decision time, so it is a possible leak signature, not a result. n = 61 is far too small to conclude anything either way. Not-revised rows (n = 1,199): 0.6948 vs 0.6907 (worse). Watch this split in future trials; do not read it as signal.

## Training history

Latest rows first within a trial. Rejected challengers are logged alongside any champion; entries are never removed.

| Date | Change | Result | Promoted? | Why |
|---|---|---|---|---|
| 2026-10-04 | Trial 001, `scout-20261004-6a384b7d`: VADER headline features `scout_v1`, untuned XGBoost (Analyst v1 recipe), pre-registered in `001-vader-headlines-v1.toml` before training; gated vs constant base rate (one-sided paired Diebold-Mariano, margin 0.0005, 5 Newey-West lags) | Holdout log loss 0.69383 vs 0.69023 base rate (worse), accuracy 0.5397 vs 0.5389, AUC 0.503; paired t = -1.42, p = 0.9226 vs alpha_k 0.025 (also rejected at 0.0075); walk-forward beat base rate in 1 of 8 folds (a tie); Mar-Apr 2020 stress accuracy 0.484, AUC 0.454 | No: REJECT (`not_significant`), no champion | Could not conclude the model improves on the base rate by more than the margin; incumbent (base rate) retained, so The Scout runs `rule_fallback`. This was the expected outcome. Trained from a local news cache (production table not yet applied; see Decision Log). Alpha budget is the design's 0.05 total; reconciliation with the unapproved signal-roadmap ledger (0.015) is pending and needed before any trial 002. |

## Promotion-gate history (generated)

Generated from `worker/models/scout/gate_log.jsonl` by `uv run --project worker --group train -m wolfpack_worker.scout.retrain render`. Do not edit the table by hand; a test fails if it drifts from the logs.

<!-- BEGIN GENERATED: scout-gate-history. Written by worker/src/wolfpack_worker/analyst/render_history.py from gate_log.jsonl + forward_log.jsonl; do not edit by hand. -->
**1 trials registered, 0 promoted.** (A trial is a committed registration in `worker/experiments/scout/`, counted whether or not it ran. Refreshes and the one-time bootstrap are not trials.)

Log loss is the per-session mean across the 5 tickers (lower is better). "Base rate" = always predicting the training up-rate. "Reading" uses fixed words: *improved* only if the trial's paired test vs the champion's recipe (or, while no champion exists, vs the base rate) was significant at its alpha_k; *edge* only if the model's forward record covers at least 126 sessions and is significantly better than the base rate; otherwise *no detectable change*.

| Date (UTC) | Kind | Recipe / hypothesis | Holdout | Log loss: challenger / champion / base rate | Paired test vs champion | Beats base rate (point estimate)? | Significantly better than base rate? | Decision (reason) | Forward since promotion: log loss vs base rate | Reading |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-04 | trial #1 (k=1) | `5f94c8619793`: Headline tone (VADER) and news-volume surprise for the 5 tickers predict next-session open-to-open direction better than the base rate by more than 0.0005 log loss on the trailing 252-session holdout; expected to be rejected | 2025-09-26 to 2026-09-30 (252 sessions; end = last label bar) | 0.6938 / n/a (no champion) / 0.6902 | t=-1.42, p=0.923 vs alpha_k=0.025 (superiority by margin 0.0005, vs base rate; no champion) | no | no (p=0.923) | REJECT (vs base rate: not_significant) | n/a (not deployed) | no detectable change |
<!-- END GENERATED: scout-gate-history -->

## Known failure mode

Observed: the model has no usable direction signal. It lost to the constant base rate on log loss in 7 of 8 walk-forward folds and on the holdout, and in the March-April 2020 stress window it was below chance (AUC 0.454) and lost more than buy-and-hold. Its probabilities move away from the base rate without adding information, which costs log loss (inferred from the scores, not separately diagnosed). The rule fallback also trails buy-and-hold on the holdout.

## Known limitations

- **No edge is claimed.** The model was rejected by its own gate. The rule fallback trailed buy-and-hold on the holdout (+0.146 gross, +0.111 net vs +0.201).
- **Isolated numbers only.** No volatility targeting, no Deflated Sharpe Ratio, and no full-pipeline transaction costs yet; the 5 bps net figure is a flat illustrative cost. These will differ from any live result.
- **Vendor headline revisions.** Alpaca filters news by `updated_at`, stored headlines are the latest revision, and a headline can change after decision time (0.4% of articles; 4.8% of holdout rows flagged). Backtest features may be slightly more complete than what was available live. Vendor archive backfill is not measurable from historical rows.
- **Uneven coverage.** XOM, QQQ and JPM have news on only 65 to 75% of sessions overall and far less in early years; SPY and AAPL are close to complete.
- **VADER misreads finance language** (for example "short", "beat", "cut"), and in-session news is largely priced in for megacaps.
- **Holdout power.** 252 sessions with a 0.0005 margin can only detect a fairly large improvement; "not significant" is not proof of no signal, only that none was demonstrated. The signal roadmap's estimate is that AUC around 0.54 or better is needed to promote at current gate power.
- **Possible post-decision-time leak** in the revision split (n = 61); unresolved.
- **Local-cache training.** Trial 001 trained from a local, gitignored news cache. The production `news_articles` table is not yet applied and backfilled, so live news ingestion and live latency are untested.

## License / reuse

Training code, eval harness, and MLflow tracking data are all in this repo; nothing depends on anything unreleased. Headline text and URLs are not redistributed.

## Data source notes (specific to The Scout)

- **News headlines**: Alpaca news API (Benzinga-sourced), the only model input. Free tier, 2015+ history.
- **StockTwits public stream**: not a v1 model input; forward-only collection, ingestion client deferred. Unofficial/undocumented endpoint, not guaranteed stable.
- **Reddit**: not used. Its 2026 Responsible Builder Policy requires pre-approval with multi-week queues.
