# Signal roadmap (design note, 2026-10-04)

STATUS: design note, NOT yet approved by Nafis. Open questions below are unanswered. Draft Decision Log rows are pending confirmation and are NOT in the PRD Decision Log.

## Decision

In what order to spend the limited pre-registered trial budget across candidate signal directions, and what each one plugs into.

## Options

1. **Analyst recipe versions only.** Simplest, but fails. `build_shared_holdout` raises `HoldoutAlignmentError` when labels differ, and `Recipe.from_dict` pins label, universe and the `binary:logistic` objective. Label changes (horizon, cross-sectional rank, volatility) cannot be compared to v1.
2. **Model families, each with its own slice of the alpha budget (RECOMMENDED).** Same-label changes stay Analyst recipe versions. A new label is a new lineage, judged against its own base rate. The vol forecast is a shared risk model feeding sizing for every persona, not a persona. One committed ledger splits the project-wide 0.05.
3. **Each family gets the full 0.05.** With 4 families the project-wide false-promotion rate is about 19%.

## Recommended trial order

1. A1: regularized logistic regression on v1 features.
2. V1: volatility forecast.
3. A2: regime features (only if exploration supports).
4. Cross-sectional lineage (only if Nafis approves).

Vol-targeted sizing ships first as a fixed textbook rule and costs no trial.

## Honest expectation

On the current gate, a genuine daily-direction signal is very unlikely to promote. Rough estimate: n=252, 5 correlated tickers, 0.0005 margin, alpha ~0.0075 means a challenger needs ~0.002-0.003 better log loss, about AUC 0.54+ on daily direction. That is not realistic. Task T0.3 (power simulation) must check this before any trial is spent.

Plausible real signal: volatility (clusters strongly) and possibly cross-sectional ranking (removes market direction, improves gate power).

## Per-direction assessment

### #6 Regularized logistic regression on v1 features
- Chance of edge: low. Chance of PROMOTING: moderate-to-high, because the v1 refit is 0.006 worse than base rate and overconfident.
- A promotion means "stopped being confidently wrong", not signal.
- Data: existing Alpaca data. Standardization is fit on training rows only. Label unchanged.
- Vehicle: Analyst recipe, trial A1. A promoted logreg becomes a calibrated incumbent, so later trials are honest "beats near-base-rate" tests.

### #1 Vol forecast plus vol-targeted sizing
- Highest chance of real signal. Vol clustering is robust; VIX forecasts realized vol.
- Sizing evidence is mixed: Moreira-Muir found Sharpe gains; Cederburg et al. (2020) found they do not hold reliably out of sample. Sizing mostly equalizes risk and trims tails, not edge.
- Data: Alpaca OHLC plus VIX via FRED VIXCLS/VXVCLS. VIX for day t is published on FRED at t+1, so use the t-1 value.
- EWMA's unbounded recursion breaks the fixed-window parity contract in `features.py`, so use `vol_20`.
- Label: realized variance over the next 5 sessions. HAC lags >= 2h = 10. Purging works via `label_end_ts`.
- Sizing rule: no trial (untuned, like the 20/50 SMA).
- Forecast model: risk-vol family, trial V1. Not a persona.

### #2 FRED regime features
- Direction: low. Calibration: moderate (may teach the model to back off toward base rate in high-VIX years, the named 2020/2022 failure mode). Stronger as input to the vol model.
- FRED API requires a free key.
- Since April 2026 BAMLH0A0HYM2 (HY spread) has only a rolling 3-year window on FRED: do not use. Use BAA10Y (check history) or ln(HYG/IEF) from Alpaca.
- Use only market-observed daily series (VIX, VIX3M, T10Y2Y, T10Y3M, BAA10Y), which are not materially revised, so ALFRED is not needed.
- Exclude macro releases (CPI, payrolls, GDP) in v1: they need vintages and release calendars.
- Lag-1 rule. Transforms only (log VIX, log VIX/VIX3M, slope, 20-day changes), never raw rate levels.
- VIX on FRED is (c) Cboe: do not commit raw series to the public repo.
- Vehicle: Analyst feature spec v2, trial A2. Register only if it passes the exploration screening rule.

### #3 Wider universe, cross-sectional ranking
- Edge: low-to-moderate in large caps (short-term reversal and industry effects have weakened), but best gate power: a market-neutral label reduces cross-ticker error correlation, so averaging over N names cuts SE.
- Data: Alpaca free SIP history from 2016; daily is fine. Point-in-time S&P 500 membership from a public GitHub dataset (licence to check).
- Storage: 100-700 tickers is ~1.75M rows. Use a local parquet cache, not Supabase (500 MB free) and not git.
- Survivorship: choosing today's large caps is lookahead. Pick the top ~100 by trailing 60-day dollar volume among point-in-time members as of each date. Delisted names missing from Alpaca are a documented residual bias; report how many.
- Bars are split-adjusted only, so ex-dividend drops bias labels: fetch `Adjustment.ALL` into a separate store.
- Label: 1[5-day return > cross-sectional median], base rate ~0.5, HAC lags >= 10.
- New lineage `analyst-xs`, bootstrapped against its own base rate; X1 must significantly beat base rate.
- Switching the Analyst's live lineage is a human decision with a Decision Log entry.

### #3b Train on wider universe, keep label
- Edge: low; more rows mainly reduce variance.
- Holdout stays on the 5 traded tickers (the shared-row intersection does this).
- Vehicle: Analyst recipe with an optional `train_universe` key. A candidate for A2.

### #4 Longer horizons, absolute direction
- Weekly: low. Monthly: untestable. h=5 gives ~50 independent observations per ticker over 252 sessions; h=21 gives ~12. The DM test is liberal at small n.
- DROP absolute weekly/monthly direction on 5 tickers. Weekly only inside the cross-sectional lineage.

### #5 Earnings / EDGAR / post-earnings drift
- PEAD in large caps is close to gone (Martineau 2021). ~20 events in the holdout means no power. Useful for vol instead (scheduled vol jumps).
- Data: SEC EDGAR `data.sec.gov/submissions` JSON, 8-K item 2.02 plus `acceptanceDateTime`. Free, 10 req/s, needs a User-Agent with a contact address kept in a secret, not committed.
- Use the acceptance timestamp vs the 16:00 ET close. "Days since last earnings" is backward-looking and fine. A "next earnings date" derived from realized filings would be lookahead: do not use.
- Vehicle: vol feature for trial V2, not a direction trial.

### Not used
- Fama-French factors: not a feature (published 1-2 months late); attribution only.
- GDELT, Google Trends, Reddit: skip (noisy, unofficial or gated; the Scout covers news).

## Multiple testing: alpha ledger (do before trial 001 runs)

- `worker/experiments/alpha_ledger.toml` with fixed allocations summing <= 0.05: analyst-direction 0.015, analyst-xs 0.010, risk-vol 0.010, scout 0.015.
- Each family spends alpha_k = A_f/(k(k+1)) with its own experiments directory and `gate_log.jsonl`. For analyst-direction: k=1 0.0075, k=2 0.0025, k=3 0.00125; about 3 worthwhile trials.
- Unspent alpha never moves between families.
- After the first trial in any family the ledger is append-only (new families from leftover headroom only). Extend `log_guard`/CI accordingly.
- Today `ALPHA_TOTAL=0.05` is hard-coded for the Analyst. Changing it now costs nothing because k=0; after a trial it would be retroactive.
- The leaderboard DSR trial count is a separate correction.
- **Explore before you register.** Exploration on pre-holdout data is free. Register only if exploration beats the champion recipe's walk-forward mean log loss (QLIKE for vol) AND beats base rate in >= 5 of 8 folds. Write this rule into the Decision Log before using it.

## Sequence

- **Phase 0 (no trials):** ledger, power simulation, recipe schema extension, generic registration.
- **Phase 1:** vol-targeted sizing and transaction costs (no trial), then A1.
- **Phase 2:** FRED ingestion, vol family bootstrap and V1 (log-HAR OLS plus log VIX(t-1) vs the incumbent `vol_20` rule, scored on QLIKE (Patton 2011); fixed relative margin set from exploration before registering; if V1 promotes, sizing switches to the forecast). V2 (earnings timing) optional.
- **Phase 3:** A2, either feature spec v2 (regime) or `train_universe`, whichever passes screening. Only one is registered.
- **Phase 4:** cross-sectional lineage only on Nafis's go-ahead; otherwise it moves to PRD Phase 2.
- **Scout:** build IN PARALLEL (committed Week 2 scope; ingestion is coder plumbing independent of the above). Its gated trials draw from the scout allocation, so wait for T0.2. The Scout design may have assumed a full 0.05 and the ledger replaces that. Note: the Scout design file's gate alpha `spend_alpha(0.05,1)=0.025` must be revisited to 0.015/2=0.0075 under the ledger.

## Implementation details

- **Recipe schema:** add optional `model_kind` (default `"xgboost"`) and `logreg_params`. `to_dict` must leave out `model_kind` when xgboost so the v1 recipe_id (`a8e2709b0d8e`) and the refresh anchor do not change. Add a test pinning the hash. `xgb_params` only required for xgboost.
- **Logreg artifact:** coefficients plus training mean/std as JSON, never a pickle. Coefficient x standardized value gives rationale feature importances.
- **Per-label constants:** make `EMBARGO_SESSIONS` and `HAC_LAGS` per-label (HAC lags >= 2h for overlapping labels).
- **Sizing:** weight = min(1, sigma_target / vol_20_annualized); notional = $3,000 x weight; no leverage; whole shares. Rebalance only on a long/flat flip or when share count is off target by > 25% (limits cost, rationale volume, and wash-trade collisions from Decision Log 2026-09-26; revisits the flip-only executor decision of 2026-09-24). Put `vol_20` and weight in the signal payload.
- **FRED storage:** `macro_series(series_id, obs_date, value, fetched_at)` table, a few thousand rows. The feature for decision t joins `obs_date <= session t-1`. Tester asserts no value dated t is used.

## Task breakdown

| ID | Task | Owner |
|---|---|---|
| T0.1 | Decision Log rows | documentarian |
| T0.2 | Alpha ledger, family-aware `spend_alpha`, CI ledger guard | ml-trainer |
| T0.3 | Power simulation using loss variances from the pre-holdout walk-forward region only; pass probability vs effect size per family | ml-trainer |
| T0.4 | Recipe schema extension, logreg scorer and artifact, model_io and inference parity | ml-trainer |
| T0.5 | Generalize `registration.py` (pluggable recipe parser, per-family dir and log) | ml-trainer |
| T1.1 | Vol-targeted sizing in executor, PRD 6d.4 | coder |
| T1.2 | Flat bp transaction cost | coder |
| T1.3 | Explore, register, run A1; then tester, reviewer, documentarian | ml-trainer et al. |
| T2.1 | FRED client and `macro_series` table, free key as secret | coder |
| T2.2 | Vol family: label, QLIKE, log-HAR, bootstrap of `vol_20` incumbent, V1 | ml-trainer |
| T2.3 | EDGAR 8-K 2.02 ingestion (optional, for V2) | coder |
| T3.1 | Feature spec v2 / `train_universe`, explore, A2 if it passes | ml-trainer |
| T4.x | Point-in-time universe builder and parquet cache | coder |
| T4.x | `analyst-xs` lineage (only if approved) | ml-trainer |

## Risks

- **A1 can be misread.** Promotion means less overconfident, not signal. Rationale and MODEL_CARD must say so and headline `edge_vs_baseline`.
- **Exploration overfits pre-holdout years.** The gate stays honest but the pass rate drops.
- **Holdout reuse and single regime.** Every trial is judged on the same trailing 252 sessions; accepted in the 2026-10-02 entries.
- **Scope.** DSR, The Pack and The Howl are the core story; Phases 2-4 must not starve them. Complexity Principle (PRD 5a): frame this as gate-rejection evidence, not more sophistication.
- **Licensing/data.** Do not redistribute FRED/Cboe raw series; ICE HY history is truncated on FRED; a wider universe strains Supabase storage.
- **Behaviour change.** Sizing changes Trend Follower and Contrarian too: more trades, rationales and collisions.

## Open questions for Nafis (UNANSWERED)

1. Accept the ledger split (0.015 / 0.010 / 0.010 / 0.015)? It must be fixed before trial 001.
2. Cross-sectional lineage: v1 scope (at the cost of Week 3 time) or PRD Phase 2? Keep the trading universe at the shared 5 tickers (recommended)?
3. Should sizing apply to all personas as PRD 6d.4 says (this changes 20/50 SMA and Bollinger cadence)?
4. OK with the Analyst becoming a near-base-rate "less opinionated" persona if A1 promotes?
5. Sign up for a free FRED API key and provide an EDGAR User-Agent contact, to be stored as secrets.

## Draft Decision Log rows (2026-10-04, PENDING confirmation, not in the PRD)

| Date | Decision | Why | Traded off |
|---|---|---|---|
| 2026-10-04 | (a) Alpha ledger splitting 0.05 across families | Separate 0.05 budgets allow ~19% project-wide false promotion across 4 families; fixing it at k=0 is free | Lower alpha per trial (Analyst first trial 0.0075, not 0.025); unspent alpha is not transferable |
| 2026-10-04 | (b) Vehicle rule: same-label changes = Analyst recipe versions; label change = new lineage judged vs its own base rate; lineage switch is a human decision with a log entry; vol forecast is a shared risk model, not a persona | `HoldoutAlignmentError`; PRD caps launch at 4 personas | Per-family registration dirs and gate logs; `registration.py` generalization; lineages not statistically rankable against each other |
| 2026-10-04 | (c) Vol-targeted sizing = untuned `vol_20` rule (min(1, sigma_target/sigma_hat), 25% rebalance band, no leverage) outside the gate; a learned vol forecast replaces it only via a risk-vol trial scored on QLIKE; FRED features limited to market-observed daily series with lag-1, macro releases excluded | Untuned rule adds no hidden trial; `vol_20` has parity; avoids ALFRED | Sizing not validated as adding Sharpe; more trades and rationales; macro signal left out; ICE HY spread unusable |

## Sources

- FRED BAMLH0A0HYM2 (3-year limit since April 2026): https://fred.stlouisfed.org/series/BAMLH0A0HYM2
- ALFRED: https://alfred.stlouisfed.org/series?seid=BAMLH0A0HYM2
