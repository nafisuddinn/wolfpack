# Persona #4 "The Scout" — Design Note

Date: 2026-10-03 (confirmed 2026-10-04) · Author: architect · Status: CONFIRMED, ready for implementation (not started). Nafis answered all six open questions on 2026-10-04 (section 13); the matching Decision Log rows are in `PRD-and-backlog.md` section 8. Sections 3, 4, 7 and 12 were amended on 2026-10-04 to reflect the answers (StockTwits deferral, public payload, `NEWS_API_KEY`, C3 deferred).

## 1. Recommendation

- **Model input**: Alpaca/Benzinga headlines only (alpaca-py `NewsClient`; 2015+ history, 200 calls/min, headline only, no body).
- **Scoring**: VADER (pinned `vaderSentiment`).
- **Model**: pooled XGBoost across the 5 tickers, untuned Analyst-v1 hyperparameters.
- **Features (`scout_v1`)**: sentiment-only, no price features. Warmup 25 sessions.
  - `s_mean_1`: mean VADER compound over window W_t (0 if no news)
  - `has_news_1`: 1 if any headline in W_t
  - `n_surprise`: log1p(n_t) minus mean of log1p(n) over t-20..t-1
  - `s_mean_5`: count-weighted mean over W_{t-4..t}
  - `s_change`: `s_mean_5` minus count-weighted mean over t-24..t-5
  - `mkt_s_mean_1`: mean over deduplicated headlines of all 5 tickers
- **StockTwits**: out of the v1 model features (confirmed). Forward-collection only, ingestion client deferred (see section 3).

## 2. Point-in-time rule

- Decision time = `ctx.as_of` = close of session t. W_t = (close_{t-1}, close_t] on `created_at`.
- Weekend/holiday articles roll to the next session.
- Live features additionally require `first_seen_at <= run start`. Late arrivals are counted and reported, never used retroactively.
- Labels reuse `analyst.dataset.make_labels`: y_t = 1[ln(O_{t+2}/O_{t+1}) > 0]. Same 2-session embargo, `trained_through < as_of` guard, 252-session trailing holdout.
- Residual leaks that cannot be fixed, so they are measured and reported: headline revisions (`updated_at > created_at`) and vendor archive backfill.

## 3. StockTwits

CONFIRMED 2026-10-04: out of v1 model features; collected going forward only, and the ingestion client (C3) is DEFERRED, not part of the initial Scout build. Revisit after the Scout model result and when ~12 months of forward collection is worth starting. When built: store message id, time, `user_sentiment` (no body, no username); payload context only, `used_in_model = false`. Reason: the public endpoint returns 30 recent messages per call at roughly 200 req/hr/IP with no usable history. It becomes a registered trial after about 12 months of collection. Shared GitHub Actions IPs may be throttled.

## 4. Rejected alternatives

- **FinBERT for v1**: torch is ~1 GB in a daily cron. Possible future pre-registered trial.
- **LLM scoring**: memorized outcomes are hidden lookahead, non-deterministic, hits Pro limits, and breaks the no-LLM mechanical path rule from 2026-09-19.
- **`NEWS_API_KEY` secret**: if it is NewsAPI.org, the free tier has ~1 month of history and is useless here. CONFIRMED 2026-10-04: remove it, it is unused. Secret removal DONE 2026-10-04 (GitHub Actions secret and local `.env`). Pending: small `coder` pass to remove dead references (`news_api_key` in `worker/src/wolfpack_worker/config.py`, the env line in `.github/workflows/daily-trades.yml`, the placeholder in `.env.example`); harmless until then, empty string resolves to None.

## 5. Schema (one migration)

- `news_articles`: `id` bigint pk, `created_at` timestamptz, `vendor_updated_at`, `headline`, `source`, `url`, `symbols` text[] (GIN index), `first_seen_at` default now(), `ingest_mode` (`backfill` | `live`). Insert on conflict do nothing. RLS: service_role only.
- `stocktwits_messages` (DEFERRED with C3; create the table only when the client is built): `id`, `symbol`, `created_at`, `user_sentiment`, `first_seen_at`. service_role only.
- No sentiment table (scores are computed, not stored).
- Confirm persona row `the-scout` / `ml_sentiment` exists in the prod seed.

## 6. Gate

- Scout v1 is registration #1 in a new `worker/experiments/scout/` with its own trial budget.
- One-sided paired Diebold-Mariano vs `ConstantScorer`, margin 0.0005 logloss, alpha = `spend_alpha(0.05, 1)` = 0.025, 5 Newey-West lags, trailing 252 holdout.
- PROMOTE -> champion. REJECT -> logged, no champion, strategy runs in rule mode.
- Reported, not gated: yearly walk-forward folds, accuracy split by `has_news_1`, rule baseline accuracy and pre-cost return, stress window 2020 Mar-Apr.
- Refactor: parameterize the Analyst governance modules with a `PersonaPaths` dataclass instead of copying them (`model_io`, `gate_log`, `registration`, `log_guard`, `forward`, `render_history`, `gating`). No behavior change; existing Analyst tests stay unchanged.

## 7. Signal-to-trade

- **Model mode**: long if p_up > 0.5, else flat.
- **rule_fallback mode**: `s_mean_1` > +0.05 long; < -0.05 flat; otherwise or no news, omit (hold). The band is VADER's documented neutral band, not tuned.
- Stale or failed news: return `[]` and log the reason.
- **Payload fields**: mode, model_rejected, gate_record_id, p_up or rule value, features, news_cutoff, n_articles, article ids, scores and source names for the top 3 articles by |score| (CHANGED 2026-10-04: no headline text or URLs in the public payload), holdout and baseline metrics, beats_baseline_logloss, stocktwits status (`deferred` until C3 ships), late_arrivals, LIMITATIONS.
- **Public-payload rule (confirmed 2026-10-04)**: the public feed and `signal_payload` show scores, counts, source names and article ids only. Headline text and URLs are stored privately (`news_articles`, service_role only) and must not appear in the public payload, feed, or rationale text, because Benzinga redistribution terms are unclear. Rationales may cite scores and counts but must not quote headlines. Tests should assert no headline/URL string reaches the payload.

## 8. Daily job

- `refresh_news` runs in `daily_trades.main` after `refresh_prices`, in its own try/except. `refresh_stocktwits` is deferred with C3.
- `StrategyContext` gains a `news` mapping that raises `LookaheadError` on `created_at > as_of`; `truncate_news` in `run_persona`; `requires_news` attribute on strategies.
- Add `the-scout` to `REGISTRY`.

## 9. Tests (tester)

- Window boundaries: close_t exactly, +1s, weekends, early close 13:00 ET, DST (20:00 vs 21:00 UTC).
- Truncation-invariance property test; after-close canary.
- Context raises on future or naive timestamps.
- Rolling features use only prior sessions.
- Live path drops late `first_seen_at`; upsert never overwrites.
- (Deferred with C3) StockTwits failure -> still trades. News failure -> `[]`, other personas unaffected.
- Gate / backstop / `log_guard` coverage and Analyst regression.
- Labels come from `make_labels` import.

## 10. `MODEL_CARD_SCOUT.md`

The current template is wrong (describes price features and `train_scout.py`). Rewrite with: coverage per ticker/year, revision and late-arrival rates, feature spec, gate result (expect REJECT, stated plainly), rule baseline, folds, `has_news_1` split, stress window, live mode, StockTwits status, limitations (VADER misreads finance language, in-session news is priced in, holdout power).

## 11. Risks

- Little or no signal expected: megacap news is priced in during the session; ETF news is sparse.
- Revisions and vendor backfill.
- StockTwits ToS / IP blocking.
- Headline licensing.
- Supabase free-tier size (~50 MB articles, ~0.3 MB/day StockTwits).

## 12. Task breakdown

- C1 (coder): migration + store protocols
- C2 (coder): `news.py` ingestion + `news_backfill` CLI `--since 2016-01-04`
- C3 (coder): `stocktwits.py` best-effort client. DEFERRED (confirmed 2026-10-04); not part of the initial Scout build
- C4 (coder): `StrategyContext.news` / `truncate_news` / `requires_news` / refresh steps
- M0 (ml-trainer): `PersonaPaths` refactor
- M1 (ml-trainer): coverage/latency/revision report (no label correlations looked at before registration)
- M2 (ml-trainer): features + recipe + commit registration BEFORE training
- M3 (ml-trainer): train/gate/walk-forward/rule baseline/stress/MLflow
- M4 (ml-trainer): `strategies/scout.py`
- T (tester): suite
- D (documentarian): model card + Decision Log

C1, C2, C4 and M0 are parallelizable (C3 deferred).

## 13. Open questions for Nafis (ANSWERED 2026-10-04)

1. StockTwits out of v1 model features? ANSWERED: yes, out. Collected going forward only (PRD section 6 table feature list departs accordingly).
2. Gate-or-fallback vs promote-anyway? ANSWERED: gate-or-fallback. The model trades only if it passes the `PairedComparator` gate vs base rate; otherwise the untuned VADER-neutral-band rule runs, labelled `rule_fallback`.
3. Headline text/URL in public payload/feed? ANSWERED: no. Scores, counts and source names only; headlines and URLs stored privately (service_role only). Design impact: the "top 3 headlines" payload field becomes ids/scores/sources (section 7).
4. Call the undocumented StockTwits endpoint from a public repo? ANSWERED: low priority; defer the ingestion client. C3 is deferred (section 12); revisit after the model result and when ~12-month forward collection is worth starting.
5. What was `NEWS_API_KEY` for? ANSWERED: unused; remove it. NewsAPI.org free tier is useless for backtesting. Secret removal done 2026-10-04; dead-reference cleanup is a pending `coder` pass (section 4).
6. VADER vs Loughran-McDonald? ANSWERED: VADER (pinned `vaderSentiment`, MIT licence).

## 14. Decision Log entries (CONFIRMED 2026-10-04)

Logged in `PRD-and-backlog.md` section 8 with 2026-10-04 dates (plus rows for the public-payload rule and `NEWS_API_KEY` removal). Summaries below; the PRD rows are authoritative.

- StockTwits forward-collection only, not a v1 model input; ingestion deferred. Why: no usable history, ~30 msgs/call rate limits. Traded off: a second sentiment source in v1.
- Point-in-time rule (W_t on `created_at`, live `first_seen_at` filter). Why: prevents news lookahead. Traded off: late-arriving articles are never used retroactively, and some revisions/backfill leaks stay unfixable (only measured).
- VADER over FinBERT/LLM scoring. Why: light, deterministic, no lookahead via memorization. Traded off: misreads finance language.
- Gate-or-fallback (REJECT -> rule mode, no champion) instead of promoting anyway. Why: avoids implying edge. Traded off: Scout may run on an untuned rule most of the time.
- `PersonaPaths` parameterization of Analyst governance modules instead of copying. Why: one gate implementation. Traded off: a refactor touching working Analyst code (mitigated by unchanged Analyst tests).
