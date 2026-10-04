"""The Scout (Persona #4): headline-sentiment direction classifier, gate-or-fallback.

Design: docs/design/scout-design.md (CONFIRMED 2026-10-04).

MODEL-RISK LIMITATION: little or no signal is expected. Headlines for five
large, heavily covered US tickers are largely priced in during the session
they are published, ETF headlines are sparse, and VADER is a general-purpose
lexicon that misreads finance language ("short", "beat", "cut"). The Scout's
model only trades if it passes the alphagate gate against the base rate;
otherwise The Scout trades an untuned VADER-neutral-band rule, labelled
`rule_fallback`, which is no evidence of edge either. See MODEL_CARD_SCOUT.md.

Modules:
- paths.py     SCOUT_PATHS (worker/models/scout, worker/experiments/scout, MODEL_CARD_SCOUT.md)
- sentiment.py VADER headline scoring (pinned vaderSentiment)
- windows.py   point-in-time session windows W_t = (close_{t-1}, close_t] on created_at
- features.py  feature spec scout_v1 (sentiment-only; no price features)
- recipe.py    Scout recipes (Analyst Recipe validated against the Scout spec registry)
- dataset.py   labeled dataset (labels = analyst.dataset.make_labels)
- train.py     training core + reported-not-gated diagnostics + MLflow
- gating.py    alphagate trial gate vs the base rate (no champion yet)
- coverage.py  M1 news coverage / revision report (no labels)
- retrain.py   CLI: report | experiment | monitor
"""
