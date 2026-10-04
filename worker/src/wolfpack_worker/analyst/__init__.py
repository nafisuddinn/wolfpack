"""The Analyst (Persona #3): XGBoost next-session direction classifier.

MODEL-RISK LIMITATION: this model is a process demonstration, not an alpha
claim. No persona has demonstrated meaningful real-world predictive edge on
public daily price data; nothing in this package should be read as implying
otherwise. See MODEL_CARD.md for measured holdout results vs. baselines.

Modules:
- features.py  — FEATURE_SPEC_VERSION'd feature builder (training AND inference)
- dataset.py   — labels, dataset assembly, chronological split/embargo,
                 walk-forward folds, split-artifact data guard
- metrics.py   — classification metrics + baselines, isolated long/flat backtest
- recipe.py    — Recipe (all model-determining settings) + recipe_id; v1 = worker/recipes/analyst/v1.toml
- model_io.py  — champion model.json/manifest.json read/write + integrity checks, including
                 the gate-log backstop (no PROMOTE record -> the champion does not load)
- gate_log.py  — plain-JSON reader of alphagate's gate_log.jsonl (no alphagate import)
- train.py     — training core; `python -m wolfpack_worker.analyst.train` reproduces the
                 v1 report only and never writes a champion
- gating.py    — alphagate promotion gate: bootstrap / refresh / trial; the only champion writer
- registration.py — experiment pre-registration + trial counting
- explore.py   — walk-forward strictly before the gate holdout (MLflow the-analyst-dev)
- forward.py   — forward monitoring of each champion since promotion
- render_history.py — MODEL_CARD.md gate-history table, generated from the logs
- paths.py     — PersonaPaths: per-persona governance locations + parsers (ANALYST_PATHS);
                 the governance modules above take `paths=` so The Scout reuses them
- retrain.py   — CLI: `python -m wolfpack_worker.analyst.retrain {refresh|experiment|explore|monitor|bootstrap}`
"""
