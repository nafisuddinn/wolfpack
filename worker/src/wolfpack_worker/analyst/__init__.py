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
- model_io.py  — champion model.json/manifest.json read/write + integrity checks
- train.py     — `python -m wolfpack_worker.analyst.train [--promote]`
"""
