"""Holdout metrics + required baselines, and the isolated long/flat backtest.

Baselines (reported next to every metric):
- baseline_accuracy: always predict the majority class of the TRAINING
  up-rate (i.e. "always up" if train up-rate > 0.5, else "always down").
- baseline_logloss: log loss of predicting the training up-rate as a
  constant probability for every test row.
The model only "beats baseline" if its log loss is lower than
baseline_logloss. Accuracy alone isn't enough: on a market that goes up
~53% of days, "always long" already scores ~53%.

Only numpy is needed except for AUC, which uses scikit-learn (train group).

The backtest here is deliberately labelled "isolated": equal-weight long/flat
per ticker on the label's open-to-open log return, with no volatility
targeting and no position sizing. The gross figure is pre-cost; the net
figure subtracts a flat illustrative cost per position change. It is NOT the
full-pipeline number (vol-targeted sizing + the project's transaction-cost
assumption + DSR are separate Week 3 backlog items) and must not be reported
as one. No Sharpe/DSR is computed here on purpose.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

THRESHOLD = 0.5
_EPS = 1e-15


def _logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, _EPS, 1 - _EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def predict_label(p: np.ndarray) -> np.ndarray:
    """Long/up iff p > THRESHOLD; exact ties are flat/down."""
    return (np.asarray(p) > THRESHOLD).astype(int)


def classification_metrics(y: np.ndarray, p: np.ndarray, train_up_rate: float) -> dict[str, Any]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if y.shape != p.shape or y.size == 0:
        raise ValueError("classification_metrics: y and p must be same non-zero length")
    if not 0.0 < train_up_rate < 1.0:
        raise ValueError(f"train_up_rate must be in (0,1), got {train_up_rate}")

    accuracy = float(np.mean(predict_label(p) == y))
    logloss = _logloss(y, p)
    brier = float(np.mean((p - y) ** 2))
    if len(np.unique(y)) == 2:
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(y, p))
    else:
        auc = float("nan")

    majority = 1.0 if train_up_rate > THRESHOLD else 0.0
    baseline_accuracy = float(np.mean(y == majority))
    baseline_logloss = _logloss(y, np.full_like(y, train_up_rate))

    return {
        "n": int(y.size),
        "test_up_rate": float(y.mean()),
        "train_up_rate": float(train_up_rate),
        "accuracy": accuracy,
        "auc": auc,
        "logloss": logloss,
        "brier": brier,
        "baseline_accuracy": baseline_accuracy,
        "baseline_logloss": baseline_logloss,
        "baseline_brier": float(np.mean((train_up_rate - y) ** 2)),
        "beats_baseline_logloss": bool(logloss < baseline_logloss),
    }


def long_flat_backtest(preds: pd.DataFrame, cost_bps_per_side: float) -> dict[str, Any]:
    """Isolated equal-weight long/flat backtest on the holdout.

    `preds` needs ts, ticker, p_up, fwd_logret (the label's open(t+1) ->
    open(t+2) log return, i.e. what the position decided at t earns). Per
    ticker: position_t = 1[p_up > 0.5]; gross = sum(position_t *
    fwd_logret_t); a "flip" is any change in position, starting from flat.
    Net subtracts cost_bps_per_side per flip (log-return approximation).
    """
    per: dict[str, dict[str, float]] = {}
    cost = cost_bps_per_side / 1e4
    for ticker, g in preds.sort_values("ts", kind="mergesort").groupby("ticker", sort=True):
        pos = predict_label(g["p_up"].to_numpy())
        ret = g["fwd_logret"].to_numpy(dtype=float)
        prev = np.concatenate([[0], pos[:-1]])
        flips = int(np.sum(pos != prev))
        gross = float(np.sum(pos * ret))
        per[ticker] = {
            "n": int(len(g)),
            "fraction_long": float(pos.mean()),
            "flips": flips,
            "gross_logret": gross,
            "net_logret": gross - flips * cost,
            "buy_hold_logret": float(np.sum(ret)),
        }
    keys = ("fraction_long", "flips", "gross_logret", "net_logret", "buy_hold_logret")
    out: dict[str, Any] = {
        "label": "isolated, pre-cost (gross); net uses a flat illustrative cost",
        "cost_bps_per_side": float(cost_bps_per_side),
        "per_ticker": per,
    }
    for k in keys:
        out[f"mean_{k}"] = float(np.mean([v[k] for v in per.values()]))
    return out
