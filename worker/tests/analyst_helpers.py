"""Shared synthetic-data helpers for The Analyst's tests.

Deterministic (seeded) OHLCV generators — no network, no DB. Prices follow a
geometric random walk so every log-return-based feature is well defined.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

UNIVERSE = ("SPY", "QQQ", "AAPL", "JPM", "XOM")


def make_ohlcv(
    n: int,
    *,
    seed: int = 0,
    start: str = "2016-01-04",
    start_price: float = 100.0,
    daily_vol: float = 0.01,
    index: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if index is None:
        # Alpaca daily bars are stamped at midnight America/New_York, which
        # is 04:00/05:00 UTC; the exact hour doesn't matter here, only that
        # the index is tz-aware and strictly increasing.
        index = pd.bdate_range(start=start, periods=n, tz="UTC") + pd.Timedelta(hours=5)
    r = rng.normal(0.0002, daily_vol, size=n)
    close = start_price * np.exp(np.cumsum(r))
    # Open = previous close nudged by a small gap, so open-to-open returns
    # (the label) differ from close-to-close returns (the features).
    gap = rng.normal(0.0, daily_vol / 4, size=n)
    open_ = np.concatenate([[start_price], close[:-1]]) * np.exp(gap)
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, daily_vol / 2, size=n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, daily_vol / 2, size=n)))
    volume = rng.integers(1_000_000, 5_000_000, size=n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


def make_universe_bars(n: int, *, seed: int = 0, start: str = "2016-01-04") -> dict[str, pd.DataFrame]:
    index = pd.bdate_range(start=start, periods=n, tz="UTC") + pd.Timedelta(hours=5)
    return {
        ticker: make_ohlcv(n, seed=seed + i, index=index, start_price=50.0 + 40 * i)
        for i, ticker in enumerate(UNIVERSE)
    }


def gate_record(
    model_version: str,
    model_sha256: str,
    *,
    record_id: str = "rec-0001",
    decision: str = "promote",
    chronology_checked: bool = True,
    kind: str = "bootstrap",
    comparator_name: str = "margin",
    context: dict | None = None,
    challenger_score: dict | None = None,
) -> dict:
    """An alphagate GateRecord dict as one gate_log.jsonl line deserializes,
    with the fields the backstop checks. Tests use this instead of calling
    alphagate so the daily-path backstop is tested without alphagate."""
    return {
        "schema_version": 1,
        "record_id": record_id,
        "decided_at": "2026-10-01T12:00:00+00:00",
        "decision": decision,
        "reason_code": "no_incumbent" if decision == "promote" else "not_significant",
        "challenger_id": model_version,
        "chronology_checked": chronology_checked,
        "comparator_name": comparator_name,
        "challenger_score": challenger_score if challenger_score is not None else {
            "value": 0.69, "samples": [0.69, 0.69], "details": {"baseline_logloss": 0.69}},
        "challenger_metadata": {"model_sha256": model_sha256},
        "context": {"kind": kind, **(context or {})},
    }


def recipe_toml(recipe_dict: dict) -> str:
    """A [recipe] table (TOML) from a recipe dict (scalars + lists only)."""
    import json

    lines = ["[recipe]"]
    for k, v in recipe_dict.items():
        if k != "xgb_params":
            lines.append(f"{k} = {json.dumps(v)}")
    lines.append("[recipe.xgb_params]")
    for k, v in recipe_dict["xgb_params"].items():
        lines.append(f"{k} = {json.dumps(v)}")
    return "\n".join(lines) + "\n"


def write_registration(experiments_dir, n: int, recipe_dict: dict, *, hypothesis: str = "h",
                       slug: str = "x", abandoned: str | None = None):
    from pathlib import Path

    d = Path(experiments_dir)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{n:03d}-{slug}.toml"
    head = f'trial_number = {n}\nregistered = 2026-10-02\nhypothesis = "{hypothesis}"\n'
    if abandoned:
        head += f'abandoned = "{abandoned}"\n'
    p.write_text(head + "\n" + recipe_toml(recipe_dict))
    return p


def append_jsonl(path, record: dict) -> None:
    import json
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def write_gated_champion(champion_dir, model_bytes: bytes, manifest: dict, *, record_id: str = "rec-0001") -> dict:
    """Write a champion AND the PROMOTE record load_champion requires, in the
    layout the real code uses: <root>/champion/ + <root>/gate_log.jsonl."""
    from pathlib import Path

    from wolfpack_worker.analyst.model_io import sha256_bytes, write_champion

    champion_dir = Path(champion_dir)
    manifest = {**manifest, "gate_record_ids": [record_id]}
    out = write_champion(champion_dir, model_bytes, manifest)
    append_jsonl(
        champion_dir.parent / "gate_log.jsonl",
        gate_record(manifest["model_version"], sha256_bytes(model_bytes), record_id=record_id),
    )
    return out


def make_predictable_universe_bars(
    n: int, *, seed: int = 0, phi: float = 0.35, start: str = "2016-01-04"
) -> dict[str, pd.DataFrame]:
    """Synthetic bars WITH learnable signal, for exercising the promotion path.

    Each ticker's close-to-close log return is AR(1): r_t = phi*r_{t-1} + e_t,
    and each open equals the previous close. Then the label
    y_t = 1[ln(O_{t+2}/O_{t+1}) > 0] = 1[r_{t+1} > 0] = 1[phi*r_t + e > 0],
    which the r_1 feature (= r_t) predicts. Real market data has nothing like
    this; it exists only so a test can watch a genuinely better model pass
    the gate (and a no-better one fail it).
    """
    index = pd.bdate_range(start=start, periods=n, tz="UTC") + pd.Timedelta(hours=5)
    out = {}
    for i, ticker in enumerate(UNIVERSE):
        rng = np.random.default_rng(seed + 101 * i)
        e = rng.normal(0.0, 0.01, size=n)
        r = np.empty(n)
        r[0] = e[0]
        for t in range(1, n):
            r[t] = phi * r[t - 1] + e[t]
        close = (50.0 + 40 * i) * np.exp(np.cumsum(r))
        open_ = np.concatenate([[close[0] / np.exp(r[0])], close[:-1]])
        high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.003, size=n)))
        low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.003, size=n)))
        volume = rng.integers(1_000_000, 5_000_000, size=n).astype(float)
        out[ticker] = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=index
        )
    return out


def weak_recipe_dict(seed: int = 7) -> dict:
    """A recipe that can barely learn anything (1 tiny boosting round):
    predictions sit at ~the base rate. Used as a stand-in 'champion' recipe."""
    from wolfpack_worker.analyst.recipe import load_v1_recipe

    d = load_v1_recipe().to_dict()
    d["num_boost_round"] = 1
    d["xgb_params"] = {**d["xgb_params"], "learning_rate": 0.01, "max_depth": 1, "seed": seed}
    return d
