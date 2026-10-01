"""Unit tests for The Analyst's feature engineering (analyst/features.py).

Each of the 12 features is checked against an independent, deliberately
naive pure-Python reference implementation (loops over plain lists) rather
than against a re-statement of the vectorized code — a bug in the
vectorized code shouldn't be able to hide by being copied into the test.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from analyst_helpers import make_ohlcv, make_universe_bars
from wolfpack_worker.analyst.features import (
    FEATURE_NAMES,
    FEATURE_SPEC_VERSION,
    RSI_PERIOD,
    RSI_WINDOW_RETURNS,
    WARMUP_BARS,
    build_features,
    ticker_features,
)


# ---------------------------------------------------------------------------
# Naive reference implementations (plain Python, no numpy/pandas rolling)
# ---------------------------------------------------------------------------


def _mean(xs):
    return sum(xs) / len(xs)


def _pstd(xs):
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _ref_rsi(r_window):
    """Wilder RSI over exactly RSI_WINDOW_RETURNS log returns: seed with the
    simple mean of the first RSI_PERIOD gains/losses, then Wilder-smooth
    (avg = (avg*(n-1) + x)/n) across the rest of the window."""
    assert len(r_window) == RSI_WINDOW_RETURNS
    gains = [max(x, 0.0) for x in r_window]
    losses = [max(-x, 0.0) for x in r_window]
    ag = _mean(gains[:RSI_PERIOD])
    al = _mean(losses[:RSI_PERIOD])
    for g, l in zip(gains[RSI_PERIOD:], losses[RSI_PERIOD:]):
        ag = (ag * (RSI_PERIOD - 1) + g) / RSI_PERIOD
        al = (al * (RSI_PERIOD - 1) + l) / RSI_PERIOD
    if ag + al == 0:
        return 50.0
    return 100.0 * ag / (ag + al)


def _ref_own_features(df: pd.DataFrame, t: int) -> dict[str, float]:
    c = [float(x) for x in df["close"]]
    h = [float(x) for x in df["high"]]
    l = [float(x) for x in df["low"]]
    v = [float(x) for x in df["volume"]]
    lc = [math.log(x) for x in c]
    r = [float("nan")] + [lc[i] - lc[i - 1] for i in range(1, len(c))]
    vol5 = _pstd(r[t - 4 : t + 1])
    vol20 = _pstd(r[t - 19 : t + 1])
    return {
        "r_1": r[t],
        "r_5": math.log(c[t] / c[t - 5]),
        "r_20": math.log(c[t] / c[t - 20]),
        "vol_20": vol20,
        "vol_ratio_5_20": vol5 / vol20,
        "ma_spread_20_50": math.log(_mean(c[t - 19 : t + 1]) / _mean(c[t - 49 : t + 1])),
        "z_20_logp": (lc[t] - _mean(lc[t - 19 : t + 1])) / _pstd(lc[t - 19 : t + 1]),
        "rsi_14": _ref_rsi(r[t - RSI_WINDOW_RETURNS + 1 : t + 1]),
        "hl_range": math.log(h[t] / l[t]),
        "logvol_ratio_20": math.log(v[t] / _mean(v[t - 19 : t + 1])),
    }


# ---------------------------------------------------------------------------
# Spec constants
# ---------------------------------------------------------------------------


def test_feature_spec_is_the_twelve_designed_features_in_order():
    assert FEATURE_SPEC_VERSION == "v1"
    assert FEATURE_NAMES == (
        "r_1",
        "r_5",
        "r_20",
        "vol_20",
        "vol_ratio_5_20",
        "ma_spread_20_50",
        "z_20_logp",
        "rsi_14",
        "hl_range",
        "logvol_ratio_20",
        "spy_r_1",
        "spy_r_5",
    )
    assert WARMUP_BARS == 50
    # The RSI's fixed window must fit inside the warmup (50 bars -> 49 returns)
    assert RSI_WINDOW_RETURNS == WARMUP_BARS - 1


def test_no_feature_is_a_raw_price_level():
    """CLAUDE.md: log returns, not raw price. Scaling every price column by
    a constant (a pure price-level change) must leave every feature
    unchanged — a raw-price feature would scale with it."""
    df = make_ohlcv(120, seed=3)
    scaled = df.copy()
    for col in ("open", "high", "low", "close"):
        scaled[col] = scaled[col] * 37.5
    a = ticker_features(df)
    b = ticker_features(scaled)
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9, atol=1e-12)


# ---------------------------------------------------------------------------
# Known values, each of the 12 features
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("t", [49, 50, 77, 119])
def test_own_features_match_naive_reference(t):
    df = make_ohlcv(120, seed=11)
    feats = ticker_features(df)
    expected = _ref_own_features(df, t)
    for name, value in expected.items():
        assert feats[name].iloc[t] == pytest.approx(value, rel=1e-10, abs=1e-12), name


def test_hand_computed_values_on_a_simple_series():
    # Doubling on the last bar: r_1 = ln 2 exactly.
    n = 60
    close = [100.0] * (n - 1) + [200.0]
    idx = pd.bdate_range("2020-01-01", periods=n, tz="UTC")
    df = pd.DataFrame(
        {
            "open": close,
            "high": [c * 1.1 for c in close],
            "low": close,
            "close": close,
            "volume": [1000.0] * (n - 1) + [4000.0],
        },
        index=idx,
    )
    f = ticker_features(df).iloc[-1]
    assert f["r_1"] == pytest.approx(math.log(2))
    assert f["r_5"] == pytest.approx(math.log(2))
    assert f["r_20"] == pytest.approx(math.log(2))
    assert f["hl_range"] == pytest.approx(math.log(1.1))
    # mean20(V) = (19*1000 + 4000)/20 = 1150
    assert f["logvol_ratio_20"] == pytest.approx(math.log(4000 / 1150))
    # Only one non-zero return (a gain) in the whole RSI window: RSI = 100.
    assert f["rsi_14"] == pytest.approx(100.0)
    # SMA20 = (19*100+200)/20 = 105, SMA50 = (49*100+200)/50 = 102
    assert f["ma_spread_20_50"] == pytest.approx(math.log(105 / 102))


def test_flat_price_gives_nan_for_scale_free_ratios_not_inf():
    """Zero volatility makes vol_ratio/z undefined. They must be NaN (so the
    row is dropped / ticker held), never inf or a silently-wrong 0."""
    n = 60
    idx = pd.bdate_range("2020-01-01", periods=n, tz="UTC")
    df = pd.DataFrame(
        {"open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "volume": 5.0}, index=idx
    )
    f = ticker_features(df).iloc[-1]
    assert f["vol_20"] == 0.0
    assert math.isnan(f["vol_ratio_5_20"])
    assert math.isnan(f["z_20_logp"])
    assert f["rsi_14"] == 50.0
    assert not np.isinf(ticker_features(df).to_numpy()).any()


def test_zero_volume_bar_is_nan_not_minus_inf():
    df = make_ohlcv(60, seed=1)
    df.iloc[-1, df.columns.get_loc("volume")] = 0.0
    f = ticker_features(df).iloc[-1]
    assert math.isnan(f["logvol_ratio_20"])


def test_warmup_rows_are_nan_and_first_full_row_is_bar_50():
    df = make_ohlcv(80, seed=2)
    feats = ticker_features(df)
    assert feats.iloc[: WARMUP_BARS - 1].isna().any(axis=1).all()
    assert not feats.iloc[WARMUP_BARS - 1].isna().any()


def test_spy_features_join_on_same_date():
    bars = make_universe_bars(90, seed=5)
    out = build_features(bars)
    spy_own = ticker_features(bars["SPY"])
    for ticker in ("QQQ", "XOM"):
        pd.testing.assert_series_equal(
            out[ticker]["spy_r_1"], spy_own["r_1"], check_names=False
        )
        pd.testing.assert_series_equal(
            out[ticker]["spy_r_5"], spy_own["r_5"], check_names=False
        )
    assert list(out["AAPL"].columns) == list(FEATURE_NAMES)


def test_missing_spy_bar_makes_spy_features_nan_on_that_date_only():
    bars = make_universe_bars(90, seed=5)
    missing_ts = bars["SPY"].index[70]
    bars["SPY"] = bars["SPY"].drop(index=missing_ts)
    out = build_features(bars)
    row = out["AAPL"].loc[missing_ts]
    assert math.isnan(row["spy_r_1"]) and math.isnan(row["spy_r_5"])
    # Own features unaffected.
    assert not row[list(FEATURE_NAMES[:10])].isna().any()


def test_no_spy_at_all_means_all_spy_features_nan():
    bars = make_universe_bars(90, seed=5)
    del bars["SPY"]
    out = build_features(bars)
    assert out["QQQ"]["spy_r_1"].isna().all()


# ---------------------------------------------------------------------------
# Lookahead / train-serve parity
# ---------------------------------------------------------------------------


def test_truncation_features_at_t_identical_whether_or_not_future_bars_exist():
    """Core lookahead test: features for bar t computed from bars[:t+1] must
    be bit-identical to features for bar t computed from the full series
    (which has bars after t). Checked for every t, exact equality."""
    bars = make_universe_bars(140, seed=9)
    full = build_features(bars)
    for t in range(WARMUP_BARS - 5, 140):
        prefix = {k: v.iloc[: t + 1] for k, v in bars.items()}
        part = build_features(prefix)
        for ticker in bars:
            a = full[ticker].iloc[t]
            b = part[ticker].iloc[-1]
            pd.testing.assert_series_equal(a, b, check_exact=True, check_names=False)


def test_garbage_future_bars_cannot_change_past_features():
    bars = make_universe_bars(120, seed=4)
    cutoff = 80
    clean = build_features(bars)
    corrupted = {}
    for k, v in bars.items():
        v = v.copy()
        v.iloc[cutoff + 1 :] = 1e12
        corrupted[k] = v
    dirty = build_features(corrupted)
    for ticker in bars:
        pd.testing.assert_frame_equal(
            clean[ticker].iloc[: cutoff + 1], dirty[ticker].iloc[: cutoff + 1], check_exact=True
        )


def test_inference_on_short_lookback_matches_training_on_full_history_exactly():
    """Train/serve parity: the daily strategy only sees ~80 bars; training
    sees ~10 years. The latest bar's features must be bit-identical either
    way (this is why RSI uses a fixed window, not an unbounded recursion)."""
    bars = make_universe_bars(2600, seed=21)
    full = build_features(bars)
    for lookback in (WARMUP_BARS, 80, 90):
        short = build_features({k: v.iloc[-lookback:] for k, v in bars.items()})
        for ticker in bars:
            pd.testing.assert_series_equal(
                full[ticker].iloc[-1], short[ticker].iloc[-1], check_exact=True, check_names=False
            )
