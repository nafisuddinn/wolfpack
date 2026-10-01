"""The Analyst's feature engineering — one function for training AND inference.

Every feature is a log return, a log ratio, or a scale-free statistic of log
returns / log prices (CLAUDE.md: log returns, not raw price). No raw price
level is ever a feature: multiplying every OHLC price by a constant leaves
every feature unchanged (tested).

Lookahead discipline: the feature row for bar t only uses bars <= t. The
daily run happens after the close of bar t, so bar t itself IS available;
the *label* (dataset.py) is what must be strictly after t.

Train/serve parity: every rolling statistic is computed per-window from a
fixed-length slice (numpy sliding windows), not from running sums or an
unbounded recursion. That makes a bar's features bit-identical whether they
were computed from ten years of history (training) or from the daily
strategy's 80-bar lookback (inference) — tested exactly, not approximately.

With C/H/L/V = close/high/low/volume and r_t = ln(C_t / C_{t-1}):

 1. r_1              r_t
 2. r_5              ln(C_t / C_{t-5})
 3. r_20             ln(C_t / C_{t-20})
 4. vol_20           population std (ddof=0) of r over the last 20 bars
 5. vol_ratio_5_20   vol_5 / vol_20   (NaN if vol_20 ~= 0, i.e. <= 1e-12)
 6. ma_spread_20_50  ln(SMA20(C) / SMA50(C))
 7. z_20_logp        (ln C_t - mean20(ln C)) / std20(ln C)  (ddof=0; NaN if std <= 1e-12)
 8. rsi_14           Wilder RSI(14) on r_t (not on raw price differences),
                     over a FIXED window of the last 49 returns: seeded with
                     the simple mean of the window's first 14 gains/losses,
                     then Wilder-smoothed across the remaining 35. Scale 0-100;
                     50 if the window has no non-zero returns.
 9. hl_range         ln(H_t / L_t)
10. logvol_ratio_20  ln(V_t / mean20(V))  (NaN if V_t <= 0)
11. spy_r_1          SPY's r_1 on the same bar timestamp (NaN if SPY has no bar)
12. spy_r_5          SPY's r_5 on the same bar timestamp (NaN if SPY has no bar)

Warmup: 50 bars (SMA50 and the 49-return RSI window). Rows before that are
NaN and are dropped in training / cause a "hold" at inference.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

FEATURE_SPEC_VERSION = "v1"

OWN_FEATURE_NAMES: tuple[str, ...] = (
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
)
MARKET_FEATURE_NAMES: tuple[str, ...] = ("spy_r_1", "spy_r_5")
FEATURE_NAMES: tuple[str, ...] = OWN_FEATURE_NAMES + MARKET_FEATURE_NAMES

MARKET_TICKER = "SPY"
WARMUP_BARS = 50
RSI_PERIOD = 14
# Fixed RSI window length in returns. 49 returns need 50 bars, so the RSI
# fits exactly inside the SMA50 warmup and adds no extra history
# requirement. A fixed window (rather than Wilder's usual recursion from the
# first bar ever seen) is what keeps training and inference identical.
RSI_WINDOW_RETURNS = WARMUP_BARS - 1


def _rolling(values: np.ndarray, window: int, fn) -> np.ndarray:
    """Apply `fn(windows, axis=-1)` to each trailing window of `values`.

    out[t] = fn(values[t-window+1 : t+1]); NaN for t < window-1. Each window
    is reduced independently, so out[t] never depends on anything before
    t-window+1 or after t (no running sums carrying float error or
    information across windows).
    """
    out = np.full(values.shape[0], np.nan)
    if values.shape[0] >= window:
        windows = sliding_window_view(values, window)
        out[window - 1 :] = fn(windows, axis=-1)
    return out


# Denominators that are standard deviations are treated as zero below this.
# A window of identical values can produce a std of ~1e-16 (float rounding
# in the mean), which would otherwise turn "undefined" into a meaningless
# ratio like 1.0. Real 20-bar log-price/return stds are >= ~1e-4.
_STD_EPS = 1e-12


def _safe_div(num: np.ndarray, den: np.ndarray, den_eps: float = 0.0) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    out[~np.isfinite(out)] = np.nan
    out[np.abs(den) <= den_eps] = np.nan
    return out


def _safe_log(x: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(x)
    out[~np.isfinite(out)] = np.nan
    return out


def _rsi_weights() -> np.ndarray:
    """Linear-filter weights equivalent to the seeded Wilder recursion.

    Seed avg = mean(x_0..x_13); then 35 steps of avg <- (13*avg + x)/14.
    Unrolled: avg = a^35 * seed + sum_{k=0}^{34} (1/14) a^k * x_{48-k},
    with a = 13/14. The test suite checks this against the literal loop.
    """
    n, w = RSI_PERIOD, RSI_WINDOW_RETURNS
    a = (n - 1) / n
    steps = w - n
    weights = np.empty(w)
    weights[:n] = (a**steps) / n
    ages = np.arange(steps - 1, -1, -1)  # position n..w-1 has age steps-1..0
    weights[n:] = (1.0 / n) * a**ages
    return weights


_RSI_W = _rsi_weights()


def _fixed_filter(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """out[t] = sum_j weights[j] * x[t - len(weights) + 1 + j], NaN for t < w-1.

    Accumulated term by term in a fixed order (oldest first) with elementwise
    ops only — deliberately NOT a BLAS matmul, whose reduction order can vary
    with array length/alignment and would break bit-exact train/serve parity.
    """
    w = weights.shape[0]
    n = x.shape[0]
    out = np.full(n, np.nan)
    if n < w:
        return out
    acc = np.zeros(n - w + 1)
    for j in range(w):
        acc = acc + weights[j] * x[j : n - w + 1 + j]
    out[w - 1 :] = acc
    return out


def _rsi(r: np.ndarray) -> np.ndarray:
    gains = np.where(np.isnan(r), np.nan, np.maximum(r, 0.0))
    losses = np.where(np.isnan(r), np.nan, np.maximum(-r, 0.0))
    ag = _fixed_filter(gains, _RSI_W)
    al = _fixed_filter(losses, _RSI_W)
    total = ag + al
    with np.errstate(divide="ignore", invalid="ignore"):
        rsi = 100.0 * ag / total
    rsi = np.where(total == 0, 50.0, rsi)
    rsi[np.isnan(total)] = np.nan
    return rsi


def ticker_features(df: pd.DataFrame) -> pd.DataFrame:
    """The 10 own-ticker features for every bar of one ticker's OHLCV frame.

    `df` must have an ascending index and open/high/low/close/volume columns.
    Returns a frame on the same index with OWN_FEATURE_NAMES columns.
    """
    c = df["close"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    lo = df["low"].to_numpy(dtype=float)
    v = df["volume"].to_numpy(dtype=float)
    n = c.shape[0]

    logc = _safe_log(c)
    r = np.full(n, np.nan)
    if n > 1:
        r[1:] = logc[1:] - logc[:-1]

    def lag_diff(k: int) -> np.ndarray:
        out = np.full(n, np.nan)
        if n > k:
            out[k:] = logc[k:] - logc[:-k]
        return out

    vol_5 = _rolling(r, 5, np.std)
    vol_20 = _rolling(r, 20, np.std)
    sma_20 = _rolling(c, 20, np.mean)
    sma_50 = _rolling(c, 50, np.mean)
    mean_logc_20 = _rolling(logc, 20, np.mean)
    std_logc_20 = _rolling(logc, 20, np.std)
    mean_v_20 = _rolling(v, 20, np.mean)

    feats = {
        "r_1": r,
        "r_5": lag_diff(5),
        "r_20": lag_diff(20),
        "vol_20": vol_20,
        "vol_ratio_5_20": _safe_div(vol_5, vol_20, _STD_EPS),
        "ma_spread_20_50": _safe_log(_safe_div(sma_20, sma_50)),
        "z_20_logp": _safe_div(logc - mean_logc_20, std_logc_20, _STD_EPS),
        "rsi_14": _rsi(r),
        "hl_range": _safe_log(_safe_div(h, lo)),
        "logvol_ratio_20": _safe_log(_safe_div(v, mean_v_20)),
    }
    return pd.DataFrame(feats, index=df.index, columns=list(OWN_FEATURE_NAMES))


def build_features(
    bars: Mapping[str, pd.DataFrame], market_ticker: str = MARKET_TICKER
) -> dict[str, pd.DataFrame]:
    """All 12 features (FEATURE_NAMES order) for every ticker in `bars`.

    The market features (spy_r_1/spy_r_5) are the market ticker's own r_1/r_5
    joined on the exact same bar timestamp — never forward-filled, so a
    missing market bar yields NaN (-> row dropped in training, ticker held at
    inference) rather than silently reusing a stale value.
    """
    market = bars.get(market_ticker)
    if market is not None and len(market) > 0:
        market_own = ticker_features(market)
        market_feats = market_own[["r_1", "r_5"]].rename(
            columns={"r_1": "spy_r_1", "r_5": "spy_r_5"}
        )
    else:
        market_feats = None

    out: dict[str, pd.DataFrame] = {}
    for ticker, df in bars.items():
        if df is None or len(df) == 0:
            continue
        own = ticker_features(df)
        if market_feats is not None:
            joined = market_feats.reindex(own.index)
        else:
            joined = pd.DataFrame(np.nan, index=own.index, columns=list(MARKET_FEATURE_NAMES))
        out[ticker] = pd.concat([own, joined], axis=1)[list(FEATURE_NAMES)]
    return out
