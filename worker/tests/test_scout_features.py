"""The Scout's point-in-time news windows and feature spec scout_v1.

The leak these tests exist for: a headline published after the close of
session t ("XYZ falls 6% after ...") must not reach the decision made at
close_t. Every window boundary case from the design (section 9) is pinned,
plus truncation invariance (features for t are identical whether or not
anything after close_t exists), an after-close canary, and exact parity
between a full-history build and the daily strategy's short lookback.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from scout_helpers import (
    NEG,
    POS,
    UNIVERSE,
    articles,
    bar_index,
    business_sessions,
    make_bars,
    make_sessions,
    ny,
    random_articles,
)
from wolfpack_worker.scout import features as F
from wolfpack_worker.scout.dataset import build_scout_dataset
from wolfpack_worker.scout.sentiment import NEUTRAL_BAND, score_headlines, vader_compound
from wolfpack_worker.scout.windows import assign_windows, bar_sessions

UTC = timezone.utc


# --- VADER --------------------------------------------------------------------------------


def test_vader_is_deterministic_and_signed():
    assert vader_compound(POS) > NEUTRAL_BAND
    assert vader_compound(NEG) < -NEUTRAL_BAND
    assert vader_compound("") == 0.0
    a = score_headlines([POS, NEG, "Apple to hold annual meeting"])
    b = score_headlines([POS, NEG, "Apple to hold annual meeting"])
    assert np.array_equal(a, b) and a.dtype == float


def test_vader_is_the_pinned_version():
    import importlib.metadata as md

    assert md.version("vaderSentiment") == "3.3.2"


# --- windows ---------------------------------------------------------------------------------


# 2026-03-06 (Fri, EST: close 21:00 UTC), weekend, Mon 2026-03-09 (EDT from Sun 03-08: close 20:00 UTC),
# Thu 2026-11-26 Thanksgiving holiday, Fri 2026-11-27 early close 13:00 ET.
WINTER_SUMMER = make_sessions([date(2026, 3, 5), date(2026, 3, 6), date(2026, 3, 9), date(2026, 3, 10)])
THANKSGIVING = make_sessions([date(2026, 11, 24), date(2026, 11, 25), date(2026, 11, 27), date(2026, 11, 30)],
                             early_close=[date(2026, 11, 27)])


def _w(sessions, *ts):
    return list(assign_windows(pd.DatetimeIndex(pd.to_datetime(list(ts), utc=True)), sessions))


def test_close_exactly_is_in_that_sessions_window_and_one_second_later_is_not():
    s = WINTER_SUMMER
    assert s[1].close == datetime(2026, 3, 6, 21, 0, tzinfo=UTC)  # EST close
    assert _w(s, s[1].close, s[1].close + timedelta(seconds=1)) == [1, 2]


def test_weekend_articles_roll_to_the_next_session():
    s = WINTER_SUMMER
    assert _w(s, datetime(2026, 3, 7, 15, tzinfo=UTC), datetime(2026, 3, 8, 23, tzinfo=UTC)) == [2, 2]


def test_dst_changes_the_utc_close():
    s = WINTER_SUMMER
    assert s[2].close == datetime(2026, 3, 9, 20, 0, tzinfo=UTC)  # EDT close: 20:00 UTC
    # 20:30 UTC on Monday 03-09 is AFTER that close (16:30 EDT) -> Tuesday's window ...
    assert _w(s, datetime(2026, 3, 9, 20, 30, tzinfo=UTC)) == [3]
    # ... while 20:30 UTC on Friday 03-06 is BEFORE that close (15:30 EST) -> Friday's window.
    assert _w(s, datetime(2026, 3, 6, 20, 30, tzinfo=UTC)) == [1]


def test_early_close_and_holiday():
    s = THANKSGIVING
    assert s[2].close == ny(date(2026, 11, 27), 13)  # 13:00 ET early close
    # 13:30 ET on the early-close day is after the close -> next session (Mon 11-30).
    assert _w(s, ny(date(2026, 11, 27), 13, 30)) == [3]
    # Thanksgiving Day (no session) rolls into Friday's window.
    assert _w(s, ny(date(2026, 11, 26), 12)) == [2]
    assert _w(s, ny(date(2026, 11, 27), 12, 59)) == [2]


def test_articles_outside_the_calendar_get_no_window():
    s = WINTER_SUMMER
    # At or before the first close: belongs to a window this calendar can't bound.
    assert _w(s, s[0].close, s[0].close - timedelta(hours=1), s[-1].close + timedelta(seconds=1)) == [-1, -1, -1]


def test_bar_timestamps_map_to_their_session():
    s = WINTER_SUMMER
    assert list(bar_sessions(bar_index(s), s)) == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="not a session"):
        bar_sessions(pd.DatetimeIndex([pd.Timestamp("2026-03-07T05:00:00Z")]), s)


def test_naive_inputs_are_refused():
    with pytest.raises(ValueError):
        assign_windows(pd.DatetimeIndex(["2026-03-06 15:00"]), WINTER_SUMMER)


# --- feature formulas --------------------------------------------------------------------------


SESS = business_sessions("2026-01-05", 40)


def _feats(arts, sessions=SESS, universe=UNIVERSE):
    return F.build_scout_features(arts, sessions, universe)


def _at(sessions, i, hh=15):
    """A time inside window i: 15:00 ET on session i's date."""
    return ny(sessions[i].date, hh)


def test_spec_registry_and_names():
    spec = F.get_scout_feature_spec("scout_v1")
    assert spec.names == ("s_mean_1", "has_news_1", "n_surprise", "s_mean_5", "s_change", "mkt_s_mean_1")
    assert spec.warmup_sessions == 25
    with pytest.raises(KeyError):
        F.get_scout_feature_spec("v1")


def test_hand_computed_features():
    s = SESS
    t = 30
    rows = [
        # window t: two AAPL headlines (0.5, -0.1); one shared AAPL+SPY (0.3)
        (1, _at(s, t), ("AAPL",), None, 0.5),
        (2, _at(s, t, 10), ("AAPL",), None, -0.1),
        (3, _at(s, t, 11), ("AAPL", "SPY"), None, 0.3),
        # window t-2: one AAPL headline 0.9
        (4, _at(s, t - 2), ("AAPL",), None, 0.9),
        # window t-10 (in the s_change base window t-24..t-5): AAPL -0.4, -0.2
        (5, _at(s, t - 10), ("AAPL",), None, -0.4),
        (6, _at(s, t - 10, 12), ("AAPL", "MSFT"), None, -0.2),
        # window t-30 is outside every lookback of t
        (7, _at(s, 0) - timedelta(days=3), ("AAPL",), None, 1.0),
        # non-universe article: ignored everywhere
        (8, _at(s, t), ("MSFT",), None, -1.0),
    ]
    f = _feats(articles(rows))
    aapl = f["AAPL"].loc[s[t].date]
    assert aapl["s_mean_1"] == pytest.approx((0.5 - 0.1 + 0.3) / 3)
    assert aapl["has_news_1"] == 1.0
    # n_t = 3; prior 20 windows (t-20..t-1) hold counts: t-2 -> 1, t-10 -> 2, rest 0
    expected_base = (np.log1p(1) + np.log1p(2)) / 20
    assert aapl["n_surprise"] == pytest.approx(np.log1p(3) - expected_base)
    # s_mean_5 over windows t-4..t: scores 0.5,-0.1,0.3,0.9 (count-weighted mean)
    s5 = (0.5 - 0.1 + 0.3 + 0.9) / 4
    assert aapl["s_mean_5"] == pytest.approx(s5)
    # base window t-24..t-5: -0.4, -0.2
    assert aapl["s_change"] == pytest.approx(s5 - (-0.3))
    # market: deduplicated headlines of all universe tickers in W_t: ids 1,2,3 (3 counted once)
    assert aapl["mkt_s_mean_1"] == pytest.approx((0.5 - 0.1 + 0.3) / 3)
    spy = f["SPY"].loc[s[t].date]
    assert spy["s_mean_1"] == pytest.approx(0.3) and spy["mkt_s_mean_1"] == aapl["mkt_s_mean_1"]
    # No news at all for XOM: zeros, not NaN, once warmed up.
    xom = f["XOM"].loc[s[t].date]
    assert xom["s_mean_1"] == 0.0 and xom["has_news_1"] == 0.0 and xom["s_mean_5"] == 0.0
    assert xom["n_surprise"] == 0.0 and xom["s_change"] == 0.0


def test_warmup_rows_are_nan():
    f = _feats(random_articles(SESS, seed=1))
    df = f["AAPL"]
    assert df.iloc[:25].isna().all().all()  # sessions 0..24
    assert not df.iloc[25:].isna().any().any()


def test_no_raw_count_level_is_a_feature():
    """Discussion volume enters only as a change (n_surprise), never as a raw
    count (CLAUDE.md: changes, not raw levels)."""
    s = SESS
    t = 30
    base = [(i, _at(s, k), ("AAPL",), None, 0.2) for i, k in enumerate(range(5, t + 1), start=1)]
    doubled = base + [(1000 + i, _at(s, k, 11), ("AAPL",), None, 0.2) for i, k in enumerate(range(5, t + 1))]
    a = _feats(articles(base))["AAPL"].loc[s[t].date]
    b = _feats(articles(doubled))["AAPL"].loc[s[t].date]
    # Twice the coverage every day, same tone: no feature changes except via log1p shape of n_surprise.
    for name in ("s_mean_1", "has_news_1", "s_mean_5", "s_change", "mkt_s_mean_1"):
        assert a[name] == pytest.approx(b[name])
    assert abs(b["n_surprise"]) < 1e-12 and abs(a["n_surprise"]) < 1e-12  # steady flow -> no surprise


# --- point-in-time properties ---------------------------------------------------------------------


def _truncated(arts, sessions, t):
    close = sessions[t].close
    return arts.loc[arts["created_at"] <= close].reset_index(drop=True), sessions[: t + 1]


def test_truncation_invariance_property():
    """Features for session t computed from everything (including later
    news and later sessions) == features computed from only what existed at
    close_t. Exact equality, over many t and random headline layouts."""
    s = business_sessions("2025-10-01", 70)
    for seed in range(4):
        arts = random_articles(s, seed=seed, per_session=4.0)
        full = _feats(arts, s)
        for t in range(25, 70, 6):
            a_t, s_t = _truncated(arts, s, t)
            part = _feats(a_t, s_t)
            for tk in UNIVERSE:
                pd.testing.assert_series_equal(full[tk].loc[s[t].date], part[tk].loc[s[t].date], check_exact=True)


def test_after_close_canary():
    """A strongly positive headline one second after close_t must not move
    any feature at t, and must move t+1."""
    s = SESS
    t = 30
    arts = random_articles(s, seed=3)
    canary = articles([(999_999, s[t].close + timedelta(seconds=1), UNIVERSE, POS, 1.0)])
    with_canary = pd.concat([arts, canary], ignore_index=True).sort_values(["created_at", "id"]).reset_index(drop=True)
    before, after = _feats(arts), _feats(with_canary)
    for tk in UNIVERSE:
        pd.testing.assert_series_equal(before[tk].loc[s[t].date], after[tk].loc[s[t].date], check_exact=True)
        assert after[tk].loc[s[t + 1].date, "s_mean_1"] != before[tk].loc[s[t + 1].date, "s_mean_1"]


def test_rolling_features_ignore_news_older_than_their_windows():
    s = SESS
    t = 35
    arts = random_articles(s, seed=5)
    cutoff_old = s[t - 25].close  # window t-25 and earlier are outside every lookback of t
    shifted = arts.copy()
    old = shifted["created_at"] <= cutoff_old
    shifted.loc[old, "score"] = -shifted.loc[old, "score"]
    a, b = _feats(arts), _feats(shifted)
    for tk in UNIVERSE:
        pd.testing.assert_series_equal(a[tk].loc[s[t].date], b[tk].loc[s[t].date], check_exact=True)


def test_short_lookback_parity_with_full_history():
    """The daily strategy computes features from only the last 26 sessions;
    they must be bit-identical to the full-history (training) values."""
    s = business_sessions("2025-06-02", 90)
    arts = random_articles(s, seed=11, per_session=6.0)
    full = _feats(arts, s)
    t = 89
    window_sessions = s[t - F.WARMUP_SESSIONS: t + 1]
    lo = window_sessions[0].close
    short_arts = arts.loc[(arts["created_at"] > lo) & (arts["created_at"] <= s[t].close)].reset_index(drop=True)
    short = _feats(short_arts, window_sessions)
    for tk in UNIVERSE:
        pd.testing.assert_series_equal(full[tk].loc[s[t].date], short[tk].loc[s[t].date], check_exact=True)
    assert F.lookback_sessions_needed() == F.WARMUP_SESSIONS == len(window_sessions) - 1


def test_scores_computed_from_headlines_when_absent():
    s = SESS
    t = 30
    rows = [(1, _at(s, t), ("AAPL",), POS), (2, _at(s, t, 11), ("AAPL",), NEG)]
    f = _feats(articles(rows))
    assert f["AAPL"].loc[s[t].date, "s_mean_1"] == pytest.approx((vader_compound(POS) + vader_compound(NEG)) / 2)


def test_build_refuses_unsorted_or_naive_sessions():
    with pytest.raises(ValueError):
        _feats(articles([]), tuple(reversed(SESS)))


# --- dataset ----------------------------------------------------------------------------------------


def test_dataset_labels_come_from_the_analyst_make_labels(monkeypatch):
    import wolfpack_worker.scout.dataset as D
    from wolfpack_worker.analyst import dataset as analyst_dataset

    assert D.make_labels is analyst_dataset.make_labels
    s = business_sessions("2025-06-02", 60)
    bars = make_bars(s)
    ds = build_scout_dataset(bars, random_articles(s, seed=2), s)
    lab = analyst_dataset.make_labels(bars["AAPL"])
    sub = ds.loc[ds["ticker"] == "AAPL"].set_index("ts")
    assert (sub["y"] == lab.loc[sub.index, "y"]).all()
    assert (sub["label_end_ts"] == lab.loc[sub.index, "label_end_ts"]).all()
    # chronological, (ts, ticker) sorted, warmup rows dropped, last 2 bars unlabeled
    assert ds["ts"].is_monotonic_increasing
    assert ds["ts"].min() == bar_index(s)[25]
    assert ds["ts"].max() == bar_index(s)[-3]
    assert set(F.SCOUT_FEATURE_NAMES) <= set(ds.columns)


def test_dataset_rows_use_only_news_up_to_their_own_close():
    """Dataset-level canary: a headline 1s after close_t changes no row at t."""
    s = business_sessions("2025-06-02", 60)
    bars = make_bars(s)
    arts = random_articles(s, seed=4)
    t = 40
    canary = articles([(999_999, s[t].close + timedelta(seconds=1), UNIVERSE, POS, 1.0)])
    a = build_scout_dataset(bars, arts, s)
    b = build_scout_dataset(bars, pd.concat([arts, canary], ignore_index=True), s)
    ts = bar_index(s)[t]
    pd.testing.assert_frame_equal(a.loc[a["ts"] == ts].reset_index(drop=True),
                                  b.loc[b["ts"] == ts].reset_index(drop=True))


# --- recipe -----------------------------------------------------------------------------------------


def test_scout_recipe_is_analyst_v1_with_the_scout_spec():
    from wolfpack_worker.analyst.recipe import Recipe, RecipeError, load_v1_recipe
    from wolfpack_worker.scout.paths import SCOUT_PATHS
    from wolfpack_worker.scout.recipe import parse_scout_recipe, scout_v1_recipe_dict

    d = scout_v1_recipe_dict()
    r = parse_scout_recipe(d)
    v1 = load_v1_recipe()
    assert r.feature_spec_version == "scout_v1"
    assert dict(r.xgb_params) == dict(v1.xgb_params) and r.num_boost_round == v1.num_boost_round
    assert r.label == v1.label and r.universe == v1.universe and r.train_since == v1.train_since
    assert SCOUT_PATHS.recipe_parser(d) == r
    # Registries are separate: neither persona can register the other's spec.
    with pytest.raises(RecipeError):
        Recipe.from_dict(d)
    with pytest.raises(RecipeError):
        parse_scout_recipe(v1.to_dict())
    # The Analyst's v1 recipe id is unchanged by the refactor.
    assert v1.recipe_id == "a8e2709b0d8e"
    assert SCOUT_PATHS.feature_spec_lookup("scout_v1").names == F.SCOUT_FEATURE_NAMES
