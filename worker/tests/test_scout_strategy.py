"""Persona #4 The Scout: strategies/scout.py (gate-or-fallback, public payload).

* No champion -> rule_fallback (untuned VADER neutral band): > +0.05 long,
  < -0.05 flat, otherwise / no news -> hold (omitted).
* A gated champion -> model mode: long iff p_up > 0.5.
* Failed or stale news -> [] (and the reason is logged); other personas
  are unaffected.
* The public payload carries scores, counts, source names and article ids
  only: never headline text or URLs (Decision Log 2026-10-04).
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from analyst_helpers import append_jsonl, write_registration
from fakes import FakeBroker, InMemoryPriceStore, InMemoryTradeRepo
from scout_helpers import POS, SECRET, UNIVERSE, articles, bar_index, business_sessions, make_bars, ny, synthetic
from wolfpack_worker.analyst.model_io import ModelIntegrityError
from wolfpack_worker.daily_trades import run_persona
from wolfpack_worker.execution import FixedNotionalSizer
from wolfpack_worker.news import LocalNewsStore, NewsArticle, NewsRefreshResult
from wolfpack_worker.scout.dataset import build_scout_dataset
from wolfpack_worker.scout.features import WARMUP_SESSIONS
from wolfpack_worker.scout.paths import SCOUT_PATHS
from wolfpack_worker.strategies import REGISTRY
from wolfpack_worker.strategies.base import LookaheadError, NewsSnapshot, StrategyContext, truncate_bars
from wolfpack_worker.strategies.scout import LIMITATIONS, Scout
from wolfpack_worker.strategies.trend_follower import TrendFollower

SESS = business_sessions("2025-11-03", 60)
T = len(SESS) - 1
AS_OF = SESS[T].close


@pytest.fixture()
def paths(tmp_path):
    return replace(SCOUT_PATHS, models_dir=tmp_path / "models" / "scout",
                   experiments_dir=tmp_path / "experiments" / "scout", model_card_path=tmp_path / "CARD.md")


def _snapshot(arts, sessions=SESS, t=T, cutoff_delay=timedelta(minutes=30)):
    window = sessions[t - WARMUP_SESSIONS: t + 1]
    close = sessions[t].close
    sub = arts.loc[(arts["created_at"] > window[0].close) & (arts["created_at"] <= close)].reset_index(drop=True)
    sub["first_seen_at"] = sub["created_at"]
    return NewsSnapshot(articles=sub, sessions=tuple(window), cutoff=close + cutoff_delay, status="ok")


def _ctx(arts, bars=None, sessions=SESS, t=T):
    bars = bars or make_bars(sessions)
    return StrategyContext(as_of=sessions[t].close, universe=UNIVERSE, bars=truncate_bars(bars, sessions[t].close),
                           news=_snapshot(arts, sessions, t))


def _background():
    """One neutral headline for every ticker on most sessions (score 0)."""
    rows, aid = [], 1
    for i, s in enumerate(SESS[:-1]):
        for tk in UNIVERSE:
            rows.append((aid, ny(s.date, 11), (tk,), "neutral", 0.0))
            aid += 1
    return rows


def _today(rows):
    d = SESS[T].date
    return articles(_background() + [(100_000 + i, ny(d, 10 + i), syms, f"{SECRET} {i} https://example.invalid/x",
                                      score) for i, (syms, score) in enumerate(rows)])


def test_registry_has_the_scout():
    assert REGISTRY["the-scout"] is Scout
    s = Scout()
    assert s.slug == "the-scout" and s.requires_news and s.news_lookback_sessions == WARMUP_SESSIONS


def test_rule_fallback_long_flat_and_hold(paths):
    arts = _today([(("AAPL",), 0.6), (("AAPL",), 0.2),   # AAPL mean 0.4 -> long
                   (("XOM",), -0.7),                      # XOM -0.7 -> flat
                   (("JPM",), 0.04)])                     # JPM inside the band -> hold
    targets = {t.ticker: t for t in Scout(paths=paths).evaluate(_ctx(arts))}
    assert targets["AAPL"].target_exposure == 1.0 and targets["XOM"].target_exposure == 0.0
    assert "JPM" not in targets and "SPY" not in targets  # SPY: no news today -> hold
    p = targets["AAPL"].payload
    assert p["mode"] == "rule_fallback" and p["model_rejected"] is False and p["gate_status"] == "not_yet_gated"
    assert p["rule_value"] == pytest.approx(0.4) and p["rule_band"] == 0.05 and p["p_up"] is None
    assert p["n_articles"] == 2 and p["stocktwits"] == "deferred" and p["limitations"] == LIMITATIONS
    assert [a["score"] for a in p["top_articles"]] == [0.6, 0.2]
    assert set(p["top_articles"][0]) == {"id", "score", "source"}
    assert p["news_cutoff"] == (AS_OF + timedelta(minutes=30)).isoformat()
    assert p["features"]["has_news_1"] == 1.0 and set(p["features"]) == {
        "s_mean_1", "has_news_1", "n_surprise", "s_mean_5", "s_change", "mkt_s_mean_1"}
    assert targets["AAPL"].signal_ts == bar_index(SESS)[T]


def test_payload_never_contains_headline_text_or_urls(paths):
    arts = _today([(("AAPL", "SPY"), 0.9), (("XOM",), -0.9)])
    for t in Scout(paths=paths).evaluate(_ctx(arts)):
        blob = json.dumps(t.payload, default=str)
        assert SECRET not in blob and "http" not in blob and "example.invalid" not in blob


def test_rejected_trial_is_reported_in_rule_mode(paths):
    rec = {"record_id": "rec-rej", "decision": "reject", "reason_code": "not_significant", "challenger_id": "scout-x",
           "challenger_score": {"value": 0.6935, "details": {"accuracy": 0.51, "auc": 0.49, "baseline_logloss": 0.6921}},
           "champion_score": {"value": 0.6921}, "comparator_stats": {"p": 0.8},
           "context": {"kind": "trial", "role": "vs_base_rate", "trial_number": 1, "alpha_k": 0.025}}
    append_jsonl(paths.gate_log_path, rec)
    arts = _today([(("AAPL",), 0.6)])
    (t,) = Scout(paths=paths).evaluate(_ctx(arts))
    p = t.payload
    assert p["mode"] == "rule_fallback" and p["model_rejected"] is True and p["gate_record_id"] == "rec-rej"
    assert p["holdout_logloss"] == 0.6935 and p["baseline_logloss"] == 0.6921
    assert p["beats_baseline_logloss"] is False and p["holdout_accuracy"] == 0.51


def test_failed_or_stale_news_returns_nothing_and_logs_why(paths, caplog):
    ctx = StrategyContext(as_of=AS_OF, universe=UNIVERSE, bars=make_bars(SESS),
                          news=NewsSnapshot.unavailable("failed", "news refresh failed: boom"))
    with caplog.at_level(logging.WARNING):
        assert Scout(paths=paths).evaluate(ctx) == []
    assert "boom" in caplog.text
    ctx2 = StrategyContext(as_of=AS_OF, universe=UNIVERSE, bars=make_bars(SESS), news=None)
    assert Scout(paths=paths).evaluate(ctx2) == []


def test_a_ticker_without_todays_bar_is_held(paths):
    bars = make_bars(SESS)
    bars["AAPL"] = bars["AAPL"].iloc[:-1]
    arts = _today([(("AAPL",), 0.6), (("XOM",), 0.6)])
    targets = {t.ticker for t in Scout(paths=paths).evaluate(_ctx(arts, bars=bars))}
    assert "AAPL" not in targets and "XOM" in targets


def test_promote_record_without_a_champion_fails_loudly(paths):
    append_jsonl(paths.gate_log_path, {"record_id": "r", "decision": "promote", "challenger_id": "scout-x",
                                       "context": {"kind": "trial"}})
    with pytest.raises(ModelIntegrityError, match="champion"):
        Scout(paths=paths).evaluate(_ctx(_today([(("AAPL",), 0.6)])))


# --- model mode (a gated champion trained on planted synthetic signal) -------------------------------


@pytest.fixture()
def promoted(paths):
    pytest.importorskip("alphagate")
    from wolfpack_worker.analyst.registration import parse_registration
    from wolfpack_worker.scout import gating as G
    from wolfpack_worker.scout.recipe import scout_v1_recipe_dict

    sessions, bars, arts = synthetic(signal=True)
    ds = build_scout_dataset(bars, arts, sessions)
    reg = parse_registration(write_registration(paths.experiments_dir, 1, scout_v1_recipe_dict()), paths=paths)
    out = G.run_scout_trial(ds, reg, as_of=sessions[-1].close, git_commit="t", paths=paths,
                            gate_config=G.ScoutGateConfig(0.05, "t"))
    assert out.promoted
    return sessions, bars, arts, out


def test_model_mode_uses_the_champion_and_matches_training_features(paths, promoted):
    sessions, bars, arts, out = promoted
    t = len(sessions) - 1
    ctx = _ctx(arts, bars=bars, sessions=sessions, t=t)
    targets = Scout(paths=paths).evaluate(ctx)
    assert len(targets) == 5
    ds = build_scout_dataset(bars, arts, sessions)  # full history, as in training
    champ_names = out.manifest["feature_names"]
    import xgboost as xgb

    from wolfpack_worker.analyst.model_io import load_champion

    booster = load_champion(paths=paths).booster
    for tp in targets:
        p = tp.payload
        assert p["mode"] == "model" and p["model_version"] == out.manifest["model_version"]
        assert p["gate_record_id"] == out.record.record_id and p["model_rejected"] is False
        assert tp.target_exposure == (1.0 if p["p_up"] > 0.5 else 0.0)
        row = ds.loc[(ds["ts"] == tp.signal_ts) & (ds["ticker"] == tp.ticker)]
        if len(row):  # the last two bars have no label in ds; compare where present
            assert [p["features"][n] for n in champ_names] == list(row[champ_names].iloc[0])
        x = np.array([[p["features"][n] for n in champ_names]])
        assert p["p_up"] == pytest.approx(float(booster.predict(xgb.DMatrix(x, feature_names=champ_names))[0]))


def test_model_trained_through_must_precede_as_of(paths, promoted):
    sessions, bars, arts, out = promoted
    early = len(sessions) - 300  # inside the training period
    with pytest.raises(LookaheadError, match="trained"):
        Scout(paths=paths).evaluate(_ctx(arts, bars=bars, sessions=sessions, t=early))


# --- through run_persona ------------------------------------------------------------------------------


def _store_for(rows_df, first_seen):
    store = LocalNewsStore()
    store.insert_new([NewsArticle(id=int(r.id), created_at=r.created_at.to_pydatetime(), vendor_updated_at=None,
                                  headline=r.headline, source=r.source, url=None, symbols=tuple(r.symbols))
                      for r in rows_df.itertuples()], first_seen_at=first_seen, ingest_mode="backfill")
    return store


def test_run_persona_news_failure_skips_scout_but_not_others(paths):
    bars = make_bars(SESS)
    price_store = InMemoryPriceStore(bars=dict(bars))
    broker = FakeBroker(sessions=list(SESS))
    kw = dict(broker=broker, price_store=price_store, trade_repo=InMemoryTradeRepo(), as_of=AS_OF,
              universe=UNIVERSE, sizer=FixedNotionalSizer(), run_id="r", dry_run=True, news_store=LocalNewsStore(),
              news_refresh=NewsRefreshResult("failed", error="network down"))
    assert run_persona(strategy=Scout(paths=paths), **kw) == []
    tf_intents = run_persona(strategy=TrendFollower(), **kw)
    tf_alone = run_persona(strategy=TrendFollower(), **{**kw, "news_refresh": None})
    assert [i.ticker for i in tf_intents] == [i.ticker for i in tf_alone]


def test_run_persona_end_to_end_rule_mode(paths):
    # Through the store there is no precomputed score: VADER scores the text.
    arts = articles(_background() + [(100_000, ny(SESS[T].date, 10), ("AAPL",), f"{SECRET} {POS}")])
    arts = arts.drop(columns=["score"])
    bars = make_bars(SESS)
    store = _store_for(arts, first_seen=AS_OF + timedelta(minutes=20))
    intents = run_persona(strategy=Scout(paths=paths), broker=FakeBroker(sessions=list(SESS)),
                          price_store=InMemoryPriceStore(bars=dict(bars)), trade_repo=InMemoryTradeRepo(),
                          as_of=AS_OF, universe=UNIVERSE, sizer=FixedNotionalSizer(), run_id="r", dry_run=True,
                          news_store=store, news_refresh=NewsRefreshResult("ok", cutoff=AS_OF + timedelta(minutes=30)))
    assert [(i.ticker, i.side) for i in intents] == [("AAPL", "buy")]
    blob = json.dumps(dict(intents[0].payload), default=str)
    assert SECRET not in blob and "http" not in blob and intents[0].payload["mode"] == "rule_fallback"
