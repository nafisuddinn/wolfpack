"""The Analyst strategy (`strategies/analyst.py`) + champion model loading
(`analyst/model_io.py`)."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from analyst_helpers import UNIVERSE, make_universe_bars, write_gated_champion
from fakes import FakeBroker, InMemoryPriceStore, InMemoryTradeRepo
from wolfpack_worker.analyst.dataset import build_dataset
from wolfpack_worker.analyst.features import FEATURE_NAMES, FEATURE_SPEC_VERSION
from wolfpack_worker.analyst.model_io import (
    DEFAULT_CHAMPION_DIR,
    ModelIntegrityError,
    load_champion,
)
from wolfpack_worker.daily_trades import run_persona
from wolfpack_worker.execution import FixedNotionalSizer
from wolfpack_worker.strategies import REGISTRY
from wolfpack_worker.strategies.analyst import (
    LIMITATIONS,
    LOOKBACK_BARS,
    THRESHOLD,
    Analyst,
    exposure_from_p_up,
)
from wolfpack_worker.strategies.base import LookaheadError, StrategyContext, truncate_bars

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Fixtures: a tiny real XGBoost champion written to a temp dir
# ---------------------------------------------------------------------------


def _tiny_manifest(trained_through: pd.Timestamp) -> dict:
    return {
        "model_version": "analyst-20260101-deadbeef",
        "mlflow_run_id": "test-run-id",
        "recipe_id": "testrecipe00",
        "trained_through": trained_through.isoformat(),
        "feature_spec_version": FEATURE_SPEC_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "xgboost_version": xgb.__version__,
        "threshold": 0.5,
        "test_metrics": {
            "accuracy": 0.51,
            "logloss": 0.693,
            "auc": 0.5,
            "brier": 0.25,
            "baseline_accuracy": 0.53,
            "baseline_logloss": 0.691,
            "beats_baseline_logloss": False,
        },
        "top_importances_gain": {"r_1": 1.0},
    }


def _train_tiny_booster(bars) -> xgb.Booster:
    ds = build_dataset(bars)
    dtrain = xgb.DMatrix(ds[list(FEATURE_NAMES)].to_numpy(), label=ds["y"].to_numpy(),
                         feature_names=list(FEATURE_NAMES))
    return xgb.train(
        {"objective": "binary:logistic", "max_depth": 2, "eta": 0.3, "seed": 0,
         "tree_method": "hist", "nthread": 1},
        dtrain,
        num_boost_round=10,
    )


@pytest.fixture(scope="module")
def bars():
    return make_universe_bars(400, seed=17, start="2024-01-02")


@pytest.fixture()
def champion_dir(tmp_path, bars):
    # <tmp>/champion + <tmp>/gate_log.jsonl: load_champion now also requires
    # the alphagate PROMOTE record next to the champion directory.
    booster = _train_tiny_booster({k: v.iloc[:200] for k, v in bars.items()})
    trained_through = bars["SPY"].index[199]
    champ = tmp_path / "champion"
    write_gated_champion(champ, bytes(booster.save_raw("json")), _tiny_manifest(trained_through))
    return champ


def _ctx(bars, as_of_idx=-1, lookback=LOOKBACK_BARS + 10):
    as_of = bars["SPY"].index[as_of_idx] + pd.Timedelta(hours=16)  # ~after the close
    truncated = truncate_bars(bars, as_of)
    window = {k: v.iloc[-lookback:] for k, v in truncated.items()}
    return StrategyContext(as_of=as_of.to_pydatetime(), universe=UNIVERSE, bars=window)


# ---------------------------------------------------------------------------
# Identity / registry
# ---------------------------------------------------------------------------


def test_slug_version_lookback_and_registered_last():
    s = Analyst()
    assert s.slug == "the-analyst"
    assert s.version == "analyst/v1"
    assert s.lookback_bars == LOOKBACK_BARS == 80
    # Registry order decides who skips on a same-ticker collision in the shared
    # paper account (later persona skips; Decision Log 2026-09-26). The Scout
    # was appended after The Analyst so no existing persona's behaviour changed.
    assert list(REGISTRY) == ["trend-follower", "contrarian", "the-analyst", "the-scout"]
    assert REGISTRY["the-analyst"] is Analyst


def test_slug_matches_supabase_seed():
    seed = (REPO_ROOT / "supabase" / "seed.sql").read_text()
    assert re.search(r"'the-analyst'", seed)


def test_exposure_rule_is_long_above_threshold_flat_otherwise_ties_flat():
    assert THRESHOLD == 0.5
    assert exposure_from_p_up(0.5000001) == 1.0
    assert exposure_from_p_up(0.5) == 0.0
    assert exposure_from_p_up(0.2) == 0.0


# ---------------------------------------------------------------------------
# Model file integrity
# ---------------------------------------------------------------------------


def test_load_champion_roundtrip(champion_dir):
    champ = load_champion(champion_dir)
    assert champ.manifest["feature_names"] == list(FEATURE_NAMES)
    assert len(champ.manifest["model_sha256"]) == 64


def test_sha256_mismatch_raises(champion_dir):
    model_path = champion_dir / "model.json"
    model_path.write_bytes(model_path.read_bytes() + b" ")
    with pytest.raises(ModelIntegrityError, match="sha256"):
        load_champion(champion_dir)


def test_feature_spec_version_mismatch_raises(champion_dir):
    m = json.loads((champion_dir / "manifest.json").read_text())
    m["feature_spec_version"] = "v0"
    (champion_dir / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ModelIntegrityError, match="feature_spec_version"):
        load_champion(champion_dir)


def test_feature_names_mismatch_raises(champion_dir):
    m = json.loads((champion_dir / "manifest.json").read_text())
    m["feature_names"] = list(reversed(m["feature_names"]))
    (champion_dir / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ModelIntegrityError, match="feature_names"):
        load_champion(champion_dir)


def test_missing_model_raises_not_silently_holds(tmp_path, bars):
    with pytest.raises(ModelIntegrityError, match="missing"):
        Analyst(model_dir=tmp_path).evaluate(_ctx(bars))


def test_committed_champion_exists_and_verifies():
    """A missing/corrupt committed champion is a bug, not a 'hold' — the
    daily cron would raise on it, so catch it here first."""
    champ = load_champion(DEFAULT_CHAMPION_DIR)
    m = champ.manifest
    assert m["feature_spec_version"] == FEATURE_SPEC_VERSION
    assert m["model_version"].startswith("analyst-")
    assert m["mlflow_run_id"]
    pd.Timestamp(m["trained_through"])  # parseable


# ---------------------------------------------------------------------------
# Lookahead guard on the model itself
# ---------------------------------------------------------------------------


def test_model_trained_through_after_as_of_raises_lookahead(champion_dir, bars):
    # trained_through = bar 199; deciding at bar 150 would use a model that
    # has seen labels from after the decision date.
    with pytest.raises(LookaheadError):
        Analyst(model_dir=champion_dir).evaluate(_ctx(bars, as_of_idx=150))


def test_model_trained_through_equal_to_as_of_raises_lookahead(champion_dir, bars):
    m = json.loads((champion_dir / "manifest.json").read_text())
    ctx = _ctx(bars, as_of_idx=300)
    m["trained_through"] = pd.Timestamp(ctx.as_of).isoformat()
    (champion_dir / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(LookaheadError):
        Analyst(model_dir=champion_dir).evaluate(ctx)


# ---------------------------------------------------------------------------
# evaluate(): outputs + payload
# ---------------------------------------------------------------------------

EXPECTED_PAYLOAD_KEYS = {
    "strategy", "version", "model_version", "mlflow_run_id", "trained_through",
    "p_up", "threshold", "regime", "features", "top_contributions",
    "holdout_accuracy", "baseline_accuracy", "holdout_logloss", "baseline_logloss",
    "beats_baseline_logloss", "base_log_odds", "last_close", "bar_ts", "as_of", "bars_used", "limitations",
}


def test_evaluate_payload_shape_and_values(champion_dir, bars):
    ctx = _ctx(bars)
    targets = Analyst(model_dir=champion_dir).evaluate(ctx)
    assert [t.ticker for t in targets] == list(UNIVERSE)
    champ = load_champion(champion_dir)
    for t in targets:
        p = t.payload
        assert set(p) == EXPECTED_PAYLOAD_KEYS
        json.dumps(p)  # self-contained, JSON-serializable
        assert p["strategy"] == "xgb_direction_classifier"
        assert p["version"] == "analyst/v1"
        assert p["model_version"] == champ.manifest["model_version"]
        assert p["mlflow_run_id"] == "test-run-id"
        assert 0.0 < p["p_up"] < 1.0
        assert p["threshold"] == 0.5
        assert t.target_exposure == (1.0 if p["p_up"] > 0.5 else 0.0)
        assert p["regime"] == ("model_bullish" if p["p_up"] > 0.5 else "model_bearish")
        assert list(p["features"]) == list(FEATURE_NAMES)
        assert len(p["top_contributions"]) == 4
        mags = [abs(c["contribution"]) for c in p["top_contributions"]]
        assert mags == sorted(mags, reverse=True)
        assert all(c["feature"] in FEATURE_NAMES for c in p["top_contributions"])
        assert p["holdout_accuracy"] == 0.51
        assert p["baseline_logloss"] == 0.691
        assert p["beats_baseline_logloss"] is False
        assert p["last_close"] == float(bars[t.ticker]["close"].iloc[-1])
        assert t.signal_ts == ctx.bars[t.ticker].index[-1]
        assert p["bars_used"] == LOOKBACK_BARS
        assert "no" in p["limitations"].lower() and "edge" in p["limitations"].lower()
        assert p["limitations"] == LIMITATIONS


def test_p_up_matches_direct_model_prediction_and_contributions_sum_to_logit(champion_dir, bars):
    ctx = _ctx(bars)
    targets = Analyst(model_dir=champion_dir).evaluate(ctx)
    champ = load_champion(champion_dir)
    from wolfpack_worker.analyst.features import build_features

    feats = build_features({k: v for k, v in ctx.bars.items()})
    for t in targets:
        x = feats[t.ticker].iloc[[-1]][list(FEATURE_NAMES)].to_numpy()
        d = xgb.DMatrix(x, feature_names=list(FEATURE_NAMES))
        p = float(champ.booster.predict(d)[0])
        assert t.payload["p_up"] == pytest.approx(p, abs=1e-7)
        contribs = champ.booster.predict(d, pred_contribs=True)[0]
        assert float(contribs.sum()) == pytest.approx(math.log(p / (1 - p)), abs=1e-4)
        for c in t.payload["top_contributions"]:
            j = FEATURE_NAMES.index(c["feature"])
            assert c["contribution"] == pytest.approx(float(contribs[j]), abs=1e-7)


def test_nan_spy_feature_leaves_ticker_out_that_day(champion_dir, bars):
    ctx = _ctx(bars)
    spy = ctx.bars["SPY"].iloc[:-1]  # SPY's latest bar missing
    new_bars = dict(ctx.bars)
    new_bars["SPY"] = spy
    ctx2 = StrategyContext(as_of=ctx.as_of, universe=ctx.universe, bars=new_bars)
    targets = Analyst(model_dir=champion_dir).evaluate(ctx2)
    tickers = {t.ticker for t in targets}
    # Non-SPY tickers: spy_r_1/spy_r_5 NaN on the latest bar -> held.
    assert tickers.isdisjoint({"QQQ", "AAPL", "JPM", "XOM"})
    # SPY's own latest bar is a stale session -> also held (not traded on
    # yesterday's data).
    assert "SPY" not in tickers


def test_insufficient_history_ticker_is_left_out(champion_dir, bars):
    ctx = _ctx(bars)
    new_bars = dict(ctx.bars)
    new_bars["JPM"] = ctx.bars["JPM"].iloc[-30:]
    ctx2 = StrategyContext(as_of=ctx.as_of, universe=ctx.universe, bars=new_bars)
    tickers = {t.ticker for t in Analyst(model_dir=champion_dir).evaluate(ctx2)}
    assert tickers == set(UNIVERSE) - {"JPM"}


def test_garbage_future_bars_do_not_change_evaluate_output(champion_dir, bars):
    ctx = _ctx(bars, as_of_idx=320)
    expected = Analyst(model_dir=champion_dir).evaluate(ctx)
    garbage = {}
    for k, v in bars.items():
        v = v.copy()
        v.iloc[321:, :4] = 1e12
        garbage[k] = v
    as_of = ctx.as_of
    truncated = truncate_bars(garbage, as_of)
    window = {k: v.iloc[-(LOOKBACK_BARS + 10):] for k, v in truncated.items()}
    ctx2 = StrategyContext(as_of=as_of, universe=UNIVERSE, bars=window)
    assert Analyst(model_dir=champion_dir).evaluate(ctx2) == expected


def test_run_persona_end_to_end_with_fakes(champion_dir, bars):
    as_of = (bars["SPY"].index[-1] + pd.Timedelta(hours=16)).to_pydatetime()
    store = InMemoryPriceStore(bars=dict(bars))
    repo = InMemoryTradeRepo()
    broker = FakeBroker()
    intents = run_persona(
        strategy=Analyst(model_dir=champion_dir),
        broker=broker,
        price_store=store,
        trade_repo=repo,
        as_of=as_of,
        universe=UNIVERSE,
        sizer=FixedNotionalSizer(),
        run_id="test",
        dry_run=False,
    )
    assert all(i.side == "buy" for i in intents)  # flat -> long only on first run
    for ins in repo.inserts:
        assert ins["signal_payload"]["strategy"] == "xgb_direction_classifier"
