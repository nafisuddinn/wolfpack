"""Feature-spec registry (append-only, v1 frozen) and the Recipe.

The v1 feature spec is the one the committed champion was trained on. If
its output changes by even one rounded digit, every v1 model silently
receives different inputs at inference than it was trained on, so the
output is pinned to a hash computed from the pre-registry code.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from analyst_helpers import make_universe_bars
from wolfpack_worker.analyst import features
from wolfpack_worker.analyst.features import FEATURE_SPECS, FeatureSpec, get_feature_spec
from wolfpack_worker.analyst.recipe import (
    LABELS,
    V1_RECIPE_PATH,
    Recipe,
    RecipeError,
    load_recipe_file,
    load_v1_recipe,
)

# sha256 of v1 features on make_universe_bars(300, seed=123), each value
# formatted with 10 significant decimals (so a last-ulp libm difference
# between macOS and Linux CI can't flip it) and NaN as "nan". Computed from
# features.py BEFORE the registry refactor (commit 5f2b107).
V1_FEATURES_DIGEST = "6c33008edfae1808a3e92e02137e1c84c43b58989cacee989d545f709417a248"

# recipe_id of recipes/analyst/v1.toml. Changing ANY recipe field changes
# this id, i.e. it is a different recipe (a new trial), never "v1 edited".
V1_RECIPE_ID = "a8e2709b0d8e"


def _digest(feats) -> str:
    h = hashlib.sha256()
    for t in sorted(feats):
        f = feats[t]
        h.update(t.encode())
        h.update(",".join(f.columns).encode())
        h.update(",".join(str(int(x)) for x in f.index.asi8).encode())
        for v in f.to_numpy(dtype=float).ravel():
            h.update(("nan" if np.isnan(v) else f"{v:.10e}").encode() + b";")
    return h.hexdigest()


# --- feature-spec registry --------------------------------------------------------


def test_v1_spec_output_is_frozen():
    spec = get_feature_spec("v1")
    assert _digest(spec.build_fn(make_universe_bars(300, seed=123))) == V1_FEATURES_DIGEST


def test_v1_spec_names_and_warmup_are_frozen():
    spec = FEATURE_SPECS["v1"]
    assert isinstance(spec, FeatureSpec)
    assert spec.names == (
        "r_1", "r_5", "r_20", "vol_20", "vol_ratio_5_20", "ma_spread_20_50",
        "z_20_logp", "rsi_14", "hl_range", "logvol_ratio_20", "spy_r_1", "spy_r_5",
    )
    assert spec.warmup_bars == 50


def test_legacy_v1_aliases_still_point_at_v1():
    assert features.FEATURE_SPEC_VERSION == "v1"
    assert features.FEATURE_NAMES == FEATURE_SPECS["v1"].names
    assert FEATURE_SPECS["v1"].build_fn is features.build_features


def test_unknown_spec_raises_keyerror_with_known_versions():
    with pytest.raises(KeyError, match="v1"):
        get_feature_spec("v0")


def test_registry_is_read_only():
    with pytest.raises(TypeError):
        FEATURE_SPECS["v2"] = FEATURE_SPECS["v1"]  # type: ignore[index]


# --- recipe ------------------------------------------------------------------------


def test_v1_recipe_matches_the_first_champion_settings():
    r = load_v1_recipe()
    assert r.feature_spec_version == "v1"
    assert r.label == "next_open_to_open_sign_v1" and r.label in LABELS
    assert r.num_boost_round == 300
    assert r.threshold == 0.5
    assert r.universe == ("SPY", "QQQ", "AAPL", "JPM", "XOM")
    assert r.train_since == "2016-01-04"
    assert dict(r.xgb_params) == {
        "objective": "binary:logistic",
        "max_depth": 3,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 20,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "seed": 42,
    }


def test_v1_recipe_matches_committed_champion_manifest_params():
    manifest = json.loads(
        (Path(__file__).resolve().parents[1] / "models/analyst/champion/manifest.json").read_text()
    )
    r = load_v1_recipe()
    assert {**r.xgb_params, "num_boost_round": r.num_boost_round} == manifest["model_params"]


def test_v1_recipe_id_is_pinned():
    assert load_v1_recipe().recipe_id == V1_RECIPE_ID


def test_recipe_id_is_canonical_and_order_independent():
    r = load_v1_recipe()
    d = r.to_dict()
    shuffled = dict(reversed(list(d.items())))
    shuffled["xgb_params"] = dict(reversed(list(d["xgb_params"].items())))
    assert Recipe.from_dict(shuffled).recipe_id == r.recipe_id
    assert len(r.recipe_id) == 12
    assert r.recipe_id == hashlib.sha256(r.canonical_json().encode()).hexdigest()[:12]
    assert json.loads(r.canonical_json()) == d


@pytest.mark.parametrize(
    "change",
    [
        {"num_boost_round": 301},
        {"train_since": "2017-01-03"},
        {"xgb_params": {"max_depth": 4}},
        {"xgb_params": {"seed": 43}},
    ],
)
def test_any_change_gives_a_different_recipe_id(change):
    d = load_v1_recipe().to_dict()
    for k, v in change.items():
        if isinstance(v, dict):
            d[k] = {**d[k], **v}
        else:
            d[k] = v
    assert Recipe.from_dict(d).recipe_id != load_v1_recipe().recipe_id


@pytest.mark.parametrize(
    "patch,match",
    [
        ({"feature_spec_version": "v9"}, "feature_spec_version"),
        ({"label": "close_to_close"}, "label"),
        ({"threshold": 0.55}, "threshold"),
        ({"universe": ["SPY", "QQQ"]}, "universe"),
        ({"num_boost_round": 0}, "num_boost_round"),
        ({"train_since": "yesterday"}, "train_since"),
        ({"surprise": 1}, "unknown"),
        ({"xgb_params": {"objective": "reg:squarederror"}}, "objective"),
        ({"xgb_params": {"objective": "binary:logistic", "nthread": 8}}, "nthread"),
        ({"xgb_params": {"objective": "binary:logistic", "max_depth": [3]}}, "scalar"),
    ],
)
def test_recipe_validation(patch, match):
    d = load_v1_recipe().to_dict()
    d.update(patch)
    with pytest.raises(RecipeError, match=match):
        Recipe.from_dict(d)


def test_missing_key_is_rejected():
    d = load_v1_recipe().to_dict()
    del d["threshold"]
    with pytest.raises(RecipeError, match="missing"):
        Recipe.from_dict(d)


def test_load_recipe_file_accepts_plain_recipe_and_registration_tables(tmp_path):
    plain = V1_RECIPE_PATH.read_text()
    (tmp_path / "plain.toml").write_text(plain)
    assert load_recipe_file(tmp_path / "plain.toml") == load_v1_recipe()
    reg = 'trial_number = 1\nregistered = 2026-10-01\nhypothesis = "x"\n\n' + "\n".join(
        ("[recipe.xgb_params]" if line.strip() == "[xgb_params]" else line)
        for line in plain.splitlines()
    )
    # Top-level recipe keys must sit under [recipe] in a registration file.
    reg = reg.replace('hypothesis = "x"\n', 'hypothesis = "x"\n\n[recipe]\n')
    (tmp_path / "reg.toml").write_text(reg)
    assert load_recipe_file(tmp_path / "reg.toml") == load_v1_recipe()
