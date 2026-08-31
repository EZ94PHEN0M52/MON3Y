#!/usr/bin/env python3
"""Unit tests for distributional / dual-head count models (Phase 1)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from distributional import (  # noqa: E402
    DISTRIBUTIONAL_MARKETS,
    DUAL_HEAD_MARKETS,
    apply_dist_edge_probabilities,
    fit_rate_model,
    market_supports_dual_head,
    market_supports_distributional,
    market_uses_dist_edge_probabilities,
    predict_rate,
    score_distributional_prop,
)
from prop_scoring import MARKET_STAT_MAP  # noqa: E402
from utils import distributional_models_dir  # noqa: E402


def test_pitcher_walks_in_distributional_markets() -> None:
    assert "pitcher_walks" in DISTRIBUTIONAL_MARKETS
    config = DISTRIBUTIONAL_MARKETS["pitcher_walks"]
    assert config["stat"] == "walks"
    assert config["role"] == "pitcher"
    assert MARKET_STAT_MAP["pitcher_walks"] == "walks"


def test_dual_head_markets_include_pitcher_outs() -> None:
    assert DUAL_HEAD_MARKETS == frozenset({
        "pitcher_strikeouts",
        "pitcher_walks",
        "pitcher_outs",
    })
    assert market_supports_dual_head("pitcher_strikeouts")
    assert market_supports_dual_head("pitcher_walks")
    assert market_supports_dual_head("pitcher_outs")
    assert not market_supports_dual_head("batter_hits")
    assert market_supports_distributional("pitcher_outs")


def test_distributional_model_path_convention() -> None:
    dist_dir = distributional_models_dir("v2")
    assert dist_dir.name == "dist"
    assert dist_dir.parent.name == "v2"
    assert (dist_dir / "pitcher_walks.pkl").name == "pitcher_walks.pkl"


def test_fit_and_score_returns_predicted_count() -> None:
    rng = np.random.default_rng(42)
    n = 80
    features = ["walks_l5", "walks_season", "line"]

    X = pd.DataFrame({
        "walks_l5": rng.uniform(0.5, 3.0, n),
        "walks_season": rng.uniform(1.0, 4.0, n),
        "line": rng.uniform(1.5, 3.5, n),
    })
    y = pd.Series(rng.poisson(2.0, n).astype(float))

    model = fit_rate_model(X, y)
    package = {
        "market": "pitcher_walks",
        "stat": "walks",
        "role": "pitcher",
        "features": features,
        "model": model,
    }

    feature_row = pd.Series({
        "walks_l5": 2.1,
        "walks_season": 2.4,
    })
    prop = pd.Series({
        "line": 2.5,
        "side": "over",
        "market": "pitcher_walks",
    })

    rate = predict_rate(package, feature_row)
    assert rate >= 0.0

    scores = score_distributional_prop(
        prop,
        feature_row,
        package,
    )

    assert "predicted_count" in scores
    assert scores["predicted_count"] == scores["predicted_rate"]
    assert scores["predicted_count"] == rate
    assert 0.0 <= scores["over_probability"] <= 1.0
    assert scores["model_probability"] == scores["over_probability"]


def test_walks_use_dist_edge_probabilities() -> None:
    assert market_uses_dist_edge_probabilities("pitcher_walks")
    assert not market_uses_dist_edge_probabilities("pitcher_strikeouts")

    clf_scores = {
        "over_probability": 0.21,
        "under_probability": 0.79,
        "model_probability": 0.79,
        "raw_model_probability": 0.66,
        "edge": 0.30,
        "ev": 0.25,
    }
    dist_scores = {
        "over_probability": 0.68,
        "under_probability": 0.32,
        "model_probability": 0.68,
        "predicted_count": 2.35,
    }
    prop = pd.Series({
        "line": 1.5,
        "side": "over",
        "odds": -110,
        "market": "pitcher_walks",
    })

    updated = apply_dist_edge_probabilities(
        clf_scores,
        dist_scores,
        prop,
    )

    assert updated["clf_over_probability"] == 0.21
    assert updated["over_probability"] == 0.68
    assert updated["model_probability"] == 0.68
    assert updated["dist_over_probability"] == 0.68
    assert updated["edge"] != clf_scores["edge"]


def main() -> int:
    test_pitcher_walks_in_distributional_markets()
    test_dual_head_markets_include_pitcher_outs()
    test_distributional_model_path_convention()
    test_fit_and_score_returns_predicted_count()
    test_walks_use_dist_edge_probabilities()
    print("Distributional Phase 1 tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
