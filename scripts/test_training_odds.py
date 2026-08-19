#!/usr/bin/env python3
"""Unit tests for training_odds consensus and feature matching."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training_odds import (  # noqa: E402
    build_consensus_lines,
    match_props_to_features,
    stat_to_market_key,
)
from train import (  # noqa: E402
    DERIVED_LINE_FEATURES,
    create_training_rows,
    model_feature_columns,
)


def test_stat_to_market_key_batter() -> None:
    assert stat_to_market_key("hits", "batter") == "batter_hits"
    assert stat_to_market_key("rbi", "batter") == "batter_rbis"
    assert stat_to_market_key("strikeouts", "pitcher") == (
        "pitcher_strikeouts"
    )


def test_build_consensus_lines_median_and_devig() -> None:
    props = pd.DataFrame([
        {
            "event_id": "evt1",
            "commence_time": "2025-04-01T22:00:00Z",
            "player": "Mike Trout",
            "market": "batter_hits",
            "bookmaker": "BookA",
            "bookmaker_key": "booka",
            "side": "Over",
            "line": 1.5,
            "odds": -110,
        },
        {
            "event_id": "evt1",
            "commence_time": "2025-04-01T22:00:00Z",
            "player": "Mike Trout",
            "market": "batter_hits",
            "bookmaker": "BookA",
            "bookmaker_key": "booka",
            "side": "Under",
            "line": 1.5,
            "odds": -110,
        },
        {
            "event_id": "evt1",
            "commence_time": "2025-04-01T22:00:00Z",
            "player": "Mike Trout",
            "market": "batter_hits",
            "bookmaker": "BookB",
            "bookmaker_key": "bookb",
            "side": "Over",
            "line": 0.5,
            "odds": -120,
        },
        {
            "event_id": "evt1",
            "commence_time": "2025-04-01T22:00:00Z",
            "player": "Mike Trout",
            "market": "batter_hits",
            "bookmaker": "BookB",
            "bookmaker_key": "bookb",
            "side": "Under",
            "line": 0.5,
            "odds": 100,
        },
    ])

    consensus = build_consensus_lines(props)

    assert len(consensus) == 1
    row = consensus.iloc[0]
    assert row["game_date"] == "2025-04-01"
    assert row["player"] == "Mike Trout"
    assert row["consensus_line"] == 1.0
    assert 0.0 < row["market_implied_over_prob"] < 1.0


def test_match_props_to_features_join() -> None:
    props = pd.DataFrame([
        {
            "event_id": "evt1",
            "commence_time": "2025-04-01T22:00:00Z",
            "player": "Mike Trout",
            "market": "batter_hits",
            "bookmaker": "BookA",
            "bookmaker_key": "booka",
            "side": "Over",
            "line": 1.5,
            "odds": -110,
        },
        {
            "event_id": "evt1",
            "commence_time": "2025-04-01T22:00:00Z",
            "player": "Mike Trout",
            "market": "batter_hits",
            "bookmaker": "BookA",
            "bookmaker_key": "booka",
            "side": "Under",
            "line": 1.5,
            "odds": -110,
        },
    ])

    features = pd.DataFrame([
        {
            "game_date": "2025-04-01",
            "batter": 123,
            "player_name": "Mike Trout",
            "opponent": "NYY",
            "is_home": 1,
            "hits": 2,
            "hits_season": 1.1,
            "hits_l3": 1.0,
        },
    ])

    feature_columns = ["hits_l3"]

    matched = match_props_to_features(
        features,
        props,
        "batter",
        "hits",
        feature_columns,
    )

    assert len(matched) == 1
    assert matched.iloc[0]["target"] == 1
    assert matched.iloc[0]["line"] == 1.5
    assert abs(matched.iloc[0]["line_vs_season_avg"] - 0.4) < 1e-6


def test_create_training_rows_auto_fallback() -> None:
    features = pd.DataFrame([
        {
            "game_date": "2025-04-01",
            "batter": 123,
            "player_name": "Mike Trout",
            "opponent": "NYY",
            "is_home": 1,
            "hits": 2,
            "hits_season": 1.0,
            "hits_l3": 1.0,
        },
    ])

    synthetic = create_training_rows(
        features,
        "batter",
        "hits",
        [0.5],
        ["hits_l3"],
        line_source="synthetic",
        historical_props=pd.DataFrame(),
    )

    assert len(synthetic) == 1
    assert pd.isna(
        synthetic.iloc[0]["market_implied_over_prob"]
    )
    assert synthetic.iloc[0]["line_vs_season_avg"] == -0.5


def test_model_feature_columns_include_derived() -> None:
    cols = model_feature_columns("v2")["batter"]
    for feature in DERIVED_LINE_FEATURES:
        assert feature in cols


def main() -> int:
    test_stat_to_market_key_batter()
    test_build_consensus_lines_median_and_devig()
    test_match_props_to_features_join()
    test_create_training_rows_auto_fallback()
    test_model_feature_columns_include_derived()
    print("training_odds tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
