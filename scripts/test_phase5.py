#!/usr/bin/env python3
"""Unit tests for Phase 5: game line features and stolen bases."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from game_lines import (  # noqa: E402
    GAME_LINE_FEATURES,
    build_consensus_game_lines,
    enrich_feature_row_with_game_lines,
    merge_game_lines_into_features,
)
from odds_api import GAME_MARKETS, PROP_MARKETS  # noqa: E402
from prop_scoring import MARKET_STAT_MAP, MODEL_MAP  # noqa: E402
from train import (  # noqa: E402
    BATTER_FEATURES,
    BATTER_MARKETS,
    PARQUET_FEATURE_SCHEMA_VERSION,
    feature_columns_for_version,
)


def _sample_game_lines():
    return pd.DataFrame([
        {
            "event_id": "evt1",
            "commence_time": "2026-08-19T23:00:00Z",
            "home_team": "New York Yankees",
            "away_team": "Boston Red Sox",
            "bookmaker": "BookA",
            "bookmaker_key": "booka",
            "market": "totals",
            "player": None,
            "side": "Over",
            "line": 8.5,
            "odds": -110,
        },
        {
            "event_id": "evt1",
            "commence_time": "2026-08-19T23:00:00Z",
            "home_team": "New York Yankees",
            "away_team": "Boston Red Sox",
            "bookmaker": "BookA",
            "bookmaker_key": "booka",
            "market": "totals",
            "player": None,
            "side": "Under",
            "line": 8.5,
            "odds": -110,
        },
        {
            "event_id": "evt1",
            "commence_time": "2026-08-19T23:00:00Z",
            "home_team": "New York Yankees",
            "away_team": "Boston Red Sox",
            "bookmaker": "BookA",
            "bookmaker_key": "booka",
            "market": "spreads",
            "player": None,
            "side": "New York Yankees",
            "line": -1.5,
            "odds": -110,
        },
        {
            "event_id": "evt1",
            "commence_time": "2026-08-19T23:00:00Z",
            "home_team": "New York Yankees",
            "away_team": "Boston Red Sox",
            "bookmaker": "BookA",
            "bookmaker_key": "booka",
            "market": "spreads",
            "player": None,
            "side": "Boston Red Sox",
            "line": 1.5,
            "odds": -110,
        },
    ])


def test_game_markets_defined() -> None:
    assert "totals" in GAME_MARKETS
    assert "spreads" in GAME_MARKETS


def test_excluded_prop_markets_not_fetched() -> None:
    from odds_api import EXCLUDED_LIVE_PROP_MARKETS, PROP_MARKETS

    for market in EXCLUDED_LIVE_PROP_MARKETS:
        assert market not in PROP_MARKETS

    assert "batter_stolen_bases" in MARKET_STAT_MAP
    assert MARKET_STAT_MAP["batter_stolen_bases"] == "stolen_bases"
    assert "stolen_bases" in BATTER_MARKETS
    assert "batter_stolen_bases" not in MODEL_MAP
    assert "batter_home_runs" not in MODEL_MAP


def test_schema_version_bumped() -> None:
    assert PARQUET_FEATURE_SCHEMA_VERSION == "3"


def test_game_line_features_in_v2_columns() -> None:
    cols = feature_columns_for_version("v2")
    for feature in GAME_LINE_FEATURES:
        assert feature in cols["batter"]
        assert feature in cols["pitcher"]


def test_stolen_bases_features_in_batter_columns() -> None:
    for suffix in ("_l3", "_l5", "_l10", "_season"):
        col = f"stolen_bases{suffix}"
        assert col in BATTER_FEATURES
        assert col in feature_columns_for_version("v2")["batter"]


def test_build_consensus_game_lines() -> None:
    consensus = build_consensus_game_lines(
        _sample_game_lines()
    )

    assert len(consensus) == 1
    row = consensus.iloc[0]
    assert row["game_total_line"] == 8.5
    assert row["game_run_line_home"] == -1.5
    assert row["game_run_line_away"] == 1.5
    assert 0.0 < row["game_implied_total_over_prob"] < 1.0


def test_merge_game_lines_into_features() -> None:
    players = pd.DataFrame([
        {
            "game_date": "2026-08-19",
            "team": "New York Yankees",
            "opponent": "Boston Red Sox",
            "is_home": 1,
            "home_team": "New York Yankees",
            "away_team": "Boston Red Sox",
            "hits_season": 1.1,
        },
        {
            "game_date": "2026-08-19",
            "team": "Boston Red Sox",
            "opponent": "New York Yankees",
            "is_home": 0,
            "home_team": "New York Yankees",
            "away_team": "Boston Red Sox",
            "hits_season": 0.9,
        },
    ])

    merged = merge_game_lines_into_features(
        players,
        _sample_game_lines(),
    )

    home_row = merged[
        merged["team"].eq("New York Yankees")
    ].iloc[0]
    away_row = merged[
        merged["team"].eq("Boston Red Sox")
    ].iloc[0]

    assert home_row["game_total_line"] == 8.5
    assert home_row["game_run_line"] == -1.5
    assert away_row["game_run_line"] == 1.5
    assert not np.isnan(home_row["game_implied_total_over_prob"])


def test_enrich_feature_row_with_game_lines() -> None:
    feature_row = pd.Series({
        "team": "New York Yankees",
        "is_home": 1,
        "hits_season": 1.0,
    })
    prop = pd.Series({
        "commence_time": "2026-08-19T23:00:00Z",
        "home_team": "New York Yankees",
        "away_team": "Boston Red Sox",
    })
    consensus = build_consensus_game_lines(
        _sample_game_lines()
    )

    enriched = enrich_feature_row_with_game_lines(
        feature_row,
        prop,
        consensus,
    )

    assert enriched["game_total_line"] == 8.5
    assert enriched["game_run_line"] == -1.5
    assert not np.isnan(
        enriched["game_implied_total_over_prob"]
    )


def main() -> int:
    test_game_markets_defined()
    test_excluded_prop_markets_not_fetched()
    test_schema_version_bumped()
    test_game_line_features_in_v2_columns()
    test_stolen_bases_features_in_batter_columns()
    test_build_consensus_game_lines()
    test_merge_game_lines_into_features()
    test_enrich_feature_row_with_game_lines()
    print("Phase 5 tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
