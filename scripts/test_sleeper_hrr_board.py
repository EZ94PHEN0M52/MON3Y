#!/usr/bin/env python3
"""Unit tests for Sleeper H+R+RBI 0.5 board."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ui.sleeper_hrr_board import (  # noqa: E402
    HRR_MARKET,
    TARGET_LINE,
    _game_context_from_sleeper_row,
    build_sleeper_hrr_board_df,
    filter_sleeper_hrr_half_props,
)


def test_filter_keeps_only_hrr_half_line() -> None:
    df = pd.DataFrame([
        {
            "player": "A",
            "market": HRR_MARKET,
            "sleeper_stat": "hits_runs_rbis",
            "line": 0.5,
        },
        {
            "player": "B",
            "market": HRR_MARKET,
            "sleeper_stat": "hits_runs_rbis",
            "line": 1.5,
        },
        {
            "player": "C",
            "market": "batter_hits",
            "sleeper_stat": "hits",
            "line": 0.5,
        },
    ])
    filtered = filter_sleeper_hrr_half_props(df)
    assert len(filtered) == 1
    assert filtered.iloc[0]["player"] == "A"
    assert float(filtered.iloc[0]["line"]) == TARGET_LINE


def test_build_board_adds_enrichment_columns() -> None:
    df = pd.DataFrame([
        {
            "player": "Aaron Judge",
            "market": HRR_MARKET,
            "sleeper_stat": "hits_runs_rbis",
            "line": 0.5,
            "game": "Boston Red Sox @ New York Yankees",
            "game_start": "2026-09-06T23:05:00+00:00",
            "player_team": "NYY",
            "stat_display": "Hits Runs Rbis",
            "over_multiplier": 1.8,
            "under_multiplier": 1.8,
            "pick_popularity": 0.55,
            "status": "active",
        },
    ])
    board = build_sleeper_hrr_board_df(df, version="v2")
    assert not board.empty
    for col in (
        "player_link",
        "l5_hit_rate",
        "h2h_avg",
        "arsenal_woba",
        "xwoba",
        "wrc_plus",
        "total_bases_log",
        "opposing_sp",
    ):
        assert col in board.columns


def test_game_context_maps_sleeper_abbrs_to_probables_names() -> None:
    row = pd.Series({
        "game": "Rockies @ Tigers",
        "game_start": "2026-09-11T22:40:00+00:00",
        "home_team": "DET",
        "away_team": "COL",
    })
    ctx = _game_context_from_sleeper_row(row)
    assert ctx is not None
    assert ctx["home_team"] == "Detroit Tigers"
    assert ctx["away_team"] == "Colorado Rockies"
    assert ctx["game_date"] == "2026-09-11"


def main() -> None:
    test_filter_keeps_only_hrr_half_line()
    test_build_board_adds_enrichment_columns()
    test_game_context_maps_sleeper_abbrs_to_probables_names()
    print("All sleeper H+R+RBI board tests passed.")


if __name__ == "__main__":
    main()
