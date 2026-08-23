#!/usr/bin/env python3
"""Unit tests for player stat history helpers (incl. H2H filter)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import ui.player_stats as ps  # noqa: E402
from ui.player_stats import (  # noqa: E402
    _filter_games_vs_opponent,
    _odds_team_to_statcast_abbr,
    get_last_n_games,
    opponent_display_name,
    slate_opponent_abbr,
)


def test_odds_team_to_statcast_abbr() -> None:
    assert _odds_team_to_statcast_abbr("New York Yankees") == "NYY"
    assert _odds_team_to_statcast_abbr("Toronto Blue Jays") == "TOR"
    assert _odds_team_to_statcast_abbr("NYY") == "NYY"


def test_filter_games_vs_opponent() -> None:
    frame = pd.DataFrame([
        {"game_date": "2026-08-01", "opponent": "NYY", "hits": 1},
        {"game_date": "2026-08-05", "opponent": "BOS", "hits": 2},
        {"game_date": "2026-08-10", "opponent": "NYY", "hits": 0},
    ])
    filtered = _filter_games_vs_opponent(frame, "NYY")
    assert len(filtered) == 2
    assert filtered["hits"].tolist() == [1, 0]


def test_slate_opponent_abbr_with_patched_team_lookup() -> None:
    original = ps.lookup_player_team_abbr
    ps.lookup_player_team_abbr = lambda *args, **kwargs: "TOR"
    try:
        opp = slate_opponent_abbr(
            "Vladimir Guerrero Jr.",
            home_team="New York Yankees",
            away_team="Toronto Blue Jays",
            kind="batter",
            version="v2",
        )
        assert opp == "NYY"
        assert opponent_display_name(opp) == "New York Yankees"
    finally:
        ps.lookup_player_team_abbr = original


def test_get_last_n_games_h2h_filter(tmp_path) -> None:
    path = tmp_path / "batter_features_v2_2026-03-25_2026-08-21.parquet"
    pd.DataFrame([
        {
            "game_date": "2026-08-01",
            "player_name": "Test Player",
            "opponent": "NYY",
            "hits": 1,
        },
        {
            "game_date": "2026-08-05",
            "player_name": "Test Player",
            "opponent": "BOS",
            "hits": 2,
        },
        {
            "game_date": "2026-08-10",
            "player_name": "Test Player",
            "opponent": "NYY",
            "hits": 3,
        },
    ]).to_parquet(path, index=False)

    original = ps.find_latest_feature_path
    ps.find_latest_feature_path = lambda kind, version: path
    try:
        all_games = get_last_n_games(
            "Test Player",
            "batter_hits",
            n=10,
        )
        assert len(all_games) == 3

        h2h = get_last_n_games(
            "Test Player",
            "batter_hits",
            n=10,
            opponent_abbr="NYY",
        )
        assert len(h2h) == 2
        assert h2h["hits"].tolist() == [1, 3]
    finally:
        ps.find_latest_feature_path = original


def main() -> None:
    test_odds_team_to_statcast_abbr()
    print("test_odds_team_to_statcast_abbr: ok")

    test_filter_games_vs_opponent()
    print("test_filter_games_vs_opponent: ok")

    test_slate_opponent_abbr_with_patched_team_lookup()
    print("test_slate_opponent_abbr_with_patched_team_lookup: ok")

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        test_get_last_n_games_h2h_filter(Path(tmp))
    print("test_get_last_n_games_h2h_filter: ok")

    print("All player_stats tests passed.")


if __name__ == "__main__":
    main()
