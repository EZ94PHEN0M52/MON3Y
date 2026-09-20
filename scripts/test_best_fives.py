"""Unit tests for Best 5s perfect-L5 helpers."""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ui.player_stats import (  # noqa: E402
    is_perfect_l5_over,
    is_perfect_l5_pp_fantasy,
    is_perfect_l5_prop,
)
from ui.best_fives_board import (  # noqa: E402
    build_perfect_l5_pp_fantasy_df,
    build_perfect_l5_props_df,
)


def test_is_perfect_l5_over_requires_five_games():
    assert is_perfect_l5_over([2, 2, 2, 2, 2], 1.5) is True
    assert is_perfect_l5_over([2, 2, 2, 2], 1.5) is False
    assert is_perfect_l5_over([2, 2, 2, 2, 1], 1.5) is False
    assert is_perfect_l5_over([2, 2, 2, 2, 1.5], 1.5) is False


def test_build_perfect_l5_props_df_filters_and_dedupes():
    df = pd.DataFrame(
        [
            {
                "player": "A",
                "market": "batter_hits",
                "line": 0.5,
                "l5_pct": 1.0,
                "bookmaker": "PrizePicks",
                "bookmaker_key": "prizepicks",
                "odds": -110,
                "edge": 0.05,
                "ev": 0.04,
                "side": "Over",
                "model_probability": 0.6,
                "over_probability": 0.6,
                "under_probability": 0.4,
                "game": "Away @ Home",
                "commence_time": "2026-09-19T19:00:00Z",
            },
            {
                "player": "A",
                "market": "batter_hits",
                "line": 0.5,
                "l5_pct": 1.0,
                "bookmaker": "FanDuel",
                "bookmaker_key": "fanduel",
                "odds": -105,
                "edge": 0.08,
                "ev": 0.07,
                "side": "Over",
                "model_probability": 0.6,
                "over_probability": 0.6,
                "under_probability": 0.4,
                "game": "Away @ Home",
                "commence_time": "2026-09-19T19:00:00Z",
            },
            {
                "player": "B",
                "market": "batter_hits",
                "line": 0.5,
                "l5_pct": 1.0,
                "bookmaker": "DraftKings",
                "bookmaker_key": "draftkings",
                "odds": -110,
                "edge": 0.02,
                "ev": 0.01,
                "side": "Over",
                "model_probability": 0.55,
                "over_probability": 0.55,
                "under_probability": 0.45,
                "game": "Away @ Home",
                "commence_time": "2026-09-19T19:00:00Z",
            },
        ]
    )

    with patch(
        "ui.best_fives_board.is_perfect_l5_prop",
        side_effect=lambda player, market, line, version="v2": True,
    ):
        result = build_perfect_l5_props_df(df, "v2")

    # All books; one best-EV row per player/market (A→FanDuel, B→DraftKings).
    assert len(result) == 2
    by_player = result.set_index("player")
    assert by_player.loc["A", "bookmaker"] == "FanDuel"
    assert by_player.loc["B", "bookmaker"] == "DraftKings"


def test_build_perfect_l5_pp_fantasy_df():
    df = pd.DataFrame(
        [
            {
                "player": "Hot Bat",
                "market": "batter_hits",
                "line": 0.5,
                "game": "Away @ Home",
                "commence_time": "2026-09-19T19:00:00Z",
            },
            {
                "player": "Cold Bat",
                "market": "batter_hits",
                "line": 0.5,
                "game": "Away @ Home",
                "commence_time": "2026-09-19T19:00:00Z",
            },
        ]
    )

    with patch(
        "ui.best_fives_board.is_perfect_l5_pp_fantasy",
        side_effect=lambda player, version="v2": player == "Hot Bat",
    ), patch(
        "ui.best_fives_board.lookup_prizepicks_fantasy_line",
        return_value=6.5,
    ), patch(
        "ui.best_fives_board.rolling_pp_fantasy_over_rates",
        return_value=(1.0, 0.8),
    ), patch(
        "ui.best_fives_board.format_prizepicks_fantasy_line",
        return_value="6.5",
    ), patch(
        "ui.best_fives_board.lookup_batter_hand",
        return_value="R",
    ):
        result = build_perfect_l5_pp_fantasy_df(df, "v2")

    assert len(result) == 1
    assert result.iloc[0]["player"] == "Hot Bat"
    assert result.iloc[0]["l5_l10_pct"] == "100% / 80%"


def test_is_perfect_l5_prop_and_fantasy_wrappers():
    with patch(
        "ui.player_stats.market_stat_values",
        return_value=[2.0, 2.0, 2.0, 2.0, 2.0],
    ):
        assert is_perfect_l5_prop("P", "batter_hits", 0.5) is True

    with patch(
        "ui.player_stats.lookup_prizepicks_fantasy_line",
        return_value=6.5,
    ), patch(
        "pp_fantasy_scores.player_pp_fantasy_score_values",
        return_value=[8.0, 9.0, 7.0, 10.0, 8.0],
    ):
        assert is_perfect_l5_pp_fantasy("P") is True

    with patch(
        "ui.player_stats.lookup_prizepicks_fantasy_line",
        return_value=None,
    ):
        assert is_perfect_l5_pp_fantasy("P") is False


if __name__ == "__main__":
    test_is_perfect_l5_over_requires_five_games()
    test_build_perfect_l5_props_df_filters_and_dedupes()
    test_build_perfect_l5_pp_fantasy_df()
    test_is_perfect_l5_prop_and_fantasy_wrappers()
    print("OK")
