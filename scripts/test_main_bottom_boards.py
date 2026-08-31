#!/usr/bin/env python3
"""Tests for main-page bottom boards."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ui.hitters_life_highlights import batting_average_has_board_highlight
from ui.main_bottom_boards import (
    _sort_hot_batter_score_rows,
    build_hot_batter_score_df,
    build_market_top_props_df,
)


def test_build_market_top_props_df_three_per_side_per_market() -> None:
    props = pd.DataFrame(
        [
            {
                "player": f"P{i}",
                "market": "batter_hits",
                "over_probability": 0.50 + i * 0.01,
                "under_probability": 0.40 + i * 0.01,
                "edge": 0.05,
            }
            for i in range(5)
        ]
        + [
            {
                "player": f"Q{i}",
                "market": "batter_total_bases",
                "over_probability": 0.60 + i * 0.01,
                "under_probability": 0.30 + i * 0.01,
                "edge": 0.04,
            }
            for i in range(4)
        ]
    )

    built = build_market_top_props_df(props, per_market=3)
    assert len(built) == 12  # 2 markets × (3 over + 3 under)
    assert set(built["market"]) == {"batter_hits", "batter_total_bases"}
    assert built["_side_rank"].value_counts().to_dict() == {"Over": 6, "Under": 6}


def test_batting_average_has_board_highlight_rules() -> None:
    assert batting_average_has_board_highlight(
        "Szn .280 · L5 .300 · L10 .295"
    )
    assert batting_average_has_board_highlight(
        "Szn .280 · L5 .260 · L10 .300"
    )
    assert batting_average_has_board_highlight(
        "Szn .310 · L5 .250 · L10 .260"
    )
    assert not batting_average_has_board_highlight(
        "Szn .250 · L5 .240 · L10 .260"
    )


def test_build_hot_batter_score_df_filters_and_limits() -> None:
    from unittest.mock import patch

    slate = pd.DataFrame(
        [
            {
                "player": "Alpha",
                "market": "batter_hits",
                "batter_score": 90.0,
                "batter_score_label": "Full",
                "game": "A @ B",
            },
            {
                "player": "Beta",
                "market": "batter_hits",
                "batter_score": 80.0,
                "batter_score_label": "Full",
                "game": "A @ B",
            },
            {
                "player": "Gamma",
                "market": "batter_hits",
                "batter_score": 70.0,
                "batter_score_label": "Full",
                "game": "A @ B",
            },
        ]
    )

    def fake_l5_top(players, version, top_n=15):
        return {"Alpha", "Beta"}

    def fake_batting_avg(player, version):
        if player == "Alpha":
            return "Szn .280 · L5 .300 · L10 .295"
        if player == "Beta":
            return "Szn .250 · L5 .240 · L10 .260"
        return "Szn .250 · L5 .240 · L10 .260"

    with patch(
        "ui.main_bottom_boards._prepare_batter_score_slate",
        return_value=slate,
    ), patch(
        "ui.main_bottom_boards._l5_avg_top_players",
        side_effect=fake_l5_top,
    ), patch(
        "ui.main_bottom_boards.format_batting_average_column",
        side_effect=fake_batting_avg,
    ), patch(
        "ui.main_bottom_boards._build_batter_score_row",
        side_effect=lambda row, version: {
            "player": row["player"],
            "_batter_score": row["batter_score"],
            "batter_score_display": str(row["batter_score"]),
            "batter_score_v2_display": str(row["batter_score"]),
            "player_link": row["player"],
            "game_time": "7:05p",
            "opposing_sp": "SP",
            "vs_pitcher": "—",
            "pp_fantasy_line": "—",
            "ud_fantasy_line": "—",
            "l5_l10_pct": "—",
            "_pp_line": None,
            "_ud_line": None,
            "_l5_pct": float("nan"),
            "_l10_pct": float("nan"),
        },
    ), patch(
        "ui.main_bottom_boards.format_total_bases_game_log",
        return_value="2 1 0 1 3",
    ):
        built = build_hot_batter_score_df(
            pd.DataFrame(),
            "v2",
            limit=10,
        )

    assert len(built) == 1
    assert built.iloc[0]["player"] == "Alpha"
    assert built.iloc[0]["total_bases_log"] == "2 1 0 1 3"


def test_hot_batter_score_ties_use_batting_avg_priority() -> None:
    rows = [
        {
            "player": "Yellow",
            "_batter_score": 80.1,
            "_batting_average": "Szn .310 · L5 .250 · L10 .260",
            "_total_bases_log": "1 0 1 1 1",
        },
        {
            "player": "Green",
            "_batter_score": 80.0,
            "_batting_average": "Szn .280 · L5 .260 · L10 .300",
            "_total_bases_log": "1 0 1 1 1",
        },
        {
            "player": "Orange",
            "_batter_score": 80.05,
            "_batting_average": "Szn .270 · L5 .310 · L10 .270",
            "_total_bases_log": "1 0 1 1 1",
        },
        {
            "player": "Blue",
            "_batter_score": 80.15,
            "_batting_average": "Szn .250 · L5 .240 · L10 .260",
            "_total_bases_log": "2 3 2 1 0",
        },
        {
            "player": "ClearLeader",
            "_batter_score": 90.0,
            "_batting_average": "Szn .310 · L5 .250 · L10 .260",
            "_total_bases_log": "1 0 1 1 1",
        },
    ]

    ordered = _sort_hot_batter_score_rows(rows)
    assert [row["player"] for row in ordered] == [
        "ClearLeader",
        "Blue",
        "Green",
        "Orange",
        "Yellow",
    ]


if __name__ == "__main__":
    test_build_market_top_props_df_three_per_side_per_market()
    test_batting_average_has_board_highlight_rules()
    test_build_hot_batter_score_df_filters_and_limits()
    test_hot_batter_score_ties_use_batting_avg_priority()
    print("OK")
