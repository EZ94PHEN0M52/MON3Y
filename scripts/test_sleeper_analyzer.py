#!/usr/bin/env python3
"""Unit tests for Sleeper Prop Analyzer add-on."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fetch_sleeper_props import parse_sleeper_apify_items  # noqa: E402
from sleeper_analyzer.data import game_key, props_for_team  # noqa: E402
from sleeper_analyzer.hit_rates import HIT_RATE_RULE, supports_hit_rate  # noqa: E402


def test_parse_includes_game_context_fields() -> None:
    rows = parse_sleeper_apify_items(
        [{
            "playerName": "CJ Abrams",
            "playerTeam": "WSH",
            "playerPosition": "SS",
            "playerImage": "https://example.com/a.jpg",
            "stat": "hits",
            "line": 0.5,
            "status": "active",
            "league": "mlb",
            "gameId": "123",
            "gameStatus": "pre_game",
            "venueName": "Nationals Park",
            "awayTeam": "WSH",
            "homeTeam": "LAD",
            "awayTeamName": "Washington Nationals",
            "homeTeamName": "Los Angeles Dodgers",
        }],
        fetched_at="2026-09-06T06:00:00+00:00",
    )
    assert rows[0]["game_id"] == "123"
    assert rows[0]["venue_name"] == "Nationals Park"
    assert rows[0]["player_image"] == "https://example.com/a.jpg"


def test_game_key_prefers_game_id() -> None:
    row = pd.Series({
        "game_id": "999",
        "away_team": "WSH",
        "home_team": "LAD",
        "game_start": "2026-09-06T23:00:00+00:00",
    })
    assert game_key(row) == "id:999"


def test_props_for_team_filters() -> None:
    df = pd.DataFrame([
        {"player": "A", "player_team": "WSH"},
        {"player": "B", "player_team": "LAD"},
    ])
    wsh = props_for_team(df, "WSH")
    assert len(wsh) == 1
    assert wsh.iloc[0]["player"] == "A"


def test_hit_rate_rule_documents_strict_over() -> None:
    assert "strictly exceeded" in HIT_RATE_RULE.lower()
    assert supports_hit_rate("batter_hits")
    assert not supports_hit_rate("sleeper_singles")


def test_best_l5_prop_picks_highest_rate() -> None:
    from unittest.mock import patch

    from sleeper_analyzer.hit_rates import best_l5_prop_for_player

    props = pd.DataFrame([
        {"player": "A", "market": "batter_hits", "sleeper_stat": "hits", "line": 0.5},
        {"player": "A", "market": "batter_total_bases", "sleeper_stat": "total_bases", "line": 1.5},
    ])

    fake_rows = pd.DataFrame([
        {"stat_display": "Hits", "line": 0.5, "l5_pct": 0.4, "sleeper_stat": "hits"},
        {"stat_display": "Total Bases", "line": 1.5, "l5_pct": 0.8, "sleeper_stat": "total_bases"},
    ])

    with patch("sleeper_analyzer.hit_rates.build_player_prop_rows", return_value=fake_rows):
        best = best_l5_prop_for_player(props, version="v2")

    assert best is not None
    assert best["stat_display"] == "Total Bases"
    assert best["l5_pct"] == 0.8


def main() -> None:
    test_parse_includes_game_context_fields()
    test_game_key_prefers_game_id()
    test_props_for_team_filters()
    test_hit_rate_rule_documents_strict_over()
    test_best_l5_prop_picks_highest_rate()
    print("All sleeper analyzer tests passed.")


if __name__ == "__main__":
    main()
