#!/usr/bin/env python3
"""Unit tests for Sleeper props fetch/parser (Apify payload)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fetch_sleeper_props import (  # noqa: E402
    parse_sleeper_apify_items,
    _map_market,
)


SAMPLE_ITEMS = [
    {
        "platform": "sleeper",
        "projection_id": "9001",
        "line": 1.5,
        "stat": "hits",
        "stat_display": "Hits",
        "status": "active",
        "player_name": "Aaron Judge",
        "player_team": "NYY",
        "player_position": "RF",
        "league": "MLB",
        "game_start": "2026-09-06T23:05:00+00:00",
        "home_team": "NYY",
        "away_team": "BOS",
        "home_team_name": "New York Yankees",
        "away_team_name": "Boston Red Sox",
        "over_multiplier": 1.85,
        "under_multiplier": 1.85,
        "pick_popularity": 0.62,
        "pick_count_over": 80,
        "pick_count_under": 49,
    },
    {
        "projectionId": "9002",
        "line": 5.5,
        "stat": "strike_outs",
        "statDisplay": "Strikeouts",
        "status": "active",
        "playerName": "Gerrit Cole",
        "playerTeam": "NYY",
        "playerPosition": "SP",
        "league": "mlb",
        "gameStart": "2026-09-06T23:05:00+00:00",
        "awayTeamName": "Boston Red Sox",
        "homeTeamName": "New York Yankees",
        "overMultiplier": 1.9,
        "underMultiplier": 1.75,
    },
    {
        "projection_id": "9003",
        "line": 99.5,
        "stat": "hits",
        "status": "closed",
        "player_name": "Ghost",
        "league": "MLB",
    },
    {
        "projection_id": "9004",
        "line": 25.5,
        "stat": "points",
        "status": "active",
        "player_name": "LeBron James",
        "league": "NBA",
    },
]


def test_parse_sleeper_apify_items_mlb_active() -> None:
    rows = parse_sleeper_apify_items(
        SAMPLE_ITEMS,
        fetched_at="2026-09-06T06:00:00+00:00",
    )

    assert len(rows) == 2
    assert rows[0]["player"] == "Aaron Judge"
    assert rows[0]["market"] == "batter_hits"
    assert rows[0]["line"] == 1.5
    assert rows[0]["over_multiplier"] == 1.85
    assert rows[0]["pick_popularity"] == 0.62
    assert rows[0]["game"] == "Boston Red Sox @ New York Yankees"
    assert rows[0]["bookmaker_key"] == "sleeper"

    assert rows[1]["player"] == "Gerrit Cole"
    assert rows[1]["market"] == "pitcher_strikeouts"
    assert rows[1]["source"] == "apify_sleeper"


def test_map_market_strikeouts_by_position() -> None:
    assert _map_market("strike_outs", "SP") == "pitcher_strikeouts"
    assert _map_market("strike_outs", "RF") == "batter_strikeouts"


def test_parse_normalizes_game_start_milliseconds() -> None:
    rows = parse_sleeper_apify_items(
        [{
            "player_name": "Aaron Judge",
            "player_team": "NYY",
            "player_position": "RF",
            "stat": "hits",
            "line": 1.5,
            "status": "active",
            "league": "mlb",
            "game_start": 1788747000000,
            "home_team": "NYY",
            "away_team": "BOS",
        }],
        fetched_at="2026-09-06T06:00:00+00:00",
    )
    assert rows[0]["game_start"] == "2026-09-07T02:10:00+00:00"


def main() -> None:
    test_parse_sleeper_apify_items_mlb_active()
    test_map_market_strikeouts_by_position()
    test_parse_normalizes_game_start_milliseconds()
    print("test_parse_sleeper_apify_items_mlb_active: ok")
    print("test_map_market_strikeouts_by_position: ok")
    print("All sleeper props tests passed.")


if __name__ == "__main__":
    main()
