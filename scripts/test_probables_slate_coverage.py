"""Tests for probables ↔ props slate alignment and SP hole detection."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from batter_score_data import analyze_batter_score_sp_coverage  # noqa: E402
from fetch_probables import (  # noqa: E402
    analyze_probables_slate_coverage,
    lookup_opposing_sp,
    missing_probables_dates_for_slate,
)
from utils import game_date_from_commence, slate_dates_from_props  # noqa: E402


def _prop_row(
    *,
    home_team: str,
    away_team: str,
    commence_time: str,
    player: str = "Test Hitter",
) -> dict:
    return {
        "event_id": "evt1",
        "home_team": home_team,
        "away_team": away_team,
        "commence_time": commence_time,
        "player": player,
        "market": "batter_hits",
        "side": "Over",
        "line": 0.5,
        "odds": -110,
    }


def test_slate_date_uses_eastern_not_utc() -> None:
    props = pd.DataFrame([
        _prop_row(
            home_team="Colorado Rockies",
            away_team="Cleveland Guardians",
            commence_time="2026-08-23T00:11:00Z",
        )
    ])

    assert slate_dates_from_props(props) == {"2026-08-22"}
    assert game_date_from_commence("2026-08-23T00:11:00Z") == "2026-08-22"


def test_missing_probables_dates_detects_timezone_skew() -> None:
    props = pd.DataFrame([
        _prop_row(
            home_team="Colorado Rockies",
            away_team="Cleveland Guardians",
            commence_time="2026-08-23T00:11:00Z",
        )
    ])
    probables = pd.DataFrame([
        {
            "game_date": "2026-08-23",
            "home_team": "Colorado Rockies",
            "away_team": "Cleveland Guardians",
            "home_sp_name": "Gabriel Hughes",
            "away_sp_name": "Tanner Bibee",
            "home_sp_id": 1,
            "away_sp_id": 2,
        }
    ])

    assert missing_probables_dates_for_slate(props, probables) == {"2026-08-22"}

    result = analyze_probables_slate_coverage(props, probables)
    assert result["ok"] is True
    assert result["games_with_sp"] == 1
    assert result["missing_probables_dates"] == ["2026-08-22"]
    assert any("adjacent schedule date" in msg for msg in result["warnings"])


def test_analyze_probables_slate_coverage_flags_missing_row() -> None:
    props = pd.DataFrame([
        _prop_row(
            home_team="San Diego Padres",
            away_team="Minnesota Twins",
            commence_time="2026-08-22T23:41:00Z",
        )
    ])
    probables = pd.DataFrame([
        {
            "game_date": "2026-08-22",
            "home_team": "Colorado Rockies",
            "away_team": "Cleveland Guardians",
            "home_sp_name": "Gabriel Hughes",
            "away_sp_name": "Tanner Bibee",
            "home_sp_id": 1,
            "away_sp_id": 2,
        }
    ])

    result = analyze_probables_slate_coverage(props, probables)

    assert not result["ok"]
    assert result["games_with_sp"] == 0
    assert result["games_missing_sp"][0]["game"] == (
        "Minnesota Twins @ San Diego Padres"
    )


def test_lookup_opposing_sp_uses_adjacent_schedule_date() -> None:
    probables = pd.DataFrame([
        {
            "game_date": "2026-08-23",
            "home_team": "Colorado Rockies",
            "away_team": "Cleveland Guardians",
            "home_sp_name": "Gabriel Hughes",
            "away_sp_name": "Tanner Bibee",
            "home_sp_id": 687312,
            "away_sp_id": 676440,
        }
    ])

    sp_name, sp_id = lookup_opposing_sp(
        probables,
        "2026-08-22",
        "Colorado Rockies",
        "Cleveland Guardians",
        "CLE",
    )

    assert sp_name == "Gabriel Hughes"
    assert sp_id == 687312


def test_analyze_batter_score_sp_coverage_flags_all_sp_tbd() -> None:
    df = pd.DataFrame([
        {
            "player": "Hitter A",
            "game": "Team A @ Team B",
            "market": "batter_hits",
            "batter_score_label": "Partial · SP TBD",
        },
        {
            "player": "Hitter B",
            "game": "Team A @ Team B",
            "market": "batter_total_bases",
            "batter_score_label": "Partial · SP TBD",
        },
    ])

    result = analyze_batter_score_sp_coverage(df)

    assert not result["ok"]
    assert result["sp_tbd"] == 2
    assert result["warnings"]


if __name__ == "__main__":
    test_slate_date_uses_eastern_not_utc()
    test_missing_probables_dates_detects_timezone_skew()
    test_analyze_probables_slate_coverage_flags_missing_row()
    test_lookup_opposing_sp_uses_adjacent_schedule_date()
    test_analyze_batter_score_sp_coverage_flags_all_sp_tbd()
    print("All probables slate coverage tests passed.")
