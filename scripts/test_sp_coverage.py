"""Unit tests for starting-pitcher prop coverage warnings."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils import (  # noqa: E402
    analyze_sp_prop_coverage,
    normalize_player_key,
    warn_sp_prop_coverage,
)


def _sp_row(
    *,
    event_id: str,
    home_team: str,
    away_team: str,
    player: str,
    market: str = "pitcher_strikeouts",
) -> dict:
    return {
        "event_id": event_id,
        "home_team": home_team,
        "away_team": away_team,
        "player": player,
        "market": market,
        "side": "Over",
        "line": 5.5,
        "odds": -110,
    }


def test_full_sp_coverage_is_ok() -> None:
    props = pd.DataFrame([
        _sp_row(
            event_id="g1",
            home_team="New York Yankees",
            away_team="Boston Red Sox",
            player="Gerrit Cole",
        ),
        _sp_row(
            event_id="g1",
            home_team="New York Yankees",
            away_team="Boston Red Sox",
            player="Chris Sale",
        ),
        _sp_row(
            event_id="g2",
            home_team="Los Angeles Dodgers",
            away_team="San Francisco Giants",
            player="Yoshinobu Yamamoto",
        ),
        _sp_row(
            event_id="g2",
            home_team="Los Angeles Dodgers",
            away_team="San Francisco Giants",
            player="Logan Webb",
        ),
    ])

    result = analyze_sp_prop_coverage(props, lag_threshold=0)

    assert result["ok"] is True
    assert result["game_count"] == 2
    assert result["pitcher_count"] == 4
    assert result["warnings"] == []


def test_missing_sp_triggers_warning_and_game_detail() -> None:
    props = pd.DataFrame([
        _sp_row(
            event_id="g1",
            home_team="New York Yankees",
            away_team="Boston Red Sox",
            player="Gerrit Cole",
        ),
        _sp_row(
            event_id="g2",
            home_team="Los Angeles Dodgers",
            away_team="San Francisco Giants",
            player="Yoshinobu Yamamoto",
        ),
        _sp_row(
            event_id="g2",
            home_team="Los Angeles Dodgers",
            away_team="San Francisco Giants",
            player="Logan Webb",
        ),
    ])
    probables = pd.DataFrame([
        {
            "home_team": "New York Yankees",
            "away_team": "Boston Red Sox",
            "home_sp_name": "Gerrit Cole",
            "away_sp_name": "Chris Sale",
        },
    ])

    result = analyze_sp_prop_coverage(
        props,
        probables=probables,
        lag_threshold=0,
    )

    assert result["ok"] is False
    assert result["game_count"] == 2
    assert result["pitcher_count"] == 3
    assert len(result["games_missing_sp"]) == 1
    assert result["games_missing_sp"][0]["game"] == (
        "Boston Red Sox @ New York Yankees"
    )
    assert any(
        "away SP Chris Sale" in side
        for side in result["games_missing_sp"][0]["missing"]
    )


def test_lag_threshold_allows_small_gap() -> None:
    props = pd.DataFrame([
        _sp_row(
            event_id="g1",
            home_team="New York Yankees",
            away_team="Boston Red Sox",
            player="Gerrit Cole",
        ),
        _sp_row(
            event_id="g1",
            home_team="New York Yankees",
            away_team="Boston Red Sox",
            player="Chris Sale",
        ),
        _sp_row(
            event_id="g2",
            home_team="Los Angeles Dodgers",
            away_team="San Francisco Giants",
            player="Yoshinobu Yamamoto",
        ),
    ])

    result = analyze_sp_prop_coverage(props, lag_threshold=2)

    assert result["ok"] is True
    assert result["pitcher_count"] == 3
    assert result["expected_min_pitchers"] == 2
    assert len(result["games_missing_sp"]) == 1


def test_warn_sp_prop_coverage_prints_without_aborting() -> None:
    props = pd.DataFrame([
        _sp_row(
            event_id="g1",
            home_team="New York Yankees",
            away_team="Boston Red Sox",
            player="Gerrit Cole",
        ),
    ])

    import io
    import contextlib

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        warn_sp_prop_coverage(props, lag_threshold=0, context="test")

    output = buffer.getvalue()
    assert "WARNING: SP prop coverage (test)" in output
    assert "Pitchers with SP props: 1" in output


def test_initial_period_mismatch_counts_pitcher_once() -> None:
    props = pd.DataFrame([
        _sp_row(
            event_id="g1",
            home_team="New York Yankees",
            away_team="Boston Red Sox",
            player="J.T. Ginn",
            market="pitcher_strikeouts",
        ),
        _sp_row(
            event_id="g1",
            home_team="New York Yankees",
            away_team="Boston Red Sox",
            player="JT Ginn",
            market="pitcher_outs",
        ),
        _sp_row(
            event_id="g1",
            home_team="New York Yankees",
            away_team="Boston Red Sox",
            player="Chris Sale",
        ),
    ])

    result = analyze_sp_prop_coverage(props, lag_threshold=0)

    assert normalize_player_key("J.T. Ginn") == normalize_player_key("JT Ginn")
    assert result["pitcher_count"] == 2
    assert result["ok"] is True
