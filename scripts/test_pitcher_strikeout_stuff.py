#!/usr/bin/env python3
"""Tests for separate Statcast stuff strikeout model."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pitcher_strikeout_stuff import (  # noqa: E402
    fit_stuff_strikeout_model,
    format_stuff_strikeout_display,
    score_stuff_strikeout_prop,
)
from pitcher_stuff import (  # noqa: E402
    build_pitcher_stuff_games,
    flag_pitch_stuff,
    pitch_in_zone,
)


def _sample_pitcher_games() -> pd.DataFrame:
    rows = []

    for game_idx in range(6):
        for pitch_idx in range(60):
            is_whiff = pitch_idx % 5 == 0
            outside = pitch_idx % 7 == 0
            rows.append(
                {
                    "game_date": f"2026-08-{10 + game_idx:02d}",
                    "game_pk": 900000 + game_idx,
                    "pitcher": 12345,
                    "description": (
                        "swinging_strike"
                        if is_whiff
                        else "called_strike"
                    ),
                    "plate_x": 1.2 if outside else 0.1,
                    "plate_z": 2.5,
                    "sz_top": 3.5,
                    "sz_bot": 1.5,
                    "release_speed": 95.0 + pitch_idx * 0.1,
                }
            )

    statcast = pd.DataFrame(rows)
    stuff = build_pitcher_stuff_games(statcast)

    games = pd.DataFrame(
        [
            {
                "game_date": f"2026-08-{10 + i:02d}",
                "game_pk": 900000 + i,
                "pitcher": 12345,
                "player_name": "Test Pitcher",
                "team": "TST",
                "opponent": "OPP",
                "is_home": 1,
                "home_team": "TST",
                "away_team": "OPP",
                "strikeouts": 4 + (i % 3),
                "walks": 1,
                "home_runs_allowed": 0,
                "hit_by_pitch": 0,
                "hits_allowed": 4,
                "outs": 15,
                "earned_runs": 2,
                "batters_faced": 22,
            }
            for i in range(6)
        ]
    )

    from build_features import add_rolling_features
    from pitcher_stuff import (
        add_pitcher_stuff_rolling_features,
        merge_stuff_into_pitcher_games,
    )

    merged = merge_stuff_into_pitcher_games(
        games,
        stuff,
    )

    merged = add_pitcher_stuff_rolling_features(
        merged
    )

    return add_rolling_features(
        merged,
        "pitcher",
        "batters_faced",
    )


def test_pitch_in_zone_uses_custom_sz() -> None:
    inside = pitch_in_zone(
        pd.Series([0.0]),
        pd.Series([2.5]),
        pd.Series([3.5]),
        pd.Series([1.5]),
    )
    outside = pitch_in_zone(
        pd.Series([1.5]),
        pd.Series([2.5]),
        pd.Series([3.5]),
        pd.Series([1.5]),
    )

    assert bool(inside[0])
    assert not bool(outside[0])


def test_flag_pitch_stuff_marks_chase_swings() -> None:
    frame = pd.DataFrame(
        [
            {
                "description": "swinging_strike",
                "plate_x": 1.5,
                "plate_z": 2.0,
                "sz_top": 3.5,
                "sz_bot": 1.5,
                "release_speed": 96.0,
            }
        ]
    )
    flagged = flag_pitch_stuff(frame)

    assert flagged.iloc[0]["is_swinging_strike"]
    assert flagged.iloc[0]["chase_swing"]


def test_fit_and_score_stuff_strikeout_model() -> None:
    games = _sample_pitcher_games()
    package = fit_stuff_strikeout_model(
        games,
    )

    assert package["metrics"]["n_rows"] >= 3
    assert np.isfinite(
        package["metrics"]["r2"]
    )

    latest = games.sort_values(
        "game_date"
    ).iloc[-1]

    scores = score_stuff_strikeout_prop(
        {
            "line": 5.5,
            "side": "over",
        },
        latest,
        package,
    )

    assert np.isfinite(
        scores["stuff_predicted_count"]
    )
    assert 0.0 <= scores["stuff_over_probability"] <= 1.0


def test_format_stuff_strikeout_display() -> None:
    text = format_stuff_strikeout_display(
        6.2,
        0.58,
    )
    assert "6.2 K" in text
    assert "58% Over" in text


if __name__ == "__main__":
    test_pitch_in_zone_uses_custom_sz()
    test_flag_pitch_stuff_marks_chase_swings()
    test_fit_and_score_stuff_strikeout_model()
    test_format_stuff_strikeout_display()
    print("OK")
