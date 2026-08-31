#!/usr/bin/env python3
"""Tests for Statcast-derived pitcher stuff metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pitcher_stuff import (  # noqa: E402
    add_rolling_rate_features,
    build_pitcher_stuff_games,
    flag_pitch_stuff,
    pitch_in_zone,
)


def test_pitch_in_zone_uses_custom_sz() -> None:
    in_zone = pitch_in_zone(
        pd.Series([0.2]),
        pd.Series([2.5]),
        pd.Series([3.8]),
        pd.Series([1.6]),
    )
    assert bool(in_zone.iloc[0])


def test_flag_pitch_stuff_counts() -> None:
    frame = pd.DataFrame(
        [
            {
                "description": "swinging_strike",
                "plate_x": 0.0,
                "plate_z": 2.5,
                "sz_top": 3.8,
                "sz_bot": 1.6,
                "release_speed": 95.0,
            },
            {
                "description": "ball",
                "plate_x": 1.5,
                "plate_z": 2.5,
                "sz_top": 3.8,
                "sz_bot": 1.6,
                "release_speed": 92.0,
            },
            {
                "description": "foul",
                "plate_x": 1.2,
                "plate_z": 2.5,
                "sz_top": 3.8,
                "sz_bot": 1.6,
                "release_speed": 94.0,
            },
        ]
    )

    flagged = flag_pitch_stuff(frame)

    assert flagged["is_swinging_strike"].sum() == 1
    assert flagged["is_swing"].sum() == 2
    assert flagged["outside_zone"].sum() == 2
    assert flagged["chase_swing"].sum() == 1


def test_build_pitcher_stuff_games_aggregates() -> None:
    frame = pd.DataFrame(
        [
            {
                "game_date": "2026-08-01",
                "game_pk": 1,
                "pitcher": 100,
                "description": "swinging_strike",
                "plate_x": 0.0,
                "plate_z": 2.5,
                "sz_top": 3.8,
                "sz_bot": 1.6,
                "release_speed": 95.0,
            },
            {
                "game_date": "2026-08-01",
                "game_pk": 1,
                "pitcher": 100,
                "description": "called_strike",
                "plate_x": 0.1,
                "plate_z": 2.6,
                "sz_top": 3.8,
                "sz_bot": 1.6,
                "release_speed": 94.0,
            },
            {
                "game_date": "2026-08-01",
                "game_pk": 1,
                "pitcher": 100,
                "description": "ball",
                "plate_x": 1.5,
                "plate_z": 2.5,
                "sz_top": 3.8,
                "sz_bot": 1.6,
                "release_speed": 93.0,
            },
        ]
    )

    games = build_pitcher_stuff_games(frame)
    row = games.iloc[0]

    assert int(row["pitches"]) == 3
    assert int(row["swinging_strikes"]) == 1
    assert int(row["called_strikes"]) == 1
    assert np.isclose(row["swstr_pct"], 1 / 3)
    assert np.isclose(row["csw_pct"], 2 / 3)


def test_rolling_rate_excludes_current_game() -> None:
    frame = pd.DataFrame(
        [
            {
                "game_date": "2026-08-01",
                "pitcher": 100,
                "swinging_strikes": 1,
                "pitches": 10,
            },
            {
                "game_date": "2026-08-02",
                "pitcher": 100,
                "swinging_strikes": 2,
                "pitches": 10,
            },
            {
                "game_date": "2026-08-03",
                "pitcher": 100,
                "swinging_strikes": 5,
                "pitches": 10,
            },
        ]
    )

    rolled = add_rolling_rate_features(
        frame,
        "pitcher",
        "swinging_strikes",
        "pitches",
        "swstr_pct",
    )

    last = rolled.iloc[-1]
    assert np.isclose(last["swstr_pct_l3"], 0.15)
    assert np.isclose(last["swstr_pct_season"], 0.15)


if __name__ == "__main__":
    test_pitch_in_zone_uses_custom_sz()
    test_flag_pitch_stuff_counts()
    test_build_pitcher_stuff_games_aggregates()
    test_rolling_rate_excludes_current_game()
    print("OK")
