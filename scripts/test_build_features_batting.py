#!/usr/bin/env python3
"""Tests for batter SB/HBP aggregation in build_features."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from build_features import (  # noqa: E402
    derive_stolen_bases_from_des,
    supplement_batter_games_from_mlb,
)
from pp_fantasy_scores import compute_pp_fantasy_score_from_stats  # noqa: E402


def test_derive_stolen_bases_from_des_uses_base_state() -> None:
    frame = pd.DataFrame(
        [
            {
                "game_date": "2026-08-22",
                "game_pk": 824073,
                "des": (
                    "Jac Caglianone strikes out swinging. "
                    "Bobby Witt Jr. steals (25) 2nd base."
                ),
                "on_1b": 677951,
                "on_2b": pd.NA,
                "on_3b": pd.NA,
            },
            {
                "game_date": "2026-08-22",
                "game_pk": 824073,
                "des": (
                    "Raynel Delgado called out on strikes and "
                    "Yainer Diaz caught stealing 2nd, catcher to shortstop."
                ),
                "on_1b": 673237,
                "on_2b": pd.NA,
                "on_3b": pd.NA,
            },
        ]
    )

    steals = derive_stolen_bases_from_des(frame)
    assert len(steals) == 1
    assert int(steals.iloc[0]["batter"]) == 677951
    assert int(steals.iloc[0]["stolen_bases"]) == 1


def test_hbp_in_pp_fantasy_score() -> None:
    assert compute_pp_fantasy_score_from_stats(
        1,
        0,
        1.0,
        rbi=1,
        walks=1,
        hit_by_pitch=1,
    ) == 9.0


def test_supplement_batter_games_from_mlb_overrides_stats(
    monkeypatch,
) -> None:
    import build_features as bf

    result = pd.DataFrame(
        [
            {
                "game_date": "2026-08-18",
                "game_pk": 824001,
                "batter": 677951,
                "player_name": "Bobby Witt",
                "stolen_bases": 0,
                "hit_by_pitch": 0,
            }
        ]
    )

    supplement = pd.DataFrame(
        [
            {
                "game_pk": 824001,
                "batter": 677951,
                "stolen_bases": 0,
                "hit_by_pitch": 1,
            }
        ]
    )

    monkeypatch.setattr(
        bf,
        "build_mlb_boxscore_batting",
        lambda game_pks: supplement,
    )

    updated = supplement_batter_games_from_mlb(result)
    assert int(updated.iloc[0]["hit_by_pitch"]) == 1
    assert int(updated.iloc[0]["stolen_bases"]) == 0


if __name__ == "__main__":
    test_derive_stolen_bases_from_des_uses_base_state()
    test_hbp_in_pp_fantasy_score()
    print("OK")
