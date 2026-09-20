"""Unit tests for batting-order / platoon opportunity tracker."""

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from batter_opportunity import (  # noqa: E402
    batting_order_from_lineup,
    build_opportunity_snapshot,
    expected_pa_for_slot,
    opportunity_index_from_pa,
    volume_adjusted_score,
    volume_from_game_log,
)


def test_expected_pa_decreases_down_order():
    assert expected_pa_for_slot(1) > expected_pa_for_slot(5)
    assert expected_pa_for_slot(5) > expected_pa_for_slot(9)
    assert expected_pa_for_slot(99) is None


def test_batting_order_from_lineup():
    lineup = [
        "Shohei Ohtani",
        "Mookie Betts",
        "Freddie Freeman",
        "Will Smith",
        "Teoscar Hernandez",
        "Max Muncy",
        "Tommy Edman",
        "Andy Pages",
        "Miguel Rojas",
    ]
    assert batting_order_from_lineup("Freddie Freeman", lineup) == 3
    assert batting_order_from_lineup("Miguel Rojas", lineup) == 9
    assert batting_order_from_lineup("Not In Lineup", lineup) is None


def test_volume_adjusted_score_scales_by_slot():
    # Slot 9 PA / league avg ≈ 3.75/4.15 ≈ 0.904
    adj = volume_adjusted_score(80.0, 3.75)
    assert adj is not None
    assert 70.0 < adj < 80.0
    assert volume_adjusted_score(80.0, None) is None


def test_opportunity_index_slot1_near_100():
    assert opportunity_index_from_pa(4.55) == 100.0
    assert opportunity_index_from_pa(3.75) == 0.0


def test_platoon_flag_from_opposite_lineups():
    vs_rhp = ["A Lefty", "Everyday Guy", "C Righty"]
    vs_lhp = ["Everyday Guy", "C Righty", "Other"]
    snap = build_opportunity_snapshot(
        "A Lefty",
        lineup_names=vs_rhp,
        lineup_source="default",
        opposite_lineup_names=vs_lhp,
        game_dates=["2026-09-01", "2026-09-03", "2026-09-05"],
        total_bases=[2, 0, 1],
        as_of=datetime(2026, 9, 10),
    )
    assert snap.batting_order == 1
    assert snap.is_platoon_candidate is True
    assert snap.volume_label == "Platoon"
    assert snap.is_low_volume is True


def test_low_start_rate_marks_platoon():
    # 4 games in 14 days vs ~12 team games → start rate ~0.33
    dates = [f"2026-09-{d:02d}" for d in (1, 3, 5, 7)]
    tbs = [1, 2, 0, 1]
    games, tb_sum, tb_week, rate = volume_from_game_log(
        dates,
        tbs,
        as_of=datetime(2026, 9, 14),
    )
    assert games == 4
    assert rate < 0.55
    snap = build_opportunity_snapshot(
        "Sparse Bat",
        lineup_names=["Sparse Bat"] + [f"P{i}" for i in range(8)],
        lineup_source="official",
        game_dates=dates,
        total_bases=tbs,
        as_of=datetime(2026, 9, 14),
    )
    assert snap.is_low_volume is True
    assert snap.volume_label in {"Platoon", "Bench", "Low vol"}
