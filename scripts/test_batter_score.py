"""
Unit tests for Batter Score (Phase A + B).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from batter_score import (  # noqa: E402
    MIN_PA_H2H,
    PHASE_A_GATES,
    PHASE_B_GATES,
    BatterInputs,
    ComponentGates,
    GameLine,
    Weights,
    compute_batter_score_partial,
    compute_batter_score_phase_b,
    pitcher_form_index,
    renormalize_weights,
)
import pandas as pd  # noqa: E402


def _sample_games():
    return [
        GameLine("2026-08-18", "SEA", 2, 3, 1),
        GameLine("2026-08-17", "SEA", 1, 1, 0),
        GameLine("2026-08-16", "SEA", 3, 5, 1),
        GameLine("2026-08-15", "HOU", 0, 0, 1),
        GameLine("2026-08-14", "HOU", 1, 2, 0),
        GameLine("2026-08-13", "HOU", 2, 2, 2),
        GameLine("2026-08-11", "TEX", 1, 1, 0),
        GameLine("2026-08-10", "TEX", 0, 0, 0),
        GameLine("2026-08-09", "TEX", 2, 4, 1),
        GameLine("2026-08-08", "LAA", 1, 1, 1),
    ]


def test_renormalize_phase_a_weights():
    weights = Weights()
    active = renormalize_weights(weights, PHASE_A_GATES)

    assert abs(sum(active.values()) - 1.0) < 1e-6
    assert abs(active["season_baseline"] - 0.30 / 0.55) < 1e-6
    assert abs(active["recent_form"] - 0.25 / 0.55) < 1e-6
    assert "matchup_grade" not in active
    assert "pitcher_form" not in active


def test_renormalize_phase_b_weights():
    weights = Weights()
    active = renormalize_weights(weights, PHASE_B_GATES)

    assert abs(sum(active.values()) - 1.0) < 1e-6
    assert abs(active["season_baseline"] - 0.30 / 0.70) < 1e-6
    assert abs(active["recent_form"] - 0.25 / 0.70) < 1e-6
    assert abs(active["pitcher_form"] - 0.15 / 0.70) < 1e-6
    assert "matchup_grade" not in active


def test_compute_batter_score_partial_form_only():
    batter = BatterInputs(
        name="Test Batter",
        season_avg_raw_points=3.8,
        game_log=_sample_games(),
    )

    result = compute_batter_score_partial(batter)

    assert result.is_partial is True
    assert result.partial_label == "Form only"
    assert result.matchup_grade is None
    assert result.pitcher_form is None
    assert 0.0 <= result.batter_score <= 100.0
    assert result.season_baseline > 0
    assert result.recent_form > 0


def test_sp_tbd_label():
    batter = BatterInputs(
        name="Test Batter",
        season_avg_raw_points=3.8,
        game_log=_sample_games(),
    )

    result = compute_batter_score_partial(
        batter,
        sp_tbd=True,
    )

    assert result.partial_label == "Partial · SP TBD"


def test_phase_b_with_sp_era():
    batter = BatterInputs(
        name="Test Batter",
        season_avg_raw_points=3.8,
        game_log=_sample_games(),
        opponent_pitcher_era_l5=2.25,
        opposing_sp_name="Ace Pitcher",
    )

    result = compute_batter_score_phase_b(batter)

    assert result.pitcher_form is not None
    assert result.partial_label == "Partial"
    assert result.opposing_sp_name == "Ace Pitcher"
    assert 0.0 <= result.batter_score <= 100.0

    active = result.active_weights
    expected = (
        active["season_baseline"] * result.season_baseline
        + active["recent_form"] * result.recent_form
        + active["pitcher_form"] * result.pitcher_form
    )
    assert abs(result.batter_score - expected) < 1e-6


def test_h2h_omitted_below_min_pa():
    batter = BatterInputs(
        name="Test Batter",
        season_avg_raw_points=3.8,
        game_log=_sample_games(),
        opponent_pitcher_era_l5=4.00,
        h2h_pa=5,
        h2h_avg_raw_points=5.0,
    )

    era_only = pitcher_form_index(batter)
    assert era_only == pitcher_form_index(
        BatterInputs(
            name="Test Batter",
            season_avg_raw_points=3.8,
            game_log=_sample_games(),
            opponent_pitcher_era_l5=4.00,
        )
    )


def test_h2h_blended_at_min_pa():
    batter = BatterInputs(
        name="Test Batter",
        season_avg_raw_points=3.8,
        game_log=_sample_games(),
        opponent_pitcher_era_l5=4.00,
        h2h_pa=MIN_PA_H2H,
        h2h_avg_raw_points=6.0,
    )

    blended = pitcher_form_index(batter)
    era_only = pitcher_form_index(
        BatterInputs(
            name="Test Batter",
            season_avg_raw_points=3.8,
            game_log=_sample_games(),
            opponent_pitcher_era_l5=4.00,
        )
    )

    assert blended > era_only


def test_partial_score_equals_renormalized_blend():
    batter = BatterInputs(
        name="Test Batter",
        season_avg_raw_points=4.0,
        game_log=_sample_games(),
    )

    result = compute_batter_score_partial(batter)
    active = result.active_weights

    expected = (
        active["season_baseline"] * result.season_baseline
        + active["recent_form"] * result.recent_form
    )

    assert abs(result.batter_score - expected) < 1e-6


def test_min_pa_h2h_constant():
    assert MIN_PA_H2H == 10


def test_gated_full_score_requires_matchup_inputs():
    from batter_score import compute_batter_score, PitchTypeMatchup

    batter = BatterInputs(
        name="Test Batter",
        season_avg_raw_points=3.8,
        game_log=_sample_games(),
        opponent_pitcher_arsenal=[
            PitchTypeMatchup("Fastball", 1.0, 0.360, 0.275),
        ],
        opponent_pitcher_era_l5=3.85,
    )

    gates = ComponentGates(
        season_baseline=True,
        recent_form=True,
        matchup_grade=True,
        pitcher_form=True,
    )

    result = compute_batter_score(batter, gates=gates)

    assert result.is_partial is False
    assert result.partial_label == "Full"
    assert result.matchup_grade is not None
    assert result.pitcher_form is not None


def test_dedupe_best_prop_one_row_per_player_market():
    from odds_aggregation import dedupe_best_prop

    props = pd.DataFrame(
        {
            "player": ["A", "A", "A", "B"],
            "market": ["batter_hits", "batter_hits", "batter_runs_scored", "batter_hits"],
            "bookmaker": ["Book1", "Book2", "Book1", "Book1"],
            "side": ["Over", "Over", "Over", "Under"],
            "line": [0.5, 1.5, 0.5, 0.5],
            "ev": [0.10, 0.20, 0.05, 0.15],
        }
    )

    deduped = dedupe_best_prop(props)

    assert len(deduped) == 3
    assert deduped.loc[
        (deduped["player"] == "A") & (deduped["market"] == "batter_hits"),
        "ev",
    ].iloc[0] == 0.20


def test_pitch_code_to_bucket():
    from pitch_matchup import pitch_code_to_bucket

    assert pitch_code_to_bucket("FF") == "Fastball"
    assert pitch_code_to_bucket("sl") == "Slider"
    assert pitch_code_to_bucket("KC") == "Curveball"
    assert pitch_code_to_bucket("CH") == "Changeup"
    assert pitch_code_to_bucket("KN") == "Other"


def test_build_opponent_pitcher_arsenal_synthetic():
    from pitch_matchup import build_opponent_pitcher_arsenal

    statcast = pd.DataFrame(
        {
            "batter": [1, 1, 1, 1, 2, 2, 2, 2],
            "pitcher": [9, 9, 9, 9, 9, 9, 9, 9],
            "pitch_type": ["FF", "FF", "SL", "SL", "FF", "FF", "SL", "CU"],
            "type": ["X", "X", "X", "X", "X", "X", "X", "X"],
            "events": [
                "single",
                "field_out",
                "single",
                "strikeout",
                "single",
                "field_out",
                "single",
                "field_out",
            ],
            "woba_value": [0.40, 0.10, 0.35, 0.05, 0.30, 0.10, 0.32, 0.08],
            "game_date": [
                "2026-08-01",
                "2026-08-01",
                "2026-08-01",
                "2026-08-01",
                "2026-08-02",
                "2026-08-02",
                "2026-08-02",
                "2026-08-02",
            ],
        }
    )

    arsenal = build_opponent_pitcher_arsenal(statcast, batter_id=1, pitcher_id=9)

    assert len(arsenal) >= 2
    assert abs(sum(item.usage_pct for item in arsenal) - 1.0) < 0.02


def test_compute_batter_score_phase_d():
    from batter_score import PitchTypeMatchup, compute_batter_score_phase_d

    batter = BatterInputs(
        name="Test Batter",
        season_avg_raw_points=3.8,
        game_log=_sample_games(),
        opponent_pitcher_arsenal=[
            PitchTypeMatchup("Fastball", 0.60, 0.360, 0.275),
            PitchTypeMatchup("Slider", 0.40, 0.290, 0.230),
        ],
        opponent_pitcher_era_l5=3.85,
        opposing_sp_name="Test SP",
    )

    result = compute_batter_score_phase_d(batter)

    assert result.is_partial is False
    assert result.partial_label == "Full"
    assert result.matchup_grade is not None
    assert result.pitcher_form is not None
    assert "matchup_grade" in result.active_weights


def test_canonical_odds_team_key_sp_lookup_with_abbr():
    from fetch_probables import lookup_opposing_sp
    from utils import canonical_odds_team_key

    assert canonical_odds_team_key("SF") == "san francisco giants"
    assert canonical_odds_team_key("San Francisco Giants") == "san francisco giants"

    probables = pd.DataFrame(
        {
            "game_date": ["2026-08-19"],
            "home_team": ["Cleveland Guardians"],
            "away_team": ["San Francisco Giants"],
            "home_sp_name": ["Parker Messick"],
            "away_sp_name": ["Matt Wilkinson"],
            "home_sp_id": [800048],
            "away_sp_id": [801234],
        }
    )

    sp_name, sp_id = lookup_opposing_sp(
        probables,
        "2026-08-19",
        "Cleveland Guardians",
        "San Francisco Giants",
        "SF",
    )

    assert sp_name == "Parker Messick"
    assert sp_id == 800048


def test_build_game_context_from_game_string():
    from batter_score_data import build_game_context

    ctx = build_game_context(
        game="Boston Red Sox @ New York Yankees",
        commence_time="2026-08-19T23:05:00Z",
    )

    assert ctx is not None
    assert ctx["home_team"] == "New York Yankees"
    assert ctx["away_team"] == "Boston Red Sox"
    assert ctx["game_date"] == "2026-08-19"


def test_enrich_with_batter_score_columns():
    from batter_score_data import enrich_with_batter_score

    props = pd.DataFrame(
        {
            "player": ["Nobody Here"],
            "market": ["batter_hits"],
            "line": [0.5],
            "game": ["Team A @ Team B"],
            "commence_time": ["2026-08-19T23:05:00Z"],
        }
    )

    enriched = enrich_with_batter_score(props, version="v2")

    assert "batter_score" in enriched.columns
    assert "batter_score_label" in enriched.columns


def test_infer_player_kind():
    from ui.player_stats import infer_player_kind

    assert infer_player_kind(["batter_hits"]) == "batter"
    assert infer_player_kind(["pitcher_strikeouts"]) == "pitcher"


def test_markets_for_kind_includes_stolen_bases():
    from ui.player_stats import markets_for_kind

    batter_markets = markets_for_kind("batter")
    assert "batter_stolen_bases" in batter_markets


def test_coerce_mlb_id_handles_nan():
    import math

    from utils import coerce_mlb_id

    assert coerce_mlb_id(None) is None
    assert coerce_mlb_id(float("nan")) is None
    assert coerce_mlb_id(694973) == 694973
    assert coerce_mlb_id(694973.0) == 694973


def test_pitcher_rows_by_sp_nan_id_falls_back_to_name():
    from batter_score_data import _pitcher_rows_by_sp

    # NaN sp_id must not raise; falls through to name or None
    result = _pitcher_rows_by_sp(float("nan"), "Paul Skenes", version="v2")
    # May be None if no data in env — must not raise ValueError
    assert result is None or not result.empty


if __name__ == "__main__":
    test_renormalize_phase_a_weights()
    test_renormalize_phase_b_weights()
    test_compute_batter_score_partial_form_only()
    test_sp_tbd_label()
    test_phase_b_with_sp_era()
    test_h2h_omitted_below_min_pa()
    test_h2h_blended_at_min_pa()
    test_partial_score_equals_renormalized_blend()
    test_min_pa_h2h_constant()
    test_gated_full_score_requires_matchup_inputs()
    test_dedupe_best_prop_one_row_per_player_market()
    test_pitch_code_to_bucket()
    test_build_opponent_pitcher_arsenal_synthetic()
    test_compute_batter_score_phase_d()
    test_canonical_odds_team_key_sp_lookup_with_abbr()
    test_build_game_context_from_game_string()
    test_enrich_with_batter_score_columns()
    test_infer_player_kind()
    test_markets_for_kind_includes_stolen_bases()
    test_coerce_mlb_id_handles_nan()
    test_pitcher_rows_by_sp_nan_id_falls_back_to_name()
    print("All batter score tests passed.")
