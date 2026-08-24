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


def test_min_pa_h2h_board_constant():
    from ui.batter_score_board import MIN_PA_H2H_BOARD

    assert MIN_PA_H2H_BOARD == 3


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


def test_build_opponent_pitcher_arsenal_detailed_synthetic():
    from pitch_matchup import (
        build_opponent_pitcher_arsenal,
        build_opponent_pitcher_arsenal_detailed,
    )

    statcast = pd.DataFrame(
        {
            "batter": [1, 1, 1, 1, 1, 1],
            "pitcher": [9, 9, 9, 9, 9, 9],
            "pitch_type": ["FF", "FF", "SI", "SL", "ST", "CH"],
            "pitch_name": [
                "4-Seam Fastball",
                "4-Seam Fastball",
                "Sinker",
                "Slider",
                "Sweeper",
                "Changeup",
            ],
            "type": ["X", "X", "X", "X", "X", "X"],
            "events": [
                "single",
                "field_out",
                "single",
                "field_out",
                "single",
                "field_out",
            ],
            "woba_value": [0.40, 0.10, 0.35, 0.05, 0.32, 0.08],
            "game_date": ["2026-08-01"] * 6,
        }
    )

    bucket_arsenal = build_opponent_pitcher_arsenal(statcast, batter_id=1, pitcher_id=9)
    detailed = build_opponent_pitcher_arsenal_detailed(
        statcast,
        batter_id=1,
        pitcher_id=9,
    )

    assert len(detailed) >= len(bucket_arsenal)
    assert any(item.pitch_type == "Sinker" for item in detailed)
    assert any(item.pitch_type == "Sweeper" for item in detailed)
    assert abs(sum(item.usage_pct for item in detailed) - 1.0) < 0.02


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


def test_lookup_opposing_sp_accepts_adjacent_schedule_date():
    from fetch_probables import lookup_opposing_sp

    probables = pd.DataFrame(
        {
            "game_date": ["2026-08-23"],
            "home_team": ["Colorado Rockies"],
            "away_team": ["Cleveland Guardians"],
            "home_sp_name": ["Gabriel Hughes"],
            "away_sp_name": ["Tanner Bibee"],
            "home_sp_id": [687312],
            "away_sp_id": [676440],
        }
    )

    sp_name, sp_id = lookup_opposing_sp(
        probables,
        "2026-08-22",
        "Colorado Rockies",
        "Cleveland Guardians",
        "CLE",
    )

    assert sp_name == "Gabriel Hughes"
    assert sp_id == 687312


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


def test_build_game_context_uses_eastern_schedule_date():
    from batter_score_data import build_game_context

    ctx = build_game_context(
        game="Cleveland Guardians @ Colorado Rockies",
        commence_time="2026-08-23T00:11:00Z",
        home_team="Colorado Rockies",
        away_team="Cleveland Guardians",
    )

    assert ctx is not None
    assert ctx["game_date"] == "2026-08-22"


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


def test_batter_score_validation_loader():
    import json
    import tempfile

    import batter_score_data as bsd

    validated_payload = {
        "validated": True,
        "sample_size": 500,
        "spearman_correlation": 0.22,
        "timestamp": "2026-08-19T00:00:00+00:00",
    }
    missing_payload = {"validated": False, "sample_size": 0}

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "batter_score_validation.json"
        original = bsd.BATTER_SCORE_VALIDATION_PATH

        try:
            bsd.BATTER_SCORE_VALIDATION_PATH = path
            bsd.clear_batter_score_validation_cache()
            assert bsd.is_batter_score_validated() is False

            path.write_text(json.dumps(validated_payload), encoding="utf-8")
            bsd.clear_batter_score_validation_cache()

            loaded = bsd.load_batter_score_validation()
            assert loaded["validated"] is True
            assert loaded["sample_size"] == 500
            assert bsd.is_batter_score_validated() is True

            path.write_text(json.dumps(missing_payload), encoding="utf-8")
            bsd.clear_batter_score_validation_cache()
            assert bsd.is_batter_score_validated() is False
        finally:
            bsd.BATTER_SCORE_VALIDATION_PATH = original
            bsd.clear_batter_score_validation_cache()


def test_score_batter_as_of_point_in_time():
    from batter_score_data import (
        actual_raw_points_on_date,
        score_batter_as_of,
    )

    rows = []
    for idx in range(12):
        rows.append(
            {
                "game_date": f"2026-08-{idx + 1:02d}",
                "player_name": "Backtest Batter",
                "hits": 1,
                "total_bases": 2,
                "walks": 1,
                "team": "AAA",
            }
        )

    frame = pd.DataFrame(rows)
    target = "2026-08-12"

    actual = actual_raw_points_on_date(frame, target)
    assert actual == 4.0  # hits(1) + tb(2) + bb(1)

    scored = score_batter_as_of(frame, target)
    assert scored is not None
    assert 0.0 <= scored.batter_score <= 100.0


def test_format_vs_pitcher_h2h_and_era_fallback():
    from batter_score import BatterScoreResult
    from ui.batter_score_board import MIN_PA_H2H_BOARD, _format_vs_pitcher

    h2h = BatterScoreResult(
        batter_name="Test",
        season_baseline=50.0,
        recent_form=50.0,
        matchup_grade=None,
        pitcher_form=50.0,
        batter_score=50.0,
        h2h_pa=MIN_PA_H2H_BOARD,
        h2h_hits=2,
        h2h_ab=7,
        opposing_sp_name="Ace",
        opposing_sp_era_l5=3.10,
    )
    assert _format_vs_pitcher(h2h) == "2/7 .286"

    below_board_min = BatterScoreResult(
        batter_name="Test",
        season_baseline=50.0,
        recent_form=50.0,
        matchup_grade=None,
        pitcher_form=50.0,
        batter_score=50.0,
        h2h_pa=MIN_PA_H2H_BOARD - 1,
        h2h_hits=2,
        h2h_ab=7,
        opposing_sp_name="Ace",
        opposing_sp_era_l5=2.75,
    )
    assert _format_vs_pitcher(below_board_min) == "SP ERA L5 2.75"

    era_only = BatterScoreResult(
        batter_name="Test",
        season_baseline=50.0,
        recent_form=50.0,
        matchup_grade=None,
        pitcher_form=50.0,
        batter_score=50.0,
        opposing_sp_name="Ace",
        opposing_sp_era_l5=2.75,
    )
    assert _format_vs_pitcher(era_only) == "SP ERA L5 2.75"

    zero_avg = BatterScoreResult(
        batter_name="Test",
        season_baseline=50.0,
        recent_form=50.0,
        matchup_grade=None,
        pitcher_form=50.0,
        batter_score=50.0,
        h2h_pa=MIN_PA_H2H_BOARD,
        h2h_hits=0,
        h2h_ab=3,
        opposing_sp_name="Ace",
        opposing_sp_era_l5=3.10,
    )
    assert _format_vs_pitcher(zero_avg) == "0/3 .000"


def test_compute_h2h_stats_kirk_vs_seymour():
    from pathlib import Path

    from batter_score_data import (
        _compute_h2h_stats,
        _load_merged_statcast,
        _merged_statcast_cache_key,
    )

    statcast = sorted(Path("data/raw").glob("statcast_*.parquet"))
    if not statcast:
        return

    kirk_id = 672386
    seymour_id = 693855

    # Scoring path still uses the latest statcast shard only.
    pa, avg_raw, hits, ab = _compute_h2h_stats(kirk_id, seymour_id)
    assert pa == 3
    assert hits == 0
    assert ab == 3
    assert avg_raw == 0.0

    # Board display merges all shards (includes 2025 Kirk vs Seymour).
    merged = _load_merged_statcast(_merged_statcast_cache_key())
    board_pa, _, board_hits, board_ab = _compute_h2h_stats(
        kirk_id,
        seymour_id,
        statcast=merged,
    )
    assert board_pa >= 7
    assert board_hits == 2
    assert board_ab == 7


def test_build_all_batter_score_df_includes_all_players():
    from unittest.mock import patch

    from batter_score import BatterScoreResult
    from ui.batter_score_board import build_all_batter_score_df

    props = pd.DataFrame(
        {
            "player": ["Alice", "Alice", "Bob", "Carol"],
            "market": ["batter_hits"] * 4,
            "line": [0.5, 1.5, 0.5, 0.5],
            "game": ["A @ B", "A @ B", "C @ D", "A @ B"],
            "commence_time": ["2026-08-20T23:05:00Z"] * 4,
            "batter_score": [72.0, 72.0, 88.5, 65.0],
            "batter_score_label": ["", "", "Partial", ""],
            "l5_l10_pct": ["60% / 50%", "40% / 30%", "70% / 60%", "—"],
        }
    )

    mock_result = BatterScoreResult(
        batter_name="Bob",
        season_baseline=50.0,
        recent_form=50.0,
        matchup_grade=None,
        pitcher_form=50.0,
        batter_score=88.5,
        opposing_sp_name="Starter",
    )

    with patch(
        "ui.batter_score_board.lookup_batter_score",
        return_value=mock_result,
    ):
        all_df = build_all_batter_score_df(props, version="v2")

    assert len(all_df) == 3
    assert list(all_df["batter_score_display"]) == [
        "88.5 (Partial)",
        "72.0",
        "65.0",
    ]
    assert set(all_df["_game"]) == {"A @ B", "C @ D"}


def test_build_all_batter_score_df_filters_by_game():
    from unittest.mock import patch

    from batter_score import BatterScoreResult
    from ui.batter_score_board import build_all_batter_score_df

    props = pd.DataFrame(
        {
            "player": ["Alice", "Bob"],
            "market": ["batter_hits", "batter_hits"],
            "line": [0.5, 0.5],
            "game": ["A @ B", "C @ D"],
            "commence_time": ["2026-08-20T23:05:00Z"] * 2,
            "batter_score": [80.0, 70.0],
            "batter_score_label": ["", ""],
            "l5_l10_pct": ["—", "—"],
        }
    )

    mock_result = BatterScoreResult(
        batter_name="Test",
        season_baseline=50.0,
        recent_form=50.0,
        matchup_grade=None,
        pitcher_form=50.0,
        batter_score=75.0,
    )

    with patch(
        "ui.batter_score_board.lookup_batter_score",
        return_value=mock_result,
    ):
        all_df = build_all_batter_score_df(props, version="v2")

    game_b = all_df[all_df["_game"] == "A @ B"]
    assert len(game_b) == 1
    assert game_b.iloc[0]["batter_score_display"] == "80.0"


def test_build_top_batter_score_df_ranks_unique_players():
    from unittest.mock import patch

    from batter_score import BatterScoreResult
    from ui.batter_score_board import build_top_batter_score_df

    props = pd.DataFrame(
        {
            "player": ["Alice", "Alice", "Bob", "Carol"],
            "market": ["batter_hits"] * 4,
            "line": [0.5, 1.5, 0.5, 0.5],
            "game": ["A @ B"] * 4,
            "commence_time": ["2026-08-20T23:05:00Z"] * 4,
            "batter_score": [72.0, 72.0, 88.5, 65.0],
            "batter_score_label": ["", "", "Partial", ""],
            "l5_l10_pct": ["60% / 50%", "40% / 30%", "70% / 60%", "—"],
        }
    )

    mock_result = BatterScoreResult(
        batter_name="Bob",
        season_baseline=50.0,
        recent_form=50.0,
        matchup_grade=None,
        pitcher_form=50.0,
        batter_score=88.5,
        opposing_sp_name="Starter",
    )

    with patch(
        "ui.batter_score_board.lookup_batter_score",
        return_value=mock_result,
    ):
        top = build_top_batter_score_df(props, version="v2")

    assert list(top["batter_score_display"]) == [
        "88.5 (Partial)",
        "72.0",
        "65.0",
    ]
    assert top.iloc[0]["opposing_sp"] == "Starter"
    assert top.iloc[0]["l5_l10_pct"] == "70% / 60%"


def test_style_batter_score_board_highlights_combo_row():
    from ui.batter_score_board import (
        STYLE_FANTASY_EQUAL,
        STYLE_FANTASY_LOWER,
        STYLE_L5_L10_GREEN,
        STYLE_L5_L10_YELLOW,
        STYLE_ROW_HIGHLIGHT,
        STYLE_VS_PITCHER_AVG,
        style_batter_score_board,
    )

    combo = pd.DataFrame(
        [
            {
                "player_link": "/player#Combo",
                "pp_fantasy_line": "5",
                "ud_fantasy_line": "4.5",
                "l5_l10_pct": "85% / 90%",
                "_pp_line": 5.0,
                "_ud_line": 4.5,
                "_l5_pct": 0.85,
                "_l10_pct": 0.90,
            }
        ]
    )
    combo_html = style_batter_score_board(combo).to_html()
    assert STYLE_FANTASY_LOWER in combo_html
    assert STYLE_L5_L10_GREEN in combo_html
    assert STYLE_ROW_HIGHLIGHT in combo_html

    equal = pd.DataFrame(
        [
            {
                "player_link": "/player#Equal",
                "pp_fantasy_line": "5",
                "ud_fantasy_line": "5",
                "l5_l10_pct": "85% / 70%",
                "_pp_line": 5.0,
                "_ud_line": 5.0,
                "_l5_pct": 0.85,
                "_l10_pct": 0.70,
            }
        ]
    )
    equal_html = style_batter_score_board(equal).to_html()
    assert STYLE_FANTASY_EQUAL in equal_html
    assert STYLE_L5_L10_YELLOW in equal_html
    assert STYLE_FANTASY_LOWER not in equal_html
    assert STYLE_ROW_HIGHLIGHT not in equal_html

    pp_lower = pd.DataFrame(
        [
            {
                "player_link": "/player#PPLower",
                "pp_fantasy_line": "4.5",
                "ud_fantasy_line": "5",
                "l5_l10_pct": "70% / 60%",
                "_pp_line": 4.5,
                "_ud_line": 5.0,
                "_l5_pct": 0.70,
                "_l10_pct": 0.60,
            }
        ]
    )
    pp_lower_html = style_batter_score_board(pp_lower).to_html()
    assert STYLE_FANTASY_LOWER in pp_lower_html
    assert STYLE_FANTASY_EQUAL not in pp_lower_html
    assert STYLE_ROW_HIGHLIGHT not in pp_lower_html

    hot_h2h = pd.DataFrame(
        [
            {
                "player_link": "/player#HotH2H",
                "vs_pitcher": "4/10 .400",
                "pp_fantasy_line": "5",
                "ud_fantasy_line": "5",
                "l5_l10_pct": "50% / 40%",
                "_pp_line": 5.0,
                "_ud_line": 5.0,
                "_l5_pct": 0.50,
                "_l10_pct": 0.40,
            }
        ]
    )
    hot_h2h_html = style_batter_score_board(hot_h2h).to_html()
    assert STYLE_VS_PITCHER_AVG in hot_h2h_html

    era_fallback = pd.DataFrame(
        [
            {
                "player_link": "/player#ERA",
                "vs_pitcher": "SP ERA L5 2.75",
                "pp_fantasy_line": "5",
                "ud_fantasy_line": "5",
                "l5_l10_pct": "50% / 40%",
                "_pp_line": 5.0,
                "_ud_line": 5.0,
                "_l5_pct": 0.50,
                "_l10_pct": 0.40,
            }
        ]
    )
    era_html = style_batter_score_board(era_fallback).to_html()
    assert STYLE_VS_PITCHER_AVG not in era_html


def test_batter_score_pick_card_highlights():
    from ui.batter_score_highlights import (
        STYLE_FANTASY_LOWER,
        STYLE_L5_L10_GREEN,
        STYLE_VS_PITCHER_AVG,
        format_batter_score_pick_details_html,
    )
    from ui.pick_builder import batter_score_row_to_pick

    row = pd.Series(
        {
            "player": "Hot Bat",
            "_game": "NYY @ BOS",
            "game_time": "7:05 PM ET",
            "opposing_sp": "Starter",
            "vs_pitcher": "4/10 .400",
            "pp_fantasy_line": "5",
            "ud_fantasy_line": "4.5",
            "l5_l10_pct": "85% / 90%",
            "_pp_line": 5.0,
            "_ud_line": 4.5,
            "_l5_pct": 0.85,
            "_l10_pct": 0.90,
            "batter_score_display": "88.5",
            "_batter_score": 88.5,
        }
    )
    pick = batter_score_row_to_pick(row)
    html = format_batter_score_pick_details_html(pick)

    assert STYLE_FANTASY_LOWER in html
    assert STYLE_L5_L10_GREEN in html
    assert STYLE_VS_PITCHER_AVG in html
    assert "4/10 .400" in html


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
    test_min_pa_h2h_board_constant()
    test_gated_full_score_requires_matchup_inputs()
    test_dedupe_best_prop_one_row_per_player_market()
    test_pitch_code_to_bucket()
    test_build_opponent_pitcher_arsenal_synthetic()
    test_build_opponent_pitcher_arsenal_detailed_synthetic()
    test_compute_batter_score_phase_d()
    test_canonical_odds_team_key_sp_lookup_with_abbr()
    test_lookup_opposing_sp_accepts_adjacent_schedule_date()
    test_build_game_context_from_game_string()
    test_build_game_context_uses_eastern_schedule_date()
    test_enrich_with_batter_score_columns()
    test_infer_player_kind()
    test_markets_for_kind_includes_stolen_bases()
    test_coerce_mlb_id_handles_nan()
    test_pitcher_rows_by_sp_nan_id_falls_back_to_name()
    test_batter_score_validation_loader()
    test_score_batter_as_of_point_in_time()
    test_format_vs_pitcher_h2h_and_era_fallback()
    test_compute_h2h_stats_kirk_vs_seymour()
    test_build_all_batter_score_df_includes_all_players()
    test_build_all_batter_score_df_filters_by_game()
    test_build_top_batter_score_df_ranks_unique_players()
    test_style_batter_score_board_highlights_combo_row()
    test_batter_score_pick_card_highlights()
    print("All batter score tests passed.")
