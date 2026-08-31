"""Tests for Hitter's Life board helpers."""

import pandas as pd

from fetch_rotowire_lineups import parse_rotowire_lineups_html
from hitters_life_data import (
    _avg_from_games,
    hitters_life_player_link,
    match_player_to_lineup,
)


def test_parse_rotowire_lineups_html():
    html = """
    <div class="col border-rb mb-20">
        <div class="border-tb heading-primary bg-cod-gray white pad-5-10">Today's Lineup</div>
        <ol class="list is-rankings pad-5-10">
            <li class="md-text"><a href="/baseball/player/aaron-judge-13923">Aaron Judge</a></li>
            <li class="md-text"><a href="/baseball/player/ben-rice-18769">Ben Rice</a></li>
        </ol>
    </div>
    <div class="col border-rb mb-20">
        <div class="border-tb heading-primary bg-cod-gray white pad-5-10">Default vs. RHP</div>
        <ol class="list is-rankings pad-5-10">
            <li class="md-text"><a href="/baseball/player/brice-turang-15415">Brice Turang</a></li>
            <li class="md-text"><a href="/baseball/player/jackson-chourio-16755">Jackson Chourio</a></li>
        </ol>
    </div>
    <div class="col border-rb mb-20">
        <div class="border-tb heading-primary bg-cod-gray white pad-5-10">Default vs. LHP</div>
        <ol class="list is-rankings pad-5-10">
            <li class="md-text"><a href="/baseball/player/jackson-chourio-16755">Jackson Chourio</a></li>
        </ol>
    </div>
    """
    lineups = parse_rotowire_lineups_html(html)
    assert lineups["OFFICIAL"] == ["Aaron Judge", "Ben Rice"]
    assert lineups["RHP"] == ["Brice Turang", "Jackson Chourio"]
    assert lineups["LHP"] == ["Jackson Chourio"]


def test_avg_from_games_windows():
    games = (
        ("2026-04-01", 1, 4),
        ("2026-04-02", 2, 4),
        ("2026-04-03", 0, 3),
        ("2026-04-04", 3, 5),
        ("2026-04-05", 1, 4),
        ("2026-04-06", 2, 5),
    )
    season = _avg_from_games(games)
    l5 = _avg_from_games(games, window=5)
    assert season is not None
    assert abs(season - (9 / 25)) < 1e-6
    assert l5 is not None
    assert abs(l5 - (8 / 21)) < 1e-6


def test_match_player_to_lineup_fuzzy():
    lineup = ["Jackson Chourio", "Brice Turang"]
    assert match_player_to_lineup("J. Chourio", lineup) is False
    assert match_player_to_lineup("Jackson Chourio", lineup) is True


def test_validate_official_lineup():
    from fetch_rotowire_lineups import (
        merge_official_lineups,
        validate_official_lineup,
        lineup_for_team,
        OFFICIAL_LINEUP_HAND,
    )

    names = [f"Player {i}" for i in range(9)]
    ok, msg = validate_official_lineup(names)
    assert ok is True
    assert msg == "ok"

    ok, msg = validate_official_lineup(names[:5], min_players=8)
    assert ok is False
    assert "need at least 8" in msg

    ok, msg = validate_official_lineup(
        names,
        slate_players={"player1", "player2"},
        min_slate_overlap=3,
    )
    assert ok is False

    merged = merge_official_lineups(
        pd.DataFrame(),
        {"NYY": names},
    )
    lineup, source = lineup_for_team(merged, "NYY", "R")
    assert source == "official"
    assert len(lineup) == 9
    assert merged["vs_hand"].eq(OFFICIAL_LINEUP_HAND).all()


def test_hitters_life_player_link_shows_name_only():
    link = hitters_life_player_link("Aaron Judge")
    assert link.endswith("#Aaron Judge")
    assert "player=Aaron+Judge" in link or "player=Aaron%20Judge" in link


def test_sp_arsenal_column():
    from hitters_life_data import (
        build_vs_pitcher_fields,
        format_sp_arsenal_column,
        format_vs_pitcher_cell,
    )

    assert format_sp_arsenal_column(
        {
            "4-Seam Fastball": 0.55,
            "Slider": 0.30,
            "Changeup": 0.15,
        }
    ) == "4-Seam Fastball · Slider · Changeup"

    fields = build_vs_pitcher_fields(
        "Player",
        "v2",
        None,
        None,
        sp_display="Cole (R)",
    )
    assert fields["opposing_sp"] == "Cole (R)"
    assert fields["h2h_avg"] is None

    cell = format_vs_pitcher_cell(
        "Player",
        "v2",
        None,
        None,
        sp_display="Cole (R)",
    )
    assert cell == "Cole (R)"
    assert "\n" not in cell


def test_aggregate_pitcher_arsenal_usage_detailed():
    import pandas as pd

    from pitch_matchup import aggregate_pitcher_arsenal_usage_detailed

    statcast = pd.DataFrame(
        {
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
            "game_date": ["2026-04-01"] * 6,
        }
    )
    usage = aggregate_pitcher_arsenal_usage_detailed(statcast, 9, last_n_starts=5)
    assert "4-Seam Fastball" in usage
    assert "Sinker" in usage
    assert "Sweeper" in usage
    assert "Fastball" not in usage
    assert abs(sum(usage.values()) - 1.0) < 0.01


def test_format_h2h_avg_display():
    from hitters_life_data import format_h2h_avg_display

    assert format_h2h_avg_display(0.400, hits=4, ab=10) == "4/10 .400"
    assert format_h2h_avg_display(None) == "—"
    assert format_h2h_avg_display(0.250, hits=1, ab=4) == "1/4 .250"


def test_build_hitters_life_row_includes_batter_score_v1():
    from unittest.mock import MagicMock, patch

    from hitters_life_data import build_hitters_life_row

    mock_result = MagicMock(
        opposing_sp_name=None,
        batter_score=68.4,
        partial_label="",
    )
    mock_result_v2 = MagicMock(
        batter_score=72.1,
        partial_label="Full",
    )
    row = pd.Series(
        {
            "player": "Test Hitter",
            "game": "Away @ Home",
            "commence_time": "2026-08-24T19:05:00Z",
            "home_team": "Home",
            "away_team": "Away",
        }
    )

    with patch(
        "batter_score_data.lookup_batter_score",
        return_value=mock_result,
    ), patch(
        "batter_score_data.lookup_batter_score_v2",
        return_value=mock_result_v2,
    ), patch(
        "hitters_life_data.build_vs_pitcher_fields",
        return_value={
            "opposing_sp": "—",
            "h2h_avg": None,
            "h2h_hits": None,
            "h2h_ab": None,
            "sp_era_l5": None,
        },
    ), patch(
        "hitters_life_data.lookup_pitch_bucket_woba",
        return_value=None,
    ), patch(
        "hitters_life_data.lookup_arsenal_weighted_woba",
        return_value=None,
    ), patch(
        "hitters_life_data.lookup_sp_arsenal_usage",
        return_value={},
    ), patch(
        "hitters_life_data.format_batting_average_column",
        return_value="Szn .280 · L5 .300 · L10 .290",
    ), patch(
        "hitters_life_data.format_total_bases_game_log",
        return_value="1 2 0 1 3",
    ), patch(
        "hitters_life_data._lookup_batter_team_abbr",
        return_value="NYY",
    ):
        built = build_hitters_life_row(row, "v2", pitch_bucket="Fastball")

    assert built["batter_score_v1_display"] == "68.4"
    assert built["batter_score_v2_display"] == "72.1 (Full)"
    assert built["_batter_score"] == 68.4


def test_format_total_bases_game_log():
    from hitters_life_data import format_total_bases_game_log

    log = format_total_bases_game_log("Yordan Alvarez", version="v2", n=5)
    assert log != "—"
    assert ":" not in log
    tokens = log.split()
    assert len(tokens) == 5
    assert tokens[0] == "1"
    assert tokens[-1] == "1"


def test_hitters_life_board_highlights():
    from ui.batter_score_highlights import STYLE_VS_PITCHER_AVG
    from ui.hitters_life_highlights import (
        STYLE_BAT_AVG_GREEN,
        STYLE_BAT_AVG_ORANGE,
        STYLE_BAT_AVG_YELLOW,
        STYLE_TB_LOG_BLUE,
        STYLE_TB_LOG_ORANGE,
        STYLE_TB_LOG_RED,
        STYLE_TB_LOG_YELLOW,
        batting_average_style,
        h2h_avg_style,
        parse_total_bases_game_log,
        style_hitters_life_board,
        total_bases_log_style,
    )

    assert parse_total_bases_game_log("1 3 5 10 2") == [1, 3, 5, 10, 2]
    assert total_bases_log_style("1 2 4 5 1") == STYLE_VS_PITCHER_AVG
    assert total_bases_log_style("1 3 5 10 2") == STYLE_VS_PITCHER_AVG
    assert total_bases_log_style("1 2 0 4 1") == ""
    assert total_bases_log_style("0 2 4 5 1") == STYLE_TB_LOG_ORANGE
    assert total_bases_log_style("2 1 1 1 1") == STYLE_TB_LOG_YELLOW
    assert total_bases_log_style("2 2 0 1 0") == ""
    assert total_bases_log_style("0 0 1 2 3") == ""
    assert total_bases_log_style("4 3 2 1 2") == STYLE_TB_LOG_BLUE
    assert total_bases_log_style("2 2 2 1 1") == STYLE_TB_LOG_BLUE
    assert total_bases_log_style("2 2 0 1 1") == STYLE_TB_LOG_YELLOW
    assert total_bases_log_style("1 0 4 2 1") == ""
    assert total_bases_log_style("6 0 2 1 5") == STYLE_TB_LOG_ORANGE
    assert total_bases_log_style("0 1 3 3 6") == STYLE_TB_LOG_ORANGE
    assert total_bases_log_style("3 3 1 1 1") == STYLE_VS_PITCHER_AVG
    assert total_bases_log_style("4 3 3 2 1") == STYLE_TB_LOG_RED
    assert batting_average_style("Szn .310 · L5 .280 · L10 .270") == STYLE_BAT_AVG_YELLOW
    assert batting_average_style("Szn .290 · L5 .310 · L10 .270") == STYLE_BAT_AVG_ORANGE
    assert batting_average_style("Szn .290 · L5 .260 · L10 .295") == STYLE_BAT_AVG_GREEN
    assert batting_average_style("Szn .290 · L5 .240 · L10 .295") == ""
    assert batting_average_style("Szn .290 · L5 .300 · L10 .295") == STYLE_BAT_AVG_GREEN
    assert h2h_avg_style(0.400) == STYLE_VS_PITCHER_AVG
    assert h2h_avg_style(0.250) == ""

    board = pd.DataFrame(
        [
            {
                "player": "Hot Bat",
                "player_link": "/?player=Hot+Bat#Hot Bat",
                "opposing_sp": "Starter (R)",
                "h2h_avg": "4/10 .400",
                "_h2h_avg": 0.400,
                "batting_average": "Szn .310 · L5 .290 · L10 .280",
                "total_bases_log": "1 2 4 5 1",
            }
        ]
    )
    html = style_hitters_life_board(board).to_html()
    assert STYLE_VS_PITCHER_AVG in html

    red_board = pd.DataFrame(
        [
            {
                "player": "Slugger",
                "player_link": "/?player=Slugger#Slugger",
                "opposing_sp": "Starter (R)",
                "h2h_avg": "—",
                "batting_average": "Szn .290 · L5 .310 · L10 .270",
                "total_bases_log": "4 3 3 2 1",
            }
        ]
    )
    red_html = style_hitters_life_board(red_board).to_html()
    assert STYLE_TB_LOG_RED in red_html
    assert red_html.count(STYLE_TB_LOG_RED) >= 2


if __name__ == "__main__":
    test_parse_rotowire_lineups_html()
    test_avg_from_games_windows()
    test_match_player_to_lineup_fuzzy()
    test_hitters_life_player_link_shows_name_only()
    test_sp_arsenal_column()
    test_aggregate_pitcher_arsenal_usage_detailed()
    test_format_h2h_avg_display()
    test_format_total_bases_game_log()
    test_build_hitters_life_row_includes_batter_score_v1()
    test_hitters_life_board_highlights()
    print("OK")
