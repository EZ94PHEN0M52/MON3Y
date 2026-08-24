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


def test_hitters_life_player_link_shows_name_only():
    link = hitters_life_player_link("Aaron Judge")
    assert link.endswith("#Aaron Judge")
    assert "player=Aaron+Judge" in link or "player=Aaron%20Judge" in link


def test_sp_arsenal_column():
    from hitters_life_data import format_sp_arsenal_column, format_vs_pitcher_cell

    assert format_sp_arsenal_column(
        {
            "4-Seam Fastball": 0.55,
            "Slider": 0.30,
            "Changeup": 0.15,
        }
    ) == "4-Seam Fastball · Slider · Changeup"

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


def test_hitters_life_board_highlights():
    from ui.batter_score_highlights import (
        STYLE_VS_PITCHER_AVG,
        h2h_avg_from_vs_pitcher,
    )
    from ui.hitters_life_highlights import (
        batting_average_style,
        parse_total_bases_game_log,
        style_hitters_life_board,
        total_bases_log_style,
    )

    assert parse_total_bases_game_log("1 3 5 10 2") == [1, 3, 5, 10, 2]
    assert total_bases_log_style("1 2 4 5 1") == STYLE_VS_PITCHER_AVG
    assert total_bases_log_style("1 3 5 10 2") == STYLE_VS_PITCHER_AVG
    assert total_bases_log_style("1 2 0 4 1") == ""
    assert batting_average_style("Szn .310 · L5 .280 · L10 .270") == STYLE_VS_PITCHER_AVG
    assert batting_average_style("Szn .290 · L5 .310 · L10 .270") == ""

    assert h2h_avg_from_vs_pitcher("Starter · 4/10 .400\nᶠᵇ ˢˡ") == 0.400

    board = pd.DataFrame(
        [
            {
                "player": "Hot Bat",
                "player_link": "/?player=Hot+Bat#Hot Bat",
                "vs_pitcher": "Starter · 4/10 .400",
                "batting_average": "Szn .310 · L5 .290 · L10 .280",
                "total_bases_log": "1 2 4 5 1",
            }
        ]
    )
    html = style_hitters_life_board(board).to_html()
    assert STYLE_VS_PITCHER_AVG in html


if __name__ == "__main__":
    test_parse_rotowire_lineups_html()
    test_avg_from_games_windows()
    test_match_player_to_lineup_fuzzy()
    test_hitters_life_player_link_shows_name_only()
    test_sp_arsenal_column()
    test_aggregate_pitcher_arsenal_usage_detailed()
    test_hitters_life_board_highlights()
    print("OK")
