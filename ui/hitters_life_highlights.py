"""Conditional cell highlights for the Hitter's Life board."""

from __future__ import annotations

import pandas as pd

from ui.batter_score_highlights import (
    STYLE_FANTASY_EQUAL,
    STYLE_FANTASY_LOWER,
    STYLE_L5_L10_YELLOW,
    STYLE_VS_PITCHER_AVG,
    VS_PITCHER_AVG_THRESHOLD,
)

L5_AVG_HOT_THRESHOLD = 0.299
L10_AVG_WARM_THRESHOLD = 0.290
L5_AVG_WARM_THRESHOLD = 0.250

STYLE_BAT_AVG_GREEN = STYLE_VS_PITCHER_AVG
STYLE_BAT_AVG_ORANGE = STYLE_FANTASY_LOWER
STYLE_BAT_AVG_YELLOW = STYLE_L5_L10_YELLOW

STYLE_TB_LOG_BLUE = STYLE_FANTASY_EQUAL
STYLE_TB_LOG_ORANGE = STYLE_FANTASY_LOWER
STYLE_TB_LOG_YELLOW = STYLE_L5_L10_YELLOW
STYLE_TB_LOG_RED = "background-color: rgba(244, 67, 54, 0.55)"

TB_LOG_COLOR_LEGEND = (
    (STYLE_TB_LOG_BLUE, "soarer", "2+ TB in the three most recent games"),
    (
        STYLE_VS_PITCHER_AVG,
        "money",
        "no zero in L5; at least two games with 2+ TB",
    ),
    (
        STYLE_TB_LOG_ORANGE,
        "hot",
        "exactly one zero in L5; at least three games with 2+ TB",
    ),
    (
        STYLE_TB_LOG_YELLOW,
        "warm",
        "last two games mix 1 TB and 2+ TB, or one zero with only two 2+ TB "
        "games both in the most recent pair",
    ),
)


def _batting_avg_window_from_cell(text, label: str) -> float | None:
    """Parse ``L5 .280`` / ``L10 .290`` / ``Szn .310`` from a batting AVG cell."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None

    cell = str(text).strip()
    if not cell or cell == "—":
        return None

    prefix = f"{label} "
    for part in cell.split("·"):
        part = part.strip()
        if not part.startswith(prefix):
            continue

        token = part[len(prefix):].strip()
        if token == "—" or not token.startswith("."):
            return None

        try:
            return float(f"0{token}")
        except ValueError:
            return None

    return None


def season_avg_from_batting_average_cell(text) -> float | None:
    """Parse season AVG from ``Szn .310 · L5 ...``."""
    return _batting_avg_window_from_cell(text, "Szn")


def l5_avg_from_batting_average_cell(text) -> float | None:
    return _batting_avg_window_from_cell(text, "L5")


def l10_avg_from_batting_average_cell(text) -> float | None:
    return _batting_avg_window_from_cell(text, "L10")


def _batting_avg_warm_combo(l5_avg: float | None, l10_avg: float | None) -> bool:
    return (
        l5_avg is not None
        and l10_avg is not None
        and l10_avg > L10_AVG_WARM_THRESHOLD
        and l5_avg > L5_AVG_WARM_THRESHOLD
    )


def _batting_avg_hot_l5(l5_avg: float | None) -> bool:
    return l5_avg is not None and l5_avg > L5_AVG_HOT_THRESHOLD


def batting_average_style(batting_average_text) -> str:
    """
    Batting average column colors (priority: green > orange > yellow).

    - Green: L10 > .290 and L5 > .250
    - Orange: L5 > .299
    - Yellow: season > .300 when neither rolling rule applies
    """
    season_avg = season_avg_from_batting_average_cell(batting_average_text)
    l5_avg = l5_avg_from_batting_average_cell(batting_average_text)
    l10_avg = l10_avg_from_batting_average_cell(batting_average_text)

    warm_combo = _batting_avg_warm_combo(l5_avg, l10_avg)
    hot_l5 = _batting_avg_hot_l5(l5_avg)

    if warm_combo:
        return STYLE_BAT_AVG_GREEN
    if hot_l5:
        return STYLE_BAT_AVG_ORANGE
    if season_avg is not None and season_avg > VS_PITCHER_AVG_THRESHOLD:
        return STYLE_BAT_AVG_YELLOW
    return ""


def batting_average_has_board_highlight(batting_average_text) -> bool:
    """True when the batting AVG cell would get green, orange, or yellow."""
    return bool(batting_average_style(batting_average_text))


def batting_average_highlight_rank(batting_average_text) -> int:
    """
    Sort rank for batting-average highlights (higher = stronger).

    Matches column color priority: green (2) > orange (1) > yellow (0).
    """
    style = batting_average_style(batting_average_text)
    if style == STYLE_BAT_AVG_GREEN:
        return 2
    if style == STYLE_BAT_AVG_ORANGE:
        return 1
    if style == STYLE_BAT_AVG_YELLOW:
        return 0
    return -1


def parse_total_bases_game_log(text) -> list[int]:
    """
    Parse a TB log cell like ``1 3 5 10 2`` into integers.

    Each space-separated token is one game total (``10`` is one value, not 1 and 0).
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return []

    cell = str(text).strip()
    if not cell or cell == "—":
        return []

    values: list[int] = []
    for token in cell.split():
        try:
            values.append(int(token))
        except ValueError:
            return []
    return values


def tb_log_is_super_rare(values: list[int]) -> bool:
    """Three+ games with 3+ TB and the most recent game has 2+ TB."""
    if not values or values[0] < 2:
        return False
    return sum(1 for value in values if value >= 3) >= 3


def _matches_tb_log_blue(values: list[int]) -> bool:
    return (
        len(values) >= 3
        and values[0] >= 2
        and values[1] >= 2
        and values[2] >= 2
    )


def _matches_tb_log_green(values: list[int]) -> bool:
    return (
        all(value != 0 for value in values)
        and sum(1 for value in values if value >= 2) >= 2
    )


def _matches_tb_log_orange(values: list[int]) -> bool:
    if sum(1 for value in values if value == 0) != 1:
        return False

    return sum(1 for value in values if value >= 2) >= 3


def _matches_tb_log_yellow(values: list[int]) -> bool:
    if len(values) < 2:
        return False

    # Path 1: most recent two mix exactly 1 TB and 2+ TB (e.g. 2 1 1 1 1).
    recent_two = (values[0], values[1])
    low = min(recent_two)
    high = max(recent_two)
    if low >= 1 and high >= 2 and low < 2:
        return True

    # Path 2: one off game, only two 2+ TB in L5, both in the last two
    # (e.g. 2 2 0 1 1). Stops at two 2+ games so orange (three+) wins first.
    two_plus = sum(1 for value in values if value >= 2)
    return (
        sum(1 for value in values if value == 0) == 1
        and two_plus == 2
        and values[0] >= 2
        and values[1] >= 2
    )


def total_bases_log_style(total_bases_text) -> str:
    """
    TB log (L5, left = most recent). Priority: blue > green > orange > yellow.

    Super-rare red (3+ games with 3+ TB and most recent 2+ TB) overrides all
    other TB colors and is omitted from the legend.
    """
    values = parse_total_bases_game_log(total_bases_text)
    if not values:
        return ""

    if tb_log_is_super_rare(values):
        return STYLE_TB_LOG_RED

    if _matches_tb_log_blue(values):
        return STYLE_TB_LOG_BLUE
    if _matches_tb_log_green(values):
        return STYLE_VS_PITCHER_AVG
    if _matches_tb_log_orange(values):
        return STYLE_TB_LOG_ORANGE
    if _matches_tb_log_yellow(values):
        return STYLE_TB_LOG_YELLOW

    return ""


def tb_log_is_blue(total_bases_text) -> bool:
    """True when the TB log would get the blue soarer highlight."""
    return total_bases_log_style(total_bases_text) == STYLE_TB_LOG_BLUE


def hot_batter_tiebreak_key(
    batting_average_text,
    total_bases_text=None,
) -> tuple[int, int]:
    """
    Independent highlight ranks for hot-batter score ties.

    TB-log colors and batting-average colors stay separate:
    - first key: blue TB soarer (1) or not (0)
    - second key: batting AVG green (2) > orange (1) > yellow (0)
    """
    return (
        1 if tb_log_is_blue(total_bases_text) else 0,
        batting_average_highlight_rank(batting_average_text),
    )


def render_tb_log_color_legend() -> None:
    """Render TB per game (L5) color key for the batting average board."""
    import streamlit as st

    chips = []
    for style, label, _help in TB_LOG_COLOR_LEGEND:
        chips.append(
            f'<span style="{style}; padding: 2px 8px; border-radius: 4px; '
            f'margin-right: 6px; font-size: 0.85em;">{label}</span>'
        )
    st.markdown(
        "**TB per game (L5) colors:** " + " ".join(chips),
        unsafe_allow_html=True,
    )


def h2h_avg_style(h2h_avg) -> str:
    if h2h_avg is None or (isinstance(h2h_avg, float) and pd.isna(h2h_avg)):
        return ""
    try:
        value = float(h2h_avg)
    except (TypeError, ValueError):
        return ""
    if value > VS_PITCHER_AVG_THRESHOLD:
        return STYLE_VS_PITCHER_AVG
    return ""


def style_hitters_life_board(full_df: pd.DataFrame):
    """Apply H2H AVG, batting average, and TB log cell highlights."""
    if full_df.empty:
        return full_df.drop(
            columns=[
                col
                for col in full_df.columns
                if col.startswith("_") or col == "player"
            ],
            errors="ignore",
        )

    display_df = full_df.drop(
        columns=[
            col
            for col in full_df.columns
            if col.startswith("_") or col == "player"
        ],
        errors="ignore",
    ).reset_index(drop=True)

    meta_df = full_df.reset_index(drop=True)

    h2h_avg_styles = []
    batting_average_styles = []
    tb_log_styles = []
    player_link_styles = []

    for idx in range(len(meta_df)):
        row = meta_df.iloc[idx]
        h2h_avg_styles.append(
            h2h_avg_style(row.get("_h2h_avg", row.get("h2h_avg")))
        )
        batting_average_styles.append(
            batting_average_style(row.get("batting_average"))
        )
        tb_values = parse_total_bases_game_log(row.get("total_bases_log"))
        tb_log_styles.append(total_bases_log_style(row.get("total_bases_log")))
        player_link_styles.append(
            STYLE_TB_LOG_RED
            if tb_log_is_super_rare(tb_values)
            else ""
        )

    def _apply_row_styles(row):
        idx = row.name
        styles = [""] * len(row)
        columns = list(row.index)

        if h2h_avg_styles[idx] and "h2h_avg" in columns:
            styles[columns.index("h2h_avg")] = h2h_avg_styles[idx]

        if batting_average_styles[idx] and "batting_average" in columns:
            styles[columns.index("batting_average")] = batting_average_styles[idx]

        if tb_log_styles[idx] and "total_bases_log" in columns:
            styles[columns.index("total_bases_log")] = tb_log_styles[idx]

        if player_link_styles[idx] and "player_link" in columns:
            styles[columns.index("player_link")] = player_link_styles[idx]

        return styles

    return display_df.style.apply(_apply_row_styles, axis=1)
