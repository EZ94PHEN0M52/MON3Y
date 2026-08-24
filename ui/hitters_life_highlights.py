"""Conditional cell highlights for the Hitter's Life board."""

from __future__ import annotations

import pandas as pd

from ui.batter_score_highlights import (
    STYLE_VS_PITCHER_AVG,
    VS_PITCHER_AVG_THRESHOLD,
    vs_pitcher_style,
)


def season_avg_from_batting_average_cell(text) -> float | None:
    """Parse season AVG from ``Szn .310 · L5 ...``."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None

    cell = str(text).strip()
    if not cell or cell == "—":
        return None

    for part in cell.split("·"):
        part = part.strip()
        if not part.startswith("Szn "):
            continue

        token = part[4:].strip()
        if token == "—" or not token.startswith("."):
            return None

        try:
            return float(f"0{token}")
        except ValueError:
            return None

    return None


def batting_average_style(batting_average_text) -> str:
    season_avg = season_avg_from_batting_average_cell(batting_average_text)
    if season_avg is not None and season_avg > VS_PITCHER_AVG_THRESHOLD:
        return STYLE_VS_PITCHER_AVG
    return ""


def parse_total_bases_game_log(text) -> list[int]:
    """
    Parse a TB log cell like ``1 2 4 5 1`` or ``1 3 5 10 2`` into integers.

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


def total_bases_log_style(total_bases_text) -> str:
    values = parse_total_bases_game_log(total_bases_text)
    if not values:
        return ""
    if all(value != 0 for value in values):
        return STYLE_VS_PITCHER_AVG
    return ""


def style_hitters_life_board(full_df: pd.DataFrame):
    """Apply Vs pitcher, batting average, and TB log cell highlights."""
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

    vs_pitcher_styles = []
    batting_average_styles = []
    tb_log_styles = []

    for idx in range(len(meta_df)):
        row = meta_df.iloc[idx]
        vs_pitcher_styles.append(vs_pitcher_style(row.get("vs_pitcher")))
        batting_average_styles.append(
            batting_average_style(row.get("batting_average"))
        )
        tb_log_styles.append(total_bases_log_style(row.get("total_bases_log")))

    def _apply_row_styles(row):
        idx = row.name
        styles = [""] * len(row)
        columns = list(row.index)

        if vs_pitcher_styles[idx] and "vs_pitcher" in columns:
            styles[columns.index("vs_pitcher")] = vs_pitcher_styles[idx]

        if batting_average_styles[idx] and "batting_average" in columns:
            styles[columns.index("batting_average")] = batting_average_styles[idx]

        if tb_log_styles[idx] and "total_bases_log" in columns:
            styles[columns.index("total_bases_log")] = tb_log_styles[idx]

        return styles

    return display_df.style.apply(_apply_row_styles, axis=1)
