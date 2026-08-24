"""Top Batter Score section for the main prop board."""

import numpy as np
import pandas as pd
import streamlit as st

from batter_score_data import (
    build_game_context,
    is_batter_score_validated,
    lookup_batter_score,
    lookup_batter_score_v2,
    lookup_h2h_board_stats,
)
from ui.batter_score import format_batter_score_display
from ui.batter_score_highlights import (
    HIT_RATE_THRESHOLD,
    STYLE_FANTASY_EQUAL,
    STYLE_FANTASY_LOWER,
    STYLE_L5_L10_GREEN,
    STYLE_L5_L10_YELLOW,
    STYLE_ROW_HIGHLIGHT,
    STYLE_VS_PITCHER_AVG,
    fantasy_cell_styles,
    join_styles,
    l5_l10_style,
    vs_pitcher_style,
)
from ui.pick_builder import render_batter_score_add_controls
from ui.formatting import format_game_time, format_name_with_hand, player_path
from ui.glossary import GLOSSARY
from ui.player_stats import (
    BATTER_MARKETS,
    _format_l5_l10_pct,
    format_prizepicks_fantasy_line,
    format_underdog_fantasy_line,
    lookup_batter_hand,
    lookup_pitcher_hand,
    lookup_prizepicks_fantasy_line,
    lookup_underdog_fantasy_line,
    rolling_pp_fantasy_over_rates,
)

# Board "Vs pitcher" column — lower bar than MIN_PA_H2H (10) used in scoring.
MIN_PA_H2H_BOARD = 3


def _resolve_l5_l10_pct(row, version: str, *, pp_line=None) -> tuple[str, float, float]:
    """Display text plus raw L5/L10 over-rates (0–1) for styling."""
    if pp_line is None:
        pp_line = lookup_prizepicks_fantasy_line(row["player"])
    if pp_line is not None:
        l5_pct, l10_pct = rolling_pp_fantasy_over_rates(
            row["player"],
            pp_line,
            version=version,
        )
        return _format_l5_l10_pct(l5_pct, l10_pct), l5_pct, l10_pct

    l5_pct = pd.to_numeric(row.get("l5_pct"), errors="coerce")
    l10_pct = pd.to_numeric(row.get("l10_pct"), errors="coerce")
    fallback = row.get("l5_l10_pct")
    if pd.isna(l5_pct) and pd.isna(l10_pct):
        if fallback is None or (isinstance(fallback, float) and pd.isna(fallback)):
            return "—", np.nan, np.nan
        return fallback, np.nan, np.nan

    return _format_l5_l10_pct(l5_pct, l10_pct), l5_pct, l10_pct


def style_batter_score_board(full_df: pd.DataFrame):
    """Color PP/UD fantasy, Vs pitcher H2H AVG, L5-L10 cells, and combo rows."""
    meta_cols = ["_pp_line", "_ud_line", "_l5_pct", "_l10_pct"]
    if full_df.empty or any(col not in full_df.columns for col in meta_cols):
        return full_df.drop(
            columns=[
                col
                for col in full_df.columns
                if col.startswith("_") or col in {"player", "batter_score_label"}
            ],
            errors="ignore",
        )

    display_df = full_df.drop(
        columns=[
            col
            for col in full_df.columns
            if col.startswith("_") or col in {"player", "batter_score_label"}
        ],
        errors="ignore",
    )
    meta_df = full_df[meta_cols].reset_index(drop=True)
    display_df = display_df.reset_index(drop=True)
    vs_pitcher_cells = (
        display_df["vs_pitcher"]
        if "vs_pitcher" in display_df.columns
        else pd.Series([None] * len(display_df))
    )

    pp_fantasy_styles = []
    ud_fantasy_styles = []
    vs_pitcher_styles = []
    l5_l10_styles = []
    row_highlight = []

    for idx, meta in meta_df.iterrows():
        pp_line = meta["_pp_line"]
        ud_line = meta["_ud_line"]
        l5_pct = meta["_l5_pct"]
        l10_pct = meta["_l10_pct"]

        pp_style, ud_style, ud_is_lower = fantasy_cell_styles(pp_line, ud_line)
        pp_fantasy_styles.append(pp_style)
        ud_fantasy_styles.append(ud_style)

        vs_pitcher_styles.append(
            vs_pitcher_style(vs_pitcher_cells.iloc[idx])
        )

        l5_l10_styles.append(l5_l10_style(l5_pct, l10_pct))

        l5_hit = not pd.isna(l5_pct) and float(l5_pct) >= HIT_RATE_THRESHOLD
        l10_hit = not pd.isna(l10_pct) and float(l10_pct) >= HIT_RATE_THRESHOLD
        row_highlight.append(ud_is_lower and l5_hit and l10_hit)

    def _apply_row_styles(row):
        idx = row.name
        styles = [""] * len(row)
        columns = list(row.index)
        border = STYLE_ROW_HIGHLIGHT if row_highlight[idx] else ""

        if pp_fantasy_styles[idx] and "pp_fantasy_line" in columns:
            styles[columns.index("pp_fantasy_line")] = join_styles(
                pp_fantasy_styles[idx],
                border,
            )

        if ud_fantasy_styles[idx] and "ud_fantasy_line" in columns:
            styles[columns.index("ud_fantasy_line")] = join_styles(
                ud_fantasy_styles[idx],
                border,
            )

        if vs_pitcher_styles[idx] and "vs_pitcher" in columns:
            styles[columns.index("vs_pitcher")] = vs_pitcher_styles[idx]

        if l5_l10_styles[idx] and "l5_l10_pct" in columns:
            styles[columns.index("l5_l10_pct")] = join_styles(
                l5_l10_styles[idx],
                border,
            )

        if border:
            for col_idx, col in enumerate(columns):
                if styles[col_idx]:
                    continue
                styles[col_idx] = border

        return styles

    return display_df.style.apply(_apply_row_styles, axis=1)


def _render_batter_score_dataframe(df: pd.DataFrame, *, height: int):
    st.dataframe(
        style_batter_score_board(df),
        hide_index=True,
        height=height,
        column_config=_batter_score_table_column_config(),
    )


def _format_game_time(row) -> str:
    return format_game_time(row.get("game"), row.get("commence_time"))


def _format_batting_avg(hits: int, ab: int) -> str:
    """Format hits/AB and AVG like ``2/7 .286``."""
    avg_text = f"{hits / ab:.3f}".removeprefix("0")
    return f"{hits}/{ab} {avg_text}"


def _format_vs_pitcher(
    result,
    *,
    h2h_pa: int | None = None,
    h2h_hits: int | None = None,
    h2h_ab: int | None = None,
) -> str:
    """Career AVG vs opposing SP when PA threshold met; else SP ERA L5 if known."""
    if result is None:
        return "—"

    pa = h2h_pa if h2h_pa is not None else result.h2h_pa
    hits = h2h_hits if h2h_hits is not None else result.h2h_hits
    ab = h2h_ab if h2h_ab is not None else result.h2h_ab

    if pa is not None and pa >= MIN_PA_H2H_BOARD:
        if ab is not None and ab > 0:
            hit_count = hits if hits is not None else 0
            return _format_batting_avg(hit_count, ab)

    if result.opposing_sp_era_l5 is not None:
        return f"SP ERA L5 {result.opposing_sp_era_l5:.2f}"

    return "—"


def _row_game_context(row) -> dict | None:
    return build_game_context(
        game=row.get("game"),
        commence_time=row.get("commence_time"),
        home_team=row.get("home_team"),
        away_team=row.get("away_team"),
    )


def build_top_batter_score_df(
    df: pd.DataFrame,
    version: str,
    *,
    markets=None,
) -> pd.DataFrame:
    """
    Top 10 unique batters by batter_score from today's slate.

    Uses enrichment columns from load_board_data(); lookup_batter_score fills
    opposing SP and H2H fields not stored on the predictions dataframe.
    """
    ranked = _prepare_batter_score_slate(df, markets=markets)
    if ranked.empty:
        return pd.DataFrame()

    ranked = ranked.head(10)
    result = pd.DataFrame(_rows_from_batter_slate(ranked, version))
    return result.drop(columns=["_game"], errors="ignore")


def _build_batter_score_row(row, version: str) -> dict:
    game_context = _row_game_context(row)
    result = lookup_batter_score(
        row["player"],
        version=version,
        game_context=game_context,
    )
    result_v2 = lookup_batter_score_v2(
        row["player"],
        version=version,
        game_context=game_context,
    )
    h2h_pa, h2h_hits, h2h_ab = lookup_h2h_board_stats(
        row["player"],
        version=version,
        game_context=game_context,
    )

    opposing = "TBD"
    if result and result.opposing_sp_name:
        sp_hand = lookup_pitcher_hand(
            result.opposing_sp_name,
            version=version,
        )
        opposing = format_name_with_hand(
            result.opposing_sp_name,
            sp_hand,
        )

    pp_line = lookup_prizepicks_fantasy_line(row["player"])
    ud_line = lookup_underdog_fantasy_line(row["player"])
    l5_l10, l5_pct, l10_pct = _resolve_l5_l10_pct(row, version, pp_line=pp_line)
    vs_pitcher = _format_vs_pitcher(
        result,
        h2h_pa=h2h_pa,
        h2h_hits=h2h_hits,
        h2h_ab=h2h_ab,
    )

    return {
        "player": row["player"],
        "player_link": player_path(
            row["player"],
            hand=lookup_batter_hand(row["player"], version=version),
        ),
        "game_time": _format_game_time(row),
        "opposing_sp": opposing,
        "vs_pitcher": vs_pitcher,
        "pp_fantasy_line": format_prizepicks_fantasy_line(row["player"]),
        "ud_fantasy_line": format_underdog_fantasy_line(row["player"]),
        "l5_l10_pct": l5_l10,
        "batter_score_display": format_batter_score_display(
            row["batter_score"],
            row.get("batter_score_label") or "",
        ),
        "batter_score_v2_display": format_batter_score_display(
            result_v2.batter_score if result_v2 else None,
            (result_v2.partial_label if result_v2 else "") or "",
        ),
        "batter_score_label": row.get("batter_score_label") or "",
        "_game": row.get("game") or "",
        "_batter_score": row["batter_score"],
        "_pp_line": pp_line,
        "_ud_line": ud_line,
        "_l5_pct": l5_pct,
        "_l10_pct": l10_pct,
    }


def render_top_batter_scores(
    df: pd.DataFrame,
    key_prefix: str,
    *,
    version: str = "v2",
):
    """Render Top 10 Batter Score table at the bottom of the board."""
    markets = st.session_state.get(f"{key_prefix}_markets", [])
    top_df = build_top_batter_score_df(
        df,
        version,
        markets=markets or None,
    )

    st.markdown("##### Top 10 batter score")
    caption = (
        "Highest Batter Score among batters on today's slate "
        "(respects Market type filter; independent of Edge / EV filters). "
        "**PP fantasy** and **UD fantasy** are each hitter's posted DFS "
        "fantasy score line (PrizePicks and Underdog). "
        "L5 / L10 % is the over-rate vs the PP line when available; otherwise "
        "falls back to the market line on that player's top prop row. "
        "**Orange** PP/UD fantasy = lower posted line between the two books; "
        "**sky blue** = same line on both; "
        "**light green** Vs pitcher = career H2H batting average above .300; "
        "**yellow/green** L5/L10 = L5 ≥ 80% with L10 below/above 80%; "
        "**red outline** = UD lower + L5/L10 green. "
        "**Batter score v2** uses Savant pitch-type matchup (Sinker, Sweeper, etc.) "
        "instead of five pitch buckets. "
    )
    if is_batter_score_validated():
        caption += GLOSSARY["batter_score_validated"]
    else:
        caption += "Batter Score — validation pending."
    st.caption(caption)

    if top_df.empty:
        st.caption("No batter scores available for the current slate.")
        return

    _render_batter_score_dataframe(
        top_df,
        height=min(42 * len(top_df) + 38, 480),
    )


def _batter_score_table_column_config():
    return {
        "player_link": st.column_config.LinkColumn(
            "Player",
            help=(
                f"{GLOSSARY['player_link']} "
                "(L) or (R) = bat hand when known from features."
            ),
            display_text=r"#(.*)$",
        ),
        "game_time": st.column_config.TextColumn(
            "Game & time",
            help=f"{GLOSSARY['game']} {GLOSSARY['commence_time']}",
        ),
        "opposing_sp": st.column_config.TextColumn(
            "Opposing pitcher",
            help="Probable opposing starter; (L) or (R) = throwing hand when known.",
        ),
        "vs_pitcher": st.column_config.TextColumn(
            "Vs pitcher",
            help=(
                "Career batting average vs the listed opposing starter "
                f"(hits/AB) when PA ≥ {MIN_PA_H2H_BOARD}; otherwise SP ERA "
                "over the pitcher's last five starts. "
                "Light green when H2H average is above .300."
            ),
        ),
        "pp_fantasy_line": st.column_config.TextColumn(
            "PP fantasy",
            help=GLOSSARY["pp_fantasy_line"],
        ),
        "ud_fantasy_line": st.column_config.TextColumn(
            "UD fantasy",
            help=GLOSSARY["ud_fantasy_line"],
        ),
        "l5_l10_pct": st.column_config.TextColumn(
            "L5 / L10 %",
            help=(
                "Share of the player's last 5 / 10 completed games where "
                "their PrizePicks fantasy score strictly exceeded the "
                "posted PP fantasy line. When no PP line is available, "
                "shows over-rate vs the market line on that player's "
                "top prop row instead."
            ),
        ),
        "batter_score_display": st.column_config.TextColumn(
            "Batter score",
            help=GLOSSARY["batter_score"],
        ),
        "batter_score_v2_display": st.column_config.TextColumn(
            "Batter score v2",
            help=GLOSSARY["batter_score_v2"],
        ),
    }


def _prepare_batter_score_slate(
    df: pd.DataFrame,
    *,
    markets=None,
) -> pd.DataFrame:
    """Best batter_score row per player from today's batter markets."""
    if df.empty or "batter_score" not in df.columns:
        return pd.DataFrame()

    batters = df[df["market"].isin(BATTER_MARKETS)].copy()
    if markets:
        batters = batters[batters["market"].isin(markets)]
    if batters.empty:
        return pd.DataFrame()

    batters = batters.dropna(subset=["batter_score"])
    if batters.empty:
        return pd.DataFrame()

    best_idx = batters.groupby("player")["batter_score"].idxmax()
    return (
        batters.loc[best_idx]
        .sort_values("batter_score", ascending=False)
        .reset_index(drop=True)
    )


def _rows_from_batter_slate(ranked: pd.DataFrame, version: str) -> list[dict]:
    return [
        _build_batter_score_row(row, version)
        for _, row in ranked.iterrows()
    ]


def build_all_batter_score_df(
    df: pd.DataFrame,
    version: str,
    *,
    markets=None,
) -> pd.DataFrame:
    """
    All unique batters by best batter_score row from today's slate.

    Same row-building logic as build_top_batter_score_df(), without the top-10
    limit. Includes a ``_game`` column for UI filtering (dropped before display).
    """
    ranked = _prepare_batter_score_slate(df, markets=markets)
    if ranked.empty:
        return pd.DataFrame()

    return pd.DataFrame(_rows_from_batter_slate(ranked, version))


def render_game_batter_scores(
    df: pd.DataFrame,
    key_prefix: str,
    *,
    version: str = "v2",
):
    """Render all batters with a game filter below Top 10 Batter Score."""
    markets = st.session_state.get(f"{key_prefix}_markets", [])
    all_df = build_all_batter_score_df(
        df,
        version,
        markets=markets or None,
    )

    st.markdown("##### Batter score by game")
    caption = (
        "All batters on today's slate (best Batter Score row per player). "
        "**PP fantasy** and **UD fantasy** are each hitter's posted DFS "
        "fantasy score line (PrizePicks and Underdog). "
        "Select a game to narrow the list. Respects Market type filter like "
        "Top 10; independent of Edge / EV filters. "
        "Cell colors match Top 10 (orange lower PP/UD, sky blue when equal, light green H2H AVG > .300, yellow/green L5/L10, red outline combo). "
        "Batter score v2 uses Savant pitch-type matchup."
    )
    if is_batter_score_validated():
        caption += " " + GLOSSARY["batter_score_validated"]
    else:
        caption += " Batter Score — validation pending."
    st.caption(caption)

    if all_df.empty:
        st.caption("No batter scores available for the current slate.")
        return

    games = sorted(
        game for game in all_df["_game"].dropna().unique() if str(game).strip()
    )
    selected_game = st.selectbox(
        "Game",
        ["All games", *games],
        key=f"{key_prefix}_batter_game_filter",
        help="Filter batters to one matchup (away @ home).",
    )

    display_df = all_df
    if selected_game != "All games":
        display_df = all_df[all_df["_game"] == selected_game]

    display_df = display_df.reset_index(drop=True)

    render_batter_score_add_controls(display_df, key_prefix)

    _render_batter_score_dataframe(
        display_df,
        height=min(42 * len(display_df) + 38, 720),
    )
