"""Top Batter Score section for the main prop board."""

import pandas as pd
import streamlit as st

from batter_score_data import (
    build_game_context,
    is_batter_score_validated,
    lookup_batter_score,
    lookup_h2h_board_stats,
)
from ui.batter_score import format_batter_score_display
from ui.formatting import format_commence_time, player_path
from ui.glossary import GLOSSARY
from ui.player_stats import BATTER_MARKETS, batter_score_l5_l10_pct

# Board "Vs pitcher" column — lower bar than MIN_PA_H2H (10) used in scoring.
MIN_PA_H2H_BOARD = 3


def _format_game_time(row) -> str:
    game = row.get("game") or ""
    time_str = format_commence_time(row.get("commence_time"))
    if game and time_str != "—":
        return f"{game} · {time_str}"
    return game or time_str


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
    ranked = (
        batters.loc[best_idx]
        .sort_values("batter_score", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    rows = []
    for _, row in ranked.iterrows():
        game_context = _row_game_context(row)
        result = lookup_batter_score(
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
            opposing = result.opposing_sp_name

        l5_l10 = batter_score_l5_l10_pct(
            row["player"],
            version=version,
            fallback=row.get("l5_l10_pct"),
        )
        if pd.isna(l5_l10) or l5_l10 is None:
            l5_l10 = "—"

        rows.append(
            {
                "player_link": player_path(row["player"]),
                "game_time": _format_game_time(row),
                "opposing_sp": opposing,
                "vs_pitcher": _format_vs_pitcher(
                    result,
                    h2h_pa=h2h_pa,
                    h2h_hits=h2h_hits,
                    h2h_ab=h2h_ab,
                ),
                "l5_l10_pct": l5_l10,
                "batter_score_display": format_batter_score_display(
                    row["batter_score"],
                    row.get("batter_score_label") or "",
                ),
            }
        )

    return pd.DataFrame(rows)


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
        "L5 / L10 % uses each player's PrizePicks fantasy score line when "
        "available (from the daily props fetch); otherwise falls back to the "
        "market line on that player's top prop row. "
    )
    if is_batter_score_validated():
        caption += GLOSSARY["batter_score_validated"]
    else:
        caption += "Batter Score — validation pending."
    st.caption(caption)

    if top_df.empty:
        st.caption("No batter scores available for the current slate.")
        return

    st.dataframe(
        top_df,
        hide_index=True,
        height=min(42 * len(top_df) + 38, 480),
        column_config=_batter_score_table_column_config(),
    )


def _batter_score_table_column_config():
    return {
        "player_link": st.column_config.LinkColumn(
            "Player",
            help=GLOSSARY["player_link"],
            display_text=r"#(.*)$",
        ),
        "game_time": st.column_config.TextColumn(
            "Game & time",
            help=f"{GLOSSARY['game']} {GLOSSARY['commence_time']}",
        ),
        "opposing_sp": st.column_config.TextColumn(
            "Opposing pitcher",
        ),
        "vs_pitcher": st.column_config.TextColumn(
            "Vs pitcher",
            help=(
                "Career batting average vs the listed opposing starter "
                f"(hits/AB) when PA ≥ {MIN_PA_H2H_BOARD}; otherwise SP ERA "
                "over the pitcher's last five starts."
            ),
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
    rows = []
    for _, row in ranked.iterrows():
        game_context = _row_game_context(row)
        result = lookup_batter_score(
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
            opposing = result.opposing_sp_name

        l5_l10 = batter_score_l5_l10_pct(
            row["player"],
            version=version,
            fallback=row.get("l5_l10_pct"),
        )
        if pd.isna(l5_l10) or l5_l10 is None:
            l5_l10 = "—"

        rows.append(
            {
                "player_link": player_path(row["player"]),
                "game_time": _format_game_time(row),
                "opposing_sp": opposing,
                "vs_pitcher": _format_vs_pitcher(
                    result,
                    h2h_pa=h2h_pa,
                    h2h_hits=h2h_hits,
                    h2h_ab=h2h_ab,
                ),
                "l5_l10_pct": l5_l10,
                "batter_score_display": format_batter_score_display(
                    row["batter_score"],
                    row.get("batter_score_label") or "",
                ),
                "_game": row.get("game") or "",
            }
        )
    return rows


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
        "Select a game to narrow the list. Respects Market type filter like "
        "Top 10; independent of Edge / EV filters."
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

    display_df = display_df.drop(columns=["_game"], errors="ignore").reset_index(
        drop=True
    )

    st.dataframe(
        display_df,
        hide_index=True,
        height=min(42 * len(display_df) + 38, 720),
        column_config=_batter_score_table_column_config(),
    )
