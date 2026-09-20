"""Best 5s boards — perfect L5 props and perfect L5 PrizePicks fantasy hitters."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from odds_aggregation import dedupe_best_prop
from ui.board import (
    RANKING_TABLE_COLUMNS,
    _prepare_board_table_df,
    _ranking_column_config,
)
from ui.formatting import (
    format_game_time,
    player_path,
    style_probability_extremes,
)
from ui.glossary import GLOSSARY
from ui.market_filters import render_market_multiselect
from ui.player_stats import (
    BATTER_MARKETS,
    _format_l5_l10_pct,
    format_prizepicks_fantasy_line,
    is_perfect_l5_pp_fantasy,
    is_perfect_l5_prop,
    lookup_batter_hand,
    lookup_prizepicks_fantasy_line,
    rolling_pp_fantasy_over_rates,
)

PERFECT_L5_PROP_COLUMNS = list(RANKING_TABLE_COLUMNS)

PERFECT_L5_FANTASY_COLUMNS = [
    "player_link",
    "game_time",
    "pp_fantasy_line",
    "l5_l10_pct",
]


def build_perfect_l5_props_df(
    df: pd.DataFrame,
    version: str,
    *,
    markets=None,
) -> pd.DataFrame:
    """
    Today's slate props with a perfect L5 over-rate vs the posted line.

    Includes all books (sportsbooks + PrizePicks). Requires five completed
    games all strictly over the line. One row per player / market (best EV).
    """
    if df.empty or "l5_pct" not in df.columns:
        return pd.DataFrame()

    working = df.copy()
    if markets:
        working = working[working["market"].isin(markets)]
    if working.empty:
        return pd.DataFrame()

    # Fast prefilter, then confirm full 5-game window.
    candidates = working[
        pd.to_numeric(working["l5_pct"], errors="coerce").fillna(0) >= 0.999
    ]
    if candidates.empty:
        return pd.DataFrame()

    keep_keys = set()
    for player, market, line in (
        candidates[["player", "market", "line"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    ):
        if is_perfect_l5_prop(player, market, line, version=version):
            keep_keys.add((player, market, float(line)))

    if not keep_keys:
        return pd.DataFrame()

    mask = [
        (row.player, row.market, float(row.line)) in keep_keys
        for row in candidates[["player", "market", "line"]].itertuples(
            index=False
        )
    ]
    perfect = candidates.loc[mask]
    if perfect.empty:
        return pd.DataFrame()

    sort_col = "ev" if "ev" in perfect.columns else "edge"
    if sort_col not in perfect.columns:
        sort_col = "over_probability"
    deduped = dedupe_best_prop(perfect, sort_col=sort_col)
    return (
        deduped.sort_values(
            ["market", "player", "line"],
            ascending=[True, True, True],
        )
        .reset_index(drop=True)
    )


def build_perfect_l5_pp_fantasy_df(
    df: pd.DataFrame,
    version: str,
) -> pd.DataFrame:
    """Unique slate batters with a perfect L5 vs their PrizePicks fantasy line."""
    if df.empty or "market" not in df.columns:
        return pd.DataFrame()

    batters = df[df["market"].isin(BATTER_MARKETS)].copy()
    if batters.empty:
        return pd.DataFrame()

    # One row per player (prefer a batter_hits row when present).
    hits = batters[batters["market"] == "batter_hits"]
    source = hits if not hits.empty else batters
    unique = source.drop_duplicates(subset=["player"], keep="first")

    rows = []
    for _, row in unique.iterrows():
        player = row["player"]
        if not is_perfect_l5_pp_fantasy(player, version=version):
            continue

        pp_line = lookup_prizepicks_fantasy_line(player)
        l5_pct, l10_pct = rolling_pp_fantasy_over_rates(
            player,
            pp_line,
            version=version,
        )
        rows.append(
            {
                "player": player,
                "player_link": player_path(
                    player,
                    hand=lookup_batter_hand(player, version=version),
                ),
                "game_time": format_game_time(
                    row.get("game"),
                    row.get("commence_time"),
                ),
                "pp_fantasy_line": format_prizepicks_fantasy_line(player),
                "l5_l10_pct": _format_l5_l10_pct(l5_pct, l10_pct),
                "_l5_pct": l5_pct,
                "_pp_line": pp_line,
            }
        )

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    return result.sort_values("player").reset_index(drop=True)


def render_perfect_l5_props_board(
    df: pd.DataFrame,
    key_prefix: str,
    *,
    version: str = "v2",
):
    st.markdown("##### Perfect L5 props")
    st.caption(
        "Props on today's slate where the player went **over the posted line "
        "in each of the last 5 completed games** (strictly greater than the "
        "line; requires a full five-game sample). All books; one best-EV row "
        "per player and market. Respects the Market filter below. "
        "(The board under this is PrizePicks **hitter fantasy** — separate.)"
    )

    markets = st.session_state.get(f"{key_prefix}_markets", [])
    perfect = build_perfect_l5_props_df(
        df,
        version,
        markets=markets or None,
    )

    if perfect.empty:
        st.caption("No props with a perfect L5 over-rate on today's slate.")
        return

    display = _prepare_board_table_df(perfect, version=version)
    cols = [col for col in PERFECT_L5_PROP_COLUMNS if col in display.columns]
    st.caption(f"Showing **{len(perfect)}** props.")
    st.dataframe(
        style_probability_extremes(display[cols]),
        hide_index=True,
        height=min(42 * len(perfect) + 38, 560),
        column_config=_ranking_column_config(),
    )


def _perfect_fantasy_column_config():
    return {
        "player_link": st.column_config.LinkColumn(
            "Player",
            help=GLOSSARY.get("player_link", "Open player profile."),
            display_text=r"#(.*)$",
        ),
        "game_time": st.column_config.TextColumn(
            "Game & time",
            help=f"{GLOSSARY.get('game', '')} {GLOSSARY.get('commence_time', '')}",
        ),
        "pp_fantasy_line": st.column_config.TextColumn(
            "PP fantasy",
            help=GLOSSARY["pp_fantasy_line"],
        ),
        "l5_l10_pct": st.column_config.TextColumn(
            "L5 / L10 %",
            help=(
                "Share of last 5 / 10 completed games where PrizePicks fantasy "
                "score strictly exceeded the posted PP fantasy line. This "
                "board only includes batters at 100% L5 with five games."
            ),
        ),
    }


def render_perfect_l5_pp_fantasy_board(
    df: pd.DataFrame,
    key_prefix: str,
    *,
    version: str = "v2",
):
    _ = key_prefix
    st.markdown("##### Perfect L5 PrizePicks fantasy")
    st.caption(
        "Batters on today's slate whose PrizePicks hitter fantasy score went "
        "**over their posted PP fantasy line in each of the last 5 completed "
        "games** (requires a full five-game sample and a posted PP line)."
    )

    perfect = build_perfect_l5_pp_fantasy_df(df, version)
    if perfect.empty:
        st.caption(
            "No batters with a perfect L5 vs their PrizePicks fantasy line."
        )
        return

    display = perfect.drop(
        columns=[col for col in perfect.columns if col.startswith("_")],
        errors="ignore",
    )
    cols = [col for col in PERFECT_L5_FANTASY_COLUMNS if col in display.columns]
    st.caption(f"Showing **{len(perfect)}** batters.")
    st.dataframe(
        display[cols],
        hide_index=True,
        height=min(42 * len(perfect) + 38, 560),
        column_config=_perfect_fantasy_column_config(),
    )


def render_best_fives_boards(
    df: pd.DataFrame,
    key_prefix: str,
    *,
    version: str = "v2",
):
    col1, _col2, _col3 = st.columns(3)
    with col1:
        render_market_multiselect(
            df,
            key=f"{key_prefix}_markets",
            label="Market",
        )

    render_perfect_l5_props_board(df, key_prefix, version=version)
    st.divider()
    render_perfect_l5_pp_fantasy_board(df, key_prefix, version=version)
