"""Sleeper Picks page — NFL props from Apify fetch + matchup browser."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from fetch_sleeper_props import SLEEPER_PROPS_PATH
from ui.sleeper_picks_board import (
    enrich_sleeper_with_model_probs,
    load_sleeper_props,
    render_sleeper_picks_board,
)
from utils import player_features_path


@st.cache_data(show_spinner="Scoring Sleeper lines with model…", ttl=120, max_entries=6)
def _cached_sleeper_props_scored(
    parquet_mtime: float,
    features_mtime: float,
    version: str,
) -> pd.DataFrame:
    _ = parquet_mtime
    _ = features_mtime
    raw = load_sleeper_props()
    return enrich_sleeper_with_model_probs(raw, version=version)


def _list_games(df: pd.DataFrame) -> list[str]:
    if df.empty or "game" not in df.columns:
        return []
    return sorted(
        g for g in df["game"].dropna().astype(str).unique().tolist() if g.strip()
    )


def _render_matchup_browser(df: pd.DataFrame, key_prefix: str) -> None:
    st.markdown("##### Matchup browser")
    st.caption(
        "Pick a game, then a team, to scan Sleeper lines, model Over/Under %, "
        "and Over pick %."
    )
    if df.empty:
        st.info("No Sleeper props to browse.")
        return

    games = _list_games(df)
    if not games:
        st.info("No game labels in Sleeper cache.")
        return

    game = st.selectbox("Game", options=games, key=f"{key_prefix}_game")
    game_df = df[df["game"].astype(str).eq(game)].copy()

    teams = sorted(
        t
        for t in game_df["player_team"].dropna().astype(str).unique().tolist()
        if t.strip()
    )
    if not teams:
        st.info("No teams found for this game.")
        return

    team = st.segmented_control(
        "Team",
        options=teams,
        default=teams[0],
        required=True,
        key=f"{key_prefix}_team",
    )
    team_df = game_df[game_df["player_team"].astype(str).eq(team)].copy()

    players = sorted(team_df["player"].dropna().astype(str).unique().tolist())
    player = st.selectbox(
        "Player",
        options=players,
        key=f"{key_prefix}_player",
    )
    player_df = team_df[team_df["player"].astype(str).eq(player)].copy()

    kickoff = "—"
    if "game_start" in player_df.columns and not player_df.empty:
        kickoff = player_df["game_start"].iloc[0]

    st.caption(f"{game} · {kickoff} · {team} · {player}")

    show = player_df[
        [
            c
            for c in [
                "stat_display",
                "line",
                "over_probability_display",
                "under_probability_display",
                "over_multiplier",
                "under_multiplier",
                "pick_popularity",
                "injury",
                "market",
            ]
            if c in player_df.columns
        ]
    ].sort_values("stat_display")

    st.dataframe(show, hide_index=True, width="stretch")
    st.caption(
        f"{len(show)} Sleeper lines for {player}. "
        "Over/Under % = model on the Sleeper line; Over pick % = Sleeper users."
    )


def render_sleeper_picks_page(version: str = "v2") -> None:
    if st.button(
        "Back to board",
        icon=":material/arrow_back:",
        key="sleeper_back_board",
    ):
        st.query_params.clear()
        st.rerun()

    st.markdown("## Sleeper Picks")
    st.caption(
        "NFL player props from Sleeper Picks (mobile app lines), fetched via "
        "Apify. Shows model Over/Under % on the Sleeper line, multipliers, "
        "pick popularity, and injury status when provided."
    )

    version = st.sidebar.selectbox(
        "Model version (Sleeper scoring)",
        ["v2", "v1"],
        index=0 if version == "v2" else 1,
        key="sleeper_model_version",
        help="Used to score Over/Under % on each Sleeper line.",
    )

    mtime = (
        SLEEPER_PROPS_PATH.stat().st_mtime
        if SLEEPER_PROPS_PATH.exists()
        else 0.0
    )
    feat_path = player_features_path(version)
    feat_mtime = feat_path.stat().st_mtime if feat_path.exists() else 0.0

    df = _cached_sleeper_props_scored(mtime, feat_mtime, version)

    render_sleeper_picks_board(
        key_prefix="sleeper_picks_nfl",
        props_df=df,
        version=version,
    )

    st.divider()
    _render_matchup_browser(df, key_prefix="sleeper_matchup_nfl")
