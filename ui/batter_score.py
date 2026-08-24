"""Batter Score UI helpers (Phase A + B)."""

import pandas as pd
import streamlit as st

from batter_score import MIN_PA_H2H
from batter_score_data import (
    build_game_context,
    get_batter_score_game_log,
    is_batter_score_validated,
    lookup_batter_score,
)
from ui.formatting import format_name_with_hand
from ui.glossary import GLOSSARY
from ui.player_stats import lookup_pitcher_hand


def format_batter_score_display(
    score,
    label="",
) -> str:
    if pd.isna(score):
        return "—"
    text = f"{float(score):.1f}"
    if label:
        text = f"{text} ({label})"
    return text


def render_batter_score_summary(
    player_name: str,
    version: str = "v2",
    *,
    game: str = "",
    commence_time=None,
    home_team: str = "",
    away_team: str = "",
):
    """Player-page component breakdown with Phase B SP / H2H context."""
    game_context = build_game_context(
        game=game,
        commence_time=commence_time,
        home_team=home_team or None,
        away_team=away_team or None,
    )

    result = lookup_batter_score(
        player_name,
        version=version,
        game_context=game_context,
    )

    if result is None:
        st.caption(
            "Batter Score unavailable — need at least 10 completed games "
            "in feature data."
        )
        return

    label_suffix = ""
    if result.partial_label:
        label_suffix = f" · **{result.partial_label}**"

    col_title, col_help = st.columns([10, 1])
    with col_title:
        st.subheader(
            f"Batter Score: {result.batter_score:.1f}/100{label_suffix}"
        )
        if is_batter_score_validated():
            st.caption("✓ Batter Score validated")
        else:
            st.caption("Batter Score — validation pending")
    with col_help:
        with st.popover("?"):
            st.markdown(GLOSSARY["batter_score"])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            "Season baseline",
            f"{result.season_baseline:.1f}",
            help=GLOSSARY["batter_score_season_baseline"],
        )
    with c2:
        st.metric(
            "Recent form",
            f"{result.recent_form:.1f}",
            help=GLOSSARY["batter_score_recent_form"],
        )
    with c3:
        matchup_value = (
            f"{result.matchup_grade:.1f}"
            if result.matchup_grade is not None
            else "—"
        )
        st.metric(
            "Matchup grade",
            matchup_value,
            help=GLOSSARY["batter_score_matchup_grade"],
        )
    with c4:
        pitcher_value = (
            f"{result.pitcher_form:.1f}"
            if result.pitcher_form is not None
            else "—"
        )
        st.metric(
            "Pitcher form",
            pitcher_value,
            help=GLOSSARY["batter_score_pitcher_form"],
        )

    detail_parts = []
    if result.opposing_sp_name:
        sp_hand = lookup_pitcher_hand(
            result.opposing_sp_name,
            version=version,
        )
        sp_label = format_name_with_hand(
            result.opposing_sp_name,
            sp_hand,
        )
        detail_parts.append(f"Opposing SP: **{sp_label}**")
    if result.opposing_sp_era_l5 is not None:
        detail_parts.append(
            f"SP ERA (L5): **{result.opposing_sp_era_l5:.2f}**"
        )
    if result.h2h_pa is not None and result.h2h_pa >= MIN_PA_H2H:
        h2h_avg = result.h2h_avg_raw_points
        avg_text = f"{h2h_avg:.2f}" if h2h_avg is not None else "—"
        detail_parts.append(
            f"H2H vs SP: **{result.h2h_pa} PA** "
            f"(avg H+TB+BB {avg_text}/game, blended into pitcher form)"
        )
    elif result.h2h_pa is not None and 0 < result.h2h_pa < MIN_PA_H2H:
        detail_parts.append(
            f"H2H vs SP: {result.h2h_pa} PA "
            f"(below {MIN_PA_H2H} PA minimum — omitted)"
        )

    if detail_parts:
        st.caption(" · ".join(detail_parts))

    if result.is_partial:
        st.caption(GLOSSARY["batter_score_partial"])

    game_log = get_batter_score_game_log(
        player_name,
        version=version,
        n=10,
    )
    if game_log is not None and not game_log.empty:
        st.caption(
            "Last 10 games — H + TB + BB raw points (Batter Score input stat)"
        )
        st.bar_chart(game_log)
