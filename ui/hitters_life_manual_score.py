"""Manual H2H Batter Score (v1) calculator for the Hitter's Life page."""

import streamlit as st

from batter_score import MIN_PA_H2H_MANUAL
from batter_score_data import (
    parse_h2h_fraction,
    score_batter_with_manual_h2h,
)
from hitters_life_data import format_h2h_avg_display
from ui.batter_score_board import _prepare_batter_score_slate, _row_game_context
from ui.formatting import format_name_with_hand
from ui.glossary import GLOSSARY
from ui.player_stats import lookup_batter_hand, lookup_pitcher_hand


def render_manual_h2h_batter_score(
    df,
    key_prefix: str,
    *,
    version: str = "v2",
):
    """
    Live v1 Batter Score using user-entered career H2H H/AB vs today's SP.

    Separate from the Batter Score columns on the boards below.
    """
    markets = st.session_state.get(f"{key_prefix}_markets", [])
    slate = _prepare_batter_score_slate(df, markets=markets or None)

    st.markdown("##### Manual H2H Batter Score (v1)")
    st.caption(
        "Enter career batting average vs today's probable starter as **hits/AB** "
        f"(e.g. `3/8`). Everything else is pulled from cached data; your H2H "
        f"overrides Statcast (2024+) and blends into **pitcher form** (55% of "
        f"that component when AB ≥ {MIN_PA_H2H_MANUAL}). Pitcher form is 15% "
        "of the full score, so a strong career line can move the total by "
        "several points. This score is **separate** from the Batter Score "
        "columns on the boards below."
    )

    if slate.empty:
        st.info("No batters on today's slate for the current Market filter.")
        return

    players = sorted(slate["player"].dropna().unique())
    player_by_row = (
        slate.drop_duplicates(subset=["player"], keep="first")
        .set_index("player")
    )

    col_player, col_h2h = st.columns([2, 1])
    with col_player:
        player = st.selectbox(
            "Player",
            options=players,
            key=f"{key_prefix}_manual_h2h_player",
        )
    with col_h2h:
        fraction_text = st.text_input(
            "Career H2H vs SP (H/AB)",
            placeholder="3/8",
            key=f"{key_prefix}_manual_h2h_fraction",
        )

    parsed = parse_h2h_fraction(fraction_text)
    if not player or parsed is None:
        if fraction_text.strip() and parsed is None:
            st.warning("Enter H2H as hits/AB, e.g. `3/8` or `12/40`.")
        return

    hits, ab = parsed
    row = player_by_row.loc[player]
    game_context = _row_game_context(row)
    if game_context is None:
        st.warning("Could not resolve game context for this player.")
        return

    result = score_batter_with_manual_h2h(
        player,
        hits,
        ab,
        version=version,
        game_context=game_context,
    )
    if result is None:
        st.warning(
            "Batter Score unavailable — need at least 10 completed games "
            "in feature data."
        )
        return

    hand = lookup_batter_hand(player, version=version)
    player_label = format_name_with_hand(player, hand)
    h2h_display = format_h2h_avg_display(hits / ab, hits=hits, ab=ab)

    label_suffix = ""
    if result.partial_label:
        label_suffix = f" · {result.partial_label}"

    st.markdown(
        f"**{player_label}** — Manual Batter Score v1: "
        f"**{result.batter_score:.1f}/100**{label_suffix}"
    )

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

    detail_parts = [f"Manual H2H: **{h2h_display}**"]
    if result.opposing_sp_name:
        sp_hand = lookup_pitcher_hand(result.opposing_sp_name, version=version)
        sp_label = format_name_with_hand(result.opposing_sp_name, sp_hand)
        detail_parts.append(f"Opposing SP: **{sp_label}**")
    if result.opposing_sp_era_l5 is not None:
        detail_parts.append(
            f"SP ERA (L5): **{result.opposing_sp_era_l5:.2f}**"
        )
    if result.h2h_avg_raw_points is not None:
        detail_parts.append(
            f"Est. H+TB+BB vs SP: **{result.h2h_avg_raw_points:.2f}/game** "
            "(blended into pitcher form)"
        )

    st.caption(" · ".join(detail_parts))
