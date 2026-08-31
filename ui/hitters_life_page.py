"""Hitter's Life page — batter score boards and batting average board."""

import streamlit as st

from ui.batter_score_board import (
    render_game_batter_scores,
    render_top_batter_scores,
)
from ui.hitters_life_board import render_hitters_life_board
from ui.hitters_life_manual_score import render_manual_h2h_batter_score
from ui.market_filters import render_market_multiselect
from ui.player import render_back_to_board


def render_hitters_life_page(df, version):
    render_back_to_board("view")
    st.markdown("## Hitter's Life")
    st.caption(
        "Batter Score rankings and batting context for today's slate. "
        "All sections respect the Market filter only (not Edge / EV). "
        "Top 10 and by-game tables include PP/UD fantasy lines, L5/L10 %, "
        "and Batter Score v2. The batting average board adds career/recent AVG, "
        "H2H vs the probable starter, pitch-type wOBA, and a total-bases game log."
    )

    key_prefix = f"hitters_life_{version}"
    col1, col2, col3 = st.columns(3)
    with col1:
        render_market_multiselect(
            df,
            key=f"{key_prefix}_markets",
            label="Market",
        )

    render_manual_h2h_batter_score(df, key_prefix, version=version)
    st.divider()

    render_top_batter_scores(df, key_prefix, version=version)
    st.divider()
    render_game_batter_scores(df, key_prefix, version=version)
    st.divider()
    render_hitters_life_board(df, key_prefix, version=version)
