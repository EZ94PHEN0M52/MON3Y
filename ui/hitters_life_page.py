"""Hitter's Life page — batting average board."""

import streamlit as st

from ui.hitters_life_board import render_hitters_life_board
from ui.market_filters import render_market_multiselect
from ui.player import render_back_to_board


def render_hitters_life_page(df, version):
    render_back_to_board("view")
    st.markdown("## Hitter's Life")
    st.caption(
        "Batting context for today's slate: career and recent AVG, H2H vs the "
        "probable starter, pitch-type wOBA, and a short total-bases game log. "
        "Respects Market filter only (not Edge / EV). Select a game to open "
        "the Rotowire lineup filter. Light green: season AVG or H2H AVG > .300, "
        "or TB log with no zero-game."
    )

    key_prefix = f"hitters_life_{version}"
    col1, col2, col3 = st.columns(3)
    with col1:
        render_market_multiselect(
            df,
            key=f"{key_prefix}_markets",
            label="Market",
        )

    render_hitters_life_board(df, key_prefix, version=version)
