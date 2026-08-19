"""Full ranked Top Over / Top Under list pages."""

import streamlit as st

from ui.board import (
    RANKING_TABLE_COLUMNS,
    _prepare_board_table_df,
    _ranking_column_config,
    apply_top_level_filters,
)
from ui.formatting import style_probability_extremes
from ui.glossary import EDGE_CALLOUT, GLOSSARY
from ui.player import render_back_to_board


def _render_ranked_table(filtered, sort_col):
    if filtered.empty:
        st.info("No props meet the current filters.")
        return

    ranked = filtered.sort_values(sort_col, ascending=False).reset_index(drop=True)
    display = _prepare_board_table_df(ranked)[RANKING_TABLE_COLUMNS]

    st.caption(f"Showing **{len(ranked)}** props ranked by **{sort_col.replace('_', ' ')}**.")
    st.dataframe(
        style_probability_extremes(display),
        hide_index=True,
        column_config=_ranking_column_config(),
    )


def render_top_over_page(df, version):
    render_back_to_board("view")

    st.title("Top Over %")
    st.caption(GLOSSARY["top_over_list"])

    filtered = apply_top_level_filters(df, key_prefix=f"top_over_{version}")
    _render_ranked_table(filtered, "over_probability")

    st.divider()
    st.caption(EDGE_CALLOUT)
    st.caption(
        "This is a statistical research tool, "
        "not a guarantee of betting outcomes."
    )


def render_top_under_page(df, version):
    render_back_to_board("view")

    st.title("Top Under %")
    st.caption(GLOSSARY["top_under_list"])

    filtered = apply_top_level_filters(df, key_prefix=f"top_under_{version}")
    filtered = filtered[filtered["market"] != "batter_home_runs"]
    _render_ranked_table(filtered, "under_probability")

    st.divider()
    st.caption(EDGE_CALLOUT)
    st.caption(
        "This is a statistical research tool, "
        "not a guarantee of betting outcomes."
    )
