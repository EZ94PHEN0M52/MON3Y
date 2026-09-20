"""Best 5s page — perfect L5 props and PrizePicks fantasy hitters."""

import streamlit as st

from ui.best_fives_board import render_best_fives_boards
from ui.glossary import EDGE_CALLOUT, GLOSSARY
from ui.player import render_back_to_board


def render_best_fives_page(df, version: str) -> None:
    render_back_to_board("view")

    st.markdown("## Best 5s")
    st.caption(GLOSSARY["best_fives_page"])

    render_best_fives_boards(
        df,
        key_prefix=f"best_fives_{version}",
        version=version,
    )

    st.divider()
    st.caption(EDGE_CALLOUT)
    st.caption(
        "This is a statistical research tool, "
        "not a guarantee of betting outcomes."
    )
