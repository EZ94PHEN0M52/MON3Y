"""Sleeper Picks page — dedicated MLB props board from Apify fetch."""

import streamlit as st

from fetch_sleeper_props import SLEEPER_PROPS_PATH
from ui.player import render_back_to_board
from ui.sleeper_hrr_board import render_sleeper_hrr_board
from ui.sleeper_picks_board import load_sleeper_props, render_sleeper_picks_board


@st.cache_data(show_spinner="Loading Sleeper props…")
def _cached_sleeper_props(parquet_mtime: float):
    _ = parquet_mtime
    return load_sleeper_props()


def render_sleeper_picks_page(version: str) -> None:
    render_back_to_board("view")
    st.markdown("## Sleeper Picks")
    st.caption(
        "Full MLB player props from Sleeper Picks (mobile app lines), fetched "
        "via Apify on each `--props` run. Shows line, over/under multipliers, "
        "and pick popularity (% of Sleeper users on the over). "
        "Not merged into the main model board."
    )

    mtime = (
        SLEEPER_PROPS_PATH.stat().st_mtime
        if SLEEPER_PROPS_PATH.exists()
        else 0.0
    )

    render_sleeper_hrr_board(
        version=version,
        props_mtime=mtime,
        key_prefix=f"sleeper_hrr_{version}",
    )

    st.divider()

    df = _cached_sleeper_props(mtime)
    render_sleeper_picks_board(key_prefix=f"sleeper_picks_{version}", props_df=df)

    from sleeper_analyzer.wizard import render_sleeper_prop_analyzer

    render_sleeper_prop_analyzer(version, mtime)
