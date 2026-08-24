import pandas as pd
import streamlit as st

from ui.board import render_board
from ui.pick_builder import (
    render_sidebar_batter_score_pick_builder,
    render_sidebar_pick_builder,
)
from ui.player import render_player_page
from ui.top_lists import render_top_over_page, render_top_under_page
from ui.hitters_life_page import render_hitters_life_page
from ui.version_compare import render_version_compare_page
from batter_score_data import enrich_with_batter_score
from ui.formatting import compare_view_path, enrich_with_over_under_probs
from ui.market_filters import exclude_ui_markets
from ui.player_stats import enrich_with_l5_l10_pct
from utils import predictions_path


st.set_page_config(
    page_title="MLB Prop Model",
    page_icon="⚾",
    layout="wide",
)


def render_header(version):
    st.title("⚾ MLB Prop Model")

    if version == "v2":
        st.caption("V2 — Opponent strength, handedness, park proxy")
    else:
        st.caption("V1 — Rolling player form (frozen baseline)")


def load_predictions(version):
    predictions_file = predictions_path(version)

    try:
        return pd.read_csv(predictions_file)
    except FileNotFoundError:
        st.error(
            f"No predictions found at {predictions_file}. "
            f"Run: python predict.py --version {version}"
        )
        st.stop()


@st.cache_data(show_spinner=False)
def load_board_data(version, predictions_mtime):
    """
    Cached board dataframe: predictions CSV + enrichment from local parquets.

    predictions_mtime busts the cache when predict.py rewrites the CSV.
    Enrichment reads feature/probables/statcast parquets only (no live APIs).
    """
    _ = predictions_mtime
    predictions = exclude_ui_markets(load_predictions(version))
    return enrich_with_batter_score(
        enrich_with_l5_l10_pct(
            enrich_with_over_under_probs(predictions),
            version,
        ),
        version,
    )


version = st.sidebar.selectbox(
    "Model version",
    ["v2", "v1"],
    help="V2 adds opponent, handedness, and park features.",
)

st.sidebar.markdown(f"**[Version compare]({compare_view_path()})** — V1 / V2 / V3 / Main")

st.sidebar.divider()
render_sidebar_pick_builder()
st.sidebar.divider()
render_sidebar_batter_score_pick_builder()

player = st.query_params.get("player")
view = st.query_params.get("view")

if view != "compare":
    render_header(version)
else:
    st.sidebar.markdown(f"**[Version compare]({compare_view_path()})** · active")

if view == "compare":
    render_version_compare_page()
elif player:
    _predictions_file = predictions_path(version)
    _predictions_mtime = (
        _predictions_file.stat().st_mtime
        if _predictions_file.exists()
        else 0.0
    )
    df = load_board_data(version, _predictions_mtime)
    render_player_page(df, player, version)
elif view == "top_over":
    _predictions_file = predictions_path(version)
    _predictions_mtime = (
        _predictions_file.stat().st_mtime
        if _predictions_file.exists()
        else 0.0
    )
    df = load_board_data(version, _predictions_mtime)
    render_top_over_page(df, version)
elif view == "top_under":
    _predictions_file = predictions_path(version)
    _predictions_mtime = (
        _predictions_file.stat().st_mtime
        if _predictions_file.exists()
        else 0.0
    )
    df = load_board_data(version, _predictions_mtime)
    render_top_under_page(df, version)
elif view == "hitters_life":
    _predictions_file = predictions_path(version)
    _predictions_mtime = (
        _predictions_file.stat().st_mtime
        if _predictions_file.exists()
        else 0.0
    )
    df = load_board_data(version, _predictions_mtime)
    render_hitters_life_page(df, version)
else:
    _predictions_file = predictions_path(version)
    _predictions_mtime = (
        _predictions_file.stat().st_mtime
        if _predictions_file.exists()
        else 0.0
    )
    df = load_board_data(version, _predictions_mtime)
    render_board(df, version)
