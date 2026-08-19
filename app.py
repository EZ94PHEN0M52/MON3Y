import pandas as pd
import streamlit as st

from ui.board import render_board
from ui.player import render_player_page
from ui.top_lists import render_top_over_page, render_top_under_page
from ui.formatting import enrich_with_over_under_probs
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


version = st.sidebar.selectbox(
    "Model version",
    ["v2", "v1"],
    help="V2 adds opponent, handedness, and park features.",
)

df = enrich_with_l5_l10_pct(
    enrich_with_over_under_probs(load_predictions(version)),
    version,
)
player = st.query_params.get("player")
view = st.query_params.get("view")

render_header(version)

if player:
    render_player_page(df, player, version)
elif view == "top_over":
    render_top_over_page(df, version)
elif view == "top_under":
    render_top_under_page(df, version)
else:
    render_board(df, version)
