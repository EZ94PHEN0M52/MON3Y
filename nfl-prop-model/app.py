"""NFL Prop Model — Streamlit entry point."""

from pathlib import Path

import pandas as pd
import streamlit as st

from ui.board import render_board
from ui.sleeper_picks_page import render_sleeper_picks_page
from utils import predictions_path

st.set_page_config(
    page_title="NFL Prop Model",
    page_icon=":material/sports_football:",
    layout="wide",
)


@st.cache_data(show_spinner="Loading predictions…", ttl=120, max_entries=4)
def load_predictions(version: str, predictions_mtime: float) -> pd.DataFrame:
    _ = predictions_mtime
    path = predictions_path(version)
    return pd.read_csv(path)


def main() -> None:
    view = st.query_params.get("view")

    if view == "sleeper_picks":
        render_sleeper_picks_page()
        return

    st.title("NFL Prop Model")
    st.caption(
        "Research board — rolling form + injury gates vs sportsbook lines. "
        "V2 adds opponent defense and usage. Not a betting service."
    )

    version = st.sidebar.selectbox(
        "Model version",
        ["v2", "v1"],
        help="V2 adds opponent defense, usage (target/rush/snap share), home/away.",
    )

    st.sidebar.markdown("[Sleeper Picks](?view=sleeper_picks)")
    st.sidebar.divider()

    predictions_file = predictions_path(version)
    if not predictions_file.exists():
        st.error(
            f"No predictions at `{predictions_file}`. "
            f"Run: `python predict.py --version {version}`"
        )
        st.stop()

    mtime = predictions_file.stat().st_mtime
    df = load_predictions(version, mtime)

    st.sidebar.caption(
        f"Loaded {len(df):,} rows · "
        f"{Path(predictions_file).name}"
    )
    st.sidebar.divider()
    st.sidebar.markdown(
        "Refresh data:\n\n"
        "```bash\n"
        "python fetch_data.py --injuries --props\n"
        "python predict.py --version v2\n"
        "```"
    )

    render_board(df)


if __name__ == "__main__":
    main()
