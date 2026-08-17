import pandas as pd
import streamlit as st

from utils import predictions_path


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="MLB Prop Model",
    page_icon="⚾",
    layout="wide"
)


# =========================================================
# HEADER
# =========================================================

st.title(
    "⚾ MLB Prop Model"
)

version = st.sidebar.selectbox(
    "Model version",
    ["v2", "v1"],
    help="V2 adds opponent, handedness, and park features."
)

if version == "v2":

    st.caption(
        "V2 — Opponent strength, handedness, park proxy"
    )

else:

    st.caption(
        "V1 — Rolling player form (frozen baseline)"
    )


# =========================================================
# LOAD
# =========================================================

predictions_file = predictions_path(
    version
)

try:

    df = pd.read_csv(
        predictions_file
    )

except FileNotFoundError:

    st.error(
        f"No predictions found at {predictions_file}. "
        f"Run: python predict.py --version {version}"
    )

    st.stop()


# =========================================================
# FILTERS
# =========================================================

col1, col2, col3 = st.columns(3)


with col1:

    markets = st.multiselect(
        "Market",
        sorted(
            df["market"]
            .dropna()
            .unique()
        )
    )


with col2:

    min_edge = st.slider(
        "Minimum Edge",
        0.0,
        0.30,
        0.03,
        0.01
    )


with col3:

    min_ev = st.slider(
        "Minimum EV",
        0.0,
        0.50,
        0.05,
        0.01
    )


# =========================================================
# FILTER
# =========================================================

filtered = df.copy()


if markets:

    filtered = filtered[
        filtered["market"]
        .isin(markets)
    ]


filtered = filtered[
    filtered["edge"] >= min_edge
]


filtered = filtered[
    filtered["ev"] >= min_ev
]


# =========================================================
# METRICS
# =========================================================

c1, c2, c3, c4 = st.columns(4)


c1.metric(
    "Props",
    len(filtered)
)


c2.metric(
    "Best Edge",
    f"{filtered['edge'].max() * 100:.1f}%"
    if len(filtered)
    else "—"
)


c3.metric(
    "Best EV",
    f"{filtered['ev'].max() * 100:.1f}%"
    if len(filtered)
    else "—"
)


c4.metric(
    "Players",
    filtered["player"].nunique()
    if len(filtered)
    else 0
)


# =========================================================
# DISPLAY
# =========================================================

if len(filtered):

    display = filtered.copy()

    display[
        "model_probability"
    ] = (
        display[
            "model_probability"
        ] * 100
    ).round(1)

    display[
        "market_probability"
    ] = (
        display[
            "market_probability"
        ] * 100
    ).round(1)

    display[
        "edge"
    ] = (
        display["edge"] * 100
    ).round(1)

    display[
        "ev"
    ] = (
        display["ev"] * 100
    ).round(1)

    display = display[
        [
            "player",
            "market",
            "bookmaker",
            "side",
            "line",
            "odds",
            "model_probability",
            "market_probability",
            "edge",
            "ev"
        ]
    ]

    display.columns = [
        "Player",
        "Market",
        "Book",
        "Side",
        "Line",
        "Odds",
        "Model %",
        "Market %",
        "Edge %",
        "EV %"
    ]

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "No props meet the current filters."
    )


# =========================================================
# DISCLAIMER
# =========================================================

st.divider()

st.caption(
    "This is a statistical research tool, "
    "not a guarantee of betting outcomes."
)
