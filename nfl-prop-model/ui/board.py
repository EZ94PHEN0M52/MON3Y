"""Main NFL prop board with injury filters and badges."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ui.formatting import (
    enrich_display,
    market_label,
    prepare_board_table,
)
from ui.glossary import EDGE_CALLOUT, GLOSSARY


def _filter_frame(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    markets = sorted(work["market"].dropna().unique().tolist())
    selected_markets = st.multiselect(
        "Markets",
        options=markets,
        default=markets,
        format_func=market_label,
    )
    if selected_markets:
        work = work[work["market"].isin(selected_markets)]

    sides = sorted(work["side"].dropna().astype(str).str.title().unique().tolist())
    selected_sides = st.segmented_control(
        "Side",
        options=["All", *sides],
        default="All",
        required=True,
    )
    if selected_sides != "All":
        work = work[work["side"].astype(str).str.title().eq(selected_sides)]

    injury_mode = st.segmented_control(
        "Injury filter",
        options=["Hide inactive", "Active + Q", "All players"],
        default="Hide inactive",
        required=True,
        help=(
            "Hide inactive = drop Out/Doubtful/IR. "
            "Active + Q keeps Questionable. All players shows everyone."
        ),
    )
    if injury_mode == "Hide inactive":
        work = work[~work["is_inactive"].fillna(False).astype(bool)]
    elif injury_mode == "Active + Q":
        work = work[~work["is_inactive"].fillna(False).astype(bool)]

    show_q_only = st.checkbox("Only Questionable", value=False)
    if show_q_only and "is_questionable" in work.columns:
        work = work[work["is_questionable"].fillna(False).astype(bool)]

    c1, c2, c3 = st.columns(3)
    with c1:
        min_edge = st.number_input(
            "Min edge",
            min_value=-1.0,
            max_value=1.0,
            value=0.0,
            step=0.01,
            help=GLOSSARY["edge"],
        )
    with c2:
        min_ev = st.number_input(
            "Min EV",
            min_value=-2.0,
            max_value=2.0,
            value=0.0,
            step=0.01,
            help=GLOSSARY["ev"],
        )
    with c3:
        min_prob = st.number_input(
            "Min model %",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.01,
            help=GLOSSARY["model_probability"],
        )

    if "edge" in work.columns:
        work = work[work["edge"].fillna(-999) >= min_edge]
    if "ev" in work.columns:
        work = work[work["ev"].fillna(-999) >= min_ev]
    if "model_probability" in work.columns:
        work = work[work["model_probability"].fillna(0) >= min_prob]

    search = st.text_input("Player search", placeholder="e.g. Mahomes")
    if search.strip():
        needle = search.strip().lower()
        work = work[
            work["player"].astype(str).str.lower().str.contains(needle, na=False)
        ]

    return work


def _metrics(df: pd.DataFrame) -> None:
    inactive_n = int(df["is_inactive"].fillna(False).astype(bool).sum()) if "is_inactive" in df.columns else 0
    q_n = int(df["is_questionable"].fillna(False).astype(bool).sum()) if "is_questionable" in df.columns else 0
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows", f"{len(df):,}")
    m2.metric("Players", f"{df['player'].nunique():,}" if "player" in df.columns else "—")
    m3.metric("Inactive on board", f"{inactive_n:,}")
    m4.metric("Questionable on board", f"{q_n:,}")


def render_board(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        st.warning("Predictions file is empty. Re-run `python predict.py --version v1`.")
        return

    st.info(EDGE_CALLOUT)

    st.markdown("[Sleeper Picks](?view=sleeper_picks)")

    with st.expander("Filters", expanded=True):
        filtered = _filter_frame(df)

    _metrics(filtered)

    if filtered.empty:
        st.warning("No rows match the current filters.")
        return

    display = enrich_display(filtered)
    table = prepare_board_table(display)

    sort_default = "edge" if "edge" in filtered.columns else filtered.columns[0]
    sort_col = st.selectbox(
        "Sort by",
        options=[
            c
            for c in [
                "edge",
                "ev",
                "model_probability",
                "line",
                "player",
                "market",
            ]
            if c in filtered.columns
        ],
        index=0,
    )
    ascending = st.toggle("Ascending", value=False)

    sorted_df = filtered.sort_values(sort_col, ascending=ascending, na_position="last")
    table = prepare_board_table(enrich_display(sorted_df))

    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        column_config={
            "Player": st.column_config.TextColumn("Player"),
            "Injury": st.column_config.TextColumn(
                "Injury",
                help=GLOSSARY["injury"],
                width="small",
            ),
            "Market": st.column_config.TextColumn("Market"),
            "Side": st.column_config.TextColumn("Side", width="small"),
            "Line": st.column_config.NumberColumn("Line", format="%.1f"),
            "Odds": st.column_config.NumberColumn("Odds", format="%d"),
            "Model %": st.column_config.TextColumn(
                "Model %",
                help=GLOSSARY["model_probability"],
            ),
            "Book %": st.column_config.TextColumn(
                "Book %",
                help=GLOSSARY["market_implied"],
            ),
            "Edge": st.column_config.TextColumn("Edge", help=GLOSSARY["edge"]),
            "EV": st.column_config.TextColumn("EV", help=GLOSSARY["ev"]),
            "Book": st.column_config.TextColumn("Book"),
            "Game": st.column_config.TextColumn("Game"),
            "Kickoff": st.column_config.TextColumn("Kickoff"),
        },
    )

    with st.expander("Glossary"):
        for key, text in GLOSSARY.items():
            st.markdown(f"**{key.replace('_', ' ').title()}** — {text}")

    csv = sorted_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered CSV",
        data=csv,
        file_name="nfl_predictions_filtered.csv",
        mime="text/csv",
    )
