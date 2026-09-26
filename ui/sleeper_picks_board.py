"""Sleeper Picks board — MLB props from sleeper_props.parquet."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ui.table_display import show_dataframe

from fetch_sleeper_props import SLEEPER_PROPS_PATH
from ui.formatting import format_commence_time, format_pct, market_label


DISPLAY_COLUMNS = [
    "player",
    "game",
    "stat_display",
    "line",
    "over_multiplier",
    "under_multiplier",
    "pick_popularity",
    "player_team",
    "game_start",
    "status",
]


def _column_config() -> dict:
    return {
        "player": st.column_config.TextColumn("Player", width="medium"),
        "game": st.column_config.TextColumn("Game", width="medium"),
        "stat_display": st.column_config.TextColumn("Stat", width="small"),
        "line": st.column_config.NumberColumn("Line", format="%.1f"),
        "over_multiplier": st.column_config.NumberColumn("Over mult", format="%.2fx"),
        "under_multiplier": st.column_config.NumberColumn("Under mult", format="%.2fx"),
        "pick_popularity": st.column_config.TextColumn("Over pick %"),
        "player_team": st.column_config.TextColumn("Team", width="small"),
        "game_start": st.column_config.TextColumn("Start (ET)"),
        "status": st.column_config.TextColumn("Status", width="small"),
    }


def load_sleeper_props() -> pd.DataFrame:
    if not SLEEPER_PROPS_PATH.exists():
        return pd.DataFrame()

    df = pd.read_parquet(SLEEPER_PROPS_PATH)
    if df.empty:
        return df

    display = df.copy()
    display["pick_popularity"] = display["pick_popularity"].map(
        lambda v: format_pct(v) if pd.notna(v) else "—"
    )
    display["game_start"] = display["game_start"].map(format_commence_time)
    if "stat_display" not in display.columns:
        display["stat_display"] = display.get("sleeper_stat", "").map(
            lambda s: str(s).replace("_", " ").title() if pd.notna(s) else "—"
        )
    return display


def _apply_filters(df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    if df.empty:
        return df

    filtered = df.copy()

    stats = sorted(
        s for s in filtered["stat_display"].dropna().unique().tolist()
        if str(s).strip()
    )
    if stats:
        selected_stats = st.multiselect(
            "Stat",
            options=stats,
            default=stats,
            key=f"{key_prefix}_stats",
        )
        if selected_stats:
            filtered = filtered[filtered["stat_display"].isin(selected_stats)]

    teams = sorted(
        t for t in filtered["player_team"].dropna().unique().tolist()
        if str(t).strip()
    )
    if teams:
        selected_teams = st.multiselect(
            "Team",
            options=teams,
            default=teams,
            key=f"{key_prefix}_teams",
        )
        if selected_teams:
            filtered = filtered[filtered["player_team"].isin(selected_teams)]

    player_query = st.text_input(
        "Player search",
        key=f"{key_prefix}_player",
        placeholder="Partial name match…",
    ).strip()
    if player_query:
        needle = player_query.casefold()
        filtered = filtered[
            filtered["player"].astype(str).str.casefold().str.contains(
                needle,
                regex=False,
            )
        ]

    return filtered.reset_index(drop=True)


def render_sleeper_picks_board(
    key_prefix: str = "sleeper_picks",
    props_df: pd.DataFrame | None = None,
) -> None:
    raw = props_df if props_df is not None else load_sleeper_props()

    if raw.empty:
        st.warning(
            "No Sleeper props loaded. Add **APIFY_TOKEN** to `.env`, then run "
            "`python fetch_data.py --props` or `./run_daily.sh`."
        )
        st.caption(
            f"Expected file: `{SLEEPER_PROPS_PATH}`"
        )
        return

    fetched = raw["fetched_at"].dropna().max() if "fetched_at" in raw.columns else None
    if fetched is not None and pd.notna(fetched):
        st.caption(
            f"**{len(raw):,}** MLB props from Sleeper Picks "
            f"(fetched {format_commence_time(fetched)})."
        )
    else:
        st.caption(f"**{len(raw):,}** MLB props from Sleeper Picks.")

    filtered = _apply_filters(raw, key_prefix)

    if filtered.empty:
        st.info("No props match the current filters.")
        return

    show_cols = [c for c in DISPLAY_COLUMNS if c in filtered.columns]
    show_dataframe(
        filtered[show_cols],
        hide_index=True,
        column_config=_column_config(),
    )

    market_counts = (
        filtered.groupby("stat_display", dropna=False)
        .size()
        .sort_values(ascending=False)
    )
    with st.expander("Stat breakdown"):
        breakdown = market_counts.reset_index()
        breakdown.columns = ["Stat", "Count"]
        show_dataframe(breakdown, hide_index=True)

    mapped = filtered["market"].dropna().unique().tolist() if "market" in filtered.columns else []
    if mapped:
        labels = ", ".join(sorted({market_label(m) for m in mapped}))
        st.caption(f"Mapped model markets where applicable: {labels}")
