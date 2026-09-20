"""Sleeper Picks board — NFL props from sleeper_props.parquet."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from fetch_sleeper_props import SLEEPER_PROPS_PATH
from train import MARKET_STAT
from ui.formatting import format_commence_time, format_pct, market_label
from utils import normalize_version, player_features_path, version_models_dir

# Remap older / alternate Sleeper market keys in cached parquets.
MARKET_ALIASES = {
    "sleeper_passing_touchdowns": "player_pass_tds",
    "sleeper_rushing_and_receiving_yards": "player_rush_reception_yds",
    "sleeper_rushing_receiving_yards": "player_rush_reception_yds",
}

DISPLAY_COLUMNS = [
    "player",
    "injury",
    "game",
    "stat_display",
    "line",
    "over_probability_display",
    "under_probability_display",
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
        "injury": st.column_config.TextColumn("Injury", width="small"),
        "game": st.column_config.TextColumn("Game", width="medium"),
        "stat_display": st.column_config.TextColumn("Stat", width="small"),
        "line": st.column_config.NumberColumn("Line", format="%.1f"),
        "over_probability_display": st.column_config.TextColumn(
            "Over %",
            help="Model P(actual > Sleeper line). Blank when market has no model.",
            width="small",
        ),
        "under_probability_display": st.column_config.TextColumn(
            "Under %",
            help="Model P(actual ≤ Sleeper line) = 1 − Over %.",
            width="small",
        ),
        "over_multiplier": st.column_config.NumberColumn(
            "Over mult", format="%.2fx"
        ),
        "under_multiplier": st.column_config.NumberColumn(
            "Under mult", format="%.2fx"
        ),
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
    if "player_injury_status" in display.columns:
        display["injury"] = display["player_injury_status"].map(
            lambda s: str(s).strip() if pd.notna(s) and str(s).strip() else "—"
        )
    else:
        display["injury"] = "—"
    return display


def enrich_sleeper_with_model_probs(
    df: pd.DataFrame,
    version: str = "v2",
) -> pd.DataFrame:
    """
    Score each Sleeper line with the trained classifier when the market maps.

    Uses the exact Sleeper line (not sportsbook predictions.csv lines).
    """
    out = df.copy()
    out["over_probability"] = pd.NA
    out["under_probability"] = pd.NA

    if out.empty:
        out["over_probability_display"] = "—"
        out["under_probability_display"] = "—"
        return out

    version = normalize_version(version)
    feat_path = player_features_path(version)
    if not feat_path.exists() or not version_models_dir(version).exists():
        out["over_probability_display"] = "—"
        out["under_probability_display"] = "—"
        return out

    if "market" in out.columns:
        out["market"] = out["market"].replace(MARKET_ALIASES)

    from predict import (
        fuzzy_match_features,
        latest_defense_lookup,
        latest_feature_rows,
        load_model,
        overlay_slate_opponent,
        score_prop,
    )

    full_features = pd.read_parquet(feat_path)
    features = latest_feature_rows(full_features)
    seasons = sorted(
        full_features["season"].dropna().astype(int).unique().tolist()
    )
    defense_lookup = (
        latest_defense_lookup(seasons) if version == "v2" else pd.DataFrame()
    )

    models = {}
    for market in MARKET_STAT:
        bundle = load_model(market, version)
        if bundle is not None:
            models[market] = bundle

    if not models:
        out["over_probability_display"] = "—"
        out["under_probability_display"] = "—"
        return out

    overs = []
    unders = []
    for _, row in out.iterrows():
        market = row.get("market")
        line = row.get("line")
        player = row.get("player")
        if (
            market not in models
            or player is None
            or line is None
            or (isinstance(line, float) and pd.isna(line))
        ):
            overs.append(pd.NA)
            unders.append(pd.NA)
            continue

        feat = fuzzy_match_features(str(player), features)
        if feat is None:
            overs.append(pd.NA)
            unders.append(pd.NA)
            continue

        if version == "v2":
            # Prefer Sleeper game teams when Odds-style names are missing.
            prop_like = pd.Series({
                "home_team": row.get("home_team") or _infer_home(row),
                "away_team": row.get("away_team") or _infer_away(row),
            })
            feat = overlay_slate_opponent(feat, prop_like, defense_lookup)

        try:
            over_p = score_prop(models[market], feat, float(line))
            overs.append(over_p)
            unders.append(1.0 - over_p)
        except Exception:
            overs.append(pd.NA)
            unders.append(pd.NA)

    out["over_probability"] = overs
    out["under_probability"] = unders
    out["over_probability_display"] = out["over_probability"].map(
        lambda v: format_pct(v) if pd.notna(v) else "—"
    )
    out["under_probability_display"] = out["under_probability"].map(
        lambda v: format_pct(v) if pd.notna(v) else "—"
    )
    return out


def _infer_home(row: pd.Series) -> str | None:
    game = str(row.get("game") or "")
    if " @ " in game:
        return game.split(" @ ", 1)[1].strip() or None
    return None


def _infer_away(row: pd.Series) -> str | None:
    game = str(row.get("game") or "")
    if " @ " in game:
        return game.split(" @ ", 1)[0].strip() or None
    return None


def _apply_filters(df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    if df.empty:
        return df

    filtered = df.copy()

    stats = sorted(
        s
        for s in filtered["stat_display"].dropna().unique().tolist()
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
        t
        for t in filtered["player_team"].dropna().unique().tolist()
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

    hide_out = st.checkbox(
        "Hide injured (Out / Doubtful / IR when present)",
        value=True,
        key=f"{key_prefix}_hide_out",
    )
    if hide_out and "player_injury_status" in filtered.columns:
        status = filtered["player_injury_status"].astype(str).str.lower()
        inactive = status.str.contains(
            r"\bout\b|doubtful|\bir\b|injured reserve",
            regex=True,
            na=False,
        )
        filtered = filtered[~inactive]

    only_modeled = st.checkbox(
        "Only rows with model Over/Under %",
        value=False,
        key=f"{key_prefix}_only_modeled",
        help="Hide Sleeper markets we do not train (e.g. anytime TD, longest play).",
    )
    if only_modeled and "over_probability" in filtered.columns:
        filtered = filtered[filtered["over_probability"].notna()]

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
    version: str = "v2",
) -> None:
    raw = props_df if props_df is not None else load_sleeper_props()

    if raw.empty:
        st.warning(
            "No Sleeper props loaded. Add **APIFY_TOKEN** to `.env`, then run "
            "`python fetch_data.py --sleeper-props` or include it with `--props`."
        )
        st.caption(f"Expected file: `{SLEEPER_PROPS_PATH}`")
        return

    fetched = (
        raw["fetched_at"].dropna().max()
        if "fetched_at" in raw.columns
        else None
    )
    if fetched is not None and pd.notna(fetched):
        st.caption(
            f"**{len(raw):,}** NFL props from Sleeper Picks "
            f"(fetched {format_commence_time(fetched)}). "
            f"Over/Under % from model **{version}** on the Sleeper line."
        )
    else:
        st.caption(
            f"**{len(raw):,}** NFL props from Sleeper Picks. "
            f"Over/Under % from model **{version}**."
        )

    filtered = _apply_filters(raw, key_prefix)
    if filtered.empty:
        st.info("No props match the current filters.")
        return

    show_cols = [c for c in DISPLAY_COLUMNS if c in filtered.columns]
    st.dataframe(
        filtered[show_cols],
        width="stretch",
        hide_index=True,
        column_config=_column_config(),
    )

    scored = (
        int(filtered["over_probability"].notna().sum())
        if "over_probability" in filtered.columns
        else 0
    )
    st.caption(
        f"Model Over/Under % on {scored:,} / {len(filtered):,} visible rows "
        f"(mapped markets only: pass/rush/rec yards, receptions, pass TDs, "
        f"rush+rec yards)."
    )

    market_counts = (
        filtered.groupby("stat_display", dropna=False)
        .size()
        .sort_values(ascending=False)
    )
    with st.expander("Stat breakdown"):
        breakdown = market_counts.reset_index()
        breakdown.columns = ["Stat", "Count"]
        st.dataframe(breakdown, hide_index=True, width="stretch")

    if "market" in filtered.columns:
        mapped = [
            m
            for m in filtered["market"].dropna().unique().tolist()
            if not str(m).startswith("sleeper_")
        ]
        if mapped:
            labels = ", ".join(sorted({market_label(m) for m in mapped}))
            st.caption(f"Mapped model markets where applicable: {labels}")
