"""Sleeper H+R+RBI 0.5 board with Hitter's Life enrichment columns."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ui.table_display import show_dataframe

from batter_score_data import build_game_context, lookup_batter_score
from fetch_sleeper_props import SLEEPER_PROPS_PATH
from hitters_life_data import (
    build_vs_pitcher_fields,
    format_h2h_avg_display,
    format_pitch_woba,
    format_total_bases_game_log,
    format_wrc_plus_column,
    format_xwoba_column,
    hitters_life_player_link,
    lookup_arsenal_weighted_woba,
)
from sleeper_analyzer.hit_rates import HIT_RATE_RULE, compute_hit_rates
from utils import TEAM_ABBR_TO_ODDS
from ui.formatting import format_commence_time, format_name_with_hand, format_pct
from ui.hitters_life_highlights import render_tb_log_color_legend, style_hitters_life_board
from ui.player_stats import lookup_pitcher_hand

HRR_MARKET = "batter_hits_runs_rbis"
HRR_SLEEPER_STAT = "hits_runs_rbis"
TARGET_LINE = 0.5

SLEEPER_DISPLAY_COLUMNS = [
    "player_link",
    "game",
    "stat_display",
    "line",
    "l5_hit_rate",
    "over_multiplier",
    "under_multiplier",
    "pick_popularity",
    "player_team",
    "game_start",
]

ENRICHMENT_COLUMNS = [
    "opposing_sp",
    "h2h_avg",
    "arsenal_woba",
    "xwoba",
    "wrc_plus",
    "total_bases_log",
]

DISPLAY_COLUMNS = SLEEPER_DISPLAY_COLUMNS + ENRICHMENT_COLUMNS


def filter_sleeper_hrr_half_props(df: pd.DataFrame) -> pd.DataFrame:
    """Keep Sleeper Hits + Runs + RBIs props at line 0.5."""
    if df.empty:
        return df

    work = df.copy()
    line = pd.to_numeric(work.get("line"), errors="coerce")
    market_ok = (
        work["market"].astype(str).eq(HRR_MARKET)
        if "market" in work.columns
        else pd.Series(False, index=work.index)
    )
    stat_ok = (
        work["sleeper_stat"].astype(str).eq(HRR_SLEEPER_STAT)
        if "sleeper_stat" in work.columns
        else pd.Series(False, index=work.index)
    )
    mask = (market_ok | stat_ok) & line.eq(TARGET_LINE)
    return work[mask].reset_index(drop=True)


def _odds_team_name(abbr, fallback: str | None = None) -> str | None:
    """Map Sleeper team abbr to Odds API / probables team name."""
    key = str(abbr or "").strip().upper()
    if key in TEAM_ABBR_TO_ODDS:
        return TEAM_ABBR_TO_ODDS[key]
    text = str(fallback or "").strip()
    return text or None


def _game_context_from_sleeper_row(row: pd.Series) -> dict | None:
    commence = row.get("_game_start_raw")
    if commence is None or (isinstance(commence, float) and pd.isna(commence)):
        commence = row.get("game_start")

    game_str = str(row.get("game") or "")
    parsed_away = parsed_home = None
    if " @ " in game_str:
        parsed_away, parsed_home = [
            part.strip() for part in game_str.split(" @ ", 1)
        ]

    away_team = _odds_team_name(row.get("away_team"), parsed_away)
    home_team = _odds_team_name(row.get("home_team"), parsed_home)

    return build_game_context(
        game=game_str,
        commence_time=commence,
        home_team=home_team,
        away_team=away_team,
    )


def _format_sleeper_display_fields(row: pd.Series) -> dict:
    pick_pop = row.get("pick_popularity")
    game_start = row.get("_game_start_raw")
    if game_start is None or (isinstance(game_start, float) and pd.isna(game_start)):
        game_start = row.get("game_start")

    stat_display = row.get("stat_display")
    if stat_display is None or (
        isinstance(stat_display, float) and pd.isna(stat_display)
    ):
        sleeper_stat = row.get("sleeper_stat")
        stat_display = (
            str(sleeper_stat).replace("_", " ").title()
            if pd.notna(sleeper_stat)
            else "—"
        )

    return {
        "stat_display": stat_display,
        "pick_popularity": (
            format_pct(pick_pop) if pd.notna(pick_pop) else "—"
        ),
        "game_start": format_commence_time(game_start),
    }


def _format_l5_hit_rate(l5_pct: float | None) -> str:
    if l5_pct is None or pd.isna(l5_pct):
        return "—"
    return format_pct(float(l5_pct))


def enrich_sleeper_hrr_row(row: pd.Series, *, version: str) -> dict:
    player = str(row["player"])
    game_context = _game_context_from_sleeper_row(row)
    result = lookup_batter_score(
        player,
        version=version,
        game_context=game_context,
    )

    sp_display = "TBD"
    if result and result.opposing_sp_name:
        sp_hand = lookup_pitcher_hand(result.opposing_sp_name, version=version)
        sp_display = format_name_with_hand(result.opposing_sp_name, sp_hand)

    vs_pitcher = build_vs_pitcher_fields(
        player,
        version,
        game_context,
        result,
        sp_display=sp_display,
    )

    line_value = pd.to_numeric(row.get("line"), errors="coerce")
    l5_pct = None
    if pd.notna(line_value):
        l5_pct, _l10_pct = compute_hit_rates(
            player,
            HRR_MARKET,
            float(line_value),
            version=version,
        )

    built = row.to_dict()
    built.update(_format_sleeper_display_fields(row))
    built["player_link"] = hitters_life_player_link(player)
    built["l5_hit_rate"] = _format_l5_hit_rate(l5_pct)
    built["_l5_pct"] = l5_pct
    built["opposing_sp"] = vs_pitcher["opposing_sp"]
    built["h2h_avg"] = format_h2h_avg_display(
        vs_pitcher["h2h_avg"],
        hits=vs_pitcher["h2h_hits"],
        ab=vs_pitcher["h2h_ab"],
        hr=vs_pitcher.get("h2h_hr"),
    )
    built["_h2h_avg"] = vs_pitcher["h2h_avg"]
    built["arsenal_woba"] = format_pitch_woba(
        lookup_arsenal_weighted_woba(player, version, game_context),
    )
    built["xwoba"] = format_xwoba_column(player, version)
    built["wrc_plus"] = format_wrc_plus_column(player, version)
    built["total_bases_log"] = format_total_bases_game_log(
        player,
        version=version,
    )
    return built


def build_sleeper_hrr_board_df(
    props: pd.DataFrame,
    *,
    version: str,
) -> pd.DataFrame:
    filtered = filter_sleeper_hrr_half_props(props)
    if filtered.empty:
        return pd.DataFrame()

    work = filtered.copy()
    work["_game_start_raw"] = work.get("game_start")

    rows = [
        enrich_sleeper_hrr_row(row, version=version)
        for _, row in work.iterrows()
    ]
    result = pd.DataFrame(rows)
    if "_h2h_avg" in result.columns:
        result = result.sort_values(
            "_h2h_avg",
            ascending=False,
            na_position="last",
        ).reset_index(drop=True)
    return result


@st.cache_data(show_spinner="Building Sleeper H+R+RBI board…")
def load_sleeper_hrr_board(props_mtime: float, version: str) -> pd.DataFrame:
    _ = props_mtime
    if not SLEEPER_PROPS_PATH.exists():
        return pd.DataFrame()
    props = pd.read_parquet(SLEEPER_PROPS_PATH)
    return build_sleeper_hrr_board_df(props, version=version)


def _column_config() -> dict:
    from hitters_life_data import MIN_PA_H2H_BOARD

    return {
        "player_link": st.column_config.LinkColumn(
            "Player",
            help="Open player profile.",
            display_text=r"#(.*)$",
        ),
        "game": st.column_config.TextColumn("Game", width="medium"),
        "stat_display": st.column_config.TextColumn("Stat", width="small"),
        "line": st.column_config.NumberColumn("Line", format="%.1f"),
        "l5_hit_rate": st.column_config.TextColumn(
            "L5 hit rate",
            help=HIT_RATE_RULE,
        ),
        "over_multiplier": st.column_config.NumberColumn("Over mult", format="%.2fx"),
        "under_multiplier": st.column_config.NumberColumn("Under mult", format="%.2fx"),
        "pick_popularity": st.column_config.TextColumn("Over pick %"),
        "player_team": st.column_config.TextColumn("Team", width="small"),
        "game_start": st.column_config.TextColumn("Start (ET)"),
        "opposing_sp": st.column_config.TextColumn(
            "Opposing SP",
            help="Probable opposing starter (throw hand when known).",
        ),
        "h2h_avg": st.column_config.TextColumn(
            "H2H AVG",
            help=(
                "Career batting average vs this starter as hits/AB and AVG "
                f"(PA ≥ {MIN_PA_H2H_BOARD}). Light green when above .300."
            ),
        ),
        "arsenal_woba": st.column_config.TextColumn(
            "Arsenal wOBA",
            help=(
                "Usage-weighted career wOBA vs the opposing SP's pitch mix "
                "(last 5 starts), by pitch bucket."
            ),
        ),
        "xwoba": st.column_config.TextColumn(
            "xwOBA",
            help=(
                "Expected wOBA (Statcast): last 5 and last 10 games. "
                "Same methodology as the Hitter's Life board."
            ),
        ),
        "wrc_plus": st.column_config.TextColumn(
            "wRC+",
            help=(
                "Weighted runs created plus (100 = league average): last 30 "
                "and last 10 games (pooled PAs). Same as the Hitter's Life board."
            ),
        ),
        "total_bases_log": st.column_config.TextColumn(
            "TB per game (L5)",
            help=(
                "Total bases in each of the last 5 games (space-separated). "
                "Leftmost number is the most recent game. Colors match the "
                "Hitter's Life TB log board."
            ),
        ),
    }


def render_sleeper_hrr_board(
    *,
    version: str,
    props_mtime: float,
    key_prefix: str = "sleeper_hrr",
) -> None:
    st.markdown("##### Sleeper H+R+RBI 0.5")
    st.caption(
        "All Sleeper Hits + Runs + RBIs props at **0.5**, with Sleeper line "
        "columns plus L5 hit rate (stat > line), H2H AVG, Arsenal wOBA, xwOBA, wRC+, and TB per game "
        "(L5) from the same sources as the batting-average and Hitter's Life "
        "boards. Sorted by H2H AVG vs today's probable starter."
    )
    render_tb_log_color_legend()

    board = load_sleeper_hrr_board(props_mtime, version)
    if board.empty:
        st.info(
            "No Sleeper H+R+RBI 0.5 props in cache. Run "
            "`python fetch_data.py --props` or `./run_daily.sh`."
        )
        return

    st.caption(f"**{len(board):,}** props at line 0.5.")

    show_cols = [col for col in DISPLAY_COLUMNS if col in board.columns]
    styled = style_hitters_life_board(board[show_cols + ["_h2h_avg", "player"]])
    show_dataframe(
        styled,
        hide_index=True,
        column_config=_column_config(),
        key=f"{key_prefix}_table",
    )
