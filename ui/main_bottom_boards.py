"""Bottom-of-main-board sections: per-market top props and hot-bat AVG batter scores."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from batter_score_data import build_game_context
from hitters_life_data import (
    format_batting_average_column,
    format_batting_average_vs_hand_column,
    format_opposing_sp_baa_column,
    format_pitch_woba,
    format_total_bases_game_log,
    format_wrc_plus_column,
    format_xwoba_column,
    lookup_arsenal_weighted_woba,
    lookup_batting_average_windows,
)
from ui.batter_score_board import (
    _batter_score_table_column_config,
    _build_batter_score_row,
    _hand_split_avg_column_config,
    _prepare_batter_score_slate,
    style_batter_score_board,
)
from ui.formatting import market_label, style_probability_extremes
from ui.glossary import GLOSSARY
from ui.hitters_life_highlights import (
    batting_average_has_board_highlight,
    hot_batter_tiebreak_key,
)

MARKET_TOP_PROPS_PER_SIDE = 3
HOT_BATTER_L5_TOP_N = 15
HOT_BATTER_SCORE_LIMIT = 20
HOT_BATTER_SCORE_TIE_TOLERANCE = 0.2

MARKET_TOP_PROPS_COLUMNS = [
    "market",
    "side_rank",
    "player_link",
    "game_time",
    "bookmaker",
    "line",
    "side",
    "over_probability",
    "under_probability",
    "l5_l10_pct",
    "edge",
]

HOT_BATTER_SCORE_COLUMNS = [
    "player_link",
    "opposing_sp",
    "vs_pitcher",
    "arsenal_woba",
    "batting_average",
    "avg_vs_rhp",
    "avg_vs_lhp",
    "sp_baa",
    "xwoba",
    "wrc_plus",
    "pp_fantasy_line",
    "ud_fantasy_line",
    "l5_l10_pct",
    "batter_score_display",
    "batter_score_v2_display",
    "batter_score_hybrid_display",
    "total_bases_log",
]


def build_market_top_props_df(
    filtered: pd.DataFrame,
    *,
    per_market: int = MARKET_TOP_PROPS_PER_SIDE,
) -> pd.DataFrame:
    """Top *per_market* Over % and Under % props for each market in *filtered*."""
    if filtered.empty or "market" not in filtered.columns:
        return pd.DataFrame()

    rows: list[pd.Series] = []
    for market in sorted(filtered["market"].dropna().unique()):
        subset = filtered[filtered["market"].eq(market)]
        if subset.empty:
            continue

        if "over_probability" in subset.columns:
            over_top = subset.nlargest(per_market, "over_probability")
            for _, row in over_top.iterrows():
                tagged = row.copy()
                tagged["_side_rank"] = "Over"
                rows.append(tagged)

        if "under_probability" in subset.columns:
            under_top = subset.nlargest(per_market, "under_probability")
            for _, row in under_top.iterrows():
                tagged = row.copy()
                tagged["_side_rank"] = "Under"
                rows.append(tagged)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).reset_index(drop=True)


def _l5_avg_top_players(
    players: list[str],
    version: str,
    *,
    top_n: int = HOT_BATTER_L5_TOP_N,
) -> set[str]:
    """Players ranked in the top *top_n* by L5 batting average (Statcast)."""
    ranked: list[tuple[str, float]] = []
    for player in players:
        _, l5_avg, _ = lookup_batting_average_windows(player, version=version)
        if l5_avg is not None and not pd.isna(l5_avg):
            ranked.append((player, float(l5_avg)))

    ranked.sort(key=lambda item: item[1], reverse=True)
    return {player for player, _avg in ranked[:top_n]}


def _sort_hot_batter_score_rows(
    rows: list[dict],
    *,
    tie_tolerance: float = HOT_BATTER_SCORE_TIE_TOLERANCE,
) -> list[dict]:
    """
    Rank by batter score, then by independent board highlight ranks.

    Scores that match or differ by *tie_tolerance* or less: **blue TB soarer**
    first (TB-log board only), then batting-AVG colors (green > orange >
    yellow). Those color rules stay on their own columns.
    """
    if not rows:
        return rows

    def _score(row: dict) -> float:
        value = pd.to_numeric(row.get("_batter_score"), errors="coerce")
        return float(value) if pd.notna(value) else float("-inf")

    ordered = sorted(rows, key=_score, reverse=True)
    ranked: list[tuple[int, dict]] = []
    group_rank = 0
    group_score = _score(ordered[0])

    for row in ordered:
        score = _score(row)
        if abs(score - group_score) > tie_tolerance:
            group_rank += 1
            group_score = score
        ranked.append((group_rank, row))

    ranked.sort(
        key=lambda item: (
            item[0],
            tuple(
                -rank
                for rank in hot_batter_tiebreak_key(
                    item[1].get("_batting_average"),
                    item[1].get("_total_bases_log"),
                )
            ),
            -_score(item[1]),
        )
    )
    return [row for _rank, row in ranked]


def build_hot_batter_score_df(
    df: pd.DataFrame,
    version: str,
    *,
    markets=None,
    l5_top_n: int = HOT_BATTER_L5_TOP_N,
    limit: int = HOT_BATTER_SCORE_LIMIT,
) -> pd.DataFrame:
    """
    Top batter scores among hitters in the top *l5_top_n* L5 AVG who also
    match a Hitter's Life batting-average highlight (green / orange / yellow).
    """
    slate = _prepare_batter_score_slate(df, markets=markets)
    if slate.empty:
        return pd.DataFrame()

    l5_elite = _l5_avg_top_players(
        slate["player"].tolist(),
        version,
        top_n=l5_top_n,
    )
    if not l5_elite:
        return pd.DataFrame()

    qualified_rows = []
    for _, row in slate.iterrows():
        player = row["player"]
        if player not in l5_elite:
            continue

        batting_average = format_batting_average_column(player, version)
        if not batting_average_has_board_highlight(batting_average):
            continue

        built = _build_batter_score_row(row, version)
        tb_log = format_total_bases_game_log(player, version=version)
        game_context = build_game_context(
            game=row.get("game"),
            commence_time=row.get("commence_time"),
            home_team=row.get("home_team"),
            away_team=row.get("away_team"),
        )
        built["arsenal_woba"] = format_pitch_woba(
            lookup_arsenal_weighted_woba(player, version, game_context),
        )
        built["batting_average"] = batting_average
        built["avg_vs_rhp"] = format_batting_average_vs_hand_column(
            player,
            version,
            hand="R",
        )
        built["avg_vs_lhp"] = format_batting_average_vs_hand_column(
            player,
            version,
            hand="L",
        )
        built["sp_baa"] = format_opposing_sp_baa_column(
            player,
            version,
            game_context,
        )
        built["xwoba"] = format_xwoba_column(player, version)
        built["wrc_plus"] = format_wrc_plus_column(player, version)
        built["_batting_average"] = batting_average
        built["total_bases_log"] = tb_log
        built["_total_bases_log"] = tb_log
        qualified_rows.append(built)

    if not qualified_rows:
        return pd.DataFrame()

    result = pd.DataFrame(
        _sort_hot_batter_score_rows(qualified_rows)
    ).head(limit)
    return result.reset_index(drop=True)


def _hot_batter_score_column_config():
    config = _batter_score_table_column_config()
    config.update(_hand_split_avg_column_config())
    config["arsenal_woba"] = st.column_config.TextColumn(
        "Arsenal wOBA",
        help=(
            "Usage-weighted career wOBA vs the opposing SP's pitch mix "
            "(last 5 starts), by pitch bucket."
        ),
    )
    config["batting_average"] = st.column_config.TextColumn(
        "Batting average",
        help=(
            "Season and rolling AVG (Statcast). Green / orange / yellow "
            "highlights match the Hitter's Life batting board."
        ),
    )
    config["xwoba"] = st.column_config.TextColumn(
        "xwOBA",
        help=(
            "Expected wOBA (Statcast): last 5 and last 10 games. "
            "L5 is listed first for sorting. Same Savant methodology as "
            "the Hitter's Life board."
        ),
    )
    config["wrc_plus"] = st.column_config.TextColumn(
        "wRC+",
        help=(
            "Weighted runs created plus (100 = league average): last 30 "
            "and last 10 games (pooled PAs). L30 is listed first for sorting."
        ),
    )
    config["total_bases_log"] = st.column_config.TextColumn(
        "TB per game (L5)",
        help=(
            "Total bases in each of the last 5 games (space-separated). "
            "Leftmost number is the most recent game. Colors match the "
            "Hitter's Life batting board TB log only: blue soarer, green "
            "money, orange hot, yellow warm."
        ),
    )
    config["batter_score_hybrid_display"] = st.column_config.TextColumn(
        "Batter score hybrid",
        help=GLOSSARY["batter_score_hybrid"],
    )
    return config


def style_hot_batter_score_board(full_df: pd.DataFrame):
    """Batter-score styling plus TB-log and batting-AVG colors when present."""
    display_cols = [
        column
        for column in HOT_BATTER_SCORE_COLUMNS
        if column in full_df.columns
    ]
    meta_cols = [
        column
        for column in full_df.columns
        if column.startswith("_") or column in {"player", "batter_score_label"}
    ]
    return style_batter_score_board(full_df[display_cols + meta_cols])


def _market_top_props_column_config():
    from ui.board import _ranking_column_config

    config = _ranking_column_config()
    config["market"] = st.column_config.TextColumn(
        "Market",
        help="Prop market for this row.",
    )
    config["side_rank"] = st.column_config.TextColumn(
        "Top",
        help="Top 3 by Over % or Under % within the market.",
    )
    return config


def render_market_top_props_board(
    filtered: pd.DataFrame,
    key_prefix: str,
    *,
    version: str = "v2",
):
    """Top 3 Over % and Under % props for every market in the current filter set."""
    from ui.board import _prepare_board_table_df

    st.markdown("##### Top props by market")
    st.caption(
        "Top **3 Over %** and top **3 Under %** for **each market** in the "
        "current filter set (after Market / Edge / EV filters; one best book "
        "per player and market). "
        f"{GLOSSARY['over_pct']} {GLOSSARY['under_pct']}"
    )

    market_df = build_market_top_props_df(filtered)
    if market_df.empty:
        st.caption("No props in the current filter set.")
        return

    display = _prepare_board_table_df(market_df, version=version)
    display["market"] = market_df["market"].map(market_label).values
    display["side_rank"] = market_df["_side_rank"].values

    table = display[
        [col for col in MARKET_TOP_PROPS_COLUMNS if col in display.columns]
    ]

    st.dataframe(
        style_probability_extremes(table),
        hide_index=True,
        height=min(42 * len(table) + 38, 640),
        column_config=_market_top_props_column_config(),
        key=f"{key_prefix}_market_top_props",
    )


def render_hot_batter_score_board(
    df: pd.DataFrame,
    key_prefix: str,
    *,
    version: str = "v2",
):
    """Top batter scores with elite L5 AVG and batting-board color rules."""
    markets = st.session_state.get(f"{key_prefix}_markets", [])
    hot_df = build_hot_batter_score_df(
        df,
        version,
        markets=markets or None,
    )

    st.markdown("##### Hot batters — batter score")
    st.caption(
        f"Top **{HOT_BATTER_SCORE_LIMIT}** Batter Scores among hitters who rank "
        f"in the top **{HOT_BATTER_L5_TOP_N}** L5 batting averages on today's "
        "slate **and** match a Hitter's Life batting-average highlight: "
        "**green** (L10 > .290 & L5 > .250), **orange** (L5 > .299), or "
        "**yellow** (season > .300). Scores that are identical or within "
        "**0.2** are ordered by **blue TB soarer** first (TB-log board "
        "only), then batting-AVG color priority. Those color rules stay "
        "separate between boards. Includes **Arsenal wOBA**, **xwOBA**, "
        "and **wRC+** from the Hitter's Life batting board. "
        "Respects the Market type filter only (not Edge / EV). "
        f"{GLOSSARY['batter_score']} {GLOSSARY['batter_score_v2']} "
        f"{GLOSSARY['batter_score_hybrid']}"
    )

    if hot_df.empty:
        st.caption(
            "No batters meet the L5 AVG and batting-average highlight criteria."
        )
        return

    st.dataframe(
        style_hot_batter_score_board(hot_df),
        hide_index=True,
        height=min(42 * len(hot_df) + 38, 520),
        column_config=_hot_batter_score_column_config(),
        key=f"{key_prefix}_hot_batter_score",
    )
