"""Player detail page for the MLB Prop Model UI."""

from urllib.parse import unquote

import altair as alt
import pandas as pd
import streamlit as st

from ui.batter_score import render_batter_score_summary
from ui.formatting import (
    format_commence_time,
    format_odds,
    market_label,
    prepare_display_df,
    style_probability_extremes,
)
from ui.glossary import EDGE_CALLOUT, GLOSSARY
from ui.pick_builder import render_player_market_add_controls
from ui.player_stats import (
    count_games_vs_opponent,
    get_features_max_game_date,
    get_stat_history,
    infer_player_kind,
    markets_for_kind,
    opponent_display_name,
    slate_opponent_abbr,
    window_average,
    MARKET_STAT_MAP,
)


_LAST_GAMES_AXIS = alt.Axis(
    labelFontSize=15,
    labelFontWeight="bold",
    titleFontSize=15,
    titleFontWeight="bold",
)


def _last_games_bar_chart(game_log):
    """Bar chart of per-game stats with prominent axis labels."""
    chart_df = game_log.reset_index()
    game_col = chart_df.columns[0]
    stat_col = chart_df.columns[1]
    stat_title = stat_col.replace("_", " ").title()

    return (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X(
                f"{game_col}:N",
                title="Game",
                sort=list(chart_df[game_col]),
                axis=_LAST_GAMES_AXIS,
            ),
            y=alt.Y(
                f"{stat_col}:Q",
                title=stat_title,
                axis=_LAST_GAMES_AXIS,
            ),
        )
        .properties(height=250)
    )


def _metric_with_help(label, value, help_key):
    col_metric, col_help = st.columns([5, 1])
    with col_metric:
        st.metric(label, value)
    with col_help:
        with st.popover("?"):
            st.markdown(GLOSSARY[help_key])


def _market_table_column_config():
    return {
        "bookmaker": st.column_config.TextColumn(
            "Book",
            help=GLOSSARY["book"],
        ),
        "side": st.column_config.TextColumn(
            "Side",
            help=GLOSSARY["side"],
        ),
        "line": st.column_config.NumberColumn(
            "Line",
            help=GLOSSARY["line"],
            format="%.1f",
        ),
        "odds": st.column_config.TextColumn(
            "Odds",
            help=GLOSSARY["odds"],
        ),
        "over_probability": st.column_config.NumberColumn(
            "Over %",
            help=GLOSSARY["over_pct"],
            format="%.1f",
        ),
        "under_probability": st.column_config.NumberColumn(
            "Under %",
            help=GLOSSARY["under_pct"],
            format="%.1f",
        ),
        "model_probability": st.column_config.NumberColumn(
            "Model %",
            help=GLOSSARY["model_pct"],
            format="%.1f",
        ),
        "market_probability": st.column_config.NumberColumn(
            "Market %",
            help=GLOSSARY["market_pct"],
            format="%.1f",
        ),
        "devigged_market_prob": st.column_config.NumberColumn(
            "Devigged %",
            help=GLOSSARY["devigged_market_pct"],
            format="%.1f",
        ),
        "edge": st.column_config.NumberColumn(
            "Edge %",
            help=GLOSSARY["edge_pct"],
            format="%.1f",
        ),
        "consensus_edge": st.column_config.NumberColumn(
            "Consensus Edge %",
            help=GLOSSARY["consensus_edge"],
            format="%.1f",
        ),
        "best_book": st.column_config.TextColumn(
            "Best Book",
            help=GLOSSARY["best_book"],
        ),
        "best_ev": st.column_config.NumberColumn(
            "Best EV %",
            help=GLOSSARY["best_ev"],
            format="%.1f",
        ),
        "ev": st.column_config.NumberColumn(
            "EV %",
            help=GLOSSARY["ev_pct"],
            format="%.1f",
        ),
        "line_delta": st.column_config.NumberColumn(
            "Line Δ",
            help=GLOSSARY["line_delta"],
            format="%.1f",
        ),
        "steam_flag": st.column_config.TextColumn(
            "Steam",
            help=GLOSSARY["steam_flag"],
        ),
    }


def _parse_slate_teams(player_df):
    if "home_team" in player_df.columns and "away_team" in player_df.columns:
        home = player_df["home_team"].iloc[0]
        away = player_df["away_team"].iloc[0]
        if pd.notna(home) and pd.notna(away):
            return str(away), str(home)

    game = player_df["game"].iloc[0] if "game" in player_df.columns else ""
    if isinstance(game, str) and " @ " in game:
        away, home = game.split(" @ ", 1)
        return away.strip(), home.strip()

    return None, None


def _render_stat_history_section(player_name, player_df, version):
    """Dropdown stat history with L5/L10 window toggle and optional H2H filter."""
    player_kind = infer_player_kind(player_df["market"].unique())
    available_markets = markets_for_kind(player_kind)

    if not available_markets:
        return

    st.subheader("Stat history")

    max_date = get_features_max_game_date(player_kind, version=version)
    if max_date:
        st.caption(f"Game logs through {max_date}")

    away_team, home_team = _parse_slate_teams(player_df)
    slate_opponent = slate_opponent_abbr(
        player_name,
        home_team,
        away_team,
        player_kind,
        version=version,
    )
    opponent_label = opponent_display_name(slate_opponent)

    col_market, col_scope, col_window, col_help = st.columns([3, 2, 2, 1])
    with col_market:
        selected_market = st.selectbox(
            "Stat market",
            available_markets,
            format_func=market_label,
            key="player_stat_history_market",
        )
    with col_scope:
        scope = st.segmented_control(
            "Opponent filter",
            options=["All", "H2H"],
            default="All",
            key="player_stat_history_scope",
        )
    with col_window:
        window = st.segmented_control(
            "Games window",
            options=[5, 10],
            default=10,
            key="player_stat_history_window",
        )
    with col_help:
        with st.popover("?"):
            st.markdown(GLOSSARY["stat_history"])

    use_h2h = scope == "H2H"
    opponent_abbr = slate_opponent if use_h2h else None

    if use_h2h and opponent_abbr is None:
        st.caption(
            "H2H filter unavailable — could not match today's opponent "
            "from slate teams and feature history."
        )
        use_h2h = False
        opponent_abbr = None

    game_log = get_stat_history(
        player_name,
        selected_market,
        version=version,
        n=10,
        opponent_abbr=opponent_abbr,
    )

    if game_log is None or game_log.empty:
        if use_h2h:
            st.caption(
                f"No game history vs **{opponent_label}** for this stat."
            )
        else:
            st.caption("No game history available for this stat.")
        return

    _, stat_col = MARKET_STAT_MAP[selected_market]
    display_log = game_log.tail(window)

    l5_avg = window_average(game_log, stat_col, 5)
    l10_avg = window_average(game_log, stat_col, 10)

    h2h_total = (
        count_games_vs_opponent(
            player_name,
            selected_market,
            slate_opponent,
            version=version,
        )
        if use_h2h and slate_opponent
        else None
    )

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("L5 avg", f"{l5_avg:.2f}" if not pd.isna(l5_avg) else "—")
    with m2:
        st.metric("L10 avg", f"{l10_avg:.2f}" if not pd.isna(l10_avg) else "—")
    with m3:
        highlighted = l5_avg if window == 5 else l10_avg
        label = f"L{window} avg (selected)"
        st.metric(
            label,
            f"{highlighted:.2f}" if not pd.isna(highlighted) else "—",
        )

    if use_h2h:
        shown = len(display_log)
        st.caption(
            f"Last {shown} of {h2h_total} career games in logs vs "
            f"**{opponent_label}** — {market_label(selected_market)}"
        )
    else:
        st.caption(
            f"Last {window} games — {market_label(selected_market)}"
        )

    st.altair_chart(_last_games_bar_chart(display_log))


def render_back_to_board(*query_keys):
    if st.button("← Back to board", type="secondary"):
        for key in query_keys:
            if key in st.query_params:
                del st.query_params[key]
        st.rerun()


def render_back_button():
    render_back_to_board("player")


def render_player_page(df, player_name, version):
    player_name = unquote(player_name)
    render_back_button()

    player_df = df[df["player"] == player_name]

    if player_df.empty:
        st.error(f"No predictions found for **{player_name}**.")
        st.caption("The player may not be on today's slate or name may not match.")
        return

    game = player_df["game"].iloc[0]
    commence = format_commence_time(player_df["commence_time"].iloc[0])

    st.title(player_name)
    st.caption(f"{game} · First pitch: {commence}")

    st.info(EDGE_CALLOUT)

    render_batter_score_summary(
        player_name,
        version=version,
        game=game,
        commence_time=player_df["commence_time"].iloc[0],
        home_team=player_df["home_team"].iloc[0]
        if "home_team" in player_df.columns
        else "",
        away_team=player_df["away_team"].iloc[0]
        if "away_team" in player_df.columns
        else "",
    )

    st.divider()

    _render_stat_history_section(player_name, player_df, version)

    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _metric_with_help(
            "Best Edge",
            f"{player_df['edge'].max() * 100:.1f}%",
            "best_edge",
        )
    with c2:
        _metric_with_help(
            "Best EV",
            f"{player_df['ev'].max() * 100:.1f}%",
            "best_ev",
        )
    with c3:
        _metric_with_help(
            "Props",
            len(player_df),
            "prop_count",
        )
    with c4:
        _metric_with_help(
            "Markets",
            player_df["market"].nunique(),
            "market_count",
        )

    st.divider()

    markets = sorted(
        player_df["market"].unique(),
        key=lambda m: player_df[player_df["market"] == m]["edge"].max(),
        reverse=True,
    )

    table_columns = [
        "bookmaker",
        "side",
        "line",
        "odds",
        "over_probability",
        "under_probability",
        "model_probability",
        "market_probability",
        "devigged_market_prob",
        "edge",
        "consensus_edge",
        "best_book",
        "best_ev",
        "ev",
        "line_delta",
        "steam_flag",
    ]

    optional_columns = [
        col
        for col in table_columns
        if col in player_df.columns
    ]

    for market in markets:
        market_rows = (
            player_df[player_df["market"] == market]
            .sort_values("edge", ascending=False)
            .copy()
        )

        col_title, col_help = st.columns([10, 1])
        with col_title:
            st.subheader(market_label(market))
        with col_help:
            with st.popover("?"):
                st.markdown(GLOSSARY["market"])

        if "consensus_line" in market_rows.columns:
            consensus_line = market_rows["consensus_line"].dropna()
            if len(consensus_line):
                st.caption(
                    f"Consensus line: **{consensus_line.iloc[0]:.1f}** "
                    f"({GLOSSARY['consensus_line']})"
                )

        display = prepare_display_df(market_rows)
        display["odds"] = market_rows["odds"].apply(format_odds)

        if "line_delta" in display.columns:
            display["line_delta"] = display["line_delta"].astype(float)

        if "steam_flag" in display.columns:
            display["steam_flag"] = market_rows["steam_flag"].map(
                lambda value: "🔥" if bool(value) else ""
            )

        st.dataframe(
            style_probability_extremes(display[optional_columns]),
            hide_index=True,
            column_config=_market_table_column_config(),
        )

        render_player_market_add_controls(market_rows, market, player_name)

    st.divider()
    st.caption(
        "This is a statistical research tool, "
        "not a guarantee of betting outcomes."
    )
