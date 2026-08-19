"""Player detail page for the MLB Prop Model UI."""

from urllib.parse import unquote

import altair as alt
import streamlit as st

from ui.formatting import (
    format_commence_time,
    format_odds,
    market_label,
    prepare_display_df,
    style_probability_extremes,
)
from ui.glossary import EDGE_CALLOUT, GLOSSARY
from ui.player_stats import get_last_n_games


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
        "edge": st.column_config.NumberColumn(
            "Edge %",
            help=GLOSSARY["edge_pct"],
            format="%.1f",
        ),
        "ev": st.column_config.NumberColumn(
            "EV %",
            help=GLOSSARY["ev_pct"],
            format="%.1f",
        ),
    }


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
        "edge",
        "ev",
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

        game_log = get_last_n_games(player_name, market, version)
        if game_log is not None and not game_log.empty:
            col_chart, col_chart_help = st.columns([10, 1])
            with col_chart:
                st.caption("Last 10 games")
                st.altair_chart(_last_games_bar_chart(game_log))
            with col_chart_help:
                with st.popover("?"):
                    st.markdown(GLOSSARY["last_10_games"])

        display = prepare_display_df(market_rows)
        display["odds"] = market_rows["odds"].apply(format_odds)

        st.dataframe(
            style_probability_extremes(display[table_columns]),
            use_container_width=True,
            hide_index=True,
            column_config=_market_table_column_config(),
        )

    st.divider()
    st.caption(
        "This is a statistical research tool, "
        "not a guarantee of betting outcomes."
    )
