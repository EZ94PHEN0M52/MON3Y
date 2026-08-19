"""Main prop board for the MLB Prop Model UI."""

import pandas as pd
import streamlit as st

from odds_aggregation import dedupe_best_prop
from ui.formatting import (
    format_batter_score_cell,
    format_odds,
    player_path,
    prepare_display_df,
    style_probability_extremes,
    top_list_path,
)
from ui.glossary import EDGE_CALLOUT, GLOSSARY
from ui.market_filters import render_market_multiselect
from ui.pick_builder import render_board_add_controls

BOARD_TABLE_COLUMNS = [
    "player_link",
    "game",
    "market",
    "bookmaker",
    "side",
    "line",
    "odds",
    "over_probability",
    "under_probability",
    "model_probability",
    "calibrated_probability",
    "market_probability",
    "devigged_market_prob",
    "l5_l10_pct",
    "batter_score_display",
    "edge",
    "consensus_edge",
    "best_book",
    "best_ev",
    "ev",
    "line_delta",
    "steam_flag",
]

RANKING_TABLE_COLUMNS = [
    "player_link",
    "market",
    "bookmaker",
    "line",
    "side",
    "over_probability",
    "under_probability",
    "edge",
]

PROP_ID_COLUMNS = ("player", "market", "bookmaker", "line")

STAT_COLUMN_LABELS = ("H2H",)

BASE_HEADER_SPECS = [
    {"label": "Player", "field": "player", "filter": "text"},
    {"label": "Game", "field": "game", "filter": "multiselect"},
    {"label": "Market", "field": "market", "filter": "multiselect", "display": "market"},
    {"label": "Book", "field": "bookmaker", "filter": "multiselect"},
    {"label": "Side", "field": "side", "filter": "side"},
    {"label": "Line", "field": "line", "filter": "range"},
    {"label": "Odds", "field": "odds", "filter": "range"},
    {
        "label": "Over %",
        "field": "over_probability",
        "filter": "min_pct",
        "glossary": "filter_over_pct",
    },
    {
        "label": "Under %",
        "field": "under_probability",
        "filter": "min_pct",
        "glossary": "filter_under_pct",
    },
    {"label": "Model %", "field": "model_probability", "filter": "min_pct"},
    {
        "label": "Calibrated %",
        "field": "calibrated_probability",
        "filter": "min_pct",
        "glossary": "calibrated_pct",
    },
    {"label": "Market %", "field": "market_probability", "filter": "min_pct"},
    {
        "label": "Devigged %",
        "field": "devigged_market_prob",
        "filter": "min_pct",
        "glossary": "devigged_market_pct",
    },
    {
        "label": "L5 / L10 %",
        "field": "l5_pct",
        "filter": "min_pct",
        "glossary": "l5_l10_pct",
    },
    {
        "label": "Batter Score",
        "field": "batter_score",
        "filter": "min_num",
        "glossary": "batter_score",
    },
    {"label": "Edge %", "field": "edge", "filter": "min_pct"},
    {
        "label": "Consensus Edge %",
        "field": "consensus_edge",
        "filter": "min_pct",
        "glossary": "consensus_edge",
    },
    {
        "label": "Best Book",
        "field": "best_book",
        "filter": "multiselect",
        "glossary": "best_book",
    },
    {
        "label": "Best EV %",
        "field": "best_ev",
        "filter": "min_pct",
        "glossary": "best_ev",
    },
    {"label": "EV %", "field": "ev", "filter": "min_pct"},
    {
        "label": "Line Δ",
        "field": "line_delta",
        "filter": "range",
        "glossary": "line_delta",
    },
    {
        "label": "Steam",
        "field": "steam_flag",
        "filter": "steam",
        "glossary": "steam_flag",
    },
]


def _optional_stat_columns(columns):
    found = []
    seen_labels = set()

    for col in columns:
        lower = col.lower()
        for label in STAT_COLUMN_LABELS:
            suffix = label.lower()
            if lower == suffix or lower.endswith(f"_{suffix}"):
                if label not in seen_labels:
                    found.append((col, label))
                    seen_labels.add(label)
                break

    return found


def _header_specs(df):
    specs = list(BASE_HEADER_SPECS)
    for column, label in _optional_stat_columns(df.columns):
        specs.append(
            {
                "label": label,
                "field": column,
                "filter": "min_num",
                "glossary": f"filter_{label.lower()}",
            }
        )
    return specs


def _init_board_state(key_prefix):
    sort_col_key = f"{key_prefix}_sort_col"
    sort_asc_key = f"{key_prefix}_sort_asc"

    if sort_col_key not in st.session_state:
        st.session_state[sort_col_key] = "ev"
        st.session_state[sort_asc_key] = False


def _sort_indicator(key_prefix, field):
    if st.session_state.get(f"{key_prefix}_sort_col") != field:
        return ""
    if st.session_state.get(f"{key_prefix}_sort_asc", True):
        return " ↑"
    return " ↓"


def _filter_active(key_prefix, field, df=None):
    value = st.session_state.get(f"{key_prefix}_filter_{field}")
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple)):
        if len(value) == 2 and df is not None and field in df.columns:
            low, high = value
            bounds = (float(df[field].min()), float(df[field].max()))
            return low > bounds[0] or high < bounds[1]
        return len(value) > 0
    if isinstance(value, dict):
        return value.get("active", False)
    return bool(value)


def _available_table_columns(filtered):
    optional_stats = _optional_stat_columns(filtered.columns)
    display = _prepare_board_table_df(filtered)
    columns = [
        column
        for column in BOARD_TABLE_COLUMNS
        if column in display.columns
    ]
    columns.extend(column for column, _ in optional_stats)
    return columns, optional_stats


def _init_board_filter_state(key_prefix):
    if f"{key_prefix}_markets" not in st.session_state:
        st.session_state[f"{key_prefix}_markets"] = []
    if f"{key_prefix}_min_edge" not in st.session_state:
        st.session_state[f"{key_prefix}_min_edge"] = 0.03
    if f"{key_prefix}_min_ev" not in st.session_state:
        st.session_state[f"{key_prefix}_min_ev"] = 0.05


def _column_filter_specs(df):
    return [spec for spec in _header_specs(df) if spec["field"] != "market"]


def _render_filter_popover(df, key_prefix):
    _init_board_filter_state(key_prefix)
    visible_key = f"{key_prefix}_visible_columns"
    min_edge_key = f"{key_prefix}_min_edge"
    min_ev_key = f"{key_prefix}_min_ev"

    table_columns, optional_stats = _available_table_columns(df)
    if visible_key not in st.session_state:
        st.session_state[visible_key] = list(table_columns)

    popover_filters = 0
    if st.session_state.get(f"{key_prefix}_min_edge", 0.03) > 0.03:
        popover_filters += 1
    if st.session_state.get(f"{key_prefix}_min_ev", 0.05) > 0.05:
        popover_filters += 1

    popover_label = ":material/tune: Filters & columns"
    if popover_filters:
        popover_label += f" ({popover_filters})"

    with st.popover(popover_label):
        st.slider(
            "Minimum Edge",
            min_value=0.0,
            max_value=0.30,
            step=0.01,
            help=GLOSSARY["min_edge"],
            key=min_edge_key,
        )
        st.slider(
            "Minimum EV",
            min_value=0.0,
            max_value=0.50,
            step=0.01,
            help=GLOSSARY["min_ev"],
            key=min_ev_key,
        )

        st.divider()
        st.markdown("**Columns**")
        st.multiselect(
            "Show columns",
            table_columns,
            help=GLOSSARY["column_filters"],
            key=visible_key,
        )

    visible_columns = [
        column
        for column in st.session_state.get(visible_key, table_columns)
        if column in table_columns
    ]
    if not visible_columns:
        visible_columns = list(table_columns)

    return visible_columns, optional_stats


def _render_column_filters_panel(df, key_prefix):
    specs = _column_filter_specs(df)
    active_count = sum(
        1 for spec in specs if _filter_active(key_prefix, spec["field"], df)
    )
    expander_label = "Filter by column"
    if active_count:
        expander_label += f" ({active_count} active)"

    with st.expander(expander_label, expanded=False):
        cols = st.columns(3)
        for index, spec in enumerate(specs):
            with cols[index % 3]:
                _render_header_filter(
                    df,
                    spec,
                    key_prefix,
                    widget_label=spec["label"],
                )


def _render_sort_controls(df, key_prefix):
    _init_board_state(key_prefix)
    specs = _header_specs(df)
    sort_col_key = f"{key_prefix}_sort_col"
    sort_asc_key = f"{key_prefix}_sort_asc"

    sort_fields = [spec["field"] for spec in specs]
    current_field = st.session_state.get(sort_col_key, "ev")
    if current_field not in sort_fields:
        current_field = sort_fields[0] if sort_fields else "ev"
        st.session_state[sort_col_key] = current_field

    st.caption(GLOSSARY["header_click_help"])

    control_row = st.container(horizontal=True, vertical_alignment="bottom")
    with control_row:
        selected_idx = st.selectbox(
            "Sort by",
            options=range(len(specs)),
            format_func=lambda index: specs[index]["label"],
            index=sort_fields.index(current_field),
            key=f"{key_prefix}_sort_select",
        )
        st.session_state[sort_col_key] = sort_fields[selected_idx]

        st.toggle(
            "Ascending",
            key=sort_asc_key,
        )


def _render_header_filter(df, spec, key_prefix, *, widget_label=None):
    field = spec["field"]
    filter_type = spec["filter"]
    filter_key = f"{key_prefix}_filter_{field}"
    glossary_key = spec.get("glossary") or f"filter_{field}"
    label_visibility = "visible" if widget_label else "collapsed"
    label = widget_label or ""

    if filter_type == "text":
        return st.text_input(
            label or "Contains",
            "",
            help=GLOSSARY.get(glossary_key, GLOSSARY["filter_player"]),
            key=filter_key,
            label_visibility=label_visibility,
        )

    if filter_type == "multiselect":
        options = sorted(df[field].dropna().unique())
        return st.multiselect(
            label or "Values",
            options,
            help=GLOSSARY.get(glossary_key, GLOSSARY.get(f"filter_{field}", "")),
            key=filter_key,
            label_visibility=label_visibility,
        )

    if filter_type == "side":
        return st.selectbox(
            label or "Side",
            ["All", "Over", "Under"],
            help=GLOSSARY["filter_side"],
            key=filter_key,
            label_visibility=label_visibility,
        )

    if filter_type == "steam":
        return st.selectbox(
            label or "Steam",
            ["All", "Steam only", "No steam"],
            help=GLOSSARY["steam_flag"],
            key=filter_key,
            label_visibility=label_visibility,
        )

    if filter_type == "range":
        low = float(df[field].min())
        high = float(df[field].max())
        selected = st.slider(
            label or "Range",
            low,
            high,
            (low, high),
            help=GLOSSARY.get(f"filter_{field}", GLOSSARY["filter_line"]),
            key=filter_key,
            label_visibility=label_visibility,
        )
        return selected

    if filter_type == "min_pct":
        pct = st.number_input(
            label or "Min %",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=0.5,
            help=GLOSSARY.get(glossary_key, ""),
            key=filter_key,
            label_visibility=label_visibility,
        )
        return pct / 100.0 if pct > 0 else None

    if filter_type == "min_num":
        minimum = st.number_input(
            label or "Min",
            min_value=0.0,
            value=0.0,
            step=0.1,
            help=GLOSSARY.get(glossary_key, ""),
            key=filter_key,
            label_visibility=label_visibility,
        )
        return minimum if minimum > 0 else None

    return None


def _apply_header_filters(df, key_prefix):
    result = df.copy()
    specs = _header_specs(df)

    for spec in specs:
        field = spec["field"]
        if field == "market":
            continue
        filter_type = spec["filter"]
        value = st.session_state.get(f"{key_prefix}_filter_{field}")

        if filter_type == "text":
            if value and str(value).strip():
                result = result[
                    result[field].str.contains(
                        str(value).strip(),
                        case=False,
                        na=False,
                    )
                ]
            continue

        if filter_type == "multiselect":
            if value:
                result = result[result[field].isin(value)]
            continue

        if filter_type == "side":
            if value and value != "All":
                result = result[result[field] == value]
            continue

        if filter_type == "steam":
            if value == "Steam only":
                result = result[result[field].astype(bool)]
            elif value == "No steam":
                result = result[~result[field].astype(bool)]
            continue

        if filter_type == "range":
            if isinstance(value, (list, tuple)) and len(value) == 2:
                low, high = value
                bounds = (float(df[field].min()), float(df[field].max()))
                if low > bounds[0] or high < bounds[1]:
                    result = result[
                        (result[field] >= low) & (result[field] <= high)
                    ]
            continue

        if filter_type in ("min_pct", "min_num"):
            if value is not None and value > 0:
                result = result[result[field] >= value]

    sort_col = st.session_state.get(f"{key_prefix}_sort_col")
    sort_asc = st.session_state.get(f"{key_prefix}_sort_asc", True)

    if sort_col and sort_col in result.columns:
        result = result.sort_values(
            sort_col,
            ascending=sort_asc,
            na_position="last",
        )

    return result.reset_index(drop=True)


def _board_column_config(extra_stat_columns):
    config = {
        "player_link": st.column_config.LinkColumn(
            "Player",
            help=GLOSSARY["player_link"],
            display_text=r"#(.*)$",
        ),
        "game": st.column_config.TextColumn(
            "Game",
            help=GLOSSARY["game"],
        ),
        "market": st.column_config.TextColumn(
            "Market",
            help=GLOSSARY["market"],
        ),
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
        "calibrated_probability": st.column_config.NumberColumn(
            "Calibrated %",
            help=GLOSSARY["calibrated_pct"],
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
        "l5_l10_pct": st.column_config.TextColumn(
            "L5 / L10 %",
            help=GLOSSARY["l5_l10_pct"],
        ),
        "batter_score_display": st.column_config.TextColumn(
            "Batter Score",
            help=GLOSSARY["batter_score"],
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

    for column, label in extra_stat_columns:
        glossary_key = f"filter_{label.lower()}"
        config[column] = st.column_config.NumberColumn(
            label,
            help=GLOSSARY.get(glossary_key, f"Player {label} average."),
            format="%.2f",
        )

    return config


def _prepare_board_table_df(filtered):
    display = prepare_display_df(filtered.copy())
    display["player_link"] = filtered["player"].map(player_path)
    display["bookmaker"] = filtered["bookmaker"].values
    display["odds"] = filtered["odds"].apply(format_odds).values

    for col in (
        "over_probability",
        "under_probability",
        "model_probability",
        "calibrated_probability",
        "market_probability",
        "devigged_market_prob",
        "edge",
        "consensus_edge",
        "ev",
        "best_ev",
        "line_delta",
    ):
        if col in display.columns:
            display[col] = display[col].astype(float)

    if "steam_flag" in display.columns:
        display["steam_flag"] = filtered["steam_flag"].map(
            lambda value: "🔥" if bool(value) else ""
        )

    if "batter_score" in filtered.columns:
        display["batter_score_display"] = [
            format_batter_score_cell(score, label)
            for score, label in zip(
                filtered["batter_score"],
                filtered.get(
                    "batter_score_label",
                    pd.Series([""] * len(filtered)),
                ),
            )
        ]

    return display


def _ranking_column_config():
    return {
        "player_link": st.column_config.LinkColumn(
            "Player",
            help=GLOSSARY["player_link"],
            display_text=r"#(.*)$",
        ),
        "market": st.column_config.TextColumn(
            "Market",
            help=GLOSSARY["market"],
        ),
        "bookmaker": st.column_config.TextColumn(
            "Book",
            help=GLOSSARY["book"],
        ),
        "line": st.column_config.NumberColumn(
            "Line",
            help=GLOSSARY["line"],
            format="%.1f",
        ),
        "side": st.column_config.TextColumn(
            "Side",
            help=GLOSSARY["side"],
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
        "edge": st.column_config.NumberColumn(
            "Edge %",
            help=GLOSSARY["edge_pct"],
            format="%.1f",
        ),
    }


def _apply_data_filters(df, markets, min_edge, min_ev, *, dedupe=True):
    filtered = df.copy()

    if markets:
        filtered = filtered[filtered["market"].isin(markets)]

    filtered = filtered[filtered["edge"] >= min_edge]
    filtered = filtered[filtered["ev"] >= min_ev]

    if dedupe:
        filtered = dedupe_best_prop(filtered)

    return filtered.sort_values(
        "ev",
        ascending=False,
    )


def apply_top_level_filters(df, key_prefix="board", *, dedupe=True):
    _init_board_filter_state(key_prefix)
    col1, col2, col3 = st.columns(3)

    with col1:
        markets = render_market_multiselect(
            df,
            key=f"{key_prefix}_markets",
            label="Market",
        )

    with col2:
        min_edge = st.slider(
            "Minimum Edge",
            min_value=0.0,
            max_value=0.30,
            step=0.01,
            help=GLOSSARY["min_edge"],
            key=f"{key_prefix}_min_edge",
        )

    with col3:
        min_ev = st.slider(
            "Minimum EV",
            min_value=0.0,
            max_value=0.50,
            step=0.01,
            help=GLOSSARY["min_ev"],
            key=f"{key_prefix}_min_ev",
        )

    return _apply_data_filters(
        df,
        markets,
        min_edge,
        min_ev,
        dedupe=dedupe,
    )


def _top_props_by_probability(filtered, probability_col, n=10):
    """Return top N props by probability, one row per (player, market)."""
    if filtered.empty or probability_col not in filtered.columns:
        return filtered.iloc[0:0].copy()

    deduped = dedupe_best_prop(filtered, sort_col=probability_col)
    ranked = deduped.sort_values(probability_col, ascending=False)
    return ranked.head(n).reset_index(drop=True)


def _apply_ranking_market_filter(filtered, selected_markets):
    if selected_markets:
        return filtered[filtered["market"].isin(selected_markets)]
    return filtered


def _render_probability_rankings(filtered, key_prefix):
    if filtered.empty:
        return

    st.markdown("##### Highest model probabilities")
    st.caption(
        "Top 10 props by **Over %** and **Under %** from the current Market / "
        "Edge / EV filters (one best book per player and market). "
        "Open a title to view the **full ranked list**. "
        f"{GLOSSARY['over_pct']} "
        f"{GLOSSARY['under_pct']}"
    )

    ranking_markets = st.session_state.get(f"{key_prefix}_markets", [])
    ranking_df = _apply_ranking_market_filter(
        filtered,
        ranking_markets,
    )

    over_top = _top_props_by_probability(ranking_df, "over_probability")
    under_filtered = ranking_df[ranking_df["market"] != "batter_home_runs"]
    under_top = _top_props_by_probability(under_filtered, "under_probability")

    col_over, col_under = st.columns(2)

    with col_over:
        st.markdown(f"**[Top Over %]({top_list_path('top_over')})**")
        if len(over_top):
            over_display = _prepare_board_table_df(over_top)[RANKING_TABLE_COLUMNS]
            st.dataframe(
                style_probability_extremes(over_display),
                hide_index=True,
                height=min(42 * len(over_top) + 38, 420),
                column_config=_ranking_column_config(),
            )
        else:
            st.caption("No props in the current filter set.")

    with col_under:
        st.markdown(f"**[Top Under %]({top_list_path('top_under')})**")
        if len(under_top):
            under_display = _prepare_board_table_df(under_top)[RANKING_TABLE_COLUMNS]
            st.dataframe(
                style_probability_extremes(under_display),
                hide_index=True,
                height=min(42 * len(under_top) + 38, 420),
                column_config=_ranking_column_config(),
            )
        else:
            st.caption("No props in the current filter set.")


def _render_board_table(filtered, visible_columns=None, optional_stats=None):
    if optional_stats is None:
        _, optional_stats = _available_table_columns(filtered)
    display = _prepare_board_table_df(filtered)
    table_columns = [
        column
        for column in BOARD_TABLE_COLUMNS
        if column in display.columns
    ]
    table_columns.extend(column for column, _ in optional_stats)

    if visible_columns:
        table_columns = [
            column
            for column in table_columns
            if column in visible_columns
        ]

    st.dataframe(
        style_probability_extremes(display[table_columns]),
        hide_index=True,
        height=520,
        column_config=_board_column_config(optional_stats),
    )


def render_board(df, version):
    key_prefix = f"board_{version}"
    _init_board_filter_state(key_prefix)

    render_market_multiselect(
        df,
        key=f"{key_prefix}_markets",
        label="Market type",
    )

    visible_columns, optional_stats = _render_filter_popover(df, key_prefix)

    filtered = _apply_data_filters(
        df,
        st.session_state.get(f"{key_prefix}_markets", []),
        st.session_state.get(f"{key_prefix}_min_edge", 0.03),
        st.session_state.get(f"{key_prefix}_min_ev", 0.05),
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Props", len(filtered))
    c2.metric(
        "Best Edge",
        f"{filtered['edge'].max() * 100:.1f}%" if len(filtered) else "—",
    )
    c3.metric(
        "Best EV",
        f"{filtered['ev'].max() * 100:.1f}%" if len(filtered) else "—",
    )
    c4.metric(
        "Players",
        filtered["player"].nunique() if len(filtered) else 0,
    )

    if len(filtered):
        _render_probability_rankings(filtered, key_prefix)

        _render_sort_controls(filtered, key_prefix)
        _render_column_filters_panel(filtered, key_prefix)
        header_filtered = _apply_header_filters(filtered, key_prefix)

        if len(header_filtered):
            st.caption(
                f"Showing **{len(header_filtered)}** of **{len(filtered)}** props. "
                "Use **Market type** above for prop categories, **Filter by column** "
                "for row filters, and **Filters & columns** for edge, EV, and "
                "column visibility."
            )
            _render_board_table(
                header_filtered,
                visible_columns=visible_columns,
                optional_stats=optional_stats,
            )
            render_board_add_controls(header_filtered, key_prefix)
        else:
            st.info("No props meet the current column filters.")
    else:
        st.info("No props meet the current filters.")

    st.divider()
    st.caption(EDGE_CALLOUT)
    st.caption(
        "This is a statistical research tool, "
        "not a guarantee of betting outcomes."
    )
