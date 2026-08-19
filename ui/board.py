"""Main prop board for the MLB Prop Model UI."""

import pandas as pd
import streamlit as st

from ui.formatting import (
    format_odds,
    market_label,
    player_path,
    prepare_display_df,
    style_probability_extremes,
    top_list_path,
)
from ui.glossary import EDGE_CALLOUT, GLOSSARY

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
    "market_probability",
    "l5_l10_pct",
    "edge",
    "ev",
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
    {"label": "Market %", "field": "market_probability", "filter": "min_pct"},
    {
        "label": "L5 / L10 %",
        "field": "l5_pct",
        "filter": "min_pct",
        "glossary": "l5_l10_pct",
    },
    {"label": "Edge %", "field": "edge", "filter": "min_pct"},
    {"label": "EV %", "field": "ev", "filter": "min_pct"},
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
        st.session_state[sort_col_key] = "edge"
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


def _render_header_filter(df, spec, key_prefix):
    field = spec["field"]
    filter_type = spec["filter"]
    filter_key = f"{key_prefix}_filter_{field}"
    glossary_key = spec.get("glossary") or f"filter_{field}"

    if filter_type == "text":
        return st.text_input(
            "Contains",
            "",
            help=GLOSSARY.get(glossary_key, GLOSSARY["filter_player"]),
            key=filter_key,
            label_visibility="collapsed",
        )

    if filter_type == "multiselect":
        options = sorted(df[field].dropna().unique())
        if spec.get("display") == "market":
            return st.multiselect(
                "Values",
                options,
                format_func=market_label,
                help=GLOSSARY.get(glossary_key, GLOSSARY["filter_market"]),
                key=filter_key,
                label_visibility="collapsed",
            )
        return st.multiselect(
            "Values",
            options,
            help=GLOSSARY.get(glossary_key, GLOSSARY.get(f"filter_{field}", "")),
            key=filter_key,
            label_visibility="collapsed",
        )

    if filter_type == "side":
        return st.selectbox(
            "Side",
            ["All", "Over", "Under"],
            help=GLOSSARY["filter_side"],
            key=filter_key,
            label_visibility="collapsed",
        )

    if filter_type == "range":
        low = float(df[field].min())
        high = float(df[field].max())
        selected = st.slider(
            "Range",
            low,
            high,
            (low, high),
            help=GLOSSARY.get(f"filter_{field}", GLOSSARY["filter_line"]),
            key=filter_key,
            label_visibility="collapsed",
        )
        return selected

    if filter_type == "min_pct":
        pct = st.number_input(
            "Min %",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=0.5,
            help=GLOSSARY.get(glossary_key, ""),
            key=filter_key,
            label_visibility="collapsed",
        )
        return pct / 100.0 if pct > 0 else None

    if filter_type == "min_num":
        minimum = st.number_input(
            "Min",
            min_value=0.0,
            value=0.0,
            step=0.1,
            help=GLOSSARY.get(glossary_key, ""),
            key=filter_key,
            label_visibility="collapsed",
        )
        return minimum if minimum > 0 else None

    return None


def _render_clickable_headers(df, key_prefix):
    _init_board_state(key_prefix)
    specs = _header_specs(df)

    st.caption(GLOSSARY["header_click_help"])

    header_cols = st.columns(len(specs))
    for col, spec in zip(header_cols, specs):
        field = spec["field"]
        label = spec["label"]
        indicator = _sort_indicator(key_prefix, field)
        filter_on = _filter_active(key_prefix, field, df)
        filter_badge = " •" if filter_on else ""

        with col:
            header_label = f"{label}{indicator}{filter_badge}"
            clicked = st.button(
                header_label,
                key=f"{key_prefix}_hdr_{field}",
                use_container_width=True,
                help=GLOSSARY["header_click_sort"],
            )
            if clicked:
                sort_col_key = f"{key_prefix}_sort_col"
                sort_asc_key = f"{key_prefix}_sort_asc"
                if st.session_state[sort_col_key] == field:
                    st.session_state[sort_asc_key] = not st.session_state[sort_asc_key]
                else:
                    st.session_state[sort_col_key] = field
                    st.session_state[sort_asc_key] = True
                st.rerun()

            with st.popover("Filter", help=GLOSSARY["header_click_filter"]):
                st.markdown(f"**{label}**")
                _render_header_filter(df, spec, key_prefix)


def _apply_header_filters(df, key_prefix):
    result = df.copy()
    specs = _header_specs(df)

    for spec in specs:
        field = spec["field"]
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
        "market_probability": st.column_config.NumberColumn(
            "Market %",
            help=GLOSSARY["market_pct"],
            format="%.1f",
        ),
        "l5_l10_pct": st.column_config.TextColumn(
            "L5 / L10 %",
            help=GLOSSARY["l5_l10_pct"],
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
        "market_probability",
        "edge",
        "ev",
    ):
        if col in display.columns:
            display[col] = display[col].astype(float)

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


def _top_props_by_probability(filtered, probability_col, n=10):
    """Return top N props by probability, one row per player (highest prob kept)."""
    if filtered.empty or probability_col not in filtered.columns:
        return filtered.iloc[0:0].copy()

    ranked = filtered.sort_values(probability_col, ascending=False)
    ranked = ranked.loc[ranked.groupby("player")[probability_col].idxmax()]
    ranked = ranked.sort_values(probability_col, ascending=False)
    return ranked.head(n).reset_index(drop=True)


def _render_probability_rankings(filtered):
    if filtered.empty:
        return

    st.markdown("##### Highest model probabilities")
    st.caption(
        "Top 10 props by **Over %** and **Under %** from the current Market / "
        "Edge / EV filters (one prop per player — highest probability kept). "
        "Open a title to view the **full ranked list**. "
        f"{GLOSSARY['over_pct']} "
        f"{GLOSSARY['under_pct']}"
    )

    over_top = _top_props_by_probability(filtered, "over_probability")
    under_filtered = filtered[filtered["market"] != "batter_home_runs"]
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


def _render_board_table(filtered):
    optional_stats = _optional_stat_columns(filtered.columns)
    display = _prepare_board_table_df(filtered)
    table_columns = list(BOARD_TABLE_COLUMNS)
    table_columns.extend(column for column, _ in optional_stats)

    st.dataframe(
        style_probability_extremes(display[table_columns]),
        use_container_width=True,
        hide_index=True,
        height=520,
        column_config=_board_column_config(optional_stats),
    )


def apply_top_level_filters(df, key_prefix="board"):
    col1, col2, col3 = st.columns(3)

    with col1:
        markets = st.multiselect(
            "Market",
            sorted(df["market"].dropna().unique(), key=market_label),
            format_func=market_label,
            help=GLOSSARY["filter_market"],
            key=f"{key_prefix}_markets",
        )

    with col2:
        min_edge = st.slider(
            "Minimum Edge",
            0.0,
            0.30,
            0.03,
            0.01,
            help=GLOSSARY["min_edge"],
            key=f"{key_prefix}_min_edge",
        )

    with col3:
        min_ev = st.slider(
            "Minimum EV",
            0.0,
            0.50,
            0.05,
            0.01,
            help=GLOSSARY["min_ev"],
            key=f"{key_prefix}_min_ev",
        )

    filtered = df.copy()

    if markets:
        filtered = filtered[filtered["market"].isin(markets)]

    filtered = filtered[filtered["edge"] >= min_edge]
    filtered = filtered[filtered["ev"] >= min_ev]

    return filtered


def render_board(df, version):
    filtered = apply_top_level_filters(df, key_prefix=f"board_{version}")

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
        _render_probability_rankings(filtered)

        key_prefix = f"board_{version}"
        _render_clickable_headers(filtered, key_prefix)
        header_filtered = _apply_header_filters(filtered, key_prefix)

        if len(header_filtered):
            st.caption(
                f"Showing **{len(header_filtered)}** of **{len(filtered)}** props. "
                "Click a **column header** to sort (↑/↓). "
                "Use **Filter** under a header to narrow that column."
            )
            _render_board_table(header_filtered)
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
