"""Shared market multiselect widgets with distinct batter/pitcher walk labels."""

import streamlit as st

from prop_scoring import MODEL_MAP
from ui.formatting import market_label
from ui.glossary import GLOSSARY


def available_market_options(df):
    """Markets present in data, ordered by MODEL_MAP then alpha."""
    present = set(df["market"].dropna().unique())
    ordered = [market for market in MODEL_MAP if market in present]
    ordered.extend(
        sorted(present - set(ordered), key=market_label)
    )
    return ordered


def render_market_multiselect(df, key, label="Market type", **kwargs):
    """
    Multiselect over market keys with unique display labels
    (e.g. Batter Walks vs Pitcher Walks).
    """
    options = available_market_options(df)
    return st.multiselect(
        label,
        options,
        format_func=market_label,
        help=GLOSSARY["filter_market"],
        key=key,
        **kwargs,
    )
