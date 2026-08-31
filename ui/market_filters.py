"""Shared market multiselect widgets with distinct batter/pitcher walk labels."""

import streamlit as st

from prop_scoring import MODEL_MAP
from ui.formatting import market_label
from ui.glossary import GLOSSARY

# Trained/scored offline only; hidden from Streamlit and live Odds API fetch.
from odds_api import EXCLUDED_LIVE_PROP_MARKETS

EXCLUDED_UI_MARKETS = EXCLUDED_LIVE_PROP_MARKETS


def exclude_ui_markets(df):
    """Drop markets hidden from the UI (training/predict output may still include them)."""
    if df is None or df.empty or "market" not in df.columns:
        return df
    return df[~df["market"].isin(EXCLUDED_UI_MARKETS)].copy()


def available_market_options(df):
    """Markets present in data, ordered by MODEL_MAP then alpha."""
    present = set(df["market"].dropna().unique()) - EXCLUDED_UI_MARKETS
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
