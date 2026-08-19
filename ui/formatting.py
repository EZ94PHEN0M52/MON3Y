"""Display formatting helpers for the MLB Prop Model UI."""

import numpy as np
import pandas as pd

from ui.glossary import MARKET_LABELS

PROBABILITY_COLUMNS = (
    "model_probability",
    "over_probability",
    "under_probability",
    "market_probability",
    "edge",
    "ev",
)


def market_label(raw):
    return MARKET_LABELS.get(raw, raw.replace("_", " ").title())


def format_pct(value, decimals=1):
    if pd.isna(value):
        return "—"
    return f"{float(value) * 100:.{decimals}f}%"


def format_odds(value):
    if pd.isna(value):
        return "—"
    odds = int(float(value))
    if odds > 0:
        return f"+{odds}"
    return str(odds)


def format_commence_time(value):
    if pd.isna(value) or not value:
        return "—"
    try:
        dt = pd.to_datetime(value, utc=True).tz_convert("America/New_York")
        hour = dt.hour % 12 or 12
        am_pm = "AM" if dt.hour < 12 else "PM"
        return f"{dt.strftime('%b')} {dt.day}, {hour}:{dt.minute:02d} {am_pm} ET"
    except (ValueError, TypeError):
        return str(value)


def enrich_with_over_under_probs(df):
    """
    Ensure over_probability and under_probability exist on prediction rows.

    New predict.py output includes both columns. Legacy CSVs stored P(Over) in
    model_probability for every row (regardless of side); in that case Over %
    is taken from model_probability and Under % is 1 - Over %.
    """
    result = df.copy()

    if (
        "over_probability" in result.columns
        and "under_probability" in result.columns
    ):
        return result

    if "model_probability" not in result.columns:
        result["over_probability"] = np.nan
        result["under_probability"] = np.nan
        return result

    result["over_probability"] = pd.to_numeric(
        result["model_probability"],
        errors="coerce",
    )
    result["under_probability"] = 1.0 - result["over_probability"]
    return result


def style_probability_extremes(
    display_df,
    over_col="over_probability",
    under_col="under_probability",
):
    """Bold the highest Over % and lowest Under % in a display dataframe."""
    if display_df.empty or over_col not in display_df.columns:
        return display_df

    max_over_idx = display_df[over_col].idxmax()
    min_under_idx = display_df[under_col].idxmin()

    def _highlight(row):
        styles = [""] * len(row)
        columns = list(row.index)
        if row.name == max_over_idx and over_col in columns:
            styles[columns.index(over_col)] = "font-weight: bold"
        if row.name == min_under_idx and under_col in columns:
            styles[columns.index(under_col)] = "font-weight: bold"
        return styles

    return display_df.style.apply(_highlight, axis=1)


def prepare_display_df(df):
    """Format probability columns as percentages for display."""
    display = df.copy()

    for col in PROBABILITY_COLUMNS:
        if col in display.columns:
            display[col] = (display[col] * 100).round(1)

    if "market" in display.columns:
        display["market"] = display["market"].map(
            lambda m: market_label(m) if pd.notna(m) else m
        )

    if "odds" in display.columns:
        display["odds"] = display["odds"].apply(format_odds)

    return display


def player_path(player_name):
    from urllib.parse import urlencode

    # Fragment is ignored by the server but lets LinkColumn show the name via
    # display_text=r"#(.*)$" (LinkColumn cannot read another dataframe column).
    return "/?" + urlencode({"player": player_name}) + f"#{player_name}"


def top_list_path(view):
    from urllib.parse import urlencode

    return "/?" + urlencode({"view": view})
