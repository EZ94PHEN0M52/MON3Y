"""Display formatting helpers for the MLB Prop Model UI."""

import numpy as np
import pandas as pd

from ui.glossary import MARKET_LABELS

PROBABILITY_COLUMNS = (
    "model_probability",
    "calibrated_probability",
    "over_probability",
    "under_probability",
    "dist_over_probability",
    "market_probability",
    "devigged_market_prob",
    "edge",
    "consensus_edge",
    "ev",
    "best_ev",
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


def format_predicted_count(value, decimals=1):
    """Format Poisson regressor expected count (pitcher K / walks)."""
    if pd.isna(value):
        return "—"
    return f"{float(value):.{decimals}f}"


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


def format_game_time(game, commence_time=None) -> str:
    """Matchup plus Eastern start time, e.g. ``Away @ Home · Aug 23, 1:05 PM ET``."""
    game_text = str(game or "").strip()
    time_str = format_commence_time(commence_time)
    if game_text and time_str != "—":
        return f"{game_text} · {time_str}"
    return game_text or time_str


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

    if "predicted_count" in display.columns:
        display["predicted_count"] = pd.to_numeric(
            display["predicted_count"],
            errors="coerce",
        ).round(1)

    return display


def player_path(player_name, hand=None):
    from urllib.parse import urlencode

    # Fragment is ignored by the server but lets LinkColumn show the name via
    # display_text=r"#(.*)$" (LinkColumn cannot read another dataframe column).
    display_name = format_name_with_hand(player_name, hand)
    return "/?" + urlencode({"player": player_name}) + f"#{display_name}"


def format_name_with_hand(name, hand=None) -> str:
    """Append (L) or (R) when *hand* is known."""
    text = str(name).strip()
    if not text:
        return text

    if hand in ("L", "R"):
        return f"{text} ({hand})"

    return text


def top_list_path(view):
    from urllib.parse import urlencode

    return "/?" + urlencode({"view": view})


def hitters_life_path():
    from urllib.parse import urlencode

    return "/?" + urlencode({"view": "hitters_life"})


def compare_view_path():
    from urllib.parse import urlencode

    return "/?" + urlencode({"view": "compare"})


def format_batter_score_cell(score, label=""):
    """Board column: score with optional partial label."""
    if pd.isna(score):
        return "—"
    text = f"{float(score):.1f}"
    if isinstance(label, str) and label.strip():
        text = f"{text} · {label.strip()}"
    return text
