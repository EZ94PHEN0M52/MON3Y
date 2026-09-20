"""Display helpers for the NFL Streamlit board."""

from __future__ import annotations

import pandas as pd

MARKET_LABELS = {
    "player_pass_yds": "Pass yards",
    "player_pass_tds": "Pass TDs",
    "player_rush_yds": "Rush yards",
    "player_receptions": "Receptions",
    "player_reception_yds": "Rec yards",
    "player_rush_reception_yds": "Rush+rec yards",
}


def market_label(market: str) -> str:
    return MARKET_LABELS.get(market, str(market).replace("_", " ").title())


def format_pct(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "—"


def format_commence_time(value) -> str:
    from utils import parse_commence_datetime

    if value is None or (isinstance(value, float) and pd.isna(value)) or value == "":
        return "—"
    try:
        dt = parse_commence_datetime(value)
        if dt is None:
            return "—"
        dt = dt.tz_convert("America/New_York")
        hour = dt.hour % 12 or 12
        am_pm = "AM" if dt.hour < 12 else "PM"
        return f"{dt.strftime('%b')} {dt.day}, {hour}:{dt.minute:02d} {am_pm} ET"
    except (ValueError, TypeError):
        return str(value)


def format_edge(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    try:
        return f"{float(value) * 100:+.1f} pp"
    except (TypeError, ValueError):
        return "—"


def format_ev(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    try:
        return f"{float(value) * 100:+.1f}¢"
    except (TypeError, ValueError):
        return "—"


def injury_badge(row: pd.Series) -> str:
    if bool(row.get("is_inactive")):
        status = row.get("injury_status") or "Out"
        return f"INACTIVE ({status})"
    if bool(row.get("is_questionable")):
        return "Q"
    status = row.get("injury_status")
    if status and str(status).strip() and str(status).lower() != "nan":
        return str(status)
    return "—"


def enrich_display(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["injury_badge"] = out.apply(injury_badge, axis=1)
    if {"away_team", "home_team"}.issubset(out.columns):
        out["game"] = (
            out["away_team"].astype(str) + " @ " + out["home_team"].astype(str)
        )
    else:
        out["game"] = "—"
    if "commence_time" in out.columns:
        ts = pd.to_datetime(out["commence_time"], utc=True, errors="coerce")
        out["kickoff"] = ts.dt.tz_convert("America/New_York").dt.strftime(
            "%a %m/%d %I:%M %p ET"
        )
    else:
        out["kickoff"] = "—"
    out["market_label"] = out["market"].map(market_label)
    return out


def prepare_board_table(df: pd.DataFrame) -> pd.DataFrame:
    table = pd.DataFrame({
        "Player": df.get("player"),
        "Injury": df.get("injury_badge"),
        "Market": df.get("market_label"),
        "Side": df.get("side"),
        "Line": df.get("line"),
        "Odds": df.get("odds"),
        "Model %": df.get("model_probability").map(format_pct)
        if "model_probability" in df.columns
        else "—",
        "Book %": df.get("market_implied_probability").map(format_pct)
        if "market_implied_probability" in df.columns
        else "—",
        "Edge": df.get("edge").map(format_edge)
        if "edge" in df.columns
        else "—",
        "EV": df.get("ev").map(format_ev) if "ev" in df.columns else "—",
        "Book": df.get("bookmaker"),
        "Game": df.get("game"),
        "Kickoff": df.get("kickoff"),
    })
    return table
