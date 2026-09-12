"""Data loading and game grouping for the Sleeper Prop Analyzer."""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from fetch_sleeper_props import SLEEPER_PROPS_PATH
from utils import TEAM_ABBR_TO_ODDS, game_date_from_commence, mlb_schedule_date


def load_raw_sleeper_props() -> pd.DataFrame:
    if not SLEEPER_PROPS_PATH.exists():
        return pd.DataFrame()

    return pd.read_parquet(SLEEPER_PROPS_PATH)


def team_display_name(abbr: str, fallback_name: str | None = None) -> str:
    abbr = str(abbr or "").strip().upper()
    if abbr in TEAM_ABBR_TO_ODDS:
        return TEAM_ABBR_TO_ODDS[abbr]
    if fallback_name and str(fallback_name).strip():
        return str(fallback_name).strip()
    return abbr or "—"


def _game_schedule_date(game_start) -> str | None:
    return game_date_from_commence(game_start)


def game_key(row: pd.Series) -> str:
    game_id = row.get("game_id")
    if pd.notna(game_id) and str(game_id).strip():
        return f"id:{game_id}"

    away = str(row.get("away_team") or "").strip().upper()
    home = str(row.get("home_team") or "").strip().upper()
    slate = _game_schedule_date(row.get("game_start")) or ""
    return f"{away}@{home}@{slate}"


@lru_cache(maxsize=4)
def _games_cache(parquet_mtime: float) -> pd.DataFrame:
    _ = parquet_mtime
    props = load_raw_sleeper_props()
    if props.empty:
        return props

    work = props.copy()
    work["schedule_date"] = work["game_start"].map(_game_schedule_date)
    work["game_key"] = work.apply(game_key, axis=1)

    grouped_rows: list[dict] = []
    for key, chunk in work.groupby("game_key", sort=False):
        sample = chunk.iloc[0]
        away_abbr = str(sample.get("away_team") or "").strip().upper()
        home_abbr = str(sample.get("home_team") or "").strip().upper()
        grouped_rows.append({
            "game_key": key,
            "game_id": sample.get("game_id"),
            "game": sample.get("game") or (
                f"{team_display_name(away_abbr, sample.get('away_team_name'))} @ "
                f"{team_display_name(home_abbr, sample.get('home_team_name'))}"
            ),
            "away_team": away_abbr,
            "home_team": home_abbr,
            "away_team_name": team_display_name(
                away_abbr,
                sample.get("away_team_name"),
            ),
            "home_team_name": team_display_name(
                home_abbr,
                sample.get("home_team_name"),
            ),
            "game_start": sample.get("game_start"),
            "schedule_date": sample.get("schedule_date"),
            "venue_name": sample.get("venue_name"),
            "venue_city": sample.get("venue_city"),
            "game_status": sample.get("game_status"),
            "prop_rows": len(chunk),
        })

    games = pd.DataFrame(grouped_rows)
    if games.empty:
        return games

    return games.sort_values(["schedule_date", "game_start"], na_position="last")


def list_games(parquet_mtime: float) -> pd.DataFrame:
    return _games_cache(parquet_mtime).copy()


def available_schedule_dates(parquet_mtime: float) -> list[str]:
    games = list_games(parquet_mtime)
    if games.empty:
        return []
    dates = games["schedule_date"].dropna().astype(str).unique().tolist()
    return sorted(dates)


def default_schedule_date(parquet_mtime: float) -> str | None:
    dates = available_schedule_dates(parquet_mtime)
    if not dates:
        return None

    today = mlb_schedule_date()
    if today in dates:
        return today
    return dates[-1]


def props_for_game(parquet_mtime: float, game_key_value: str) -> pd.DataFrame:
    _ = parquet_mtime
    props = load_raw_sleeper_props()
    if props.empty or not game_key_value:
        return props.iloc[0:0].copy()

    props = props.copy()
    props["game_key"] = props.apply(game_key, axis=1)
    return props[props["game_key"] == game_key_value].reset_index(drop=True)


def props_for_team(props: pd.DataFrame, team_abbr: str) -> pd.DataFrame:
    if props.empty:
        return props
    abbr = str(team_abbr or "").strip().upper()
    return props[
        props["player_team"].astype(str).str.upper().eq(abbr)
    ].reset_index(drop=True)


def player_metadata(props: pd.DataFrame) -> pd.DataFrame:
    """One row per player with position and headshot from Sleeper props."""
    if props.empty:
        return pd.DataFrame()

    cols = ["player", "player_team", "player_position", "player_image"]
    present = [col for col in cols if col in props.columns]
    meta = (
        props[present]
        .drop_duplicates(subset=["player"], keep="first")
        .reset_index(drop=True)
    )
    return meta
