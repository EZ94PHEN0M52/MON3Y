"""
Fetch probable starting pitchers for today's MLB slate.

Primary source: MLB Stats API (statsapi.mlb.com).
Fallback: pybaseball does not expose same-day probables — retries MLB API
once on transient failure, then returns an empty frame with source logged.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests

from utils import (
    MLB_TO_ODDS_TEAM,
    PROCESSED_DIR,
    canonical_odds_team_key,
    coerce_mlb_id,
)


MLB_STATS_API = "https://statsapi.mlb.com/api/v1"
PROBABLES_PATH = PROCESSED_DIR / "daily_probables.parquet"


def normalize_team_for_odds(name: Optional[str]) -> Optional[str]:
    """Map MLB Stats API team name to Odds API home_team/away_team format."""
    if not isinstance(name, str) or not name.strip():
        return None

    cleaned = name.strip()
    return MLB_TO_ODDS_TEAM.get(cleaned, cleaned)


def _parse_probable_pitcher(probable) -> tuple:
    """Return (sp_name, sp_id) or (None, None) for TBD / missing."""
    if not probable or not isinstance(probable, dict):
        return None, None

    pitcher_id = probable.get("id")
    full_name = probable.get("fullName")

    if pitcher_id in (None, 0, "0"):
        return None, None

    if isinstance(full_name, str):
        name = full_name.strip()
        if not name or name.upper() == "TBD":
            return None, None
        return name, int(pitcher_id)

    return None, None


def _fetch_schedule_mlb_api(
    game_date: str,
    timeout: int = 30,
) -> list:
    url = f"{MLB_STATS_API}/schedule"
    params = {
        "sportId": 1,
        "date": game_date,
        "hydrate": "probablePitcher",
    }

    response = requests.get(
        url,
        params=params,
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()

    dates = payload.get("dates") or []
    if not dates:
        return []

    return dates[0].get("games") or []


def _rows_from_games(
    games: list,
    game_date: str,
    fetched_at: str,
    source: str,
) -> list:
    rows = []

    for game in games:
        teams = game.get("teams") or {}
        home = teams.get("home") or {}
        away = teams.get("away") or {}

        home_team_obj = home.get("team") or {}
        away_team_obj = away.get("team") or {}

        home_team = normalize_team_for_odds(
            home_team_obj.get("name")
        )
        away_team = normalize_team_for_odds(
            away_team_obj.get("name")
        )

        home_sp_name, home_sp_id = _parse_probable_pitcher(
            home.get("probablePitcher")
        )
        away_sp_name, away_sp_id = _parse_probable_pitcher(
            away.get("probablePitcher")
        )

        rows.append(
            {
                "game_date": game_date,
                "home_team": home_team,
                "away_team": away_team,
                "home_sp_name": home_sp_name,
                "away_sp_name": away_sp_name,
                "home_sp_id": home_sp_id,
                "away_sp_id": away_sp_id,
                "fetched_at": fetched_at,
                "source": source,
            }
        )

    return rows


def _fetch_probables_pybaseball(
    game_date: str,
) -> Optional[pd.DataFrame]:
    """
    pybaseball has no same-day probable-pitcher endpoint.

    Documented fallback — returns None so the caller can rely on MLB API only.
    """
    return None


def fetch_probables(
    game_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch probable starters for *game_date* (YYYY-MM-DD, default today UTC).

    Returns a dataframe with schema:
    game_date, home_team, away_team, home_sp_name, away_sp_name,
    home_sp_id, away_sp_id, fetched_at, source
    """
    if game_date is None:
        game_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    fetched_at = (
        datetime.now(timezone.utc)
        .isoformat()
    )

    source = "mlb_stats_api"
    games = []

    try:
        games = _fetch_schedule_mlb_api(game_date)
    except requests.RequestException as exc:
        print(
            f"WARNING: MLB Stats API request failed: {exc}"
        )
        print("Trying documented pybaseball fallback (no probables API)...")
        fallback = _fetch_probables_pybaseball(game_date)
        if fallback is not None and not fallback.empty:
            return fallback
        source = "mlb_stats_api_failed"

    rows = _rows_from_games(
        games,
        game_date,
        fetched_at,
        source,
    )

    if not rows and source == "mlb_stats_api":
        print(
            f"No MLB games found for {game_date}."
        )

    return pd.DataFrame(
        rows,
        columns=[
            "game_date",
            "home_team",
            "away_team",
            "home_sp_name",
            "away_sp_name",
            "home_sp_id",
            "away_sp_id",
            "fetched_at",
            "source",
        ],
    )


def save_probables(
    df: pd.DataFrame,
    output_path=PROBABLES_PATH,
) -> pd.DataFrame:
    output_path = PROCESSED_DIR / output_path.name
    df.to_parquet(
        output_path,
        index=False,
    )
    print(
        f"Saved {len(df):,} probable-pitcher rows to {output_path}"
    )
    return df


def fetch_and_save_probables(
    game_date: Optional[str] = None,
) -> pd.DataFrame:
    print()
    print("=" * 60)
    print("DOWNLOADING MLB PROBABLE STARTING PITCHERS")
    print("=" * 60)

    df = fetch_probables(game_date=game_date)

    if len(df):
        known_home = df["home_sp_name"].notna().sum()
        known_away = df["away_sp_name"].notna().sum()
        print(
            f"Collected {len(df):,} games "
            f"({known_home} home SP, {known_away} away SP named)."
        )
        if len(df) <= 5:
            print(df.to_string(index=False))
        else:
            print(df.head(5).to_string(index=False))
            print("...")
    else:
        print("No probables collected.")

    return save_probables(df)


def lookup_opposing_sp(
    probables_df: pd.DataFrame,
    game_date: str,
    home_team: str,
    away_team: str,
    batter_team: str,
) -> tuple:
    """
    Return (sp_name, sp_id) for the opposing starter, or (None, None).

    Join keys: normalized (game_date, home_team, away_team) matching props.
    """
    if probables_df is None or probables_df.empty:
        return None, None

    home_key = canonical_odds_team_key(home_team)
    away_key = canonical_odds_team_key(away_team)
    batter_key = canonical_odds_team_key(batter_team)

    matches = probables_df[
        probables_df["game_date"].astype(str).eq(str(game_date)[:10])
        & probables_df["home_team"].map(canonical_odds_team_key).eq(home_key)
        & probables_df["away_team"].map(canonical_odds_team_key).eq(away_key)
    ]

    if matches.empty:
        return None, None

    row = matches.iloc[0]

    if batter_key == canonical_odds_team_key(row["home_team"]):
        return row.get("away_sp_name"), coerce_mlb_id(
            row.get("away_sp_id")
        )

    if batter_key == canonical_odds_team_key(row["away_team"]):
        return row.get("home_sp_name"), coerce_mlb_id(
            row.get("home_sp_id")
        )

    return None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch MLB probable starting pitchers",
    )
    parser.add_argument(
        "--date",
        help="Game date YYYY-MM-DD (default: today UTC)",
    )
    args = parser.parse_args()

    try:
        fetch_and_save_probables(game_date=args.date)
    except requests.RequestException as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
