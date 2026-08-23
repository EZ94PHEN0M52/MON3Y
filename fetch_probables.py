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
    CURRENT_PROPS_PATH,
    MLB_TO_ODDS_TEAM,
    PROCESSED_DIR,
    canonical_odds_team_key,
    coerce_mlb_id,
    game_date_from_commence,
    mlb_schedule_date,
    require_live_fetch,
    slate_dates_from_props,
    slate_games_from_props,
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
    require_live_fetch("MLB Stats API probables")
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
    Fetch probable starters for *game_date* (YYYY-MM-DD).

    Default *game_date* is today's MLB schedule date in US Eastern, matching
    how prop commence times map to slate dates (not UTC calendar date).

    Returns a dataframe with schema:
    game_date, home_team, away_team, home_sp_name, away_sp_name,
    home_sp_id, away_sp_id, fetched_at, source
    """
    if game_date is None:
        game_date = mlb_schedule_date()

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

    if output_path.exists() and not df.empty:
        existing = pd.read_parquet(output_path)
        incoming_dates = set(df["game_date"].astype(str).unique())
        keep = existing[
            ~existing["game_date"].astype(str).isin(incoming_dates)
        ]
        df = pd.concat([keep, df], ignore_index=True)

    df.to_parquet(
        output_path,
        index=False,
    )
    print(
        f"Saved {len(df):,} probable-pitcher rows to {output_path}"
    )
    return df


def _probables_team_rows(
    probables_df: pd.DataFrame,
    home_team: str,
    away_team: str,
) -> pd.DataFrame:
    home_key = canonical_odds_team_key(home_team)
    away_key = canonical_odds_team_key(away_team)

    return probables_df[
        probables_df["home_team"].map(canonical_odds_team_key).eq(home_key)
        & probables_df["away_team"].map(canonical_odds_team_key).eq(away_key)
    ]


def _schedule_date_candidates(game_date: str) -> list[str]:
    candidates = [str(game_date)[:10]]
    try:
        anchor = pd.Timestamp(candidates[0])
        for offset in (-1, 1):
            day = (anchor + pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
            if day not in candidates:
                candidates.append(day)
    except (TypeError, ValueError):
        pass
    return candidates


def _probables_row_for_game(
    probables_df: pd.DataFrame,
    game_date: str,
    home_team: str,
    away_team: str,
):
    """Best probables row for a game, trying Eastern date ±1 day."""
    if probables_df is None or probables_df.empty:
        return None

    for candidate_date in _schedule_date_candidates(game_date):
        dated = probables_df[
            probables_df["game_date"].astype(str).eq(candidate_date)
        ]
        team_rows = _probables_team_rows(
            dated,
            home_team,
            away_team,
        )
        if not team_rows.empty:
            return team_rows.iloc[0]

    team_rows = _probables_team_rows(
        probables_df,
        home_team,
        away_team,
    )
    if len(team_rows) == 1:
        return team_rows.iloc[0]

    return None


def _probables_dates(probables_df: pd.DataFrame) -> set[str]:
    if probables_df is None or probables_df.empty:
        return set()
    return {
        str(value)
        for value in probables_df["game_date"].astype(str).unique()
    }


def missing_probables_dates_for_slate(
    props: pd.DataFrame,
    probables_df: pd.DataFrame,
) -> set[str]:
    """Schedule dates in props that have no same-day probables rows."""
    slate_dates = slate_dates_from_props(props)
    if not slate_dates:
        return set()

    stored_dates = _probables_dates(probables_df)
    return {date for date in slate_dates if date not in stored_dates}


def analyze_probables_slate_coverage(
    props: pd.DataFrame,
    probables_df: pd.DataFrame | None = None,
) -> dict:
    """
    Check that probables cover today's props slate for Batter Score SP lookup.

    Returns a summary dict; does not print or abort.
    """
    empty = {
        "ok": True,
        "game_count": 0,
        "games_with_sp": 0,
        "games_missing_sp": [],
        "warnings": [],
        "slate_dates": [],
        "probables_dates": [],
        "missing_probables_dates": [],
    }

    slate = slate_games_from_props(props)
    if slate.empty:
        return empty

    if probables_df is None:
        if PROBABLES_PATH.exists():
            try:
                probables_df = pd.read_parquet(PROBABLES_PATH)
            except Exception:
                probables_df = pd.DataFrame()
        else:
            probables_df = pd.DataFrame()

    slate_dates = sorted(slate_dates_from_props(props))
    prob_dates = sorted(_probables_dates(probables_df))
    missing_dates = sorted(
        missing_probables_dates_for_slate(props, probables_df)
    )

    games_missing_sp = []
    games_with_sp = 0
    date_skew_games = []

    for _, game in slate.iterrows():
        game_date = str(game["game_date"])
        home_team = game["home_team"]
        away_team = game["away_team"]
        label = game.get("game") or f"{away_team} @ {home_team}"

        row = _probables_row_for_game(
            probables_df,
            game_date,
            home_team,
            away_team,
        )

        missing_parts = []
        if row is None:
            missing_parts.append("no probables row for game")
        else:
            if not isinstance(row.get("home_sp_name"), str) or not row["home_sp_name"].strip():
                missing_parts.append("home SP TBD")
            if not isinstance(row.get("away_sp_name"), str) or not row["away_sp_name"].strip():
                missing_parts.append("away SP TBD")
            stored_date = str(row.get("game_date", ""))[:10]
            if stored_date and stored_date != game_date:
                date_skew_games.append(
                    f"{label} (slate {game_date}, probables {stored_date})"
                )

        if missing_parts:
            games_missing_sp.append({
                "game": label,
                "game_date": game_date,
                "missing": missing_parts,
            })
        else:
            games_with_sp += 1

    warnings = []
    if missing_dates:
        warnings.append(
            "Probables missing entire slate date(s): "
            + ", ".join(missing_dates)
            + ". Batter Score SP lookup may fall back to adjacent dates."
        )

    if date_skew_games:
        warnings.append(
            "Probables matched via adjacent schedule date for: "
            + "; ".join(date_skew_games)
            + ". Re-fetch with `python fetch_data.py --probables` to store "
            "the correct Eastern slate date."
        )

    if slate_dates and prob_dates and not set(slate_dates) & set(prob_dates):
        warnings.append(
            "No overlap between props slate dates ("
            + ", ".join(slate_dates)
            + ") and probables file dates ("
            + ", ".join(prob_dates)
            + "). Check timezone / re-fetch probables."
        )

    if games_missing_sp:
        detail_lines = []
        for item in games_missing_sp:
            detail_lines.append(
                f"  - {item['game']} ({item['game_date']}): "
                + "; ".join(item["missing"])
            )
        warnings.append(
            "Games missing probables / SP names for Batter Score:\n"
            + "\n".join(detail_lines)
        )

    game_count = len(slate)
    ok = bool(games_with_sp == game_count)

    return {
        "ok": ok,
        "game_count": game_count,
        "games_with_sp": games_with_sp,
        "games_missing_sp": games_missing_sp,
        "warnings": warnings,
        "slate_dates": slate_dates,
        "probables_dates": prob_dates,
        "missing_probables_dates": missing_dates,
    }


def warn_probables_slate_coverage(
    props: pd.DataFrame,
    probables_df: pd.DataFrame | None = None,
    *,
    context: str = "",
) -> dict:
    """
    Print non-fatal warnings when probables look misaligned with the slate.

    Advisory only — never blocks predict or Streamlit; the board still loads
    with Partial · SP TBD when SP lookup fails.
    """
    result = analyze_probables_slate_coverage(
        props,
        probables_df=probables_df,
    )

    if result["game_count"] == 0:
        return result

    prefix = "WARNING: Probables / Batter Score SP coverage"
    if context:
        prefix = f"{prefix} ({context})"

    print()
    print("=" * 60)
    print(prefix)
    print("=" * 60)
    print(
        f"Slate games: {result['game_count']} | "
        f"With both SPs: {result['games_with_sp']} | "
        f"Slate dates: {', '.join(result['slate_dates']) or '—'} | "
        f"Probables dates: {', '.join(result['probables_dates']) or '—'}"
    )

    if result["ok"]:
        print("OK — probables align with today's props slate.")
    else:
        for message in result["warnings"]:
            print(message)
        print(
            "Fix: python fetch_data.py --probables "
            "(auto-fetches missing Eastern slate dates when props cache exists)."
        )

    print()
    return result


def ensure_probables_for_props_slate(
    props: pd.DataFrame | None = None,
    *,
    primary_date: str | None = None,
) -> pd.DataFrame:
    """
    Fetch and merge probables for the Eastern schedule date(s) on today's slate.

    Always refreshes *primary_date* (default: Eastern today). Also fetches any
    props slate dates missing from daily_probables.parquet.
    """
    if props is None and CURRENT_PROPS_PATH.exists():
        try:
            props = pd.read_parquet(CURRENT_PROPS_PATH)
        except Exception:
            props = None

    if primary_date is None:
        primary_date = mlb_schedule_date()

    dates_to_fetch = {primary_date}
    if props is not None and not props.empty:
        dates_to_fetch.update(slate_dates_from_props(props))

    combined = (
        pd.read_parquet(PROBABLES_PATH)
        if PROBABLES_PATH.exists()
        else pd.DataFrame()
    )
    dates_to_fetch.update(
        missing_probables_dates_for_slate(props, combined)
        if props is not None and not props.empty
        else set()
    )

    fetched_frames = []
    for schedule_date in sorted(dates_to_fetch):
        print(f">>> Fetching probables for {schedule_date} (US Eastern slate)...")
        fetched_frames.append(
            fetch_probables(game_date=schedule_date)
        )

    if not fetched_frames:
        return combined

    incoming = pd.concat(
        fetched_frames,
        ignore_index=True,
    )
    return save_probables(incoming)


def fetch_and_save_probables(
    game_date: Optional[str] = None,
    *,
    props: pd.DataFrame | None = None,
) -> pd.DataFrame:
    print()
    print("=" * 60)
    print("DOWNLOADING MLB PROBABLE STARTING PITCHERS")
    print("=" * 60)

    if game_date is None:
        print(f"Primary schedule date (US Eastern): {mlb_schedule_date()}")
    else:
        print(f"Primary schedule date: {game_date}")

    combined = ensure_probables_for_props_slate(
        props=props,
        primary_date=game_date,
    )

    if len(combined):
        latest = combined[
            combined["game_date"].astype(str).eq(
                str(game_date or mlb_schedule_date())
            )
        ]
        if latest.empty:
            latest = combined
        known_home = latest["home_sp_name"].notna().sum()
        known_away = latest["away_sp_name"].notna().sum()
        print(
            f"Probables cache: {len(combined):,} rows across "
            f"{combined['game_date'].astype(str).nunique()} date(s); "
            f"primary day has {known_home} home / {known_away} away SP named."
        )
    else:
        print("No probables collected.")

    if props is None and CURRENT_PROPS_PATH.exists():
        try:
            props = pd.read_parquet(CURRENT_PROPS_PATH)
        except Exception:
            props = None

    if props is not None and not props.empty:
        warn_probables_slate_coverage(
            props,
            combined,
            context="after probables fetch",
        )

    return combined


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

    row = _probables_row_for_game(
        probables_df,
        game_date,
        home_team,
        away_team,
    )

    if row is None:
        return None, None

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
        help="Game date YYYY-MM-DD (default: US Eastern schedule date)",
    )
    args = parser.parse_args()

    try:
        fetch_and_save_probables(game_date=args.date)
    except requests.RequestException as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
