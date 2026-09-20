"""
Fetch Sleeper Picks NFL player props via Apify.

Requires APIFY_TOKEN in .env (https://console.apify.com/account/integrations).
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from utils import (
    PROCESSED_DIR,
    parse_commence_datetime,
    require_live_fetch,
    sleeper_props_path,
)

APIFY_ACTOR_ID = "solidcode~sleeper-player-props-scraper"
APIFY_FALLBACK_ACTOR_ID = "zen-studio~sleeper-player-props"
APIFY_SYNC_URL = (
    f"https://api.apify.com/v2/acts/{APIFY_ACTOR_ID}/run-sync-get-dataset-items"
)
APIFY_FALLBACK_SYNC_URL = (
    f"https://api.apify.com/v2/acts/{APIFY_FALLBACK_ACTOR_ID}"
    "/run-sync-get-dataset-items"
)
SLEEPER_PROPS_PATH = sleeper_props_path()

# Map Sleeper NFL stat keys → Odds API / model market keys where possible.
SLEEPER_STAT_TO_MARKET: dict[str, str] = {
    "pass_yds": "player_pass_yds",
    "passing_yards": "player_pass_yds",
    "pass_yards": "player_pass_yds",
    "pass_tds": "player_pass_tds",
    "passing_tds": "player_pass_tds",
    "pass_touchdowns": "player_pass_tds",
    "rush_yds": "player_rush_yds",
    "rushing_yards": "player_rush_yds",
    "rush_yards": "player_rush_yds",
    "receptions": "player_receptions",
    "rec": "player_receptions",
    "rec_yds": "player_reception_yds",
    "receiving_yards": "player_reception_yds",
    "reception_yds": "player_reception_yds",
    "rush_rec_yds": "player_rush_reception_yds",
    "rush_reception_yds": "player_rush_reception_yds",
    "rushing_receiving_yards": "player_rush_reception_yds",
    "rushing_and_receiving_yards": "player_rush_reception_yds",
    "passing_touchdowns": "player_pass_tds",
}


def _apify_token() -> str | None:
    token = os.getenv("APIFY_TOKEN", "").strip()
    return token or None


def _redact_secrets(text) -> str:
    if not text:
        return text
    text = str(text)
    text = re.sub(
        r"(token=)[^&\s\"']+",
        r"\1***REDACTED***",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(Bearer\s+)\S+",
        r"\1***REDACTED***",
        text,
        flags=re.IGNORECASE,
    )
    return text


def _item_get(item: dict, *keys: str):
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]
    return None


def _normalize_game_start(value):
    dt = parse_commence_datetime(value)
    if dt is None:
        return value
    return dt.isoformat()


def _normalize_apify_item(item: dict) -> dict:
    return {
        "league": _item_get(item, "league", "League"),
        "status": _item_get(item, "status", "Status"),
        "player_name": _item_get(item, "player_name", "playerName"),
        "player_team": _item_get(item, "player_team", "playerTeam"),
        "player_position": _item_get(item, "player_position", "playerPosition"),
        "stat": _item_get(item, "stat", "Stat"),
        "stat_display": _item_get(item, "stat_display", "statDisplay"),
        "line": _item_get(item, "line", "Line"),
        "over_multiplier": _item_get(item, "over_multiplier", "overMultiplier"),
        "under_multiplier": _item_get(item, "under_multiplier", "underMultiplier"),
        "pick_popularity": _item_get(item, "pick_popularity", "pickPopularity"),
        "pick_count_over": _item_get(item, "pick_count_over", "pickCountOver"),
        "pick_count_under": _item_get(item, "pick_count_under", "pickCountUnder"),
        "pick_count_total": _item_get(item, "pick_count_total", "pickCountTotal"),
        "game_start": _item_get(item, "game_start", "gameStart"),
        "home_team": _item_get(item, "home_team", "homeTeam"),
        "away_team": _item_get(item, "away_team", "awayTeam"),
        "home_team_name": _item_get(item, "home_team_name", "homeTeamName"),
        "away_team_name": _item_get(item, "away_team_name", "awayTeamName"),
        "projection_id": _item_get(item, "projection_id", "projectionId"),
        "updated_at": _item_get(item, "updated_at", "updatedAt"),
        "game_id": _item_get(item, "game_id", "gameId"),
        "game_status": _item_get(item, "game_status", "gameStatus"),
        "venue_name": _item_get(item, "venue_name", "venueName"),
        "venue_city": _item_get(item, "venue_city", "venueCity"),
        "player_image": _item_get(item, "player_image", "playerImage"),
        "player_injury_status": _item_get(
            item, "player_injury_status", "playerInjuryStatus"
        ),
    }


def _map_market(stat: str) -> str:
    stat_key = str(stat or "").strip().lower()
    return SLEEPER_STAT_TO_MARKET.get(stat_key, f"sleeper_{stat_key}")


def _game_label(item: dict) -> str:
    away = item.get("away_team_name") or item.get("away_team") or ""
    home = item.get("home_team_name") or item.get("home_team") or ""
    away = str(away).strip()
    home = str(home).strip()
    if away and home and away.lower() != "none" and home.lower() != "none":
        return f"{away} @ {home}"
    return ""


def _fill_game_labels(rows: list[dict]) -> list[dict]:
    """Apify NFL payloads often omit home/away; infer label from game_id teams."""
    by_game: dict[str, set[str]] = {}
    for row in rows:
        game_id = str(row.get("game_id") or "").strip()
        team = str(row.get("player_team") or "").strip().upper()
        if game_id and team and team not in {"NONE", "NAN"}:
            by_game.setdefault(game_id, set()).add(team)

    labels = {}
    for game_id, teams in by_game.items():
        ordered = sorted(teams)
        if len(ordered) >= 2:
            labels[game_id] = f"{ordered[0]} @ {ordered[1]}"
        elif len(ordered) == 1:
            labels[game_id] = ordered[0]

    for row in rows:
        if str(row.get("game") or "").strip():
            continue
        game_id = str(row.get("game_id") or "").strip()
        if game_id in labels:
            row["game"] = labels[game_id]
    return rows


def parse_sleeper_apify_items(
    items: list[dict],
    *,
    fetched_at: str | None = None,
) -> list[dict]:
    if not items:
        return []

    fetched_at = fetched_at or (
        datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    )

    rows: list[dict] = []
    for raw in items:
        if not isinstance(raw, dict):
            continue

        item = _normalize_apify_item(raw)
        league = str(item.get("league") or "").strip().lower()
        if league not in {"nfl", "americanfootball_nfl"}:
            continue

        status = str(item.get("status", "active")).lower()
        if status not in ("", "active"):
            continue

        player_name = str(item.get("player_name") or "").strip()
        stat = str(item.get("stat") or "").strip()
        line_value = item.get("line")
        if not player_name or not stat or line_value is None:
            continue

        try:
            line = float(line_value)
        except (TypeError, ValueError):
            continue

        over_mult = item.get("over_multiplier")
        under_mult = item.get("under_multiplier")
        pick_pop = item.get("pick_popularity")

        rows.append({
            "player": player_name,
            "market": _map_market(stat),
            "sleeper_stat": stat,
            "stat_display": (
                item.get("stat_display") or stat.replace("_", " ").title()
            ),
            "line": line,
            "over_multiplier": (
                float(over_mult) if over_mult is not None else None
            ),
            "under_multiplier": (
                float(under_mult) if under_mult is not None else None
            ),
            "pick_popularity": (
                float(pick_pop) if pick_pop is not None else None
            ),
            "pick_count_over": item.get("pick_count_over"),
            "pick_count_under": item.get("pick_count_under"),
            "pick_count_total": item.get("pick_count_total"),
            "game": _game_label(item),
            "game_start": _normalize_game_start(item.get("game_start")),
            "player_team": item.get("player_team"),
            "player_position": item.get("player_position"),
            "player_injury_status": item.get("player_injury_status"),
            "home_team": item.get("home_team"),
            "away_team": item.get("away_team"),
            "status": status or "active",
            "projection_id": item.get("projection_id"),
            "updated_at": item.get("updated_at"),
            "game_id": item.get("game_id"),
            "game_status": item.get("game_status"),
            "venue_name": item.get("venue_name"),
            "venue_city": item.get("venue_city"),
            "player_image": item.get("player_image"),
            "bookmaker": "Sleeper",
            "bookmaker_key": "sleeper",
            "source": "apify_sleeper",
            "fetched_at": fetched_at,
        })

    return _fill_game_labels(rows)


def _run_apify_actor(
    *,
    actor_id: str,
    sync_url: str,
    run_input: dict,
    timeout: int,
) -> list[dict]:
    token = _apify_token()
    if not token:
        raise ValueError(
            "APIFY_TOKEN is missing from .env — "
            "add a token from https://console.apify.com/account/integrations"
        )

    response = requests.post(
        sync_url,
        params={"token": token},
        json=run_input,
        headers={"Accept": "application/json"},
        timeout=timeout,
    )

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail = _redact_secrets(response.text[:500])
        raise requests.HTTPError(
            f"Apify Sleeper actor {actor_id} failed "
            f"({response.status_code}): {detail}",
            response=response,
        ) from exc

    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError(
            f"Unexpected Apify response from {actor_id} — "
            "expected a JSON array of prop rows."
        )
    return payload


def fetch_sleeper_props_via_apify(*, timeout: int = 300) -> list[dict]:
    require_live_fetch("Apify Sleeper player-props actor")

    fetched_at = (
        datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    )

    primary_items = _run_apify_actor(
        actor_id=APIFY_ACTOR_ID,
        sync_url=APIFY_SYNC_URL,
        run_input={"leagues": ["nfl"]},
        timeout=timeout,
    )
    rows = parse_sleeper_apify_items(primary_items, fetched_at=fetched_at)
    if rows:
        return rows

    print(
        "WARNING: Primary Apify actor returned no NFL rows; "
        f"trying fallback {APIFY_FALLBACK_ACTOR_ID}..."
    )
    fallback_items = _run_apify_actor(
        actor_id=APIFY_FALLBACK_ACTOR_ID,
        sync_url=APIFY_FALLBACK_SYNC_URL,
        run_input={"leagues": ["NFL"]},
        timeout=timeout,
    )
    return parse_sleeper_apify_items(fallback_items, fetched_at=fetched_at)


def save_sleeper_props(
    rows: list[dict],
    output_path=None,
) -> pd.DataFrame:
    output_path = output_path or SLEEPER_PROPS_PATH
    output_path = PROCESSED_DIR / Path(output_path).name

    if not rows:
        if output_path.exists():
            print(
                "WARNING: No Sleeper prop rows collected; "
                "keeping existing cache at",
                output_path,
            )
            return pd.read_parquet(output_path)
        print("WARNING: No Sleeper prop rows collected.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    dedupe_cols = ["projection_id"]
    if "projection_id" not in df.columns or df["projection_id"].isna().all():
        dedupe_cols = ["player", "sleeper_stat", "line", "game_start"]

    df = (
        df.sort_values("fetched_at")
        .drop_duplicates(subset=dedupe_cols, keep="last")
        .reset_index(drop=True)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    teams = sorted(df["player_team"].dropna().astype(str).unique())
    print(
        f"Saved {len(df):,} Sleeper prop rows across "
        f"{len(teams)} team(s):",
        output_path,
    )
    return df


def fetch_and_save_sleeper_props(output_path=None) -> pd.DataFrame:
    print()
    print("=" * 60)
    print("DOWNLOADING SLEEPER PICKS NFL PROPS (APIFY)")
    print("=" * 60)
    print("Actor: solidcode/sleeper-player-props-scraper | leagues: nfl")

    rows = fetch_sleeper_props_via_apify()
    print(f"Collected {len(rows):,} Sleeper prop rows")
    return save_sleeper_props(rows, output_path=output_path)
