"""
Fetch Underdog Fantasy MLB batter fantasy-point lines.

Underdog does not expose batter_fantasy_score through The Odds API us_dfs
feed (unlike PrizePicks). Their pick'em app loads lines from a public JSON
endpoint — one request returns the full MLB slate.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import requests

from utils import PROCESSED_DIR, require_live_fetch

UNDERDOG_API_BASE = "https://api.underdogfantasy.com/v1"
UNDERDOG_FANTASY_LINES_PATH = (
    PROCESSED_DIR / "underdog_fantasy_lines.parquet"
)
UNDERDOG_FANTASY_STAT = "fantasy_points"


def _player_name(player: dict) -> str:
    first = str(player.get("first_name") or "").strip()
    last = str(player.get("last_name") or "").strip()
    return f"{first} {last}".strip()


def parse_underdog_fantasy_payload(
    payload: dict,
    *,
    fetched_at: str | None = None,
) -> list[dict]:
    """Extract batter fantasy-point Over lines from an over_under_lines payload."""
    if not payload:
        return []

    fetched_at = fetched_at or (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )

    players = {
        player["id"]: player
        for player in payload.get("players", [])
        if isinstance(player, dict) and player.get("id")
    }
    appearances = {
        appearance["id"]: appearance
        for appearance in payload.get("appearances", [])
        if isinstance(appearance, dict) and appearance.get("id")
    }

    rows: list[dict] = []

    for line in payload.get("over_under_lines", []):
        if not isinstance(line, dict):
            continue

        over_under = line.get("over_under") or {}
        appearance_stat = over_under.get("appearance_stat") or {}

        if appearance_stat.get("stat") != UNDERDOG_FANTASY_STAT:
            continue

        if str(line.get("status", "")).lower() not in ("", "active"):
            continue

        appearance = appearances.get(
            appearance_stat.get("appearance_id"),
            {},
        )
        player = players.get(appearance.get("player_id"), {})
        name = _player_name(player)

        stat_value = line.get("stat_value")
        if not name or stat_value is None:
            continue

        try:
            line_value = float(stat_value)
        except (TypeError, ValueError):
            continue

        rows.append({
            "player": name,
            "market": "batter_fantasy_score",
            "side": "over",
            "line": line_value,
            "bookmaker": "Underdog",
            "bookmaker_key": "underdog",
            "fetched_at": fetched_at,
            "source": "underdog_api",
        })

    return rows


def fetch_underdog_fantasy_payload(
    sport_id: str = "MLB",
    *,
    timeout: int = 45,
) -> dict:
    require_live_fetch("Underdog Fantasy pick'em API")

    url = f"{UNDERDOG_API_BASE}/over_under_lines"
    response = requests.get(
        url,
        params={"sport_id": sport_id},
        headers={"Accept": "application/json"},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()

    if not isinstance(payload, dict):
        raise ValueError(
            "Unexpected Underdog API response — expected JSON object."
        )

    return payload


def fetch_underdog_fantasy_lines(
    sport_id: str = "MLB",
) -> list[dict]:
    payload = fetch_underdog_fantasy_payload(sport_id=sport_id)
    return parse_underdog_fantasy_payload(payload)


def save_underdog_fantasy_lines(
    rows: list[dict],
    output_path=UNDERDOG_FANTASY_LINES_PATH,
) -> pd.DataFrame:
    output_path = PROCESSED_DIR / output_path.name

    if not rows:
        if output_path.exists():
            print(
                "WARNING: No Underdog fantasy rows collected; "
                "keeping existing cache at",
                output_path,
            )
            return pd.read_parquet(output_path)

        print("WARNING: No Underdog fantasy rows collected.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = (
        df.sort_values("fetched_at")
        .drop_duplicates(subset=["player"], keep="last")
        .reset_index(drop=True)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    print(
        f"Saved {len(df):,} Underdog fantasy lines:",
        output_path,
    )
    return df


def fetch_and_save_underdog_fantasy_lines(
    sport_id: str = "MLB",
    output_path=UNDERDOG_FANTASY_LINES_PATH,
) -> pd.DataFrame:
    print()
    print("=" * 60)
    print("DOWNLOADING UNDERDOG FANTASY POINT LINES")
    print("=" * 60)
    print(f"Source: {UNDERDOG_API_BASE}/over_under_lines?sport_id={sport_id}")

    rows = fetch_underdog_fantasy_lines(sport_id=sport_id)
    print(f"Collected {len(rows):,} fantasy-point rows")

    return save_underdog_fantasy_lines(rows, output_path=output_path)
