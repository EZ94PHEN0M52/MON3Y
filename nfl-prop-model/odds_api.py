"""The Odds API helpers for NFL player props."""

from __future__ import annotations

import re
from datetime import datetime, timezone

import requests

from utils import (
    NFL_SPORT,
    ODDS_API_BASE,
    ODDS_API_KEY,
    require_live_fetch,
)

# V1 board markets only — keep credit burn bounded.
PROP_MARKETS = [
    "player_pass_yds",
    "player_pass_tds",
    "player_rush_yds",
    "player_receptions",
    "player_reception_yds",
    "player_rush_reception_yds",
]

PROP_REGIONS = "us,us2"

GAME_MARKETS = [
    "h2h",
    "spreads",
    "totals",
]


class OddsApiQuotaError(Exception):
    """Odds API quota exhausted or unauthorized."""


def require_api_key() -> str:
    if not ODDS_API_KEY or ODDS_API_KEY == "your_key_here":
        raise RuntimeError(
            "Set ODDS_API_KEY in nfl-prop-model/.env "
            "(see .env.example)."
        )
    return ODDS_API_KEY


def redact_api_key(text) -> str:
    if not text:
        return text
    return re.sub(
        r"(apiKey=)[^&\s\"']+",
        r"\1***REDACTED***",
        str(text),
        flags=re.IGNORECASE,
    )


def _is_quota_error(status_code, response_text) -> bool:
    if status_code == 401:
        return True
    return "OUT_OF_USAGE_CREDITS" in (response_text or "")


def odds_request(url, params, timeout=30):
    require_live_fetch("Odds API request")
    require_api_key()

    response = requests.get(url, params=params, timeout=timeout)

    if response.status_code != 200:
        safe_text = redact_api_key(response.text)
        print(response.status_code)
        print(safe_text)

        if _is_quota_error(response.status_code, response.text):
            raise OddsApiQuotaError(
                "Odds API quota exhausted or unauthorized "
                f"(HTTP {response.status_code})"
            )

        response.raise_for_status()

    return response.json()


def normalize_event(event, fetched_at=None):
    if fetched_at is None:
        fetched_at = datetime.now(timezone.utc).isoformat()

    rows = []

    for bookmaker in event.get("bookmakers", []):
        bookmaker_name = bookmaker.get("title")
        bookmaker_key = bookmaker.get("key")

        for market in bookmaker.get("markets", []):
            market_key = market.get("key")
            market_update = market.get("last_update")

            for outcome in market.get("outcomes", []):
                player = outcome.get("description")
                side = outcome.get("name")

                if market_key in ("totals", "spreads", "h2h"):
                    player = None

                rows.append({
                    "event_id": event.get("id"),
                    "commence_time": event.get("commence_time"),
                    "home_team": event.get("home_team"),
                    "away_team": event.get("away_team"),
                    "bookmaker": bookmaker_name,
                    "bookmaker_key": bookmaker_key,
                    "market": market_key,
                    "player": player,
                    "side": side,
                    "line": outcome.get("point"),
                    "odds": outcome.get("price"),
                    "last_update": market_update,
                    "fetched_at": fetched_at,
                })

    return rows


def get_events():
    url = f"{ODDS_API_BASE}/sports/{NFL_SPORT}/events"
    return odds_request(url, {"apiKey": ODDS_API_KEY})


def get_event_odds(event_id, markets):
    url = (
        f"{ODDS_API_BASE}/sports/{NFL_SPORT}/events/"
        f"{event_id}/odds"
    )
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": PROP_REGIONS,
        "markets": ",".join(markets),
        "oddsFormat": "american",
    }
    return odds_request(url, params)


def get_event_props(event_id, markets=None):
    if markets is None:
        markets = PROP_MARKETS
    return get_event_odds(event_id, markets)


def get_event_game_lines(event_id, markets=None):
    if markets is None:
        markets = GAME_MARKETS
    return get_event_odds(event_id, markets)
