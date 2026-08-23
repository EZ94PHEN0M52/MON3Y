import re
from datetime import datetime, timezone

import requests

from utils import (
    ODDS_API_KEY,
    ODDS_API_BASE,
    MLB_SPORT,
    require_live_fetch,
)


PROP_MARKETS = [
    "batter_hits",
    "batter_home_runs",
    "batter_total_bases",
    "batter_rbis",
    "batter_runs_scored",
    "batter_walks",
    "batter_hits_runs_rbis",
    "batter_stolen_bases",
    "pitcher_strikeouts",
    "pitcher_walks",
    "pitcher_hits_allowed",
    "pitcher_outs",
    "pitcher_earned_runs",
]

# PrizePicks DFS fantasy score (us_dfs region only).
PRIZEPICKS_FANTASY_MARKETS = [
    "batter_fantasy_score",
]


# US books split across two Odds API regions. Some player props (notably
# pitcher_walks) are posted only on us2 books (Fliff, Hard Rock Bet).
PROP_REGIONS = "us,us2"


# Game-level markets (totals, run lines) — used as context features, not
# standalone betting models in Phase 5.
GAME_MARKETS = [
    "totals",
    "spreads",
]


class OddsApiQuotaError(Exception):
    """Odds API quota exhausted or unauthorized."""


def redact_api_key(text):
    if not text:
        return text

    return re.sub(
        r"(apiKey=)[^&\s\"']+",
        r"\1***REDACTED***",
        str(text),
        flags=re.IGNORECASE,
    )


def _is_quota_error(status_code, response_text):
    if status_code == 401:
        return True

    return "OUT_OF_USAGE_CREDITS" in (response_text or "")


def odds_request(
    url,
    params,
    timeout=30,
):
    require_live_fetch("Odds API request")
    response = requests.get(
        url,
        params=params,
        timeout=timeout,
    )

    if response.status_code != 200:
        safe_text = redact_api_key(response.text)

        print(response.status_code)
        print(safe_text)

        if _is_quota_error(
            response.status_code,
            response.text,
        ):
            raise OddsApiQuotaError(
                "Odds API quota exhausted or "
                "unauthorized (HTTP "
                f"{response.status_code})"
            )

        response.raise_for_status()

    return response.json()


def normalize_event(
    event,
    fetched_at=None,
    snapshot_date=None,
):
    if fetched_at is None:
        fetched_at = (
            datetime.now(timezone.utc)
            .isoformat()
        )

    rows = []

    for bookmaker in event.get(
        "bookmakers",
        []
    ):
        bookmaker_name = (
            bookmaker.get("title")
        )

        bookmaker_key = (
            bookmaker.get("key")
        )

        for market in bookmaker.get(
            "markets",
            []
        ):
            market_key = (
                market.get("key")
            )

            market_update = (
                market.get("last_update")
            )

            for outcome in market.get(
                "outcomes",
                []
            ):
                player = outcome.get(
                    "description"
                )

                side = outcome.get("name")

                # Game markets use team/outcome name, not player props.
                if market_key in (
                    "totals",
                    "spreads",
                    "h2h",
                ):
                    player = None

                row = {
                    "event_id":
                        event.get("id"),

                    "commence_time":
                        event.get(
                            "commence_time"
                        ),

                    "home_team":
                        event.get(
                            "home_team"
                        ),

                    "away_team":
                        event.get(
                            "away_team"
                        ),

                    "bookmaker":
                        bookmaker_name,

                    "bookmaker_key":
                        bookmaker_key,

                    "market":
                        market_key,

                    "player":
                        player,

                    "side":
                        side,

                    "line":
                        outcome.get(
                            "point"
                        ),

                    "odds":
                        outcome.get(
                            "price"
                        ),

                    "last_update":
                        market_update,

                    "fetched_at":
                        fetched_at,
                }

                if snapshot_date is not None:
                    row["snapshot_date"] = (
                        snapshot_date
                    )

                rows.append(row)

    return rows


def get_events():
    url = (
        f"{ODDS_API_BASE}/sports/"
        f"{MLB_SPORT}/events"
    )

    params = {
        "apiKey": ODDS_API_KEY,
    }

    return odds_request(
        url,
        params,
    )


def get_event_odds(
    event_id,
    markets,
):
    url = (
        f"{ODDS_API_BASE}/sports/"
        f"{MLB_SPORT}/events/"
        f"{event_id}/odds"
    )

    params = {
        "apiKey": ODDS_API_KEY,
        "regions": PROP_REGIONS,
        "markets": ",".join(
            markets
        ),
        "oddsFormat": "american",
    }

    return odds_request(
        url,
        params,
    )


def get_event_props(
    event_id,
    markets=None,
):
    if markets is None:
        markets = PROP_MARKETS

    return get_event_odds(
        event_id,
        markets,
    )


def _get_event_prizepicks(
    event_id,
    markets,
):
    url = (
        f"{ODDS_API_BASE}/sports/"
        f"{MLB_SPORT}/events/"
        f"{event_id}/odds"
    )

    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us_dfs",
        "bookmakers": "prizepicks",
        "markets": ",".join(markets),
        "oddsFormat": "american",
    }

    return odds_request(
        url,
        params,
    )


def get_event_prizepicks_props(
    event_id,
    markets=None,
):
    """PrizePicks player props for standard markets (Odds API us_dfs region)."""
    if markets is None:
        markets = PROP_MARKETS

    return _get_event_prizepicks(
        event_id,
        markets,
    )


def get_event_prizepicks_fantasy(
    event_id,
    markets=None,
):
    """PrizePicks batter fantasy score lines (Odds API us_dfs region)."""
    if markets is None:
        markets = PRIZEPICKS_FANTASY_MARKETS

    return _get_event_prizepicks(
        event_id,
        markets,
    )


def get_event_game_lines(
    event_id,
    markets=None,
):
    if markets is None:
        markets = GAME_MARKETS

    return get_event_odds(
        event_id,
        markets,
    )
