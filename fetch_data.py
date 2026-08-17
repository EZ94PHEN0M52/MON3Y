import argparse
import time

import pandas as pd
import requests

from pybaseball import (
    statcast,
    cache
)

from utils import (
    RAW_DIR,
    PROCESSED_DIR,
    ODDS_API_KEY,
    ODDS_API_BASE,
    MLB_SPORT
)


# =========================================================
# STATCAST
# =========================================================

def fetch_statcast(
    start_date,
    end_date
):

    print()
    print("=" * 60)
    print("DOWNLOADING STATCAST")
    print("=" * 60)

    output_file = (
        RAW_DIR /
        f"statcast_{start_date}_{end_date}.parquet"
    )

    if output_file.exists():

        print(
            "Already exists:"
        )

        print(
            output_file
        )

        return pd.read_parquet(
            output_file
        )

    print(
        f"Requesting "
        f"{start_date} → {end_date}"
    )

    # Enable pybaseball caching.
    cache.enable()

    df = statcast(
        start_dt=start_date,
        end_dt=end_date,
        verbose=True,
        parallel=True
    )

    print(
        f"Downloaded "
        f"{len(df):,} rows"
    )

    df.to_parquet(
        output_file,
        index=False
    )

    print(
        "Saved:",
        output_file
    )

    return df


# =========================================================
# ODDS API
# =========================================================

PROP_MARKETS = [
    "batter_hits",
    "batter_home_runs",
    "batter_total_bases",
    "batter_rbis",
    "batter_runs_scored",

    "pitcher_strikeouts",
    "pitcher_walks",
    "pitcher_hits_allowed",
]


def odds_request(
    url,
    params
):

    response = requests.get(
        url,
        params=params,
        timeout=30
    )

    if response.status_code != 200:

        print(
            response.status_code
        )

        print(
            response.text
        )

    response.raise_for_status()

    return response.json()


# =========================================================
# GET TODAY'S MLB EVENTS
# =========================================================

def get_events():

    url = (
        f"{ODDS_API_BASE}/sports/"
        f"{MLB_SPORT}/events"
    )

    params = {
        "apiKey": ODDS_API_KEY
    }

    return odds_request(
        url,
        params
    )


# =========================================================
# GET EVENT PROPS
# =========================================================

def get_event_props(
    event_id
):

    url = (
        f"{ODDS_API_BASE}/sports/"
        f"{MLB_SPORT}/events/"
        f"{event_id}/odds"
    )

    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": ",".join(
            PROP_MARKETS
        ),
        "oddsFormat": "american"
    }

    return odds_request(
        url,
        params
    )


# =========================================================
# NORMALIZE ODDS
# =========================================================

def normalize_event(
    event
):

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

                rows.append({

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
                        outcome.get(
                            "description"
                        ),

                    "side":
                        outcome.get(
                            "name"
                        ),

                    "line":
                        outcome.get(
                            "point"
                        ),

                    "odds":
                        outcome.get(
                            "price"
                        ),

                    "last_update":
                        market_update
                })

    return rows


# =========================================================
# COLLECT ALL CURRENT PROPS
# =========================================================

def fetch_current_props():

    if not ODDS_API_KEY:

        raise RuntimeError(
            "ODDS_API_KEY is missing "
            "from .env"
        )

    print()
    print("=" * 60)
    print("DOWNLOADING CURRENT MLB PROPS")
    print("=" * 60)

    events = get_events()

    print(
        f"Found {len(events)} MLB events."
    )

    all_rows = []

    for index, event in enumerate(
        events,
        start=1
    ):

        print(
            f"[{index}/{len(events)}] "
            f"{event.get('away_team')} "
            f"@ "
            f"{event.get('home_team')}"
        )

        try:

            event_data = (
                get_event_props(
                    event["id"]
                )
            )

            rows = normalize_event(
                event_data
            )

            all_rows.extend(
                rows
            )

            # Avoid hammering the API.
            time.sleep(0.1)

        except Exception as exc:

            print(
                "ERROR:",
                exc
            )

    df = pd.DataFrame(
        all_rows
    )

    output_file = (
        PROCESSED_DIR /
        "current_props.parquet"
    )

    df.to_parquet(
        output_file,
        index=False
    )

    print()
    print(
        f"Collected "
        f"{len(df):,} prop rows"
    )

    print(
        "Saved:",
        output_file
    )

    return df


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--start",
        help="YYYY-MM-DD"
    )

    parser.add_argument(
        "--end",
        help="YYYY-MM-DD"
    )

    parser.add_argument(
        "--statcast",
        action="store_true"
    )

    parser.add_argument(
        "--props",
        action="store_true"
    )

    args = parser.parse_args()

    if args.statcast:

        if not args.start or not args.end:

            raise ValueError(
                "--start and --end "
                "are required"
            )

        fetch_statcast(
            args.start,
            args.end
        )

    if args.props:

        fetch_current_props()
