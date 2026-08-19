import argparse
import re
import sys
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
    "batter_walks",
    "batter_hits_runs_rbis",

    "pitcher_strikeouts",
    "pitcher_walks",
    "pitcher_hits_allowed",
    "pitcher_outs",
    "pitcher_earned_runs",
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
    params
):

    response = requests.get(
        url,
        params=params,
        timeout=30
    )

    if response.status_code != 200:

        safe_text = redact_api_key(response.text)

        print(
            response.status_code
        )

        print(
            safe_text
        )

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

def _exit_props_fetch_failure(
    message,
    output_file=None,
):
    print()
    print(message)

    if (
        output_file is not None
        and output_file.exists()
    ):
        try:
            cached = pd.read_parquet(
                output_file
            )
            if len(cached) > 0:
                print()
                print(
                    "WARNING: Keeping existing "
                    f"cached props "
                    f"({len(cached):,} rows) at:"
                )
                print(output_file)
                print()
                print(
                    "Re-run predictions with "
                    "cached props, or skip "
                    "fetch with "
                    "./run_daily.sh --skip-props"
                )
        except Exception:
            pass

    sys.exit(1)


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

    output_file = (
        PROCESSED_DIR /
        "current_props.parquet"
    )

    try:
        events = get_events()
    except OddsApiQuotaError as exc:
        _exit_props_fetch_failure(
            f"ERROR: {exc}\n"
            "Could not list MLB events — "
            "Odds API quota exhausted.",
            output_file,
        )

    print(
        f"Found {len(events)} MLB events."
    )

    all_rows = []
    events_failed = 0
    quota_exhausted = False

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

        except OddsApiQuotaError as exc:
            quota_exhausted = True
            print(
                "ERROR:",
                exc,
            )
            print(
                "Stopping early — Odds API "
                "quota exhausted "
                "(OUT_OF_USAGE_CREDITS)."
            )
            break

        except Exception as exc:

            events_failed += 1

            print(
                "ERROR:",
                redact_api_key(exc),
            )

    if len(all_rows) == 0:

        all_events_failed = (
            len(events) > 0
            and events_failed == len(events)
        )

        if output_file.exists():
            try:
                cached = pd.read_parquet(
                    output_file
                )
                if len(cached) > 0:
                    print()
                    print(
                        "WARNING: Collected 0 "
                        "prop rows (API quota or "
                        "fetch errors). "
                        "NOT overwriting "
                        "existing cache."
                    )
                    print()
                    print(
                        f"Using cached props "
                        f"({len(cached):,} rows) "
                        f"at:"
                    )
                    print(output_file)
                    print()
                    print(
                        "Re-run with "
                        "./run_daily.sh "
                        "--skip-props to skip "
                        "this fetch step."
                    )
                    sys.exit(1)
            except Exception:
                pass

        if quota_exhausted:
            reason = (
                "Odds API quota exhausted "
                "(OUT_OF_USAGE_CREDITS)."
            )
        elif all_events_failed or len(events) == 0:
            reason = (
                "Odds API quota exhausted "
                "(OUT_OF_USAGE_CREDITS)."
                if events_failed > 0
                or len(events) == 0
                else "No prop rows returned "
                "for today's events."
            )
        else:
            reason = (
                "No prop rows collected."
            )

        _exit_props_fetch_failure(
            f"ERROR: {reason}\n"
            "No cached current_props.parquet "
            "available — cannot continue.",
            output_file=None,
        )

    df = pd.DataFrame(
        all_rows
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
