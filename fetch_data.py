import argparse
import sys
import time

import pandas as pd

from pybaseball import (
    statcast,
    cache
)

from odds_api import (
    GAME_MARKETS,
    OddsApiQuotaError,
    get_event_game_lines,
    get_event_props,
    get_events,
    normalize_event,
    redact_api_key,
)
from odds_snapshots import save_live_snapshot
from fetch_probables import fetch_and_save_probables
from utils import (
    RAW_DIR,
    PROCESSED_DIR,
    ODDS_API_KEY,
    current_game_lines_path,
    require_live_fetch,
    statcast_needs_refresh,
    statcast_raw_path,
)


# =========================================================
# STATCAST
# =========================================================

def fetch_statcast(
    start_date,
    end_date,
    force=False,
):
    require_live_fetch("Statcast download (pybaseball)")

    print()
    print("=" * 60)
    print("DOWNLOADING STATCAST")
    print("=" * 60)

    output_file = statcast_raw_path(start_date, end_date)

    if output_file.exists() and not force:
        if not statcast_needs_refresh(start_date, end_date):
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
            "Cached Statcast stops before "
            f"{end_date}; re-fetching..."
        )
        output_file.unlink()
    elif output_file.exists() and force:
        print(
            f"Removing cached Statcast (--force): {output_file}"
        )
        output_file.unlink()

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
    require_live_fetch("live sportsbook props (Odds API)")

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

    if "fetched_at" not in df.columns:
        df["fetched_at"] = pd.NaT

    df["fetched_at"] = df["fetched_at"].fillna(
        pd.Timestamp.now(tz="UTC").isoformat()
    )

    df.to_parquet(
        output_file,
        index=False
    )

    snapshot_path = save_live_snapshot(df)

    print()
    print(
        f"Collected "
        f"{len(df):,} prop rows"
    )

    print(
        "Saved:",
        output_file
    )

    if snapshot_path is not None:
        print(
            "Snapshot:",
            snapshot_path,
        )

    return df


# =========================================================
# COLLECT CURRENT GAME LINES (TOTALS / SPREADS)
# =========================================================

def fetch_current_game_lines():
    require_live_fetch("live game lines (Odds API)")

    if not ODDS_API_KEY:

        raise RuntimeError(
            "ODDS_API_KEY is missing "
            "from .env"
        )

    print()
    print("=" * 60)
    print("DOWNLOADING CURRENT MLB GAME LINES")
    print("=" * 60)

    output_file = current_game_lines_path()

    try:
        events = get_events()
    except OddsApiQuotaError as exc:
        print(f"ERROR: {exc}")
        print(
            "Could not list MLB events — "
            "Odds API quota exhausted."
        )
        if output_file.exists():
            print(
                f"Keeping cached game lines at: {output_file}"
            )
            return pd.read_parquet(output_file)
        sys.exit(1)

    print(
        f"Found {len(events)} MLB events."
    )

    all_rows = []
    quota_exhausted = False

    for index, event in enumerate(
        events,
        start=1,
    ):

        print(
            f"[{index}/{len(events)}] "
            f"{event.get('away_team')} "
            f"@ "
            f"{event.get('home_team')}"
        )

        try:

            event_data = (
                get_event_game_lines(
                    event["id"]
                )
            )

            rows = normalize_event(
                event_data
            )

            all_rows.extend(
                rows
            )

            time.sleep(0.1)

        except OddsApiQuotaError as exc:
            quota_exhausted = True
            print("ERROR:", exc)
            print(
                "Stopping early — Odds API "
                "quota exhausted."
            )
            break

        except Exception as exc:

            print(
                "ERROR:",
                redact_api_key(exc),
            )

    if len(all_rows) == 0:

        if output_file.exists():
            try:
                cached = pd.read_parquet(
                    output_file
                )
                if len(cached) > 0:
                    print()
                    print(
                        "WARNING: Collected 0 "
                        "game line rows. "
                        "NOT overwriting cache."
                    )
                    print(output_file)
                    return cached
            except Exception:
                pass

        if quota_exhausted:
            print(
                "ERROR: Odds API quota exhausted."
            )
        else:
            print(
                "ERROR: No game line rows collected."
            )

        sys.exit(1)

    df = pd.DataFrame(
        all_rows
    )

    if "fetched_at" not in df.columns:
        df["fetched_at"] = pd.NaT

    df["fetched_at"] = df["fetched_at"].fillna(
        pd.Timestamp.now(tz="UTC").isoformat()
    )

    df.to_parquet(
        output_file,
        index=False,
    )

    print()
    print(
        f"Collected "
        f"{len(df):,} game line rows"
    )

    print(
        "Saved:",
        output_file,
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
        "--force",
        action="store_true",
        help=(
            "Re-download Statcast even when a cached parquet exists "
            "(also used when cached data stops before --end)"
        ),
    )

    parser.add_argument(
        "--props",
        action="store_true"
    )

    parser.add_argument(
        "--game-lines",
        action="store_true",
        help=(
            "Fetch game totals and run lines "
            "(totals, spreads) for today's slate"
        ),
    )

    parser.add_argument(
        "--probables",
        action="store_true",
        help=(
            "Fetch probable starting pitchers for today's slate "
            "→ daily_probables.parquet"
        ),
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
            args.end,
            force=args.force,
        )

    if args.props:

        fetch_current_props()

    if args.game_lines:

        fetch_current_game_lines()

    if args.probables:

        fetch_and_save_probables()
