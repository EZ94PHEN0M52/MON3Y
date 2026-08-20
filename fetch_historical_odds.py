import argparse
import sys
import time
from datetime import (
    datetime,
    timedelta,
    timezone,
)

import pandas as pd

from odds_api import (
    GAME_MARKETS,
    PROP_MARKETS,
    OddsApiQuotaError,
    normalize_event,
    odds_request,
    redact_api_key,
)
from utils import (
    ODDS_API_KEY,
    ODDS_API_BASE,
    MLB_SPORT,
    historical_game_lines_path,
    historical_odds_path,
    require_live_fetch,
)


def date_range(start, end):
    current = datetime.strptime(
        start,
        "%Y-%m-%d",
    ).date()

    last = datetime.strptime(
        end,
        "%Y-%m-%d",
    ).date()

    while current <= last:
        yield current.isoformat()
        current += timedelta(days=1)


def get_historical_events(
    snapshot_iso,
    commence_from=None,
    commence_to=None,
):
    url = (
        f"{ODDS_API_BASE}/historical/sports/"
        f"{MLB_SPORT}/events"
    )

    params = {
        "apiKey": ODDS_API_KEY,
        "date": snapshot_iso,
    }

    if commence_from:
        params["commenceTimeFrom"] = (
            commence_from
        )

    if commence_to:
        params["commenceTimeTo"] = (
            commence_to
        )

    payload = odds_request(
        url,
        params,
    )

    return payload.get("data") or []


def get_historical_event_odds(
    event_id,
    snapshot_iso,
    markets,
):
    url = (
        f"{ODDS_API_BASE}/historical/sports/"
        f"{MLB_SPORT}/events/"
        f"{event_id}/odds"
    )

    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": ",".join(
            markets
        ),
        "oddsFormat": "american",
        "date": snapshot_iso,
    }

    payload = odds_request(
        url,
        params,
    )

    return payload.get("data")


def fetch_date_props(
    snapshot_date,
    markets,
    sleep_seconds=0.15,
):
    fetched_at = (
        datetime.now(timezone.utc)
        .isoformat()
    )

    commence_from = (
        f"{snapshot_date}T00:00:00Z"
    )

    commence_to = (
        f"{snapshot_date}T23:59:59Z"
    )

    # End-of-day snapshot lists games scheduled that day.
    events_snapshot = (
        f"{snapshot_date}T23:59:00Z"
    )

    events = get_historical_events(
        events_snapshot,
        commence_from=commence_from,
        commence_to=commence_to,
    )

    print(
        f"  {len(events)} events on "
        f"{snapshot_date}"
    )

    all_rows = []
    events_failed = 0

    for index, event in enumerate(
        events,
        start=1,
    ):
        event_id = event["id"]

        print(
            f"  [{index}/{len(events)}] "
            f"{event.get('away_team')} @ "
            f"{event.get('home_team')}"
        )

        # Pre-game snapshot: closest odds at or before first pitch.
        odds_snapshot = event.get(
            "commence_time",
            events_snapshot,
        )

        try:
            event_data = (
                get_historical_event_odds(
                    event_id,
                    odds_snapshot,
                    markets,
                )
            )

            if not event_data:
                continue

            rows = normalize_event(
                event_data,
                fetched_at=fetched_at,
                snapshot_date=snapshot_date,
            )

            all_rows.extend(rows)

            time.sleep(sleep_seconds)

        except OddsApiQuotaError:
            raise

        except Exception as exc:
            events_failed += 1
            print(
                "  ERROR:",
                redact_api_key(exc),
            )

    if events_failed:
        print(
            f"  {events_failed} event(s) "
            "failed"
        )

    return all_rows


def fetch_historical_props(
    start_date,
    end_date,
    markets=None,
    force=False,
    dry_run=False,
):
    if not dry_run:
        require_live_fetch("historical Odds API props")

    if not ODDS_API_KEY:
        raise RuntimeError(
            "ODDS_API_KEY is missing "
            "from .env"
        )

    if markets is None:
        markets = PROP_MARKETS

    print()
    print("=" * 60)
    print("DOWNLOADING HISTORICAL MLB PROPS")
    print("=" * 60)
    print(
        f"Range: {start_date} → {end_date}"
    )
    print(
        f"Markets: {len(markets)}"
    )

    dates = list(
        date_range(
            start_date,
            end_date,
        )
    )

    if dry_run:
        print()
        print(
            "DRY RUN — would fetch "
            f"{len(dates)} date(s):"
        )
        for snapshot_date in dates:
            output_file = (
                historical_odds_path(
                    snapshot_date
                )
            )
            exists = output_file.exists()
            print(
                f"  {snapshot_date} → "
                f"{output_file} "
                f"({'skip' if exists and not force else 'fetch'})"
            )
        return

    quota_exhausted = False
    dates_fetched = 0
    dates_skipped = 0

    for snapshot_date in dates:
        output_file = historical_odds_path(
            snapshot_date
        )

        if (
            output_file.exists()
            and not force
        ):
            print()
            print(
                f"Skipping {snapshot_date} "
                "(already fetched; use "
                "--force to refetch)"
            )
            dates_skipped += 1
            continue

        print()
        print(
            f"Fetching {snapshot_date}..."
        )

        try:
            rows = fetch_date_props(
                snapshot_date,
                markets,
            )
        except OddsApiQuotaError as exc:
            quota_exhausted = True
            print()
            print(f"ERROR: {exc}")
            print(
                "Stopping early — Odds API "
                "quota exhausted."
            )
            break

        if len(rows) == 0:
            print(
                f"  No prop rows for "
                f"{snapshot_date}"
            )

            if (
                output_file.exists()
                and not force
            ):
                print(
                    "  Keeping existing file "
                    "(not overwriting with "
                    "empty data)"
                )
                continue

            # Do not write empty parquet over good data.
            if output_file.exists():
                print(
                    "  WARNING: Not "
                    "overwriting existing "
                    "file with empty data"
                )
                continue

        df = pd.DataFrame(rows)

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        df.to_parquet(
            output_file,
            index=False,
        )

        dates_fetched += 1

        print(
            f"  Saved {len(df):,} rows → "
            f"{output_file}"
        )

    print()
    print(
        f"Done. Fetched {dates_fetched} "
        f"date(s), skipped "
        f"{dates_skipped}."
    )

    if quota_exhausted:
        sys.exit(1)


def fetch_date_game_lines(
    snapshot_date,
    markets,
    sleep_seconds=0.15,
):
    return fetch_date_props(
        snapshot_date,
        markets,
        sleep_seconds=sleep_seconds,
    )


def fetch_historical_game_lines(
    start_date,
    end_date,
    markets=None,
    force=False,
    dry_run=False,
):
    if not dry_run:
        require_live_fetch("historical Odds API game lines")

    if not ODDS_API_KEY:
        raise RuntimeError(
            "ODDS_API_KEY is missing "
            "from .env"
        )

    if markets is None:
        markets = GAME_MARKETS

    print()
    print("=" * 60)
    print("DOWNLOADING HISTORICAL MLB GAME LINES")
    print("=" * 60)
    print(
        f"Range: {start_date} → {end_date}"
    )
    print(
        f"Markets: {len(markets)}"
    )

    dates = list(
        date_range(
            start_date,
            end_date,
        )
    )

    if dry_run:
        print()
        print(
            "DRY RUN — would fetch "
            f"{len(dates)} date(s):"
        )
        for snapshot_date in dates:
            output_file = (
                historical_game_lines_path(
                    snapshot_date
                )
            )
            exists = output_file.exists()
            print(
                f"  {snapshot_date} → "
                f"{output_file} "
                f"({'skip' if exists and not force else 'fetch'})"
            )
        return

    quota_exhausted = False
    dates_fetched = 0
    dates_skipped = 0

    for snapshot_date in dates:
        output_file = historical_game_lines_path(
            snapshot_date
        )

        if (
            output_file.exists()
            and not force
        ):
            print()
            print(
                f"Skipping {snapshot_date} "
                "(already fetched; use "
                "--force to refetch)"
            )
            dates_skipped += 1
            continue

        print()
        print(
            f"Fetching {snapshot_date}..."
        )

        try:
            rows = fetch_date_game_lines(
                snapshot_date,
                markets,
            )
        except OddsApiQuotaError as exc:
            quota_exhausted = True
            print()
            print(f"ERROR: {exc}")
            print(
                "Stopping early — Odds API "
                "quota exhausted."
            )
            break

        if len(rows) == 0:
            print(
                f"  No game line rows for "
                f"{snapshot_date}"
            )

            if (
                output_file.exists()
                and not force
            ):
                print(
                    "  Keeping existing file "
                    "(not overwriting with "
                    "empty data)"
                )
                continue

            if output_file.exists():
                print(
                    "  WARNING: Not "
                    "overwriting existing "
                    "file with empty data"
                )
                continue

        df = pd.DataFrame(rows)

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        df.to_parquet(
            output_file,
            index=False,
        )

        dates_fetched += 1

        print(
            f"  Saved {len(df):,} rows → "
            f"{output_file}"
        )

    print()
    print(
        f"Done. Fetched {dates_fetched} "
        f"date(s), skipped "
        f"{dates_skipped}."
    )

    if quota_exhausted:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fetch historical MLB player "
            "prop odds from The Odds API"
        ),
    )

    parser.add_argument(
        "--start",
        required=True,
        help="Start date (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--end",
        required=True,
        help="End date (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--markets",
        default=None,
        help=(
            "Comma-separated market keys "
            f"(default: all {len(PROP_MARKETS)} "
            "PROP_MARKETS)"
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Refetch dates that already "
            "have parquet files"
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print planned fetches without "
            "calling the API"
        ),
    )

    parser.add_argument(
        "--game-lines",
        action="store_true",
        help=(
            "Fetch game totals/spreads instead "
            "of player props"
        ),
    )

    args = parser.parse_args()

    markets = None

    if args.markets:
        markets = [
            m.strip()
            for m in args.markets.split(",")
            if m.strip()
        ]

    if args.game_lines:
        fetch_historical_game_lines(
            args.start,
            args.end,
            markets=markets,
            force=args.force,
            dry_run=args.dry_run,
        )
        return

    fetch_historical_props(
        args.start,
        args.end,
        markets=markets,
        force=args.force,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
