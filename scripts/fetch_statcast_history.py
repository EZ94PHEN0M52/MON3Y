#!/usr/bin/env python3
"""
Download optional Statcast history shards for career H2H / wOBA lookups.

Merged Statcast (`batter_score_data._load_merged_statcast`) automatically
concatenates every ``data/raw/statcast_*.parquet`` file. Daily ``run_daily.sh``
only maintains the current-season window; run this script once (or after adding
a new season) to backfill prior years for Hitter's Life and Batter Score.

Usage:
  python scripts/fetch_statcast_history.py
  python scripts/fetch_statcast_history.py --season 2024
  python scripts/fetch_statcast_history.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fetch_data import fetch_statcast
from utils import statcast_needs_refresh, statcast_raw_path

# Regular season + late September; enough for career wOBA / H2H vs SP.
HISTORICAL_STATCAST_SHARDS: dict[str, tuple[str, str]] = {
    "2024": ("2024-03-28", "2024-09-29"),
    "2025": ("2025-04-01", "2025-10-15"),
}


def fetch_shard(
    start_date: str,
    end_date: str,
    *,
    force: bool = False,
    dry_run: bool = False,
) -> bool:
    path = statcast_raw_path(start_date, end_date)
    needs = force or statcast_needs_refresh(start_date, end_date)

    if path.exists() and not needs:
        print(f"OK (cached): {path.name}")
        return False

    if dry_run:
        action = "would fetch" if not path.exists() else "would refresh"
        print(f"{action}: {path.name} ({start_date} → {end_date})")
        return True

    print(f"Fetching {path.name} ({start_date} → {end_date})...")
    fetch_statcast(start_date, end_date, force=force)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill Statcast parquets for career H2H / wOBA depth.",
    )
    parser.add_argument(
        "--season",
        action="append",
        choices=sorted(HISTORICAL_STATCAST_SHARDS),
        help="Fetch one season (default: all listed historical shards)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even when cached parquet looks current",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned downloads without calling pybaseball",
    )
    args = parser.parse_args()

    seasons = args.season or sorted(HISTORICAL_STATCAST_SHARDS)
    fetched = 0

    for season in seasons:
        start_date, end_date = HISTORICAL_STATCAST_SHARDS[season]
        if fetch_shard(
            start_date,
            end_date,
            force=args.force,
            dry_run=args.dry_run,
        ):
            fetched += 1

    if args.dry_run:
        print(f"\nDry run: {fetched} shard(s) would be fetched/refreshed.")
    else:
        print(
            f"\nDone. {fetched} shard(s) downloaded/refreshed. "
            "Restart Streamlit to pick up merged Statcast for H2H / wOBA."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
