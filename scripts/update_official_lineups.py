#!/usr/bin/env python3
"""
Fetch Rotowire **Today's Lineup** for slate teams and merge into cache.

Run close to first pitch via ``run_official_lineups.sh``. Boards prefer OFFICIAL
rows in ``data/processed/rotowire_lineups.parquet`` when the lineup filter is on.
"""

from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

from fetch_rotowire_lineups import (
    DEFAULT_MAX_OFFICIAL_PLAYERS,
    DEFAULT_MIN_OFFICIAL_PLAYERS,
    DEFAULT_MIN_SLATE_OVERLAP,
    _normalize_player_key,
    slate_team_abbrs_from_props,
    update_official_lineups,
)
from hitters_life_data import _lookup_batter_team_abbr
from utils import CURRENT_PROPS_PATH


def build_slate_players_by_team(version: str = "v2") -> dict[str, set[str]]:
    """Map team abbr → normalized prop-slate players using latest feature team."""
    if not CURRENT_PROPS_PATH.exists():
        return {}

    props = pd.read_parquet(CURRENT_PROPS_PATH)
    if props.empty or "player" not in props.columns:
        return {}

    grouped: dict[str, set[str]] = {}
    for player in props["player"].dropna().astype(str).unique():
        abbr = _lookup_batter_team_abbr(player, version=version)
        if not abbr:
            continue
        grouped.setdefault(abbr.upper(), set()).add(_normalize_player_key(player))

    return grouped


def _parse_team_list(raw: str) -> list[str]:
    return [
        part.strip().upper()
        for part in raw.split(",")
        if part.strip()
    ]


def _print_results(results) -> int:
    updated = unchanged = skipped = failed = 0

    for item in results:
        status = item.status
        if status == "updated":
            updated += 1
        elif status == "unchanged":
            unchanged += 1
        elif status == "skipped":
            skipped += 1
        else:
            failed += 1

        players = ", ".join(item.players[:3])
        if len(item.players) > 3:
            players = f"{players}, …"

        suffix = f" — {players}" if players else ""
        print(f"  [{item.team_abbr}] {status}: {item.message}{suffix}")

    print()
    print(
        f"Summary: {updated} updated, {unchanged} unchanged, "
        f"{skipped} skipped, {failed} failed"
    )

    if updated > 0:
        print("Saved to data/processed/rotowire_lineups.parquet")
        print("Refresh Streamlit (or reload the page) to pick up official lineups.")

    if failed and not updated and not unchanged:
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch Rotowire Today's Lineup for slate teams and update cache."
        ),
    )
    parser.add_argument(
        "--teams",
        default="",
        help="Comma-separated Rotowire team codes (default: all slate teams)",
    )
    parser.add_argument(
        "--version",
        default="v2",
        help="Feature version for slate player → team lookup (default: v2)",
    )
    parser.add_argument(
        "--min-players",
        type=int,
        default=DEFAULT_MIN_OFFICIAL_PLAYERS,
        help=f"Minimum batters in official lineup (default: {DEFAULT_MIN_OFFICIAL_PLAYERS})",
    )
    parser.add_argument(
        "--max-players",
        type=int,
        default=DEFAULT_MAX_OFFICIAL_PLAYERS,
        help=f"Maximum batters in official lineup (default: {DEFAULT_MAX_OFFICIAL_PLAYERS})",
    )
    parser.add_argument(
        "--min-slate-overlap",
        type=int,
        default=DEFAULT_MIN_SLATE_OVERLAP,
        help=(
            "Minimum lineup batters that must match today's prop slate "
            f"(default: {DEFAULT_MIN_SLATE_OVERLAP}; 0 disables)"
        ),
    )
    parser.add_argument(
        "--skip-slate-check",
        action="store_true",
        help="Do not cross-check lineup batters against prop slate",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and validate only; do not write parquet",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip timestamped backup before writing",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=1,
        help="Retry count per team on fetch failure (default: 1)",
    )
    parser.add_argument(
        "--retry-wait",
        type=float,
        default=5.0,
        help="Seconds between retries (default: 5)",
    )

    args = parser.parse_args(argv)

    if args.teams.strip():
        team_abbrs = _parse_team_list(args.teams)
    else:
        team_abbrs = slate_team_abbrs_from_props()
        if not team_abbrs:
            print(
                "No slate teams found. Run the daily pipeline first "
                "(./run_daily.sh) or pass --teams NYY,BOS,…",
                file=sys.stderr,
            )
            return 1

    slate_players = None
    if not args.skip_slate_check and args.min_slate_overlap > 0:
        slate_players = build_slate_players_by_team(version=args.version)

    print(f"Checking official lineups for {len(team_abbrs)} team(s): "
          f"{', '.join(team_abbrs)}")
    if args.dry_run:
        print("(dry run — cache will not be modified)")

    all_results = []
    pending = list(team_abbrs)

    for attempt in range(1, max(args.retries, 1) + 1):
        if attempt > 1:
            print(f"Retry attempt {attempt}/{args.retries} for "
                  f"{len(pending)} team(s)…")
            time.sleep(args.retry_wait)

        results = update_official_lineups(
            pending,
            min_players=args.min_players,
            max_players=args.max_players,
            min_slate_overlap=args.min_slate_overlap,
            slate_players_by_team=slate_players,
            dry_run=args.dry_run,
            backup=not args.no_backup,
        )

        if attempt == 1:
            all_results = results
        else:
            by_team = {item.team_abbr: item for item in all_results}
            for item in results:
                prev = by_team.get(item.team_abbr)
                if prev and prev.status == "failed" and item.status != "failed":
                    by_team[item.team_abbr] = item
            all_results = list(by_team.values())

        pending = [
            item.team_abbr
            for item in all_results
            if item.status == "failed"
        ]
        if not pending:
            break

    return _print_results(all_results)


if __name__ == "__main__":
    raise SystemExit(main())
