"""Fetch nflverse stats/injuries/schedules and Odds API props."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone

import pandas as pd

from odds_api import (
    OddsApiQuotaError,
    get_event_props,
    get_events,
    normalize_event,
    redact_api_key,
)
from utils import (
    current_injuries_path,
    current_props_path,
    injuries_raw_path,
    is_inactive_status,
    is_questionable_status,
    normalize_injury_status,
    player_stats_raw_path,
    require_live_fetch,
    rosters_weekly_raw_path,
    schedules_raw_path,
)
from features_v2 import snap_counts_raw_path, team_stats_raw_path


def _polars_to_pandas(frame) -> pd.DataFrame:
    return frame.to_pandas()


def resolve_season(season: int | None) -> int:
    if season is not None:
        return int(season)
    import nflreadpy as nfl

    return int(nfl.get_current_season())


def resolve_week(season: int, week: int | None) -> int:
    if week is not None:
        return int(week)
    import nflreadpy as nfl

    current_season = int(nfl.get_current_season())
    if season == current_season:
        return int(nfl.get_current_week())

    # Historical season: use max week present in schedule.
    sched = nfl.load_schedules([season])
    pdf = _polars_to_pandas(sched)
    return int(pdf.loc[pdf["season"] == season, "week"].max())


# =========================================================
# NFLVERSE — STATS / SCHEDULE / INJURIES / ROSTERS
# =========================================================

def fetch_player_stats(season: int, force: bool = False) -> pd.DataFrame:
    require_live_fetch("nflverse player stats")
    import nflreadpy as nfl

    output = player_stats_raw_path(season)
    print()
    print("=" * 60)
    print(f"DOWNLOADING PLAYER STATS ({season})")
    print("=" * 60)

    if output.exists() and not force:
        print(f"Already exists: {output}")
        return pd.read_parquet(output)

    frame = nfl.load_player_stats(season, summary_level="week")
    df = _polars_to_pandas(frame)
    df.to_parquet(output, index=False)
    print(f"Saved {len(df):,} rows → {output}")
    return df


def fetch_schedules(season: int, force: bool = False) -> pd.DataFrame:
    require_live_fetch("nflverse schedules")
    import nflreadpy as nfl

    output = schedules_raw_path(season)
    print()
    print("=" * 60)
    print(f"DOWNLOADING SCHEDULES ({season})")
    print("=" * 60)

    if output.exists() and not force:
        print(f"Already exists: {output}")
        return pd.read_parquet(output)

    frame = nfl.load_schedules([season])
    df = _polars_to_pandas(frame)
    df.to_parquet(output, index=False)
    print(f"Saved {len(df):,} rows → {output}")
    return df


def fetch_injuries_raw(season: int, force: bool = False) -> pd.DataFrame:
    require_live_fetch("nflverse injuries")
    import nflreadpy as nfl

    output = injuries_raw_path(season)
    print()
    print("=" * 60)
    print(f"DOWNLOADING INJURIES ({season})")
    print("=" * 60)

    if output.exists() and not force:
        print(f"Already exists: {output}")
        return pd.read_parquet(output)

    frame = nfl.load_injuries(season)
    df = _polars_to_pandas(frame)
    df.to_parquet(output, index=False)
    print(f"Saved {len(df):,} rows → {output}")
    return df


def fetch_rosters_weekly(season: int, force: bool = False) -> pd.DataFrame:
    require_live_fetch("nflverse weekly rosters")
    import nflreadpy as nfl

    output = rosters_weekly_raw_path(season)
    print()
    print("=" * 60)
    print(f"DOWNLOADING WEEKLY ROSTERS ({season})")
    print("=" * 60)

    if output.exists() and not force:
        print(f"Already exists: {output}")
        return pd.read_parquet(output)

    frame = nfl.load_rosters_weekly(season)
    df = _polars_to_pandas(frame)
    df.to_parquet(output, index=False)
    print(f"Saved {len(df):,} rows → {output}")
    return df


def fetch_snap_counts(season: int, force: bool = False) -> pd.DataFrame:
    require_live_fetch("nflverse snap counts")
    import nflreadpy as nfl

    output = snap_counts_raw_path(season)
    print()
    print("=" * 60)
    print(f"DOWNLOADING SNAP COUNTS ({season})")
    print("=" * 60)

    if output.exists() and not force:
        print(f"Already exists: {output}")
        return pd.read_parquet(output)

    frame = nfl.load_snap_counts(season)
    df = _polars_to_pandas(frame)
    df.to_parquet(output, index=False)
    print(f"Saved {len(df):,} rows → {output}")
    return df


def fetch_team_stats(season: int, force: bool = False) -> pd.DataFrame:
    require_live_fetch("nflverse team stats")
    import nflreadpy as nfl

    output = team_stats_raw_path(season)
    print()
    print("=" * 60)
    print(f"DOWNLOADING TEAM STATS ({season})")
    print("=" * 60)

    if output.exists() and not force:
        print(f"Already exists: {output}")
        return pd.read_parquet(output)

    frame = nfl.load_team_stats(season, summary_level="week")
    df = _polars_to_pandas(frame)
    df.to_parquet(output, index=False)
    print(f"Saved {len(df):,} rows → {output}")
    return df


def build_current_injuries(
    season: int,
    week: int,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """
    Build board-ready injury snapshot for a season/week.

    Joins nflverse injury reports with weekly roster status / depth chart.
    Always re-downloads injury raw data (status moves mid-week). Rosters
    re-download when --force or the cache is missing.
    """
    injuries = fetch_injuries_raw(season, force=True)
    rosters = fetch_rosters_weekly(season, force=force)

    inj = injuries.copy()
    if inj.empty:
        print("WARNING: No injury rows for season", season)
        out = pd.DataFrame()
        out.to_parquet(current_injuries_path(), index=False)
        return out

    inj = inj[inj["week"].astype(int) == int(week)].copy()
    if inj.empty:
        # Fall back to latest week available ≤ target.
        available = injuries["week"].dropna().astype(int)
        if available.empty:
            print("WARNING: No injury weeks available.")
            out = pd.DataFrame()
            out.to_parquet(current_injuries_path(), index=False)
            return out
        fallback_week = int(available[available <= int(week)].max())
        print(
            f"WARNING: No injuries for week {week}; "
            f"using week {fallback_week}."
        )
        inj = injuries[injuries["week"].astype(int) == fallback_week].copy()
        week = fallback_week

    inj["report_status_norm"] = inj["report_status"].map(
        normalize_injury_status
    )
    inj["is_inactive"] = inj["report_status"].map(is_inactive_status)
    inj["is_questionable"] = inj["report_status"].map(is_questionable_status)

    roster = rosters.copy()
    if not roster.empty and "week" in roster.columns:
        roster = roster[roster["week"].astype(int) == int(week)].copy()

    roster_cols = [
        c
        for c in [
            "gsis_id",
            "full_name",
            "team",
            "position",
            "depth_chart_position",
            "status",
            "status_description_abbr",
            "week",
        ]
        if c in roster.columns
    ]
    roster_slim = (
        roster[roster_cols].drop_duplicates(subset=["gsis_id"])
        if "gsis_id" in roster_cols and not roster.empty
        else pd.DataFrame()
    )

    if not roster_slim.empty:
        merged = inj.merge(
            roster_slim,
            on="gsis_id",
            how="left",
            suffixes=("", "_roster"),
        )
        # Prefer injury-table name/team/position; fill from roster.
        for col in ("full_name", "team", "position"):
            roster_col = f"{col}_roster"
            if roster_col in merged.columns:
                merged[col] = merged[col].fillna(merged[roster_col])
                merged = merged.drop(columns=[roster_col])
    else:
        merged = inj

    merged["season"] = season
    merged["board_week"] = week
    merged["fetched_at"] = datetime.now(timezone.utc).isoformat()

    # One row per player for the board week.
    if "gsis_id" in merged.columns:
        merged = (
            merged.sort_values(
                ["gsis_id", "report_status"],
                na_position="last",
            )
            .drop_duplicates(subset=["gsis_id"], keep="last")
            .reset_index(drop=True)
        )

    output = current_injuries_path()
    merged.to_parquet(output, index=False)

    inactive_n = int(merged["is_inactive"].fillna(False).sum())
    q_n = int(merged["is_questionable"].fillna(False).sum())
    print(
        f"Saved current injuries ({len(merged):,} players, "
        f"{inactive_n} inactive, {q_n} questionable) → {output}"
    )
    return merged


# =========================================================
# ODDS API — CURRENT PROPS
# =========================================================

def _exit_props_fetch_failure(message: str, output_file=None) -> None:
    print()
    print(message)
    if output_file is not None and output_file.exists():
        try:
            cached = pd.read_parquet(output_file)
            if len(cached) > 0:
                print()
                print(
                    "WARNING: Keeping existing cached props "
                    f"({len(cached):,} rows) at:"
                )
                print(output_file)
        except Exception:
            pass
    sys.exit(1)


def fetch_current_props() -> pd.DataFrame:
    require_live_fetch("live NFL sportsbook props (Odds API)")

    print()
    print("=" * 60)
    print("DOWNLOADING CURRENT NFL PROPS")
    print("=" * 60)

    output_file = current_props_path()

    try:
        events = get_events()
    except OddsApiQuotaError as exc:
        _exit_props_fetch_failure(
            f"ERROR: {exc}\nCould not list NFL events.",
            output_file,
        )

    print(f"Found {len(events)} NFL events.")

    all_rows = []
    events_failed = 0
    quota_exhausted = False

    for index, event in enumerate(events, start=1):
        print(
            f"[{index}/{len(events)}] "
            f"{event.get('away_team')} @ {event.get('home_team')}"
        )
        try:
            event_data = get_event_props(event["id"])
            all_rows.extend(normalize_event(event_data))
            time.sleep(0.15)
        except OddsApiQuotaError as exc:
            quota_exhausted = True
            print("ERROR:", exc)
            print("Stopping early — Odds API quota exhausted.")
            break
        except Exception as exc:
            events_failed += 1
            print("ERROR:", redact_api_key(exc))

    if len(all_rows) == 0:
        if output_file.exists():
            try:
                cached = pd.read_parquet(output_file)
                if len(cached) > 0:
                    print()
                    print(
                        "WARNING: Collected 0 prop rows. "
                        "NOT overwriting existing cache."
                    )
                    print(f"Cached props ({len(cached):,} rows): {output_file}")
                    sys.exit(1)
            except Exception:
                pass

        reason = (
            "Odds API quota exhausted."
            if quota_exhausted
            else "No prop rows collected."
        )
        _exit_props_fetch_failure(
            f"ERROR: {reason}\nNo cached current_props.parquet available.",
            output_file=None,
        )

    df = pd.DataFrame(all_rows)
    if "fetched_at" not in df.columns:
        df["fetched_at"] = pd.NaT
    df["fetched_at"] = df["fetched_at"].fillna(
        pd.Timestamp.now(tz="UTC").isoformat()
    )
    df.to_parquet(output_file, index=False)
    print(f"Saved {len(df):,} prop rows → {output_file}")
    return df


# =========================================================
# CLI
# =========================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch NFL stats, schedules, injuries, and/or Odds API props."
        )
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Download nflverse weekly player stats for the season.",
    )
    parser.add_argument(
        "--schedule",
        action="store_true",
        help="Download nflverse schedule for the season.",
    )
    parser.add_argument(
        "--injuries",
        action="store_true",
        help=(
            "Refresh injury + weekly roster data and write "
            "data/processed/current_injuries.parquet for board gates."
        ),
    )
    parser.add_argument(
        "--props",
        action="store_true",
        help="Fetch live Odds API player props for the current slate.",
    )
    parser.add_argument(
        "--sleeper-props",
        action="store_true",
        help=(
            "Fetch Sleeper Picks NFL props via Apify "
            "(requires APIFY_TOKEN). Also runs after --props when token is set."
        ),
    )
    parser.add_argument(
        "--snaps",
        action="store_true",
        help="Download nflverse snap counts (needed for V2 offense_snap_pct).",
    )
    parser.add_argument(
        "--team-stats",
        action="store_true",
        help="Download nflverse weekly team stats (needed for V2 opponent defense).",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="NFL season year (default: nflverse current season).",
    )
    parser.add_argument(
        "--week",
        type=int,
        default=None,
        help="NFL week for injury snapshot (default: current week).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even when a raw parquet cache exists.",
    )
    args = parser.parse_args()

    if not (
        args.stats
        or args.schedule
        or args.injuries
        or args.props
        or args.sleeper_props
        or args.snaps
        or args.team_stats
    ):
        parser.error(
            "Pass at least one of --stats, --schedule, --injuries, "
            "--props, --sleeper-props, --snaps, --team-stats"
        )

    season = resolve_season(args.season)

    if args.stats:
        fetch_player_stats(season, force=args.force)

    if args.schedule:
        fetch_schedules(season, force=args.force)

    if args.snaps:
        fetch_snap_counts(season, force=args.force)

    if args.team_stats:
        fetch_team_stats(season, force=args.force)

    if args.injuries:
        week = resolve_week(season, args.week)
        print(f"Injury snapshot target: season={season} week={week}")
        build_current_injuries(season, week, force=args.force)

    if args.props:
        fetch_current_props()
        # Best-effort Sleeper refresh when token is configured.
        try:
            from fetch_sleeper_props import (
                _apify_token,
                fetch_and_save_sleeper_props,
            )

            if _apify_token():
                fetch_and_save_sleeper_props()
            else:
                print(
                    "Skipping Sleeper props (APIFY_TOKEN not set). "
                    "Add it to .env or run: python fetch_data.py --sleeper-props"
                )
        except Exception as exc:
            print(f"WARNING: Sleeper props fetch failed: {exc}")

    if args.sleeper_props and not args.props:
        from fetch_sleeper_props import fetch_and_save_sleeper_props

        fetch_and_save_sleeper_props()


if __name__ == "__main__":
    main()
