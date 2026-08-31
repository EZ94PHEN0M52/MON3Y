#!/usr/bin/env python3
"""Ensure feature parquets exist and contain columns required by train.py.

Rebuild triggers (--fix):
  - missing parquet file or required column
  - parquet fingerprint mismatch (column list / PARQUET_FEATURE_SCHEMA_VERSION)
  - build_features.py or features_v2.py newer than the parquet

Does NOT rebuild for train.py / training_odds.py edits or derived model inputs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from train import (  # noqa: E402
    PARQUET_FEATURE_SCHEMA_VERSION,
    feature_columns_for_version,
    parquet_schema_is_stale,
)
from utils import (  # noqa: E402
    batter_features_path,
    features_caught_up_to_statcast,
    feature_parquet_needs_refresh,
    live_fetch_disabled,
    normalize_version,
    pitcher_features_path,
    require_live_fetch,
    statcast_needs_refresh,
    statcast_raw_path,
)

# Only files that change build_features.py output. train.py / training_odds.py
# affect training/inference logic and derived columns, not parquet contents.
SOURCE_FILES = {
    "build_features.py": ROOT / "build_features.py",
    "game_lines.py": ROOT / "game_lines.py",
    "pitcher_stuff.py": ROOT / "pitcher_stuff.py",
}


@dataclass(frozen=True)
class FeatureIssue:
    role: str
    path: Path
    missing_file: bool
    missing_columns: tuple[str, ...]
    stale_schema: bool = False
    stale_sources: tuple[str, ...] = ()
    stale_data: bool = False



def source_files_for_version(version: str) -> dict[str, Path]:
    files = dict(SOURCE_FILES)
    if normalize_version(version) == "v2":
        files["features_v2.py"] = ROOT / "features_v2.py"
    return files


def missing_columns(path: Path, required_columns: list[str]) -> list[str]:
    schema = pq.read_schema(path)
    actual = set(schema.names)
    return [col for col in required_columns if col not in actual]


def parquet_fingerprint(path: Path) -> str | None:
    metadata = pq.read_metadata(path).metadata
    if metadata is None:
        return None
    value = metadata.get(b"feature_schema_fingerprint")
    if value is None:
        return None
    return value.decode()


def stale_by_source_mtime(path: Path, version: str) -> tuple[str, ...]:
    if not path.exists():
        return ()

    parquet_mtime = path.stat().st_mtime
    stale_sources: list[str] = []

    for name, source_path in source_files_for_version(version).items():
        if (
            source_path.exists()
            and source_path.stat().st_mtime > parquet_mtime
        ):
            stale_sources.append(
                f"{name} newer than parquet"
            )

    return tuple(stale_sources)


def summarize_missing_columns(
    missing: tuple[str, ...],
) -> str:
    if not missing:
        return ""

    summaries: set[str] = set()

    for col in missing:
        matched = False
        for suffix in (
            "_l20",
            "_l10",
            "_l5",
            "_l3",
            "_season",
        ):
            if col.endswith(suffix):
                summaries.add(
                    f"{col[: -len(suffix)]}_l*"
                )
                matched = True
                break

        if not matched:
            summaries.add(col)

    return ", ".join(sorted(summaries))


def check_range(
    start_date: str,
    end_date: str,
    version: str,
) -> list[FeatureIssue]:
    version = normalize_version(version)
    feature_sets = feature_columns_for_version(version)
    issues: list[FeatureIssue] = []

    for role, columns in (
        ("batter", feature_sets["batter"]),
        ("pitcher", feature_sets["pitcher"]),
    ):
        path_fn = (
            batter_features_path
            if role == "batter"
            else pitcher_features_path
        )
        path = path_fn(start_date, end_date, version)

        if not path.exists():
            issues.append(
                FeatureIssue(
                    role=role,
                    path=path,
                    missing_file=True,
                    missing_columns=tuple(columns),
                )
            )
            continue

        cols_missing = missing_columns(path, columns)
        stored_fingerprint = parquet_fingerprint(path)
        stale_schema = parquet_schema_is_stale(
            stored_fingerprint,
            version,
        )
        stale_sources = stale_by_source_mtime(path, version)
        stale_data = feature_parquet_needs_refresh(
            path,
            start_date,
            end_date,
        )

        if cols_missing or stale_schema or stale_sources or stale_data:
            issues.append(
                FeatureIssue(
                    role=role,
                    path=path,
                    missing_file=False,
                    missing_columns=tuple(cols_missing),
                    stale_schema=stale_schema,
                    stale_sources=stale_sources,
                    stale_data=stale_data,
                )
            )

    return issues


def _issues_are_stale_data_only(issues: list[FeatureIssue]) -> bool:
    if not issues:
        return False
    return all(
        not issue.missing_file
        and not issue.missing_columns
        and not issue.stale_schema
        and not issue.stale_sources
        and issue.stale_data
        for issue in issues
    )


def describe_issue(issue: FeatureIssue) -> str:
    details: list[str] = []

    if issue.missing_file:
        details.append(f"missing file {issue.path}")
    elif issue.missing_columns:
        summary = summarize_missing_columns(
            issue.missing_columns
        )
        details.append(
            f"missing columns [{summary}] "
            f"({list(issue.missing_columns)})"
        )

    if issue.stale_schema:
        details.append(
            "parquet schema fingerprint mismatch "
            f"(expected {PARQUET_FEATURE_SCHEMA_VERSION})"
        )

    if issue.stale_sources:
        details.append(
            ", ".join(issue.stale_sources)
        )

    if issue.stale_data:
        details.append(
            "feature data stops before requested end date "
            "(Statcast may be incomplete — re-fetch and rebuild)"
        )

    return f"{issue.role}: " + "; ".join(details)


def print_issues(
    start_date: str,
    end_date: str,
    version: str,
    issues: list[FeatureIssue],
) -> None:
    print(
        f"Feature check FAILED for {start_date} → {end_date} ({version}):"
    )
    for issue in issues:
        print(f"  - {describe_issue(issue)}")


def print_rebuild_notice(
    issues: list[FeatureIssue],
) -> None:
    if any(issue.missing_file for issue in issues):
        print("Feature parquets missing, rebuilding...")
        return

    missing_summaries: set[str] = set()
    stale_schema = False
    stale_data = False
    stale_sources: set[str] = set()

    for issue in issues:
        if issue.missing_columns:
            for part in summarize_missing_columns(
                issue.missing_columns
            ).split(", "):
                if part:
                    missing_summaries.add(part)

        if issue.stale_schema:
            stale_schema = True

        if issue.stale_data:
            stale_data = True

        stale_sources.update(issue.stale_sources)

    if missing_summaries:
        joined = ", ".join(sorted(missing_summaries))
        print(
            f"New model features detected ({joined}), rebuilding..."
        )
    elif stale_data:
        print(
            "Feature data is behind the requested end date "
            "(Statcast incomplete or early-morning fetch), rebuilding..."
        )
    elif stale_schema:
        print(
            "Parquet feature columns changed since last build "
            f"(schema {PARQUET_FEATURE_SCHEMA_VERSION}), rebuilding..."
        )
    elif stale_sources:
        joined = ", ".join(sorted(stale_sources))
        print(
            f"Feature source code changed ({joined}), rebuilding..."
        )
    else:
        print("Rebuilding feature parquets...")


def remove_stale_parquets(
    issues: list[FeatureIssue],
) -> None:
    for issue in issues:
        if issue.path.exists():
            print(
                f">>> Removing stale {issue.role} parquet: "
                f"{issue.path}"
            )
            issue.path.unlink()


def run_cmd(args: list[str]) -> None:
    subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        check=True,
    )


def fix_range(
    start_date: str,
    end_date: str,
    version: str,
    issues: list[FeatureIssue],
) -> None:
    print_rebuild_notice(issues)
    remove_stale_parquets(issues)

    raw_path = statcast_raw_path(start_date, end_date)

    if statcast_needs_refresh(start_date, end_date):
        if live_fetch_disabled():
            require_live_fetch(
                f"Statcast refresh for {start_date} → {end_date}"
            )

        if raw_path.exists():
            print(
                f">>> Statcast raw stale (missing games through "
                f"{end_date}), re-fetching: {raw_path}"
            )
            raw_path.unlink()

        print(
            f">>> Fetching Statcast ({start_date} → {end_date})..."
        )
        run_cmd(
            [
                "fetch_data.py",
                "--statcast",
                "--start",
                start_date,
                "--end",
                end_date,
            ]
        )
    else:
        print(f">>> Statcast raw already current: {raw_path}")

    print(
        f">>> Building {version} features "
        f"({start_date} → {end_date})..."
    )
    run_cmd(
        [
            "build_features.py",
            "--start",
            start_date,
            "--end",
            end_date,
            "--version",
            version,
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify batter/pitcher feature parquets exist and include "
            "all columns required for training/inference. Required "
            "columns are read from train.feature_columns_for_version()."
        )
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
        "--version",
        default="v2",
        choices=["v1", "v2"],
        help="Feature set version (default: v2)",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help=(
            "Fetch missing Statcast raw data, remove stale parquets, "
            "and rebuild features when files are missing, columns "
            "drift from train.py, or source files are newer"
        ),
    )
    args = parser.parse_args()

    version = normalize_version(args.version)
    issues = check_range(args.start, args.end, version)

    if not issues:
        print(
            f"Feature check OK: {args.start} → {args.end} ({version})"
        )
        return 0

    print_issues(args.start, args.end, version, issues)

    if not args.fix:
        print(
            "Re-run with --fix to fetch Statcast and rebuild features."
        )
        return 1

    fix_range(
        args.start,
        args.end,
        version,
        issues,
    )

    remaining = check_range(args.start, args.end, version)
    if remaining:
        caught_up, statcast_max, required = features_caught_up_to_statcast(
            args.start,
            args.end,
            version,
        )
        if _issues_are_stale_data_only(remaining) and caught_up:
            print(
                "Feature check OK with Statcast posting lag: "
                f"latest game_date {statcast_max} "
                f"(MLB schedule through {required} not on Savant yet). "
                "Re-run ./run_daily.sh later when Game logs through advances."
            )
            return 0

        print_issues(args.start, args.end, version, remaining)
        print("Feature rebuild did not resolve all issues.")
        return 1

    print(
        f"Feature check OK after rebuild: "
        f"{args.start} → {args.end} ({version})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
