#!/usr/bin/env python3
"""Dry-run tests for ensure_features stale/missing-column detection."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.ensure_features import (  # noqa: E402
    check_range,
    summarize_missing_columns,
)
from train import feature_columns_for_version  # noqa: E402
from utils import batter_features_path, pitcher_features_path  # noqa: E402


def write_minimal_parquet(
    path: Path,
    columns: list[str],
    fingerprint: str | None = None,
) -> None:
    data = {col: [1.0] for col in columns}
    table = pa.Table.from_pydict(data)

    if fingerprint is not None:
        metadata = table.schema.metadata or {}
        metadata[b"feature_schema_fingerprint"] = (
            fingerprint.encode()
        )
        table = table.replace_schema_metadata(metadata)

    pq.write_table(table, path)


def test_missing_walks_columns_triggers_issue() -> None:
    start = "2099-01-01"
    end = "2099-01-02"
    version = "v2"

    required = feature_columns_for_version(version)
    batter_cols = [
        col
        for col in required["batter"]
        if not col.startswith("walks_")
    ]
    pitcher_cols = required["pitcher"][:5]

    with tempfile.TemporaryDirectory() as tmp:
        processed = Path(tmp) / "processed"
        processed.mkdir()

        batter_path = processed / (
            f"batter_features_v2_{start}_{end}.parquet"
        )
        pitcher_path = processed / (
            f"pitcher_features_v2_{start}_{end}.parquet"
        )

        write_minimal_parquet(batter_path, batter_cols)
        write_minimal_parquet(pitcher_path, pitcher_cols)

        original_batter = batter_features_path
        original_pitcher = pitcher_features_path

        def mock_batter(
            s: str,
            e: str,
            v: str = "v2",
        ) -> Path:
            return batter_path

        def mock_pitcher(
            s: str,
            e: str,
            v: str = "v2",
        ) -> Path:
            return pitcher_path

        import scripts.ensure_features as ensure_features
        import utils

        ensure_features.batter_features_path = mock_batter
        ensure_features.pitcher_features_path = mock_pitcher
        utils.batter_features_path = mock_batter
        utils.pitcher_features_path = mock_pitcher

        issues = check_range(start, end, version)

        assert issues, "expected missing walks columns to fail check"

        batter_issue = next(
            issue
            for issue in issues
            if issue.role == "batter"
        )
        missing = set(batter_issue.missing_columns)
        assert "walks_l3" in missing
        assert "walks_season" in missing

        summary = summarize_missing_columns(
            batter_issue.missing_columns
        )
        assert "walks_l*" in summary

        ensure_features.batter_features_path = original_batter
        ensure_features.pitcher_features_path = original_pitcher


def test_summarize_missing_columns_groups_windows() -> None:
    summary = summarize_missing_columns(
        (
            "walks_l3",
            "walks_l5",
            "walks_l10",
            "walks_season",
            "plate_appearances",
        )
    )
    assert summary == "plate_appearances, walks_l*"


def main() -> int:
    test_summarize_missing_columns_groups_windows()
    test_missing_walks_columns_triggers_issue()
    print("ensure_features logic tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
