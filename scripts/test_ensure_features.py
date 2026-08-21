#!/usr/bin/env python3
"""Dry-run tests for ensure_features stale/missing-column detection."""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.ensure_features import (  # noqa: E402
    SOURCE_FILES,
    check_range,
    summarize_missing_columns,
)
from train import (  # noqa: E402
    feature_columns_for_version,
    feature_schema_fingerprint,
    parquet_schema_is_stale,
    valid_parquet_fingerprints,
)
from utils import (  # noqa: E402
    batter_features_path,
    normalize_version,
    pitcher_features_path,
)


def write_minimal_parquet(
    path: Path,
    columns: list[str],
    fingerprint: str | None = None,
    game_dates: list[str] | None = None,
) -> None:
    row_count = len(game_dates) if game_dates else 1
    data = {
        col: [1.0] * row_count
        for col in columns
        if col != "game_date"
    }
    if "game_date" in columns:
        data["game_date"] = game_dates or ["2099-01-01"]
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


def test_legacy_fingerprints_include_current_schema() -> None:
    current = feature_schema_fingerprint("v2")
    assert current in valid_parquet_fingerprints("v2")
    assert not parquet_schema_is_stale(current, "v2")
    assert not parquet_schema_is_stale(None, "v2")


def test_training_odds_not_in_source_files() -> None:
    assert "training_odds.py" not in SOURCE_FILES
    assert "train.py" not in SOURCE_FILES


def test_source_mtime_ignores_train_py() -> None:
    import os

    start = "2099-03-01"
    end = "2099-03-02"
    version = "v2"

    required = feature_columns_for_version(version)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        processed = tmp_path / "processed"
        processed.mkdir()

        batter_path = processed / (
            f"batter_features_v2_{start}_{end}.parquet"
        )
        pitcher_path = processed / (
            f"pitcher_features_v2_{start}_{end}.parquet"
        )

        write_minimal_parquet(
            batter_path,
            required["batter"],
            fingerprint=feature_schema_fingerprint("v2"),
        )
        write_minimal_parquet(
            pitcher_path,
            required["pitcher"],
            fingerprint=feature_schema_fingerprint("v2"),
        )

        past = time.time() - 3600
        os.utime(batter_path, (past, past))
        os.utime(pitcher_path, (past, past))

        stale_source = tmp_path / "build_features.py"
        stale_source.write_text("# stub\n")
        os.utime(stale_source, (past, past))

        stale_v2 = tmp_path / "features_v2.py"
        stale_v2.write_text("# stub\n")
        os.utime(stale_v2, (past, past))

        stale_game_lines = tmp_path / "game_lines.py"
        stale_game_lines.write_text("# stub\n")
        os.utime(stale_game_lines, (past, past))

        import scripts.ensure_features as ensure_features

        original_source_files = (
            ensure_features.source_files_for_version
        )

        def mock_source_files(
            v: str,
        ) -> dict[str, Path]:
            files = original_source_files(v)
            files["build_features.py"] = stale_source
            files["game_lines.py"] = stale_game_lines
            if normalize_version(v) == "v2":
                files["features_v2.py"] = stale_v2
            return files

        ensure_features.source_files_for_version = (
            mock_source_files
        )

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

        ensure_features.batter_features_path = mock_batter
        ensure_features.pitcher_features_path = mock_pitcher

        issues = check_range(start, end, version)

        ensure_features.source_files_for_version = (
            original_source_files
        )

        assert not issues, (
            "expected valid parquets to pass when only train.py is newer"
        )


def test_stale_feature_data_triggers_issue() -> None:
    start = "2099-04-01"
    end = "2099-04-03"
    version = "v2"

    required = feature_columns_for_version(version)

    with tempfile.TemporaryDirectory() as tmp:
        processed = Path(tmp) / "processed"
        raw = Path(tmp) / "raw"
        processed.mkdir()
        raw.mkdir()

        batter_path = processed / (
            f"batter_features_v2_{start}_{end}.parquet"
        )
        pitcher_path = processed / (
            f"pitcher_features_v2_{start}_{end}.parquet"
        )

        write_minimal_parquet(
            batter_path,
            required["batter"] + ["game_date"],
            fingerprint=feature_schema_fingerprint("v2"),
            game_dates=["2099-04-01", "2099-04-02"],
        )
        write_minimal_parquet(
            pitcher_path,
            required["pitcher"] + ["game_date"],
            fingerprint=feature_schema_fingerprint("v2"),
            game_dates=["2099-04-01", "2099-04-02"],
        )

        current_table = pa.Table.from_pydict(
            {
                "game_date": [
                    "2099-04-01",
                    "2099-04-02",
                    "2099-04-03",
                ]
            }
        )
        pq.write_table(
            current_table,
            raw / f"statcast_{start}_{end}.parquet",
        )

        import scripts.ensure_features as ensure_features
        import utils

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

        def mock_statcast_raw(
            s: str,
            e: str,
        ) -> Path:
            return raw / f"statcast_{s}_{e}.parquet"

        ensure_features.batter_features_path = mock_batter
        ensure_features.pitcher_features_path = mock_pitcher
        utils.batter_features_path = mock_batter
        utils.pitcher_features_path = mock_pitcher
        utils.statcast_raw_path = mock_statcast_raw
        utils.RAW_DIR = raw

        issues = check_range(start, end, version)

        assert issues, "expected stale feature/statcast data to fail check"
        assert any(issue.stale_data for issue in issues)


def test_off_day_end_date_matches_statcast() -> None:
    start = "2099-05-01"
    end = "2099-05-03"
    version = "v2"

    required = feature_columns_for_version(version)

    with tempfile.TemporaryDirectory() as tmp:
        processed = Path(tmp) / "processed"
        raw = Path(tmp) / "raw"
        processed.mkdir()
        raw.mkdir()

        batter_path = processed / (
            f"batter_features_v2_{start}_{end}.parquet"
        )
        pitcher_path = processed / (
            f"pitcher_features_v2_{start}_{end}.parquet"
        )

        write_minimal_parquet(
            batter_path,
            required["batter"] + ["game_date"],
            fingerprint=feature_schema_fingerprint("v2"),
            game_dates=["2099-05-01", "2099-05-02"],
        )
        write_minimal_parquet(
            pitcher_path,
            required["pitcher"] + ["game_date"],
            fingerprint=feature_schema_fingerprint("v2"),
            game_dates=["2099-05-01", "2099-05-02"],
        )

        pq.write_table(
            pa.Table.from_pydict(
                {"game_date": ["2099-05-01", "2099-05-02"]}
            ),
            raw / f"statcast_{start}_{end}.parquet",
        )

        import scripts.ensure_features as ensure_features
        import utils

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

        def mock_statcast_raw(
            s: str,
            e: str,
        ) -> Path:
            return raw / f"statcast_{s}_{e}.parquet"

        ensure_features.batter_features_path = mock_batter
        ensure_features.pitcher_features_path = mock_pitcher
        utils.batter_features_path = mock_batter
        utils.pitcher_features_path = mock_pitcher
        utils.statcast_raw_path = mock_statcast_raw
        utils.RAW_DIR = raw

        issues = check_range(start, end, version)

        assert not issues, (
            "expected matching feature/statcast max to pass when end_date "
            "has no extra game rows"
        )


def main() -> int:
    test_summarize_missing_columns_groups_windows()
    test_missing_walks_columns_triggers_issue()
    test_legacy_fingerprints_include_current_schema()
    test_training_odds_not_in_source_files()
    test_source_mtime_ignores_train_py()
    test_stale_feature_data_triggers_issue()
    test_off_day_end_date_matches_statcast()
    print("ensure_features logic tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
