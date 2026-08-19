#!/usr/bin/env python3
"""Unit tests for intraday odds snapshots and movement features."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import odds_snapshots  # noqa: E402
from odds_movement import compute_movement_features  # noqa: E402
from odds_snapshots import (  # noqa: E402
    load_opening_snapshot,
    save_live_snapshot,
    snapshots_dir,
)


def _sample_props(
    line=1.5,
    odds=-110,
    fetched_at="2026-08-19T18:00:00+00:00",
):
    return pd.DataFrame([
        {
            "event_id": "evt1",
            "commence_time": "2026-08-19T23:00:00Z",
            "home_team": "NYY",
            "away_team": "BOS",
            "bookmaker": "BookA",
            "bookmaker_key": "booka",
            "market": "batter_hits",
            "player": "Mike Trout",
            "side": "Over",
            "line": line,
            "odds": odds,
            "fetched_at": fetched_at,
        },
        {
            "event_id": "evt1",
            "commence_time": "2026-08-19T23:00:00Z",
            "home_team": "NYY",
            "away_team": "BOS",
            "bookmaker": "BookA",
            "bookmaker_key": "booka",
            "market": "batter_hits",
            "player": "Mike Trout",
            "side": "Under",
            "line": line,
            "odds": -110,
            "fetched_at": fetched_at,
        },
    ])


def test_save_live_snapshot_writes_append_only() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        odds_snapshots.RAW_DIR = Path(tmp) / "raw"

        props = _sample_props()

        path_one = save_live_snapshot(props)
        path_two = save_live_snapshot(props)

        assert path_one is not None
        assert path_two is not None
        assert path_one != path_two
        assert path_one.name.startswith("props_")
        assert path_one.parent == snapshots_dir()

        loaded = pd.read_parquet(path_one)
        assert "fetched_at" in loaded.columns
        assert len(loaded) == 2


def test_load_opening_snapshot_earliest_before_commence() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp) / "snapshots"
        directory.mkdir()

        early = _sample_props(
            line=1.5,
            odds=-120,
            fetched_at="2026-08-19T14:00:00+00:00",
        )
        late = _sample_props(
            line=0.5,
            odds=-105,
            fetched_at="2026-08-19T20:00:00+00:00",
        )

        early.to_parquet(
            directory / "props_20260819_140000.parquet",
            index=False,
        )
        late.to_parquet(
            directory / "props_20260819_200000.parquet",
            index=False,
        )

        combined = pd.concat(
            [early, late],
            ignore_index=True,
        )

        opening = load_opening_snapshot(
            combined,
            game_date="2026-08-19",
        )

        over_row = opening[
            opening["side"] == "Over"
        ].iloc[0]

        assert over_row["opening_line"] == 1.5
        assert over_row["opening_odds"] == -120


def test_compute_movement_features_deltas_and_steam() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp) / "snapshots"
        directory.mkdir()

        opening = _sample_props(
            line=1.5,
            odds=-120,
            fetched_at="2026-08-19T14:00:00+00:00",
        )
        opening.to_parquet(
            directory / "props_20260819_140000.parquet",
            index=False,
        )

        current = _sample_props(
            line=0.5,
            odds=-110,
            fetched_at="2026-08-19T20:00:00+00:00",
        )

        result = compute_movement_features(
            current,
            directory,
            game_date="2026-08-19",
        )

        over_row = result[
            result["side"] == "Over"
        ].iloc[0]

        assert over_row["opening_line"] == 1.5
        assert abs(over_row["line_delta"] - (-1.0)) < 1e-6
        assert abs(over_row["odds_delta"] - 10.0) < 1e-6
        assert bool(over_row["steam_flag"]) is True


def test_compute_movement_features_without_snapshots() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        current = _sample_props()

        result = compute_movement_features(
            current,
            Path(tmp) / "empty",
            game_date="2026-08-19",
        )

        assert pd.isna(result.loc[0, "opening_line"])
        assert pd.isna(result.loc[0, "line_delta"])
        assert not bool(result.loc[0, "steam_flag"])


def main() -> int:
    test_save_live_snapshot_writes_append_only()
    test_load_opening_snapshot_earliest_before_commence()
    test_compute_movement_features_deltas_and_steam()
    test_compute_movement_features_without_snapshots()
    print("odds_movement tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
