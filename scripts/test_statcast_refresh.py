"""Tests for Statcast / feature parquet freshness helpers."""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils import (
    feature_parquet_needs_refresh,
    same_day_statcast_grace,
    statcast_needs_refresh,
)


def test_same_day_statcast_grace_accepts_yesterday_when_end_is_today() -> None:
    today = date(2026, 8, 30)
    yesterday = today - timedelta(days=1)
    with patch("utils.date") as mock_date:
        mock_date.today.return_value = today
        assert same_day_statcast_grace(
            "2026-08-30",
            yesterday,
            yesterday,
            today,
        )


def test_same_day_statcast_grace_rejects_when_end_is_yesterday() -> None:
    today = date(2026, 8, 30)
    with patch("utils.date") as mock_date:
        mock_date.today.return_value = today
        assert not same_day_statcast_grace(
            "2026-08-29",
            date(2026, 8, 28),
            date(2026, 8, 28),
            date(2026, 8, 29),
        )


def test_statcast_needs_refresh_honors_same_day_grace() -> None:
    today = date(2026, 8, 30)
    yesterday = today - timedelta(days=1)
    fake_path = patch("utils.statcast_raw_path")
    with patch("utils.date") as mock_date, fake_path as raw_path, patch(
        "utils.parquet_max_game_date",
        return_value=yesterday,
    ), patch(
        "utils.required_max_game_date",
        return_value=today,
    ):
        mock_date.today.return_value = today
        raw_path.return_value.exists.return_value = True
        assert not statcast_needs_refresh("2026-03-25", "2026-08-30")


def test_feature_parquet_needs_refresh_honors_same_day_grace() -> None:
    today = date(2026, 8, 30)
    yesterday = today - timedelta(days=1)
    feature_path = MagicMock()
    feature_path.exists.return_value = True
    with patch("utils.date") as mock_date, patch(
        "utils.statcast_raw_path",
        return_value=MagicMock(),
    ), patch(
        "utils.parquet_max_game_date",
        return_value=yesterday,
    ), patch(
        "utils.required_max_game_date",
        return_value=today,
    ):
        mock_date.today.return_value = today
        assert not feature_parquet_needs_refresh(
            feature_path,
            "2026-03-25",
            "2026-08-30",
        )


def test_features_caught_up_to_statcast_when_savant_lags() -> None:
    from utils import features_caught_up_to_statcast

    yesterday = date(2026, 8, 29)
    required_day = date(2026, 8, 30)
    with patch(
        "utils.statcast_raw_path",
        return_value=MagicMock(),
    ), patch(
        "utils.parquet_max_game_date",
        return_value=yesterday,
    ), patch(
        "utils.required_max_game_date",
        return_value=required_day,
    ), patch(
        "utils.batter_features_path",
        return_value=MagicMock(exists=MagicMock(return_value=True)),
    ), patch(
        "utils.pitcher_features_path",
        return_value=MagicMock(exists=MagicMock(return_value=True)),
    ):
        ok, statcast_max, required = features_caught_up_to_statcast(
            "2026-03-25",
            "2026-08-30",
            "v2",
        )
    assert ok
    assert statcast_max == yesterday
    assert required == required_day


if __name__ == "__main__":
    test_same_day_statcast_grace_accepts_yesterday_when_end_is_today()
    test_same_day_statcast_grace_rejects_when_end_is_yesterday()
    test_statcast_needs_refresh_honors_same_day_grace()
    test_feature_parquet_needs_refresh_honors_same_day_grace()
    test_features_caught_up_to_statcast_when_savant_lags()
    print("All statcast refresh tests passed.")
