"""
Unit tests for version compare window and slot loading.
"""

import sys
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ui import version_compare as vc  # noqa: E402


def test_compare_window_end_extends_past_anchor():
    anchor = date.fromisoformat(vc.PREDICT_END)
    effective = date.fromisoformat(vc._compare_window_end())
    assert effective >= anchor
    assert effective >= date.today()


def test_filter_compare_window_includes_recent_props():
    df = pd.DataFrame(
        {
            "player": ["A", "B"],
            "market": ["batter_hits", "batter_hits"],
            "commence_time": [
                "2026-08-20T18:11:00Z",
                "2026-02-01T12:00:00Z",
            ],
            "over_probability": [0.6, 0.5],
        }
    )

    windowed = vc._filter_compare_window(df)
    assert len(windowed) == 1
    assert windowed.iloc[0]["player"] == "A"


def test_legacy_v1_model_probability_counts_as_usable():
    df = pd.DataFrame(
        {
            "player": ["A"],
            "market": ["batter_hits"],
            "commence_time": ["2026-08-18T00:41:00Z"],
            "model_probability": [0.9],
        }
    )

    windowed = vc._filter_compare_window(df)
    assert vc._has_usable_probabilities(windowed)


def test_slot_needs_generation_when_outside_anchor_but_inside_effective_end():
    raw = pd.DataFrame(
        {
            "player": ["A"],
            "market": ["batter_hits"],
            "commence_time": ["2026-08-20T18:11:00Z"],
            "over_probability": [0.55],
            "under_probability": [0.45],
        }
    )

    with patch.object(vc, "_read_slot_raw_csv", return_value=raw):
        assert vc._slot_window_rows("v2") is not None
        assert not vc._slot_needs_generation(
            {"key": "v2", "label": "V2", "model_version": "v2"}
        )


if __name__ == "__main__":
    test_compare_window_end_extends_past_anchor()
    test_filter_compare_window_includes_recent_props()
    test_legacy_v1_model_probability_counts_as_usable()
    test_slot_needs_generation_when_outside_anchor_but_inside_effective_end()
    print("All version compare tests passed.")
