"""Unit tests for cross-source player name normalization and matching."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prop_scoring import fuzzy_player_match  # noqa: E402
from utils import normalize_player_key  # noqa: E402


def test_normalize_player_key_strips_initials_periods() -> None:
    assert normalize_player_key("J.T. Ginn") == normalize_player_key("JT Ginn")
    assert normalize_player_key("J.T. Realmuto") == normalize_player_key(
        "JT Realmuto"
    )


def test_fuzzy_player_match_initial_period_mismatch() -> None:
    candidates = pd.Series(["J.T. Ginn", "Emerson Hancock"])
    assert fuzzy_player_match("JT Ginn", candidates) == "J.T. Ginn"
