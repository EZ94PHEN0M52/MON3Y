#!/usr/bin/env python3
"""Unit tests for Underdog fantasy line fetch/parser."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fetch_underdog_fantasy import (  # noqa: E402
    parse_underdog_fantasy_payload,
)


SAMPLE_PAYLOAD = {
    "players": [
        {
            "id": "p1",
            "first_name": "Aaron",
            "last_name": "Judge",
        },
    ],
    "appearances": [
        {
            "id": "a1",
            "player_id": "p1",
        },
    ],
    "over_under_lines": [
        {
            "status": "active",
            "stat_value": "8.5",
            "over_under": {
                "appearance_stat": {
                    "appearance_id": "a1",
                    "stat": "fantasy_points",
                    "display_stat": "Fantasy Points",
                },
            },
        },
        {
            "status": "active",
            "stat_value": "2.5",
            "over_under": {
                "appearance_stat": {
                    "appearance_id": "a1",
                    "stat": "hits_runs_rbis",
                    "display_stat": "Hits + Runs + RBIs",
                },
            },
        },
    ],
}


def test_parse_underdog_fantasy_payload_filters_stat() -> None:
    rows = parse_underdog_fantasy_payload(
        SAMPLE_PAYLOAD,
        fetched_at="2026-08-23T06:00:00+00:00",
    )

    assert len(rows) == 1
    assert rows[0]["player"] == "Aaron Judge"
    assert rows[0]["line"] == 8.5
    assert rows[0]["market"] == "batter_fantasy_score"
    assert rows[0]["bookmaker_key"] == "underdog"


def main() -> None:
    test_parse_underdog_fantasy_payload_filters_stat()
    print("test_parse_underdog_fantasy_payload_filters_stat: ok")
    print("All underdog fantasy tests passed.")


if __name__ == "__main__":
    main()
