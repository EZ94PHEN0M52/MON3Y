#!/usr/bin/env python3
"""Rebuild pp_fantasy_game_scores.parquet from batter feature parquets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pp_fantasy_scores import (  # noqa: E402
    archive_is_stale,
    archive_max_game_date,
    rebuild_pp_fantasy_game_scores,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build PrizePicks fantasy score archive from batter game logs "
            "(source of truth for L5/L10 vs PP line)."
        )
    )
    parser.add_argument("--version", default="v2", choices=["v1", "v2"])
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even when the archive is newer than features.",
    )
    args = parser.parse_args()

    if not args.force and not archive_is_stale(args.version):
        print("PP fantasy score archive is up to date.")
        return 0

    path = rebuild_pp_fantasy_game_scores(version=args.version)
    if path is None:
        print("ERROR: No batter features available to build archive.", file=sys.stderr)
        return 1

    max_date = archive_max_game_date(args.version)
    print(f"Saved PP fantasy game scores → {path}")
    if max_date:
        print(f"Archive through game_date: {max_date}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
