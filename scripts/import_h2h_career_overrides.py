#!/usr/bin/env python3
"""
Import career H2H overrides from a pasted markdown/text table.

Example:
  .venv/bin/python scripts/import_h2h_career_overrides.py path/to/h2h_paste.txt

The text file should look like the pipe tables you paste from ESPN/StatMuse
boards (Team | Batter | Opposing SP | H/AB | AVG | HR | RBI).

Safe merge: only adds new (batter, pitcher) pairs or updates matching ones.
Unrelated existing rows in data/reference/h2h_career_overrides.csv are kept.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from h2h_career_overrides import (  # noqa: E402
    H2H_CAREER_OVERRIDES_PATH,
    import_career_h2h_from_text_file,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Parse a markdown/text H2H table file and upsert into "
            "h2h_career_overrides.csv"
        ),
    )
    parser.add_argument(
        "text_file",
        type=Path,
        help="Path to a .txt/.md file containing the pipe table",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=H2H_CAREER_OVERRIDES_PATH,
        help=f"Overrides CSV path (default: {H2H_CAREER_OVERRIDES_PATH})",
    )
    args = parser.parse_args()

    out, n, stats = import_career_h2h_from_text_file(
        args.text_file,
        csv_path=args.csv,
    )
    print(f"Parsed {n} H2H row(s) from {args.text_file}")
    print(
        f"Merge into {out}: "
        f"kept {stats['kept']} · "
        f"updated {stats['updated']} · "
        f"added {stats['added']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
