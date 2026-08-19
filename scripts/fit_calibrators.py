"""
Fit probability calibrators from historical backtest rows (Phase 6).

Scores props over a date window (or loads an existing backtest CSV), fits
per-market isotonic or Platt calibrators, and saves to
models/{version}/calibrators/{market}.pkl.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from calibration import (  # noqa: E402
    DEFAULT_METHOD,
    MIN_CALIBRATION_SAMPLES,
    fit_calibrators_from_frame,
)
from scripts.backtest import run_backtest  # noqa: E402
from utils import backtest_output_path, normalize_version  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fit isotonic or Platt calibrators per market from "
            "historical backtest rows"
        ),
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
        help="Model version (default: v2)",
    )

    parser.add_argument(
        "--method",
        default=DEFAULT_METHOD,
        choices=["isotonic", "platt"],
        help=(
            "Calibration method (default: isotonic)"
        ),
    )

    parser.add_argument(
        "--min-samples",
        type=int,
        default=MIN_CALIBRATION_SAMPLES,
        help=(
            "Minimum labeled rows per market "
            f"(default: {MIN_CALIBRATION_SAMPLES})"
        ),
    )

    parser.add_argument(
        "--from-csv",
        default=None,
        help=(
            "Optional existing backtest CSV instead of "
            "re-running backtest"
        ),
    )

    args = parser.parse_args()
    version = normalize_version(args.version)

    if args.from_csv:
        frame = pd.read_csv(args.from_csv)
    else:
        output = backtest_output_path(
            args.start,
            args.end,
        )

        frame = run_backtest(
            args.start,
            args.end,
            version=version,
        )

        if frame.empty:
            print(
                "No backtest rows — cannot fit calibrators."
            )
            sys.exit(1)

        if not output.exists():
            print(
                "Expected backtest output at",
                output,
            )
            sys.exit(1)

    required = {
        "market",
        "actual_over",
    }

    if (
        "raw_over_probability"
        not in frame.columns
        and "over_probability"
        not in frame.columns
    ):
        missing = required | {"over_probability"}
    else:
        missing = required - set(frame.columns)

    if missing:
        print(
            "Backtest frame missing columns:",
            ", ".join(sorted(missing)),
        )
        sys.exit(1)

    saved = fit_calibrators_from_frame(
        frame,
        version=version,
        method=args.method,
        min_samples=args.min_samples,
    )

    print()
    print("=" * 60)
    print("CALIBRATOR FIT SUMMARY")
    print("=" * 60)
    print(
        f"Method: {args.method} | "
        f"Min samples: {args.min_samples}"
    )
    print(
        f"Markets fitted: {len(saved)}"
    )

    for market, path in sorted(
        saved.items()
    ):
        print(f"  {market} → {path}")

    if not saved:
        print(
            "No calibrators saved — need more labeled "
            "outcomes per market."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
