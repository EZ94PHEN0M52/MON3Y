#!/usr/bin/env python3
"""Join logged predictions to post-game actuals for the learning loop."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from learning_log import (  # noqa: E402
    LEARNING_MARKETS,
    append_outcomes_log,
    join_outcomes_for_market,
    load_predictions_log,
    summarize_learning_log,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Join predictions_log rows to realized stats from feature "
            "parquets and append to outcomes_log.parquet."
        ),
    )
    parser.add_argument(
        "--market",
        default="pitcher_outs",
        choices=sorted(LEARNING_MARKETS),
    )
    parser.add_argument(
        "--start",
        required=True,
        help="First game_date to join (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="Last game_date to join (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--version",
        default="v2",
        choices=["v1", "v2"],
    )
    parser.add_argument(
        "--feature-end",
        help=(
            "Pitcher feature parquet end date (defaults to --end). "
            "Use season-end date when features cover a wider window."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary only; do not write outcomes_log.parquet",
    )

    args = parser.parse_args()

    outcomes = join_outcomes_for_market(
        args.market,
        args.start,
        args.end,
        version=args.version,
        feature_end=args.feature_end,
    )

    print()
    print("=" * 72)
    print("LEARNING OUTCOMES JOIN")
    print("=" * 72)
    print(f"Market: {args.market}")
    print(f"Window: {args.start} → {args.end} ({args.version})")
    print(f"Matched rows: {len(outcomes):,}")

    if outcomes.empty:
        print()
        preds = load_predictions_log(market=args.market)
        if preds.empty:
            print(
                "No outcomes joined — predictions_log is empty. "
                "Run predict.py first (logging is automatic for pitcher_outs)."
            )
        else:
            pred_dates = sorted(
                preds["game_date"].dropna().astype(str).unique().tolist()
            )
            print(
                f"Predictions logged: {len(preds):,} rows "
                f"({pred_dates[0]} → {pred_dates[-1]})."
            )
            outside = [
                d for d in pred_dates
                if d < args.start or d > args.end
            ]
            if outside:
                print(
                    f"  {len(outside)} prediction date(s) fall outside the "
                    f"join window ({args.start} → {args.end}). "
                    "Use --join-start / --join-end to widen, or wait until "
                    "those games finish and re-run with --end=yesterday."
                )
            print(
                "No box-score matches in window. After games complete, "
                "re-run join with --end through the played date."
            )
        return

    if not args.dry_run:
        written = append_outcomes_log(outcomes)
        print(f"Appended to outcomes_log: {written:,} rows")

    summary = summarize_learning_log(market=args.market)
    print()
    print(
        f"Log totals — predictions: {summary['predictions_logged']:,}, "
        f"outcomes: {summary['outcomes_joined']:,}"
    )


if __name__ == "__main__":
    main()
