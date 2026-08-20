"""
Backtest Batter Score vs actual H+TB+BB outcomes (validation track).

Scores each batter-game using only pre-game history (no lookahead), compares
to the same-game hits + total bases + walks composite from feature parquets,
and writes pass/fail gates to data/backtest/batter_score_validation.json.

Orthogonal to prop-model backtests — does not affect board edge or ranking.

Offline / cache-only: this script reads data/processed/ feature parquets only.
It never hits Statcast, pybaseball, Odds API, or MLB Stats API. game_context
is omitted so scoring stays Phase A (season + recent form) without probables
or raw Statcast arsenal loads. DISABLE_LIVE_FETCH=1 is enabled by default.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Block live API calls if any shared helper is invoked accidentally.
os.environ.setdefault("DISABLE_LIVE_FETCH", "1")

from batter_score_data import (  # noqa: E402
    actual_raw_points_on_date,
    score_batter_as_of,
    write_batter_score_validation,
)
from scripts.backtest import (  # noqa: E402
    find_covering_feature_path,
)
from utils import (  # noqa: E402
    BATTER_SCORE_VALIDATION_PATH,
    batter_features_path,
    normalize_version,
)


DEFAULT_MIN_SAMPLE = 100
DEFAULT_MIN_SPEARMAN = 0.15
MAX_RAW_POINTS = 6.0  # matches BatterInputs.max_raw_points_for_100 default


def load_batter_features(
    start_date: str,
    end_date: str,
    version: str = "v2",
) -> pd.DataFrame:
    version = normalize_version(version)

    path = batter_features_path(start_date, end_date, version)
    if not path.exists():
        path = find_covering_feature_path(
            start_date,
            end_date,
            version,
            "batter",
        )

    if path is None or not path.exists():
        raise FileNotFoundError(
            "No batter feature parquet covers "
            f"{start_date} → {end_date}.\n"
            "Run build_features.py for a range that includes these dates."
        )

    frame = pd.read_parquet(path)
    frame["game_date"] = (
        pd.to_datetime(frame["game_date"])
        .dt.strftime("%Y-%m-%d")
    )
    return frame


def _safe_corr(
    scores: np.ndarray,
    actuals: np.ndarray,
    method: str,
) -> float | None:
    if len(scores) < 2:
        return None

    if np.std(scores) == 0 or np.std(actuals) == 0:
        return None

    if method == "pearson":
        value, _ = pearsonr(scores, actuals)
    else:
        value, _ = spearmanr(scores, actuals)

    if np.isnan(value):
        return None

    return float(value)


def _implied_raw_points(score: float) -> float:
    return float(score) / 100.0 * MAX_RAW_POINTS


def run_batter_score_backtest(
    start_date: str,
    end_date: str,
    version: str = "v2",
    min_sample: int = DEFAULT_MIN_SAMPLE,
    min_spearman: float = DEFAULT_MIN_SPEARMAN,
    write_detail: bool = False,
) -> dict:
    """
    Score batter-games in [start_date, end_date] and evaluate validation gates.

    Target outcome: same-game H + TB + BB raw points (Batter Score input stat).
    Primary gate: Spearman rank correlation (robust to non-linear 0–100 index).
    """
    batters = load_batter_features(start_date, end_date, version)

    required = {
        "game_date",
        "player_name",
        "hits",
        "total_bases",
        "walks",
    }
    missing = required - set(batters.columns)
    if missing:
        raise ValueError(
            f"Batter features missing columns: {sorted(missing)}"
        )

    eval_rows = batters[
        batters["game_date"].ge(start_date)
        & batters["game_date"].le(end_date)
    ].copy()

    eval_rows = eval_rows.drop_duplicates(
        subset=["game_date", "player_name"],
        keep="first",
    )

    detail_rows = []
    skipped_insufficient_history = 0
    skipped_missing_actual = 0
    skipped_score_failed = 0

    by_player = {
        name: group.copy()
        for name, group in batters.groupby("player_name", sort=False)
    }

    # Batch by player: one history frame lookup per player, dates in order.
    for player_name, player_eval in eval_rows.groupby(
        "player_name",
        sort=False,
    ):
        player_history = by_player.get(player_name)

        if player_history is None:
            skipped_missing_actual += len(player_eval)
            continue

        for _, row in player_eval.sort_values(
            "game_date",
        ).iterrows():
            game_date = row["game_date"]

            actual = actual_raw_points_on_date(
                player_history,
                game_date,
            )
            if actual is None:
                skipped_missing_actual += 1
                continue

            scored = score_batter_as_of(
                player_history,
                game_date,
                game_context=None,
                version=version,
            )

            if scored is None:
                prior_count = len(
                    player_history[
                        player_history["game_date"].lt(game_date)
                    ]
                )
                if prior_count < 10:
                    skipped_insufficient_history += 1
                else:
                    skipped_score_failed += 1
                continue

            implied = _implied_raw_points(scored.batter_score)
            detail_rows.append(
                {
                    "game_date": game_date,
                    "player_name": player_name,
                    "batter_score": scored.batter_score,
                    "partial_label": scored.partial_label or "",
                    "actual_raw_points": actual,
                    "implied_raw_points": implied,
                    "abs_error": abs(implied - actual),
                }
            )

    detail = pd.DataFrame(detail_rows)
    sample_size = len(detail)

    scores = detail["batter_score"].to_numpy(dtype=float)
    actuals = detail["actual_raw_points"].to_numpy(dtype=float)

    pearson = _safe_corr(scores, actuals, "pearson")
    spearman = _safe_corr(scores, actuals, "spearman")
    mae = (
        float(detail["abs_error"].mean())
        if sample_size > 0
        else None
    )

    per_date = []
    if sample_size > 0:
        for game_date, group in detail.groupby("game_date", sort=True):
            group_scores = group["batter_score"].to_numpy(dtype=float)
            group_actuals = group["actual_raw_points"].to_numpy(dtype=float)
            per_date.append(
                {
                    "game_date": game_date,
                    "sample_size": int(len(group)),
                    "spearman_correlation": _safe_corr(
                        group_scores,
                        group_actuals,
                        "spearman",
                    ),
                    "mean_batter_score": float(group_scores.mean()),
                    "mean_actual_raw_points": float(group_actuals.mean()),
                }
            )

    validated = (
        sample_size >= min_sample
        and spearman is not None
        and spearman >= min_spearman
    )

    timestamp = (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )

    payload = {
        "validated": bool(validated),
        "sample_size": int(sample_size),
        "pearson_correlation": pearson,
        "spearman_correlation": spearman,
        "mae_implied_raw_points": mae,
        "date_range": {
            "start": start_date,
            "end": end_date,
        },
        "criteria_used": {
            "target_stat": "hits + total_bases + walks (same-game raw points)",
            "scoring": "point-in-time Batter Score (pre-game history only)",
            "primary_metric": "spearman_correlation",
            "primary_metric_rationale": (
                "Batter Score is a ranked 0–100 composite; Spearman captures "
                "monotonic ordering vs outcomes without assuming linearity."
            ),
            "gates": [
                f"sample_size >= {min_sample}",
                f"spearman_correlation >= {min_spearman}",
            ],
        },
        "thresholds": {
            "min_sample": int(min_sample),
            "min_spearman": float(min_spearman),
        },
        "skipped": {
            "insufficient_history": int(skipped_insufficient_history),
            "missing_actual": int(skipped_missing_actual),
            "score_failed": int(skipped_score_failed),
        },
        "per_date": per_date,
        "timestamp": timestamp,
        "version": normalize_version(version),
    }

    write_batter_score_validation(payload)

    if write_detail and not detail.empty:
        detail_path = (
            BATTER_SCORE_VALIDATION_PATH.with_name(
                "batter_score_validation_detail.parquet"
            )
        )
        detail.to_parquet(detail_path, index=False)
        payload["detail_path"] = str(detail_path)

    _print_summary(payload, detail)
    return payload


def _print_summary(payload: dict, detail: pd.DataFrame) -> None:
    print()
    print("=" * 80)
    print("BATTER SCORE VALIDATION")
    print("=" * 80)

    date_range = payload.get("date_range") or {}
    print(
        f"Window: {date_range.get('start')} → {date_range.get('end')} "
        f"({payload.get('version', 'v2')})"
    )
    print(f"Sample size: {payload.get('sample_size', 0):,}")

    skipped = payload.get("skipped") or {}
    print(
        "Skipped: "
        f"{skipped.get('insufficient_history', 0):,} insufficient history, "
        f"{skipped.get('score_failed', 0):,} score errors, "
        f"{skipped.get('missing_actual', 0):,} missing actuals"
    )

    print()
    print(
        f"Pearson r:  {payload.get('pearson_correlation')}"
    )
    print(
        f"Spearman ρ: {payload.get('spearman_correlation')}"
    )
    mae = payload.get("mae_implied_raw_points")
    if mae is not None:
        print(
            f"MAE (score→implied raw vs actual H+TB+BB): {mae:.3f}"
        )

    thresholds = payload.get("thresholds") or {}
    print()
    print("Gates:")
    print(
        f"  sample_size >= {thresholds.get('min_sample', DEFAULT_MIN_SAMPLE)}"
    )
    print(
        f"  spearman >= {thresholds.get('min_spearman', DEFAULT_MIN_SPEARMAN)}"
    )
    print()
    status = "PASS ✓" if payload.get("validated") else "FAIL"
    print(f"Validated: {status}")
    print()
    print("Saved:", BATTER_SCORE_VALIDATION_PATH)

    if not detail.empty and "partial_label" in detail.columns:
        label_counts = (
            detail["partial_label"]
            .replace("", "unknown")
            .value_counts()
        )
        print()
        print("Score labels in sample:")
        for label, count in label_counts.items():
            print(f"  {label}: {count:,}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Backtest Batter Score vs actual H+TB+BB outcomes and "
            "write validation gates"
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
        help="Feature parquet version (default: v2)",
    )
    parser.add_argument(
        "--min-sample",
        type=int,
        default=DEFAULT_MIN_SAMPLE,
        help=(
            "Minimum batter-game rows required to pass validation "
            f"(default: {DEFAULT_MIN_SAMPLE})"
        ),
    )
    parser.add_argument(
        "--min-spearman",
        type=float,
        default=DEFAULT_MIN_SPEARMAN,
        help=(
            "Minimum Spearman correlation vs actual H+TB+BB "
            f"(default: {DEFAULT_MIN_SPEARMAN})"
        ),
    )
    parser.add_argument(
        "--write-detail",
        action="store_true",
        help=(
            "Also write per-row detail parquet to "
            "data/backtest/batter_score_validation_detail.parquet"
        ),
    )

    args = parser.parse_args()

    run_batter_score_backtest(
        args.start,
        args.end,
        version=args.version,
        min_sample=args.min_sample,
        min_spearman=args.min_spearman,
        write_detail=args.write_detail,
    )


if __name__ == "__main__":
    main()
