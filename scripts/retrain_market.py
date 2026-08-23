#!/usr/bin/env python3
"""Retrain a single prop market from feature parquets (Track 1: pitcher_outs)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from learning_log import (  # noqa: E402
    LEARNING_MARKETS,
    load_predictions_log,
    summarize_learning_log,
)
from prop_scoring import MODEL_MAP  # noqa: E402
from train import (  # noqa: E402
    PITCHER_MARKETS,
    create_training_rows,
    feature_columns_for_version,
    model_feature_columns,
    train_model,
    validate_feature_columns,
)
from training_odds import load_historical_props  # noqa: E402
from utils import (  # noqa: E402
    normalize_version,
    resolve_feature_path,
)
from distributional import distributional_model_path  # noqa: E402


MARKET_TRAINING_TARGET = {
    "pitcher_outs": ("pitcher", "outs", PITCHER_MARKETS["outs"]),
}


def _resolve_training_spec(market: str):
    if market not in LEARNING_MARKETS:
        raise ValueError(
            f"Market {market!r} is not enabled for retrain_market. "
            f"Supported: {sorted(LEARNING_MARKETS)}"
        )

    if market not in MARKET_TRAINING_TARGET:
        raise ValueError(
            f"No training spec for {market!r}. "
            "Add an entry to MARKET_TRAINING_TARGET."
        )

    role, target, lines = MARKET_TRAINING_TARGET[market]
    return role, target, lines


def retrain_market(
    market: str,
    start_date: str,
    end_date: str,
    *,
    version: str = "v2",
    line_source: str = "auto",
    fit_distributional: bool = False,
) -> dict:
    version = normalize_version(version)
    role, target, lines = _resolve_training_spec(market)

    feature_sets = feature_columns_for_version(version)
    model_features = model_feature_columns(version)

    historical_props = None
    if line_source in ("real", "auto"):
        historical_props = load_historical_props(
            start_date,
            end_date,
        )

    batter_path = resolve_feature_path(
        start_date,
        end_date,
        version,
        role="batter",
    )
    pitcher_path = resolve_feature_path(
        start_date,
        end_date,
        version,
        role="pitcher",
    )

    if role == "pitcher":
        if not pitcher_path.exists():
            raise FileNotFoundError(
                f"No pitcher feature parquet covering "
                f"{start_date} → {end_date} ({version}). "
                "Run ensure_features.py or ./run_daily.sh first."
            )
        features_df = pd.read_parquet(pitcher_path)
        feature_columns = feature_sets["pitcher"]
        player_col = "pitcher"
        validate_feature_columns(
            features_df,
            feature_columns,
            pitcher_path,
            start_date,
            end_date,
            version,
        )
    else:
        if not batter_path.exists():
            raise FileNotFoundError(
                f"No batter feature parquet covering "
                f"{start_date} → {end_date} ({version}). "
                "Run ensure_features.py or ./run_daily.sh first."
            )
        features_df = pd.read_parquet(batter_path)
        feature_columns = feature_sets["batter"]
        player_col = "batter"
        validate_feature_columns(
            features_df,
            feature_columns,
            batter_path,
            start_date,
            end_date,
            version,
        )

    training = create_training_rows(
        features_df,
        player_col,
        target,
        lines,
        feature_columns,
        line_source=line_source,
        historical_props=historical_props,
    )

    if len(training) < 100:
        raise RuntimeError(
            f"Not enough training rows for {market}: {len(training)}"
        )

    model_name = market
    if market.startswith("pitcher_"):
        model_name = market
    elif market.startswith("batter_"):
        model_name = market
    else:
        model_name = f"{role}_{target}"

    train_model(
        training,
        model_features[role],
        model_name,
        version,
    )

    dist_paths = {}
    if fit_distributional:
        import subprocess

        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "fit_distributional.py"),
                "--start",
                start_date,
                "--end",
                end_date,
                "--version",
                version,
                "--market",
                market,
            ],
            cwd=ROOT,
            check=True,
        )
        dist_paths[market] = distributional_model_path(
            market,
            version,
        )

    return {
        "market": market,
        "model_file": MODEL_MAP.get(market),
        "training_rows": len(training),
        "train_start": start_date,
        "train_end": end_date,
        "distributional_paths": dist_paths,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Retrain one market's LightGBM classifier. Track 1 default: "
            "pitcher_outs. Does not modify board layout — only replaces "
            "the model pickle used at predict time."
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
        help="Training window start (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="Training window end (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--version",
        default="v2",
        choices=["v1", "v2"],
    )
    parser.add_argument(
        "--line-source",
        default="auto",
        choices=["real", "synthetic", "auto"],
    )
    parser.add_argument(
        "--fit-distributional",
        action="store_true",
        help=(
            "Also train Poisson regressor → models/v2/dist/pitcher_outs.pkl "
            "(populates Pred # and Dist Over % on board; edge still classifier)"
        ),
    )

    args = parser.parse_args()

    print()
    print("=" * 72)
    print("SINGLE-MARKET RETRAIN")
    print("=" * 72)
    print(f"Market: {args.market}")
    print(f"Window: {args.start} → {args.end} ({args.version})")
    print(f"Line source: {args.line_source}")

    result = retrain_market(
        args.market,
        args.start,
        args.end,
        version=args.version,
        line_source=args.line_source,
        fit_distributional=args.fit_distributional,
    )

    preds = load_predictions_log(market=args.market)
    summary = summarize_learning_log(market=args.market)

    print()
    print(f"Saved model: models/{args.version}/{result['model_file']}")
    print(f"Training rows: {result['training_rows']:,}")
    if result.get("distributional_paths"):
        for market, path in sorted(result["distributional_paths"].items()):
            print(f"Distributional: {market} → {path}")
    print()
    print(
        "Learning log — "
        f"predictions logged: {summary['predictions_logged']:,}, "
        f"outcomes joined: {summary['outcomes_joined']:,} "
        f"(from {len(preds):,} prediction rows)"
    )
    print()
    print(
        "Board impact: re-run predict.py to pick up the new model. "
        "Over %, Edge, and EV for pitcher_outs rows may change. "
        "With --fit-distributional, Pred # and Dist Over % also populate "
        "for outs rows (edge still from classifier)."
    )


if __name__ == "__main__":
    main()
