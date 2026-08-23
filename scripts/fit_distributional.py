"""
Train Poisson rate models for distributional prop scoring (Phase 6 / Phase 1).

Saves models to models/{version}/dist/{market}.pkl for batter_hits,
pitcher_strikeouts, pitcher_walks, and pitcher_outs.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from distributional import (  # noqa: E402
    DISTRIBUTIONAL_MARKETS,
    fit_rate_model,
    save_distributional_model,
)
from train import (  # noqa: E402
    feature_columns_for_version,
)
from utils import (  # noqa: E402
    batter_features_path,
    normalize_version,
    pitcher_features_path,
)


def _load_features(
    start_date: str,
    end_date: str,
    version: str,
    role: str,
) -> pd.DataFrame:
    if role == "batter":
        path = batter_features_path(
            start_date,
            end_date,
            version,
        )
    else:
        path = pitcher_features_path(
            start_date,
            end_date,
            version,
        )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing feature parquet: {path}\n"
            "Run build_features.py first."
        )

    return pd.read_parquet(path)


def fit_distributional_models(
    start_date: str,
    end_date: str,
    version: str = "v2",
    *,
    markets: list[str] | None = None,
):
    version = normalize_version(version)
    feature_sets = feature_columns_for_version(
        version
    )
    saved = {}

    market_items = DISTRIBUTIONAL_MARKETS.items()
    if markets is not None:
        allowed = set(markets)
        market_items = [
            (market, config)
            for market, config in market_items
            if market in allowed
        ]

    for market, config in market_items:
        role = config["role"]
        stat = config["stat"]
        features = feature_sets[role]

        frame = _load_features(
            start_date,
            end_date,
            version,
            role,
        )

        if stat not in frame.columns:
            print(
                f"Skipping {market}: "
                f"missing stat column {stat}"
            )
            continue

        X = (
            frame[features]
            .replace(
                [float("inf"), float("-inf")],
                pd.NA,
            )
            .fillna(0)
        )
        y = frame[stat].astype(float)

        model = fit_rate_model(X, y)

        package = {
            "market": market,
            "stat": stat,
            "role": role,
            "features": features,
            "model": model,
            "train_start": start_date,
            "train_end": end_date,
            "n_rows": len(frame),
        }

        path = save_distributional_model(
            market,
            package,
            version=version,
        )
        saved[market] = path

    return saved


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train Poisson rate models for distributional "
            "prop probabilities"
        ),
    )

    parser.add_argument(
        "--start",
        required=True,
        help="Training start date (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--end",
        required=True,
        help="Training end date (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--version",
        default="v2",
        choices=["v1", "v2"],
        help="Model version (default: v2)",
    )

    parser.add_argument(
        "--market",
        action="append",
        dest="markets",
        help=(
            "Train one distributional model (repeatable). "
            "Default: all DISTRIBUTIONAL_MARKETS."
        ),
    )

    args = parser.parse_args()

    saved = fit_distributional_models(
        args.start,
        args.end,
        version=args.version,
        markets=args.markets,
    )

    print()
    print("=" * 60)
    print("DISTRIBUTIONAL MODEL FIT SUMMARY")
    print("=" * 60)
    print(f"Models saved: {len(saved)}")

    for market, path in sorted(
        saved.items()
    ):
        print(f"  {market} → {path}")

    if not saved:
        sys.exit(1)


if __name__ == "__main__":
    main()
