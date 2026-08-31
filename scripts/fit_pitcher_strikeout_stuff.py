#!/usr/bin/env python3
"""
Train the separate Statcast stuff → strikeout model.

Saves models/v2/pitcher_strikeouts_stuff.pkl (independent of
pitcher_strikeouts.pkl).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pitcher_strikeout_stuff import (  # noqa: E402
    STUFF_K_FEATURES,
    fit_stuff_strikeout_model,
    save_stuff_strikeout_model,
)
from utils import (  # noqa: E402
    normalize_version,
    pitcher_features_path,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train SwStr/chase/velocity strikeout model "
            "(separate from main pitcher_strikeouts.pkl)"
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
        help="Model version directory (default: v2)",
    )
    args = parser.parse_args()

    version = normalize_version(
        args.version
    )
    path = pitcher_features_path(
        args.start,
        args.end,
        version,
    )

    if not path.exists():
        raise SystemExit(
            f"Missing pitcher features: {path}\n"
            "Run build_features.py first."
        )

    frame = __import__(
        "pandas"
    ).read_parquet(path)

    missing = [
        col
        for col in STUFF_K_FEATURES
        if col not in frame.columns
    ]

    if missing:
        raise SystemExit(
            "Feature parquet is missing stuff columns "
            f"{missing}. Rebuild features with build_features.py."
        )

    package = fit_stuff_strikeout_model(
        frame,
        train_start=args.start,
        train_end=args.end,
    )
    out = save_stuff_strikeout_model(
        package,
        version=version,
    )

    metrics = package["metrics"]
    print(f"Saved stuff strikeout model → {out}")
    print(
        f"Rows: {metrics['n_rows']:,} | "
        f"R² (K% ~ stuff): {metrics['r2']:.3f} | "
        f"SwStr-only R²: {metrics['swstr_only_r2']:.3f}"
    )


if __name__ == "__main__":
    main()
