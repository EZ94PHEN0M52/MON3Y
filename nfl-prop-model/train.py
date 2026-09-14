"""Train LightGBM Over/Under classifiers for NFL prop markets (V1 / V2)."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from features_v2 import v2_feature_column_names
from utils import (
    normalize_version,
    player_features_path,
    version_models_dir,
)

MARKET_STAT = {
    "player_pass_yds": "passing_yards",
    "player_pass_tds": "passing_tds",
    "player_rush_yds": "rushing_yards",
    "player_receptions": "receptions",
    "player_reception_yds": "receiving_yards",
    "player_rush_reception_yds": "rush_reception_yards",
}

MARKET_LINES = {
    "player_pass_yds": [199.5, 224.5, 249.5, 274.5, 299.5],
    "player_pass_tds": [0.5, 1.5, 2.5],
    "player_rush_yds": [39.5, 54.5, 69.5, 84.5, 99.5],
    "player_receptions": [2.5, 3.5, 4.5, 5.5, 6.5],
    "player_reception_yds": [24.5, 39.5, 54.5, 69.5, 84.5],
    "player_rush_reception_yds": [39.5, 54.5, 69.5, 84.5, 99.5],
}

MARKET_POSITIONS = {
    "player_pass_yds": {"QB"},
    "player_pass_tds": {"QB"},
    "player_rush_yds": {"RB", "QB", "WR", "HB", "FB"},
    "player_receptions": {"WR", "TE", "RB", "HB", "FB"},
    "player_reception_yds": {"WR", "TE", "RB", "HB", "FB"},
    "player_rush_reception_yds": {"RB", "WR", "TE", "HB", "FB", "QB"},
}

STAT_COLUMNS = [
    "passing_yards",
    "passing_tds",
    "rushing_yards",
    "receptions",
    "receiving_yards",
    "rush_reception_yards",
    "targets",
    "attempts",
    "carries",
]

ROLL_WINDOWS = (3, 5)


def rolling_feature_names() -> list[str]:
    cols = []
    for stat in STAT_COLUMNS:
        for window in ROLL_WINDOWS:
            cols.append(f"{stat}_l{window}")
        cols.append(f"{stat}_season")
    return cols


FEATURE_COLUMNS_V1 = rolling_feature_names() + [
    "is_home",
    "is_inactive",
    "is_questionable",
    "line",
    "line_vs_season_avg",
]

FEATURE_COLUMNS_V2 = FEATURE_COLUMNS_V1 + v2_feature_column_names()
FEATURE_COLUMNS = FEATURE_COLUMNS_V1


def feature_columns_for_version(version: str) -> list[str]:
    version = normalize_version(version)
    if version == "v2":
        return list(FEATURE_COLUMNS_V2)
    return list(FEATURE_COLUMNS_V1)


def create_training_rows(
    features: pd.DataFrame,
    market: str,
) -> pd.DataFrame:
    target = MARKET_STAT[market]
    lines = MARKET_LINES[market]
    positions = MARKET_POSITIONS.get(market)

    usable = features.copy()
    season_col = f"{target}_season"
    usable = usable[usable[season_col].notna()].copy()

    if positions and "position" in usable.columns:
        usable = usable[usable["position"].isin(positions)].copy()

    if "is_inactive" in usable.columns:
        usable = usable[usable["is_inactive"].fillna(0).astype(int).eq(0)].copy()

    rows = []
    for line in lines:
        temp = usable.copy()
        temp["line"] = line
        temp["target"] = (temp[target] > line).astype(int)
        temp["market"] = market
        temp["line_vs_season_avg"] = temp["line"] - temp[season_col]
        rows.append(temp)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def train_market(
    features: pd.DataFrame,
    market: str,
    models_dir: Path,
    feature_cols: list[str],
) -> dict:
    train_df = create_training_rows(features, market)
    if train_df.empty or train_df["target"].nunique() < 2:
        print(f"SKIP {market}: insufficient training rows")
        return {}

    cols = [c for c in feature_cols if c in train_df.columns]
    x = train_df[cols].astype(float)
    y = train_df["target"].astype(int)

    last_season = int(train_df["season"].max())
    train_mask = train_df["season"] < last_season
    if train_mask.sum() < 200 or (~train_mask).sum() < 50:
        cut = int(len(train_df) * 0.8)
        train_mask = pd.Series(False, index=train_df.index)
        train_mask.iloc[:cut] = True

    x_train, y_train = x.loc[train_mask], y.loc[train_mask]
    x_valid, y_valid = x.loc[~train_mask], y.loc[~train_mask]

    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        verbose=-1,
    )
    model.fit(x_train, y_train)

    proba = model.predict_proba(x_valid)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "market": market,
        "n_train": int(len(x_train)),
        "n_valid": int(len(x_valid)),
        "accuracy": float(accuracy_score(y_valid, pred)),
        "log_loss": float(log_loss(y_valid, proba, labels=[0, 1])),
    }
    try:
        metrics["auc"] = float(roc_auc_score(y_valid, proba))
    except ValueError:
        metrics["auc"] = float("nan")

    out_path = models_dir / f"{market}.pkl"
    joblib.dump(
        {
            "model": model,
            "feature_columns": cols,
            "market": market,
            "target_stat": MARKET_STAT[market],
            "metrics": metrics,
        },
        out_path,
    )

    print(
        f"{market}: train={metrics['n_train']:,} valid={metrics['n_valid']:,} "
        f"acc={metrics['accuracy']:.3f} auc={metrics['auc']:.3f} → {out_path.name}"
    )
    return metrics


def train_all(version: str = "v1", seasons: list[int] | None = None) -> None:
    version = normalize_version(version)

    path = player_features_path(version)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run:\n"
            f"  python build_features.py --seasons 2024 2025 2026 --version {version}"
        )

    features = pd.read_parquet(path)
    if seasons:
        features = features[features["season"].isin(seasons)].copy()

    models_dir = version_models_dir(version)
    feature_cols = feature_columns_for_version(version)

    print()
    print("=" * 60)
    print(f"TRAINING {version.upper()} MODELS → {models_dir}")
    print("=" * 60)

    for market in MARKET_STAT:
        train_market(features, market, models_dir, feature_cols)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NFL prop models.")
    parser.add_argument("--version", default="v1", choices=["v1", "v2"])
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=None,
        help="Optional season filter on the feature parquet.",
    )
    args = parser.parse_args()
    train_all(version=args.version, seasons=args.seasons)


if __name__ == "__main__":
    main()
