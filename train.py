import argparse
import hashlib
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    accuracy_score
)

from features_v2 import (
    BATTER_FEATURES_V2_EXTRA,
    PITCHER_FEATURES_V2_EXTRA,
)

from training_odds import (
    DERIVED_LINE_FEATURES,
    load_historical_props,
    match_props_to_features,
)
from utils import (
    batter_features_path,
    pitcher_features_path,
    normalize_version,
    version_models_dir,
)


# =========================================================
# CONFIG
# =========================================================

BATTER_MARKETS = {

    "hits": [
        0.5,
        1.5,
        2.5
    ],

    "home_runs": [
        0.5,
        1.5
    ],

    "total_bases": [
        0.5,
        1.5,
        2.5,
        3.5,
        4.5
    ],

    "rbi": [
        0.5,
        1.5,
        2.5
    ],

    "runs": [
        0.5,
        1.5
    ],

    "walks": [
        0.5,
        1.5
    ],

    "hits_runs_rbis": [
        0.5,
        1.5,
        2.5,
        3.5
    ],

    "stolen_bases": [
        0.5,
        1.5
    ],
}


PITCHER_MARKETS = {

    "strikeouts": [
        2.5,
        3.5,
        4.5,
        5.5,
        6.5,
        7.5,
        8.5
    ],

    "walks": [
        0.5,
        1.5,
        2.5
    ],

    "hits_allowed": [
        0.5,
        1.5,
        2.5,
        3.5,
        4.5
    ],

    "outs": [
        14.5,
        15.5,
        16.5,
        17.5,
        18.5,
        19.5,
        20.5
    ],

    "earned_runs": [
        1.5,
        2.5,
        3.5,
        4.5
    ]
}


# =========================================================
# MODEL FEATURES
#
# BATTER_FEATURES / PITCHER_FEATURES must stay in sync with the
# stat lists in build_features.py (build_all_features). V2 extras
# live in features_v2.py. scripts/ensure_features.py reads required
# columns via feature_columns_for_version() below — bump
# PARQUET_FEATURE_SCHEMA_VERSION when adding or renaming parquet columns.
# Derived model inputs (DERIVED_LINE_FEATURES) are added at train/infer time
# and do not require a parquet rebuild.
# =========================================================

# Bump only when BATTER_FEATURES / PITCHER_FEATURES (or V2 extras) change.
PARQUET_FEATURE_SCHEMA_VERSION = "3"

# Metadata alias written by build_features.py (same as parquet schema version).
FEATURE_SCHEMA_VERSION = PARQUET_FEATURE_SCHEMA_VERSION

BATTER_FEATURES = [
    "hits_l3",
    "hits_l5",
    "hits_l10",
    "hits_l20",
    "hits_season",

    "home_runs_l3",
    "home_runs_l5",
    "home_runs_l10",
    "home_runs_season",

    "total_bases_l3",
    "total_bases_l5",
    "total_bases_l10",
    "total_bases_season",

    "rbi_l3",
    "rbi_l5",
    "rbi_l10",
    "rbi_season",

    "runs_l3",
    "runs_l5",
    "runs_l10",
    "runs_season",

    "walks_l3",
    "walks_l5",
    "walks_l10",
    "walks_season",

    "hits_runs_rbis_l3",
    "hits_runs_rbis_l5",
    "hits_runs_rbis_l10",
    "hits_runs_rbis_season",

    "stolen_bases_l3",
    "stolen_bases_l5",
    "stolen_bases_l10",
    "stolen_bases_season",

    "plate_appearances",

    "is_home"
]


PITCHER_FEATURES = [
    "strikeouts_l3",
    "strikeouts_l5",
    "strikeouts_l10",
    "strikeouts_l20",
    "strikeouts_season",

    "walks_l3",
    "walks_l5",
    "walks_l10",
    "walks_season",

    "hits_allowed_l3",
    "hits_allowed_l5",
    "hits_allowed_l10",
    "hits_allowed_season",

    "outs_l3",
    "outs_l5",
    "outs_l10",
    "outs_season",

    "earned_runs_l3",
    "earned_runs_l5",
    "earned_runs_l10",
    "earned_runs_season",

    "is_home"
]


BATTER_FEATURES_V2 = (
    BATTER_FEATURES +
    BATTER_FEATURES_V2_EXTRA
)


PITCHER_FEATURES_V2 = (
    PITCHER_FEATURES +
    PITCHER_FEATURES_V2_EXTRA
)


def feature_columns_for_version(
    version
):

    version = normalize_version(
        version
    )

    if version == "v1":

        return {
            "batter": BATTER_FEATURES,
            "pitcher": PITCHER_FEATURES,
        }

    return {
        "batter": BATTER_FEATURES_V2,
        "pitcher": PITCHER_FEATURES_V2,
    }


def model_feature_columns(
    version,
):
    """
    Parquet stat features plus line-derived inputs used at train/infer time.
    """

    feature_sets = feature_columns_for_version(
        version
    )

    return {
        role: (
            cols + DERIVED_LINE_FEATURES
        )
        for role, cols in feature_sets.items()
    }


def feature_schema_fingerprint(
    version,
):

    version = normalize_version(
        version
    )

    feature_sets = feature_columns_for_version(
        version
    )

    payload = (
        f"{PARQUET_FEATURE_SCHEMA_VERSION}|{version}|"
        f"batter:{','.join(feature_sets['batter'])}|"
        f"pitcher:{','.join(feature_sets['pitcher'])}"
    )

    return hashlib.sha256(
        payload.encode()
    ).hexdigest()[
        :16
    ]


def valid_parquet_fingerprints(
    version,
) -> frozenset[str]:
    """
    Fingerprints accepted for existing parquets.

    Includes legacy hashes from when FEATURE_SCHEMA_VERSION was briefly
    bumped for derived-only model inputs (Phase 3).
    """

    version = normalize_version(
        version
    )

    feature_sets = feature_columns_for_version(
        version
    )

    batter = ",".join(
        feature_sets["batter"]
    )
    pitcher = ",".join(
        feature_sets["pitcher"]
    )

    fingerprints = {
        feature_schema_fingerprint(
            version
        )
    }

    for legacy_version in (
        "2",
        "3",
    ):
        payload = (
            f"{legacy_version}|{version}|"
            f"batter:{batter}|"
            f"pitcher:{pitcher}"
        )
        fingerprints.add(
            hashlib.sha256(
                payload.encode()
            ).hexdigest()[
                :16
            ]
        )

    return frozenset(
        fingerprints
    )


def parquet_schema_is_stale(
    stored_fingerprint,
    version,
) -> bool:
    """
    True when parquet metadata fingerprint does not match current columns.

    Parquets without metadata fingerprints are not treated as stale; required
    column presence is validated separately.
    """

    if stored_fingerprint is None:
        return False

    return stored_fingerprint not in valid_parquet_fingerprints(
        version
    )


def validate_feature_columns(
    df,
    feature_columns,
    path,
    start_date,
    end_date,
    version
):

    missing = [
        col
        for col in feature_columns
        if col not in df.columns
    ]

    if not missing:

        return

    raise ValueError(
        "Feature parquet is missing columns required for training: "
        f"{missing}. "
        f"Rebuild features with:\n"
        f"  python build_features.py "
        f"--start {start_date} --end {end_date} "
        f"--version {version}\n"
        f"File: {path}"
    )


# =========================================================
# BUILD TRAINING ROWS
# =========================================================

def create_training_rows_synthetic(
    df,
    player_col,
    target,
    line_values,
    feature_columns,
):
    rows = []

    usable = df.copy()

    usable = usable[
        usable[
            f"{target}_season"
        ].notna()
    ].copy()

    for line in line_values:

        temp = usable.copy()

        temp["line"] = line

        temp["target"] = (
            temp[target] >
            line
        ).astype(int)

        temp["market"] = target

        temp["market_implied_over_prob"] = np.nan

        temp["line_vs_season_avg"] = (
            temp["line"]
            - temp[f"{target}_season"]
        )

        keep = [
            "game_date",
            player_col,
            "opponent",
            "is_home",
            target,
            "line",
            "target",
            "market",
            "market_implied_over_prob",
            "line_vs_season_avg",
        ]

        keep += feature_columns

        temp = temp[
            list(
                dict.fromkeys(
                    keep
                )
            )
        ]

        rows.append(
            temp
        )

    if not rows:

        return pd.DataFrame()

    return pd.concat(
        rows,
        ignore_index=True,
    )


def create_training_rows(
    df,
    player_col,
    target,
    line_values,
    feature_columns,
    line_source="auto",
    historical_props=None,
):
    use_real = line_source in (
        "real",
        "auto",
    )

    if use_real and historical_props is not None:

        real_rows = match_props_to_features(
            df,
            historical_props,
            player_col,
            target,
            feature_columns,
        )

        if line_source == "real":
            return real_rows

        if len(real_rows) >= 100:
            return real_rows

        if not real_rows.empty:
            print(
                f"  Real-line rows ({len(real_rows)}) "
                "below minimum; falling back to "
                "synthetic thresholds."
            )

    return create_training_rows_synthetic(
        df,
        player_col,
        target,
        line_values,
        feature_columns,
    )


# =========================================================
# TIME SPLIT
# =========================================================

def time_split(
    df
):

    df = df.sort_values(
        "game_date"
    )

    dates = (
        df["game_date"]
        .sort_values()
        .unique()
    )

    split_index = int(
        len(dates) * 0.8
    )

    split_date = dates[
        split_index
    ]

    train = df[
        df["game_date"] <
        split_date
    ].copy()

    test = df[
        df["game_date"] >=
        split_date
    ].copy()

    return train, test


# =========================================================
# TRAIN ONE MODEL
# =========================================================

def train_model(
    training_data,
    feature_columns,
    name,
    model_version="v2"
):

    training_data = (
        training_data
        .replace(
            [
                np.inf,
                -np.inf
            ],
            np.nan
        )
    )

    # -----------------------------------------------------
    # Numeric missing values
    # -----------------------------------------------------

    X = training_data[
        feature_columns +
        ["line"]
    ].copy()

    y = training_data[
        "target"
    ]

    X = X.fillna(
        X.median(
            numeric_only=True
        )
    )

    train, test = time_split(
        pd.concat(
            [
                X,
                y.reset_index(
                    drop=True
                ),
                training_data[
                    "game_date"
                ].reset_index(
                    drop=True
                )
            ],
            axis=1
        )
    )

    feature_cols = (
        feature_columns +
        ["line"]
    )

    X_train = train[
        feature_cols
    ]

    y_train = train[
        "target"
    ]

    X_test = test[
        feature_cols
    ]

    y_test = test[
        "target"
    ]

    # -----------------------------------------------------
    # LightGBM
    # -----------------------------------------------------

    model = lgb.LGBMClassifier(

        objective="binary",

        n_estimators=400,

        learning_rate=0.03,

        num_leaves=31,

        max_depth=6,

        subsample=0.85,

        colsample_bytree=0.85,

        reg_alpha=0.1,

        reg_lambda=1.0,

        random_state=42,

        verbosity=-1
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict_proba(
        X_test
    )[:, 1]

    print()
    print(
        "=" * 60
    )

    print(
        name
    )

    print(
        "=" * 60
    )

    print(
        "Rows:",
        len(training_data)
    )

    try:

        auc = roc_auc_score(
            y_test,
            predictions
        )

        print(
            "ROC AUC:",
            round(auc, 4)
        )

    except Exception:

        pass

    try:

        ll = log_loss(
            y_test,
            predictions
        )

        print(
            "Log Loss:",
            round(ll, 4)
        )

    except Exception:

        pass

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    output = (
        version_models_dir(
            model_version
        ) /
        f"{name}.pkl"
    )

    joblib.dump(
        {
            "model": model,
            "features": feature_columns
        },
        output
    )

    print(
        "Saved:",
        output
    )

    return model


# =========================================================
# MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--start",
        required=True
    )

    parser.add_argument(
        "--end",
        required=True
    )

    parser.add_argument(
        "--version",
        default="v2",
        choices=["v1", "v2"],
        help="Model version to train (default: v2)"
    )

    parser.add_argument(
        "--line-source",
        default="auto",
        choices=["real", "synthetic", "auto"],
        help=(
            "Training line source: real book consensus lines, "
            "synthetic threshold grids, or auto (real when "
            "historical props exist, else synthetic)"
        ),
    )

    args = parser.parse_args()

    version = normalize_version(
        args.version
    )

    feature_sets = feature_columns_for_version(
        version
    )

    model_features = model_feature_columns(
        version
    )

    historical_props = None

    if args.line_source in (
        "real",
        "auto",
    ):
        historical_props = load_historical_props(
            args.start,
            args.end,
        )

        if historical_props.empty:
            print()
            print(
                "No historical props found for "
                f"{args.start} → {args.end}."
            )

            if args.line_source == "real":
                print(
                    "Cannot train with --line-source real "
                    "without historical props."
                )
                return

            print(
                "Using synthetic threshold lines."
            )
        else:
            print()
            print(
                f"Loaded {len(historical_props):,} "
                "historical prop rows for training."
            )

    batter_path = batter_features_path(
        args.start,
        args.end,
        version
    )

    pitcher_path = pitcher_features_path(
        args.start,
        args.end,
        version
    )

    batters = pd.read_parquet(
        batter_path
    )

    pitchers = pd.read_parquet(
        pitcher_path
    )

    validate_feature_columns(
        batters,
        feature_sets["batter"],
        batter_path,
        args.start,
        args.end,
        version
    )

    validate_feature_columns(
        pitchers,
        feature_sets["pitcher"],
        pitcher_path,
        args.start,
        args.end,
        version
    )

    # -----------------------------------------------------
    # BATTER MODELS
    # -----------------------------------------------------

    for target, lines in (
        BATTER_MARKETS.items()
    ):

        print()
        print(
            f"Preparing {target}"
        )

        training = create_training_rows(
            batters,
            "batter",
            target,
            lines,
            feature_sets["batter"],
            line_source=args.line_source,
            historical_props=historical_props,
        )

        if len(training) < 100:

            print(
                "Not enough rows."
            )

            continue

        real_count = (
            training["market_implied_over_prob"]
            .notna()
            .sum()
        )

        print(
            f"  Training rows: {len(training):,} "
            f"({real_count:,} with real book lines)"
        )

        train_model(
            training,
            model_features["batter"],
            f"batter_{target}",
            version
        )

    # -----------------------------------------------------
    # PITCHER MODELS
    # -----------------------------------------------------

    for target, lines in (
        PITCHER_MARKETS.items()
    ):

        print()
        print(
            f"Preparing {target}"
        )

        training = create_training_rows(
            pitchers,
            "pitcher",
            target,
            lines,
            feature_sets["pitcher"],
            line_source=args.line_source,
            historical_props=historical_props,
        )

        if len(training) < 100:

            print(
                "Not enough rows."
            )

            continue

        real_count = (
            training["market_implied_over_prob"]
            .notna()
            .sum()
        )

        print(
            f"  Training rows: {len(training):,} "
            f"({real_count:,} with real book lines)"
        )

        train_model(
            training,
            model_features["pitcher"],
            f"pitcher_{target}",
            version
        )


if __name__ == "__main__":

    main()
