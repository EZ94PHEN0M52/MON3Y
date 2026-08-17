import argparse
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
    ]
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
    ]
}


# =========================================================
# MODEL FEATURES
# =========================================================

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


# =========================================================
# BUILD TRAINING ROWS
# =========================================================

def create_training_rows(
    df,
    player_col,
    target,
    line_values,
    feature_columns
):

    rows = []

    usable = df.copy()

    # -----------------------------------------------------
    # Keep only rows where we have at least some history.
    # -----------------------------------------------------

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

        keep = [
            "game_date",
            player_col,
            "opponent",
            "is_home",
            target,
            "line",
            "target",
            "market"
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

    result = pd.concat(
        rows,
        ignore_index=True
    )

    return result


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

    args = parser.parse_args()

    version = normalize_version(
        args.version
    )

    feature_sets = feature_columns_for_version(
        version
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
            feature_sets["batter"]
        )

        if len(training) < 100:

            print(
                "Not enough rows."
            )

            continue

        train_model(
            training,
            feature_sets["batter"],
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
            feature_sets["pitcher"]
        )

        if len(training) < 100:

            print(
                "Not enough rows."
            )

            continue

        train_model(
            training,
            feature_sets["pitcher"],
            f"pitcher_{target}",
            version
        )


if __name__ == "__main__":

    main()
