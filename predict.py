import argparse

import joblib
import numpy as np
import pandas as pd

from utils import (
    batter_features_path,
    pitcher_features_path,
    predictions_path,
    normalize_version,
    version_models_dir,
    american_to_implied_probability,
    expected_value
)


# =========================================================
# MARKET → MODEL
# =========================================================

MODEL_MAP = {

    "batter_hits":
        "batter_hits.pkl",

    "batter_home_runs":
        "batter_home_runs.pkl",

    "batter_total_bases":
        "batter_total_bases.pkl",

    "batter_rbis":
        "batter_rbi.pkl",

    "batter_runs_scored":
        "batter_runs.pkl",

    "batter_walks":
        "batter_walks.pkl",

    "batter_hits_runs_rbis":
        "batter_hits_runs_rbis.pkl",

    "pitcher_strikeouts":
        "pitcher_strikeouts.pkl",

    "pitcher_walks":
        "pitcher_walks.pkl",

    "pitcher_hits_allowed":
        "pitcher_hits_allowed.pkl",

    "pitcher_outs":
        "pitcher_outs.pkl",

    "pitcher_earned_runs":
        "pitcher_earned_runs.pkl"
}


# =========================================================
# LOAD MODELS
# =========================================================

def load_model(
    filename,
    version="v2"
):

    path = (
        version_models_dir(
            version
        ) /
        filename
    )

    if not path.exists():

        return None

    return joblib.load(
        path
    )


# =========================================================
# PREPARE CURRENT BOARD
# =========================================================

def prepare_board(
    start_date,
    end_date,
    version="v2"
):

    props_path = (
        batter_features_path(
            start_date,
            end_date,
            version
        ).parent /
        "current_props.parquet"
    )

    batter_path = batter_features_path(
        start_date,
        end_date,
        version
    )

    pitcher_path = pitcher_features_path(
        start_date,
        end_date,
        version
    )

    props = pd.read_parquet(
        props_path
    )

    batters = pd.read_parquet(
        batter_path
    )

    pitchers = pd.read_parquet(
        pitcher_path
    )

    # -----------------------------------------------------
    # Current prediction date = latest available game
    # in the feature dataset.
    # -----------------------------------------------------

    batter_latest = (
        batters
        .sort_values("game_date")
        .groupby("batter")
        .tail(1)
        .copy()
    )

    pitcher_latest = (
        pitchers
        .sort_values("game_date")
        .groupby("pitcher")
        .tail(1)
        .copy()
    )

    return (
        props,
        batter_latest,
        pitcher_latest
    )


# =========================================================
# MATCH PLAYER
# =========================================================

def fuzzy_player_match(
    player_name,
    candidates
):

    if not isinstance(
        player_name,
        str
    ):

        return None

    name = (
        player_name
        .lower()
        .strip()
    )

    exact = candidates[
        candidates
        .str.lower()
        .str.strip()
        .eq(name)
    ]

    if len(exact) > 0:

        return exact.iloc[0]

    # Basic last-name fallback.
    last_name = name.split()[-1]

    matches = candidates[
        candidates
        .str.lower()
        .str.contains(
            last_name,
            na=False
        )
    ]

    if len(matches) > 0:

        return matches.iloc[0]

    return None


# =========================================================
# PREDICT
# =========================================================

def generate_predictions(
    start_date,
    end_date,
    version="v2"
):

    version = normalize_version(
        version
    )

    (
        props,
        batters,
        pitchers
    ) = prepare_board(
        start_date,
        end_date,
        version
    )

    predictions = []

    batter_names = (
        batters["player_name"]
        .dropna()
        .drop_duplicates()
    )

    pitcher_names = (
        pitchers["player_name"]
        .dropna()
        .drop_duplicates()
    )

    # -----------------------------------------------------
    # Each sportsbook outcome
    # -----------------------------------------------------

    for _, prop in props.iterrows():

        market = prop["market"]

        model_filename = (
            MODEL_MAP.get(
                market
            )
        )

        if not model_filename:

            continue

        package = load_model(
            model_filename,
            version
        )

        if package is None:

            continue

        player_name = prop[
            "player"
        ]

        # -------------------------------------------------
        # Batter
        # -------------------------------------------------

        if market.startswith(
            "batter_"
        ):

            match = fuzzy_player_match(
                player_name,
                batter_names
            )

            if match is None:

                continue

            row = batters[
                batters["player_name"]
                == match
            ].iloc[0]

        # -------------------------------------------------
        # Pitcher
        # -------------------------------------------------

        else:

            match = fuzzy_player_match(
                player_name,
                pitcher_names
            )

            if match is None:

                continue

            row = pitchers[
                pitchers["player_name"]
                == match
            ].iloc[0]

        # -------------------------------------------------
        # Build features
        # -------------------------------------------------

        features = package[
            "features"
        ]

        values = {}

        for feature in features:

            values[
                feature
            ] = row.get(
                feature,
                np.nan
            )

        values["line"] = prop[
            "line"
        ]

        X = pd.DataFrame(
            [values]
        )

        X = X.replace(
            [
                np.inf,
                -np.inf
            ],
            np.nan
        )

        X = X.fillna(0)

        proba = package["model"].predict_proba(X)[0]
        over_probability = float(proba[1])
        under_probability = float(proba[0])

        side = str(prop["side"]).strip().lower()
        model_probability = (
            over_probability
            if side == "over"
            else under_probability
        )

        # -------------------------------------------------
        # Market probability
        # -------------------------------------------------

        market_probability = (
            american_to_implied_probability(
                prop["odds"]
            )
        )

        edge = (
            model_probability -
            market_probability
        )

        ev = expected_value(
            model_probability,
            prop["odds"]
        )

        predictions.append({

            "game":
                f'{prop["away_team"]} @ '
                f'{prop["home_team"]}',

            "player":
                player_name,

            "market":
                market,

            "bookmaker":
                prop["bookmaker"],

            "side":
                prop["side"],

            "line":
                prop["line"],

            "odds":
                prop["odds"],

            "over_probability":
                over_probability,

            "under_probability":
                under_probability,

            "model_probability":
                model_probability,

            "market_probability":
                market_probability,

            "edge":
                edge,

            "ev":
                ev,

            "commence_time":
                prop["commence_time"]
        })

    result = pd.DataFrame(
        predictions
    )

    if result.empty:

        print(
            "No predictions generated."
        )

        return result

    # -----------------------------------------------------
    # Sort by model edge
    # -----------------------------------------------------

    result = result.sort_values(
        "edge",
        ascending=False
    )

    output = predictions_path(
        version
    )

    result.to_csv(
        output,
        index=False
    )

    print()
    print(
        "=" * 80
    )

    print(
        "TOP MLB PROP EDGES"
    )

    print(
        "=" * 80
    )

    display_columns = [
        "player",
        "market",
        "bookmaker",
        "side",
        "line",
        "odds",
        "over_probability",
        "under_probability",
        "model_probability",
        "market_probability",
        "edge",
        "ev",
    ]

    print(
        result[
            display_columns
        ]
        .head(25)
        .to_string(
            index=False
        )
    )

    print()
    print(
        "Saved:",
        output
    )

    return result


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--start",
        default="2026-03-25",
        help="Feature file start date (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--end",
        default="2026-08-16",
        help="Feature file end date (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--version",
        default="v2",
        choices=["v1", "v2"],
        help="Model version to use (default: v2)"
    )

    args = parser.parse_args()

    generate_predictions(
        args.start,
        args.end,
        args.version
    )
