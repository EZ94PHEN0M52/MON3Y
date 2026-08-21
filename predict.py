import argparse

import pandas as pd

from game_lines import (
    build_consensus_game_lines,
    enrich_feature_row_with_game_lines,
    load_current_game_lines,
)
from training_odds import attach_consensus_to_props
from odds_aggregation import dedupe_best_prop, enrich_predictions
from odds_movement import compute_movement_features
from odds_snapshots import snapshots_dir
from prop_scoring import (
    MODEL_MAP,
    fuzzy_player_match,
    load_model,
    score_prop,
)
from distributional import (
    load_distributional_model,
    market_supports_distributional,
    market_supports_dual_head,
    score_distributional_prop,
)
from utils import (
    batter_features_path,
    normalize_version,
    predictions_best_path,
    predictions_path,
    pitcher_features_path,
    resolve_feature_path,
    warn_sp_prop_coverage,
)


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

    if not batter_path.exists():
        raise FileNotFoundError(
            f"No batter feature parquet for {start_date} → {end_date} "
            f"({version}). Run:\n"
            f"  python scripts/ensure_features.py --start {start_date} "
            f"--end {end_date} --version {version} --fix"
        )

    if not pitcher_path.exists():
        raise FileNotFoundError(
            f"No pitcher feature parquet for {start_date} → {end_date} "
            f"({version}). Run:\n"
            f"  python scripts/ensure_features.py --start {start_date} "
            f"--end {end_date} --version {version} --fix"
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

    props = attach_consensus_to_props(props)

    warn_sp_prop_coverage(
        props,
        context="before predict",
    )

    props = compute_movement_features(
        props,
        snapshots_dir(),
    )

    game_lines = load_current_game_lines()
    game_line_consensus = build_consensus_game_lines(
        game_lines
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

        row = enrich_feature_row_with_game_lines(
            row,
            prop,
            game_line_consensus,
        )

        scores = score_prop(
            prop,
            row,
            package,
            version=version,
        )

        if market_supports_distributional(
            market
        ):
            dist_package = load_distributional_model(
                market,
                version,
            )

            if dist_package is not None:
                dist_scores = score_distributional_prop(
                    prop,
                    row,
                    dist_package,
                )

                if market_supports_dual_head(
                    market
                ):
                    scores[
                        "predicted_count"
                    ] = dist_scores[
                        "predicted_count"
                    ]
                    scores[
                        "dist_over_probability"
                    ] = dist_scores[
                        "over_probability"
                    ]
                else:
                    scores[
                        "predicted_rate"
                    ] = dist_scores[
                        "predicted_count"
                    ]

        predictions.append({

            "event_id":
                prop.get("event_id"),

            "game":
                f'{prop["away_team"]} @ '
                f'{prop["home_team"]}',

            "player":
                player_name,

            "market":
                market,

            "bookmaker":
                prop["bookmaker"],

            "bookmaker_key":
                prop.get("bookmaker_key"),

            "side":
                prop["side"],

            "line":
                prop["line"],

            "odds":
                prop["odds"],

            "commence_time":
                prop["commence_time"],

            "opening_line":
                prop.get("opening_line"),

            "opening_odds":
                prop.get("opening_odds"),

            "line_delta":
                prop.get("line_delta"),

            "odds_delta":
                prop.get("odds_delta"),

            "steam_flag":
                prop.get("steam_flag"),

            **scores,
        })

    result = pd.DataFrame(
        predictions
    )

    if result.empty:

        print(
            "No predictions generated."
        )

        return result

    result = enrich_predictions(result)

    # -----------------------------------------------------
    # Sort by devigged EV (vig-aware ranking)
    # -----------------------------------------------------

    result = result.sort_values(
        "ev",
        ascending=False
    )

    output = predictions_path(
        version
    )

    result.to_csv(
        output,
        index=False
    )

    best_output = predictions_best_path(
        version
    )

    best_rows = dedupe_best_prop(result)

    best_rows.to_csv(
        best_output,
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
        "raw_model_probability",
        "calibrated_probability",
        "market_probability",
        "devigged_market_prob",
        "edge",
        "consensus_line",
        "consensus_edge",
        "best_book",
        "best_ev",
        "ev",
        "is_best_price",
        "opening_line",
        "line_delta",
        "odds_delta",
        "steam_flag",
        "predicted_count",
        "dist_over_probability",
        "predicted_rate",
    ]

    print(
        result[
            [
                column
                for column in display_columns
                if column in result.columns
            ]
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
    print(
        "Best price rows:",
        best_output,
        f"({len(best_rows)} rows)",
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
