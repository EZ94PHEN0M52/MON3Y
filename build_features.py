import argparse

import numpy as np
import pandas as pd
from pybaseball import playerid_reverse_lookup

from utils import (
    RAW_DIR,
    batter_features_path,
    pitcher_features_path,
    normalize_version,
)


# =========================================================
# LOAD RAW STATCAST
# =========================================================

def load_statcast(
    start_date,
    end_date
):

    path = (
        RAW_DIR /
        f"statcast_{start_date}_{end_date}.parquet"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Could not find {path}"
        )

    return pd.read_parquet(
        path
    )


# =========================================================
# BASIC BATTER OUTCOMES
# =========================================================

HITS = {
    "single",
    "double",
    "triple",
    "home_run"
}


EXTRA_BASES = {
    "single": 1,
    "double": 2,
    "triple": 3,
    "home_run": 4
}


SCORING_EVENTS = HITS | {
    "sac_fly",
    "sac_fly_double_play",
}


def batter_id_to_name(
    batter_ids
):

    ids = (
        pd.Series(batter_ids)
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    if not ids:

        return {}

    lookup = playerid_reverse_lookup(
        ids,
        key_type="mlbam"
    )

    lookup["full_name"] = (
        lookup["name_first"]
        .str.title()
        + " "
        + lookup["name_last"]
        .str.title()
    )

    return dict(
        zip(
            lookup["key_mlbam"],
            lookup["full_name"]
        )
    )


def normalize_player_name(
    name
):

    if not isinstance(
        name,
        str
    ):
        return name

    if ", " in name:

        last, first = name.split(
            ", ",
            1
        )

        return (
            f"{first} {last}"
        )

    return name


def derive_rbi(
    data
):

    if "rbi" in data.columns:

        return (
            pd.to_numeric(
                data["rbi"],
                errors="coerce"
            )
            .fillna(0)
        )

    runs_on_play = (
        pd.to_numeric(
            data["post_bat_score"],
            errors="coerce"
        )
        - pd.to_numeric(
            data["bat_score"],
            errors="coerce"
        )
    ).fillna(0).clip(
        lower=0
    )

    return np.where(
        data["events"].isin(
            SCORING_EVENTS
        ),
        runs_on_play,
        0
    )


# =========================================================
# BUILD BATTER GAME LOG
# =========================================================

def build_batter_games(
    df
):

    data = df.copy()

    data = data[
        data["events"].notna()
    ].copy()

    # -----------------------------------------------------
    # Hits
    # -----------------------------------------------------

    data["hit"] = (
        data["events"]
        .isin(HITS)
        .astype(int)
    )

    # -----------------------------------------------------
    # Home runs
    # -----------------------------------------------------

    data["home_run"] = (
        data["events"]
        .eq("home_run")
        .astype(int)
    )

    # -----------------------------------------------------
    # Total bases
    # -----------------------------------------------------

    data["total_bases"] = (
        data["events"]
        .map(EXTRA_BASES)
        .fillna(0)
    )

    # -----------------------------------------------------
    # RBI
    # -----------------------------------------------------

    data["rbi_clean"] = derive_rbi(
        data
    )

    # -----------------------------------------------------
    # Batter name (Statcast player_name is the pitcher)
    # -----------------------------------------------------

    name_map = batter_id_to_name(
        data["batter"]
    )

    data["player_name"] = (
        data["batter"]
        .map(name_map)
        .map(normalize_player_name)
    )

    data = data[
        data["player_name"].notna()
    ].copy()

    # -----------------------------------------------------
    # Runs scored by batter
    #
    # We'll use the batter_runs field if available.
    # -----------------------------------------------------

    if "bat_score" in data.columns:

        data["run_proxy"] = 0

    else:

        data["run_proxy"] = 0

    # -----------------------------------------------------
    # Team / opponent
    # -----------------------------------------------------

    data["team"] = np.where(
        data["inning_topbot"].eq("Top"),
        data["away_team"],
        data["home_team"]
    )

    data["opponent"] = np.where(
        data["inning_topbot"].eq("Top"),
        data["home_team"],
        data["away_team"]
    )

    data["is_home"] = (
        data["team"] ==
        data["home_team"]
    ).astype(int)

    # -----------------------------------------------------
    # Aggregate
    # -----------------------------------------------------

    result = (

        data.groupby(
            [
                "game_date",
                "game_pk",
                "batter",
                "player_name",
                "team",
                "opponent",
                "is_home"
            ],
            as_index=False
        )

        .agg(

            hits=(
                "hit",
                "sum"
            ),

            home_runs=(
                "home_run",
                "sum"
            ),

            total_bases=(
                "total_bases",
                "sum"
            ),

            rbi=(
                "rbi_clean",
                "sum"
            ),

            plate_appearances=(
                "events",
                "count"
            )
        )
    )

    return result


# =========================================================
# PITCHER GAME LOG
# =========================================================

def build_pitcher_games(
    df
):

    data = df.copy()

    data = data[
        data["events"].notna()
    ].copy()

    # -----------------------------------------------------
    # Strikeouts
    # -----------------------------------------------------

    data["strikeout"] = (
        data["events"]
        .isin([
            "strikeout",
            "strikeout_double_play"
        ])
        .astype(int)
    )

    # -----------------------------------------------------
    # Walks
    # -----------------------------------------------------

    data["walk"] = (
        data["events"]
        .isin([
            "walk",
            "intent_walk"
        ])
        .astype(int)
    )

    # -----------------------------------------------------
    # Hits allowed
    # -----------------------------------------------------

    data["hit_allowed"] = (
        data["events"]
        .isin(HITS)
        .astype(int)
    )

    # -----------------------------------------------------
    # Pitcher team
    # -----------------------------------------------------

    data["team"] = np.where(
        data["inning_topbot"].eq("Top"),
        data["home_team"],
        data["away_team"]
    )

    data["opponent"] = np.where(
        data["inning_topbot"].eq("Top"),
        data["away_team"],
        data["home_team"]
    )

    data["is_home"] = (
        data["team"] ==
        data["home_team"]
    ).astype(int)

    data["player_name"] = (
        data["player_name"]
        .map(normalize_player_name)
    )

    # -----------------------------------------------------
    # Aggregate
    # -----------------------------------------------------

    result = (

        data.groupby(
            [
                "game_date",
                "game_pk",
                "pitcher",
                "player_name",
                "team",
                "opponent",
                "is_home"
            ],
            as_index=False
        )

        .agg(

            strikeouts=(
                "strikeout",
                "sum"
            ),

            walks=(
                "walk",
                "sum"
            ),

            hits_allowed=(
                "hit_allowed",
                "sum"
            )
        )
    )

    return result


# =========================================================
# ROLLING FEATURES
# =========================================================

def add_rolling_features(
    df,
    player_col,
    stat
):

    data = df.copy()

    data = data.sort_values(
        [
            player_col,
            "game_date"
        ]
    )

    group = (
        data
        .groupby(player_col)[stat]
    )

    # -----------------------------------------------------
    # IMPORTANT:
    #
    # shift(1) means today's game is excluded.
    # This prevents target leakage.
    # -----------------------------------------------------

    data[
        f"{stat}_l3"
    ] = (
        group
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                3,
                min_periods=1
            )
            .mean()
        )
    )

    data[
        f"{stat}_l5"
    ] = (
        group
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                5,
                min_periods=1
            )
            .mean()
        )
    )

    data[
        f"{stat}_l10"
    ] = (
        group
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                10,
                min_periods=1
            )
            .mean()
        )
    )

    data[
        f"{stat}_l20"
    ] = (
        group
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                20,
                min_periods=1
            )
            .mean()
        )
    )

    data[
        f"{stat}_season"
    ] = (
        group
        .transform(
            lambda x:
            x.shift(1)
            .expanding()
            .mean()
        )
    )

    return data


# =========================================================
# BUILD ALL FEATURES
# =========================================================

def build_all_features(
    statcast_df
):

    print(
        "Building batter game logs..."
    )

    batters = build_batter_games(
        statcast_df
    )

    print(
        "Building pitcher game logs..."
    )

    pitchers = build_pitcher_games(
        statcast_df
    )

    # -----------------------------------------------------
    # Batter rolling stats
    # -----------------------------------------------------

    batter_stats = [
        "hits",
        "home_runs",
        "total_bases",
        "rbi"
    ]

    for stat in batter_stats:

        batters = add_rolling_features(
            batters,
            "batter",
            stat
        )

    # -----------------------------------------------------
    # Pitcher rolling stats
    # -----------------------------------------------------

    pitcher_stats = [
        "strikeouts",
        "walks",
        "hits_allowed"
    ]

    for stat in pitcher_stats:

        pitchers = add_rolling_features(
            pitchers,
            "pitcher",
            stat
        )

    return batters, pitchers


# =========================================================
# SAVE
# =========================================================

def save_features(
    batters,
    pitchers,
    start_date,
    end_date,
    version="v2"
):

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

    batters.to_parquet(
        batter_path,
        index=False
    )

    pitchers.to_parquet(
        pitcher_path,
        index=False
    )

    print(
        "Saved:",
        batter_path
    )

    print(
        "Saved:",
        pitcher_path
    )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

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
        help="Feature set version (default: v2)"
    )

    args = parser.parse_args()

    version = normalize_version(
        args.version
    )

    raw = load_statcast(
        args.start,
        args.end
    )

    if version == "v1":

        batters, pitchers = (
            build_all_features(
                raw
            )
        )

    else:

        from features_v2 import (
            build_all_features_v2
        )

        batters, pitchers = (
            build_all_features_v2(
                raw
            )
        )

    save_features(
        batters,
        pitchers,
        args.start,
        args.end,
        version
    )

    print()
    print(
        f"Batters: {len(batters):,}"
    )

    print(
        f"Pitchers: {len(pitchers):,}"
    )
