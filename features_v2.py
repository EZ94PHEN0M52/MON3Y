"""
V2 feature additions: opponent team strength, handedness, park proxy.

Built on top of V1 rolling player features from build_features.py.
"""

import numpy as np
import pandas as pd

from build_features import (
    HITS,
    add_rolling_features,
    build_all_features,
)
from game_lines import (
    GAME_LINE_FEATURES,
    load_historical_game_lines,
    merge_game_lines_into_features,
)


STRIKEOUT_EVENTS = {
    "strikeout",
    "strikeout_double_play",
}


def _team_pitching_daily(
    pitcher_games
):

    return (

        pitcher_games
        .groupby(
            [
                "team",
                "game_date"
            ],
            as_index=False
        )
        .agg(
            team_strikeouts=(
                "strikeouts",
                "sum"
            ),
            team_walks=(
                "walks",
                "sum"
            ),
            team_hits_allowed=(
                "hits_allowed",
                "sum"
            ),
            team_earned_runs=(
                "earned_runs",
                "sum"
            ),
            team_starts=(
                "game_pk",
                "nunique"
            )
        )
    )


def _team_batting_daily(
    batter_games
):

    return (

        batter_games
        .groupby(
            [
                "team",
                "game_date"
            ],
            as_index=False
        )
        .agg(
            team_hits=(
                "hits",
                "sum"
            ),
            team_total_bases=(
                "total_bases",
                "sum"
            ),
            team_home_runs=(
                "home_runs",
                "sum"
            ),
            team_rbi=(
                "rbi",
                "sum"
            ),
            team_runs=(
                "runs",
                "sum"
            ),
            team_plate_appearances=(
                "plate_appearances",
                "sum"
            )
        )
    )


def _team_batting_k_daily(
    statcast_df
):

    data = statcast_df[
        statcast_df["events"].notna()
    ].copy()

    data["team"] = np.where(
        data["inning_topbot"].eq("Top"),
        data["away_team"],
        data["home_team"]
    )

    data["strikeout"] = (
        data["events"]
        .isin(STRIKEOUT_EVENTS)
        .astype(int)
    )

    return (

        data
        .groupby(
            [
                "team",
                "game_date"
            ],
            as_index=False
        )
        .agg(
            team_batter_strikeouts=(
                "strikeout",
                "sum"
            ),
            team_batter_pa=(
                "events",
                "count"
            )
        )
    )


def _add_team_rolling(
    team_daily,
    team_col,
    stats
):

    data = team_daily.sort_values(
        [
            team_col,
            "game_date"
        ]
    ).copy()

    for stat in stats:

        group = (
            data
            .groupby(team_col)[stat]
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
                    min_periods=3
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
                .expanding(
                    min_periods=3
                )
                .mean()
            )
        )

    return data


def _merge_team_features(
    player_games,
    team_features,
    opponent_col="opponent"
):

    merge_cols = [
        "team",
        "game_date"
    ]

    feature_cols = [
        col
        for col in team_features.columns
        if col not in merge_cols
    ]

    renamed = team_features.rename(
        columns={
            col: f"opp_{col}"
            for col in feature_cols
        }
    )

    renamed = renamed.rename(
        columns={
            "team": opponent_col
        }
    )

    return player_games.merge(
        renamed,
        on=[
            opponent_col,
            "game_date"
        ],
        how="left"
    )


def _batter_stand_by_game(
    statcast_df
):

    data = statcast_df[
        statcast_df["events"].notna()
    ].copy()

    if "stand" not in data.columns:

        return pd.DataFrame(
            columns=[
                "game_date",
                "game_pk",
                "batter",
                "batter_stand_L"
            ]
        )

    stand = (

        data
        .groupby(
            [
                "game_date",
                "game_pk",
                "batter"
            ],
            as_index=False
        )["stand"]
        .agg(
            lambda x:
            x.mode().iloc[0]
            if len(x.mode()) > 0
            else np.nan
        )
    )

    stand["batter_stand_L"] = (
        stand["stand"]
        .eq("L")
        .astype(int)
    )

    return stand.drop(
        columns=["stand"]
    )


def _pitcher_hand_by_game(
    statcast_df
):

    data = statcast_df[
        statcast_df["events"].notna()
    ].copy()

    if "p_throws" not in data.columns:

        return pd.DataFrame(
            columns=[
                "game_date",
                "game_pk",
                "pitcher",
                "pitcher_throws_L"
            ]
        )

    throws = (

        data
        .groupby(
            [
                "game_date",
                "game_pk",
                "pitcher"
            ],
            as_index=False
        )["p_throws"]
        .agg(
            lambda x:
            x.mode().iloc[0]
            if len(x.mode()) > 0
            else np.nan
        )
    )

    throws["pitcher_throws_L"] = (
        throws["p_throws"]
        .eq("L")
        .astype(int)
    )

    return throws.drop(
        columns=["p_throws"]
    )


def _batter_vs_hand_splits(
    statcast_df,
    batter_games
):

    data = statcast_df[
        statcast_df["events"].notna()
    ].copy()

    if "p_throws" not in data.columns:

        batter_games[
            "hits_vs_lhp_season"
        ] = np.nan

        batter_games[
            "hits_vs_rhp_season"
        ] = np.nan

        return batter_games

    data["hit"] = (
        data["events"]
        .isin(HITS)
        .astype(int)
    )

    data["vs_lhp"] = (
        data["p_throws"]
        .eq("L")
        .astype(int)
    )

    daily = (

        data
        .groupby(
            [
                "game_date",
                "game_pk",
                "batter",
                "vs_lhp"
            ],
            as_index=False
        )
        .agg(
            hits=("hit", "sum")
        )
    )

    lhp = daily[
        daily["vs_lhp"].eq(1)
    ].copy()

    rhp = daily[
        daily["vs_lhp"].eq(0)
    ].copy()

    lhp = add_rolling_features(
        lhp.rename(
            columns={
                "hits":
                    "hits_vs_lhp"
            }
        ),
        "batter",
        "hits_vs_lhp"
    )[
        [
            "game_date",
            "game_pk",
            "batter",
            "hits_vs_lhp_season"
        ]
    ]

    rhp = add_rolling_features(
        rhp.rename(
            columns={
                "hits":
                    "hits_vs_rhp"
            }
        ),
        "batter",
        "hits_vs_rhp"
    )[
        [
            "game_date",
            "game_pk",
            "batter",
            "hits_vs_rhp_season"
        ]
    ]

    merged = batter_games.merge(
        lhp,
        on=[
            "game_date",
            "game_pk",
            "batter"
        ],
        how="left"
    )

    merged = merged.merge(
        rhp,
        on=[
            "game_date",
            "game_pk",
            "batter"
        ],
        how="left"
    )

    return merged


def _home_offense_daily(
    batter_games
):

    home = batter_games[
        batter_games["is_home"].eq(1)
    ].copy()

    return (

        home
        .groupby(
            [
                "team",
                "game_date"
            ],
            as_index=False
        )
        .agg(
            home_team_hits=(
                "hits",
                "sum"
            ),
            home_team_tb=(
                "total_bases",
                "sum"
            )
        )
    )


def build_all_features_v2(
    statcast_df,
    start_date=None,
    end_date=None,
):

    batters, pitchers = build_all_features(
        statcast_df
    )

    print(
        "Adding V2 opponent team features..."
    )

    team_pitching = _add_team_rolling(
        _team_pitching_daily(
            pitchers
        ),
        "team",
        [
            "team_strikeouts",
            "team_walks",
            "team_hits_allowed",
            "team_earned_runs",
        ]
    )

    team_batting = _add_team_rolling(
        _team_batting_daily(
            batters
        ),
        "team",
        [
            "team_hits",
            "team_total_bases",
            "team_home_runs",
            "team_rbi",
            "team_runs",
        ]
    )

    team_batting_k = _add_team_rolling(
        _team_batting_k_daily(
            statcast_df
        ),
        "team",
        [
            "team_batter_strikeouts",
            "team_batter_pa"
        ]
    )

    team_batting_k[
        "team_batter_k_rate_season"
    ] = (
        team_batting_k[
            "team_batter_strikeouts_season"
        ]
        /
        team_batting_k[
            "team_batter_pa_season"
        ].replace(0, np.nan)
    )

    batters = _merge_team_features(
        batters,
        team_pitching
    )

    batters = batters.merge(
        _batter_stand_by_game(
            statcast_df
        ),
        on=[
            "game_date",
            "game_pk",
            "batter"
        ],
        how="left"
    )

    batters = _batter_vs_hand_splits(
        statcast_df,
        batters
    )

    home_offense = _add_team_rolling(
        _home_offense_daily(
            batters
        ),
        "team",
        [
            "home_team_hits",
            "home_team_tb"
        ]
    )

    batters = batters.merge(
        home_offense.rename(
            columns={
                "team": "team",
                "home_team_hits_season":
                    "park_home_hits_season",
                "home_team_tb_season":
                    "park_home_tb_season"
            }
        )[
            [
                "team",
                "game_date",
                "park_home_hits_season",
                "park_home_tb_season"
            ]
        ],
        on=[
            "team",
            "game_date"
        ],
        how="left"
    )

    pitchers = _merge_team_features(
        pitchers,
        team_batting
    )

    pitchers = _merge_team_features(
        pitchers,
        team_batting_k[
            [
                "team",
                "game_date",
                "team_batter_k_rate_season"
            ]
        ]
    )

    pitchers = pitchers.merge(
        _pitcher_hand_by_game(
            statcast_df
        ),
        on=[
            "game_date",
            "game_pk",
            "pitcher"
        ],
        how="left"
    )

    if start_date is None:
        start_date = (
            pd.to_datetime(
                statcast_df["game_date"]
            )
            .min()
            .strftime("%Y-%m-%d")
        )

    if end_date is None:
        end_date = (
            pd.to_datetime(
                statcast_df["game_date"]
            )
            .max()
            .strftime("%Y-%m-%d")
        )

    print(
        "Merging game line context features..."
    )

    game_lines = load_historical_game_lines(
        start_date,
        end_date,
    )

    batters = merge_game_lines_into_features(
        batters,
        game_lines,
    )

    pitchers = merge_game_lines_into_features(
        pitchers,
        game_lines,
    )

    return batters, pitchers


BATTER_FEATURES_V2_EXTRA = [
    "opp_team_strikeouts_season",
    "opp_team_walks_season",
    "opp_team_hits_allowed_season",
    "opp_team_earned_runs_season",
    "batter_stand_L",
    "hits_vs_lhp_season",
    "hits_vs_rhp_season",
    "park_home_hits_season",
    "park_home_tb_season",
    *GAME_LINE_FEATURES,
]


PITCHER_FEATURES_V2_EXTRA = [
    "opp_team_hits_season",
    "opp_team_total_bases_season",
    "opp_team_home_runs_season",
    "opp_team_rbi_season",
    "opp_team_runs_season",
    "opp_team_batter_k_rate_season",
    "pitcher_throws_L",
    *GAME_LINE_FEATURES,
]
