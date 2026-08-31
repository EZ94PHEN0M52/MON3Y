"""
Pitch-level Statcast metrics for pitcher strikeout modeling.

Derives SwStr%, CSW%, chase (O-Swing%), whiff%, and velocity from raw
Statcast pitch rows, then aggregates to pitcher-game logs suitable for
rolling features in build_features.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SWING_DESCRIPTIONS = frozenset({
    "swinging_strike",
    "swinging_strike_blocked",
    "foul",
    "foul_tip",
    "hit_into_play",
    "missed_bunt",
    "foul_bunt",
    "bunt_foul_tip",
})

SWINGING_STRIKE_DESCRIPTIONS = frozenset({
    "swinging_strike",
    "swinging_strike_blocked",
})

CALLED_STRIKE_DESCRIPTIONS = frozenset({
    "called_strike",
})

STUFF_GAME_GROUP_COLS = [
    "game_date",
    "game_pk",
    "pitcher",
]


def _normalize_description(
    series: pd.Series,
) -> pd.Series:

    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )


def pitch_in_zone(
    plate_x: pd.Series,
    plate_z: pd.Series,
    sz_top: pd.Series,
    sz_bot: pd.Series,
) -> pd.Series:

    x = pd.to_numeric(
        plate_x,
        errors="coerce",
    )
    z = pd.to_numeric(
        plate_z,
        errors="coerce",
    )
    top = pd.to_numeric(
        sz_top,
        errors="coerce",
    )
    bot = pd.to_numeric(
        sz_bot,
        errors="coerce",
    )

    has_sz = top.notna() & bot.notna()

    in_custom = (
        x.abs().le(0.83)
        & z.ge(bot)
        & z.le(top)
    )

    fallback = (
        x.abs().le(0.83)
        & z.ge(1.5)
        & z.le(3.5)
    )

    return np.where(
        has_sz,
        in_custom,
        fallback,
    )


def flag_pitch_stuff(
    df: pd.DataFrame,
) -> pd.DataFrame:

    data = df.copy()
    desc = _normalize_description(
        data["description"]
    )

    data["is_swing"] = desc.isin(
        SWING_DESCRIPTIONS
    )
    data["is_swinging_strike"] = desc.isin(
        SWINGING_STRIKE_DESCRIPTIONS
    )
    data["is_called_strike"] = desc.isin(
        CALLED_STRIKE_DESCRIPTIONS
    )

    data["in_zone"] = pd.Series(
        pitch_in_zone(
            data["plate_x"],
            data["plate_z"],
            data["sz_top"],
            data["sz_bot"],
        ),
        index=data.index,
    ).fillna(False).astype(bool)

    data["outside_zone"] = ~data["in_zone"]
    data["chase_swing"] = (
        data["is_swing"]
        & data["outside_zone"]
    )

    data["release_speed"] = pd.to_numeric(
        data["release_speed"],
        errors="coerce",
    )

    return data


def build_pitcher_stuff_games(
    statcast_df: pd.DataFrame,
) -> pd.DataFrame:

    required = {
        "game_date",
        "game_pk",
        "pitcher",
        "description",
        "plate_x",
        "plate_z",
        "sz_top",
        "sz_bot",
        "release_speed",
    }

    missing = required - set(
        statcast_df.columns
    )

    if missing:
        raise ValueError(
            "Statcast frame missing columns for stuff metrics: "
            f"{sorted(missing)}"
        )

    data = flag_pitch_stuff(
        statcast_df
    )

    grouped = (
        data
        .groupby(
            STUFF_GAME_GROUP_COLS,
            as_index=False,
        )
        .agg(
            pitches=(
                "description",
                "size",
            ),
            swinging_strikes=(
                "is_swinging_strike",
                "sum",
            ),
            called_strikes=(
                "is_called_strike",
                "sum",
            ),
            swings=(
                "is_swing",
                "sum",
            ),
            pitches_outside_zone=(
                "outside_zone",
                "sum",
            ),
            chase_swings=(
                "chase_swing",
                "sum",
            ),
            release_speed_sum=(
                "release_speed",
                "sum",
            ),
            release_speed_count=(
                "release_speed",
                "count",
            ),
            max_velocity=(
                "release_speed",
                "max",
            ),
        )
    )

    grouped["swstr_pct"] = (
        grouped["swinging_strikes"]
        / grouped["pitches"]
    )
    grouped["csw_pct"] = (
        (
            grouped["swinging_strikes"]
            + grouped["called_strikes"]
        )
        / grouped["pitches"]
    )
    chase_swings = grouped["chase_swings"].to_numpy(
        dtype=float,
    )
    pitches_outside = grouped[
        "pitches_outside_zone"
    ].to_numpy(dtype=float)
    grouped["chase_pct"] = np.divide(
        chase_swings,
        pitches_outside,
        out=np.full(
            len(grouped),
            np.nan,
        ),
        where=pitches_outside > 0,
    )

    swinging = grouped["swinging_strikes"].to_numpy(
        dtype=float,
    )
    swings = grouped["swings"].to_numpy(
        dtype=float,
    )
    grouped["whiff_pct"] = np.divide(
        swinging,
        swings,
        out=np.full(
            len(grouped),
            np.nan,
        ),
        where=swings > 0,
    )

    vel_sum = grouped["release_speed_sum"].to_numpy(
        dtype=float,
    )
    vel_count = grouped[
        "release_speed_count"
    ].to_numpy(dtype=float)
    grouped["avg_velocity"] = np.divide(
        vel_sum,
        vel_count,
        out=np.full(
            len(grouped),
            np.nan,
        ),
        where=vel_count > 0,
    )

    return grouped


def add_rolling_rate_features(
    df: pd.DataFrame,
    player_col: str,
    numerator_col: str,
    denominator_col: str,
    prefix: str,
) -> pd.DataFrame:

    data = df.copy()

    data = data.sort_values(
        [
            player_col,
            "game_date",
        ]
    )

    num_group = (
        data
        .groupby(player_col)[numerator_col]
    )
    den_group = (
        data
        .groupby(player_col)[denominator_col]
    )

    windows = {
        "l3": 3,
        "l5": 5,
        "l10": 10,
        "l20": 20,
    }

    for suffix, window in windows.items():

        num_roll = num_group.transform(
            lambda x, w=window:
            x.shift(1)
            .rolling(
                w,
                min_periods=1,
            )
            .sum()
        )
        den_roll = den_group.transform(
            lambda x, w=window:
            x.shift(1)
            .rolling(
                w,
                min_periods=1,
            )
            .sum()
        )

        data[f"{prefix}_{suffix}"] = np.where(
            den_roll.gt(0),
            num_roll / den_roll,
            np.nan,
        )

    num_season = num_group.transform(
        lambda x:
        x.shift(1)
        .expanding()
        .sum()
    )
    den_season = den_group.transform(
        lambda x:
        x.shift(1)
        .expanding()
        .sum()
    )

    data[f"{prefix}_season"] = np.where(
        den_season.gt(0),
        num_season / den_season,
        np.nan,
    )

    return data


def add_rolling_weighted_mean(
    df: pd.DataFrame,
    player_col: str,
    value_col: str,
    weight_col: str,
    prefix: str,
) -> pd.DataFrame:

    data = df.copy()
    data["_weighted_value"] = (
        pd.to_numeric(
            data[value_col],
            errors="coerce",
        )
        * pd.to_numeric(
            data[weight_col],
            errors="coerce",
        )
    )

    data = data.sort_values(
        [
            player_col,
            "game_date",
        ]
    )

    value_group = (
        data
        .groupby(player_col)["_weighted_value"]
    )
    weight_group = (
        data
        .groupby(player_col)[weight_col]
    )

    windows = {
        "l3": 3,
        "l5": 5,
        "l10": 10,
        "l20": 20,
    }

    for suffix, window in windows.items():

        value_roll = value_group.transform(
            lambda x, w=window:
            x.shift(1)
            .rolling(
                w,
                min_periods=1,
            )
            .sum()
        )
        weight_roll = weight_group.transform(
            lambda x, w=window:
            x.shift(1)
            .rolling(
                w,
                min_periods=1,
            )
            .sum()
        )

        data[f"{prefix}_{suffix}"] = np.where(
            weight_roll.gt(0),
            value_roll / weight_roll,
            np.nan,
        )

    value_season = value_group.transform(
        lambda x:
        x.shift(1)
        .expanding()
        .sum()
    )
    weight_season = weight_group.transform(
        lambda x:
        x.shift(1)
        .expanding()
        .sum()
    )

    data[f"{prefix}_season"] = np.where(
        weight_season.gt(0),
        value_season / weight_season,
        np.nan,
    )

    return data.drop(
        columns=["_weighted_value"]
    )


def add_rolling_max_features(
    df: pd.DataFrame,
    player_col: str,
    stat: str,
    prefix: str | None = None,
) -> pd.DataFrame:

    if prefix is None:
        prefix = stat

    data = df.copy()

    data = data.sort_values(
        [
            player_col,
            "game_date",
        ]
    )

    group = (
        data
        .groupby(player_col)[stat]
    )

    windows = {
        "l3": 3,
        "l5": 5,
        "l10": 10,
        "l20": 20,
    }

    for suffix, window in windows.items():

        data[f"{prefix}_{suffix}"] = group.transform(
            lambda x, w=window:
            x.shift(1)
            .rolling(
                w,
                min_periods=1,
            )
            .max()
        )

    data[f"{prefix}_season"] = group.transform(
        lambda x:
        x.shift(1)
        .expanding()
        .max()
    )

    return data


def merge_stuff_into_pitcher_games(
    pitcher_games: pd.DataFrame,
    stuff_games: pd.DataFrame,
) -> pd.DataFrame:

    stuff = stuff_games.copy()
    stuff["csw_numerator"] = (
        stuff["swinging_strikes"]
        + stuff["called_strikes"]
    )

    merged = pitcher_games.merge(
        stuff[
            STUFF_GAME_GROUP_COLS
            + [
                "pitches",
                "swinging_strikes",
                "csw_numerator",
                "chase_swings",
                "pitches_outside_zone",
                "swings",
                "avg_velocity",
                "max_velocity",
                "release_speed_count",
            ]
        ],
        on=STUFF_GAME_GROUP_COLS,
        how="left",
    )

    return merged


def add_pitcher_stuff_rolling_features(
    pitcher_games: pd.DataFrame,
) -> pd.DataFrame:

    data = pitcher_games.copy()

    for num_col, den_col, prefix in [
        ("swinging_strikes", "pitches", "swstr_pct"),
        ("csw_numerator", "pitches", "csw_pct"),
        ("chase_swings", "pitches_outside_zone", "chase_pct"),
        ("swinging_strikes", "swings", "whiff_pct"),
    ]:
        data = add_rolling_rate_features(
            data,
            "pitcher",
            num_col,
            den_col,
            prefix,
        )

    data = add_rolling_weighted_mean(
        data,
        "pitcher",
        "avg_velocity",
        "release_speed_count",
        "avg_velocity",
    )

    data = add_rolling_max_features(
        data,
        "pitcher",
        "max_velocity",
    )

    return data
