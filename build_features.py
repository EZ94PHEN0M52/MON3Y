import argparse
import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pybaseball import playerid_reverse_lookup

from pitcher_stuff import (
    add_pitcher_stuff_rolling_features,
    build_pitcher_stuff_games,
    merge_stuff_into_pitcher_games,
)
from utils import (
    MLB_STATS_API,
    RAW_DIR,
    batter_features_path,
    live_fetch_disabled,
    pitcher_features_path,
    normalize_version,
)

MLB_BOXSCORE_BATTING_PATH = (
    RAW_DIR / "mlb_boxscore_batting.parquet"
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


INNING_TAINT_EVENTS = {
    "field_error",
    "catcher_interf",
}


# Outs recorded by the pitcher (matches sportsbook "pitcher outs" props).
PITCHER_OUT_EVENTS = {
    "strikeout": 1,
    "field_out": 1,
    "force_out": 1,
    "fielders_choice_out": 1,
    "sac_fly": 1,
    "sac_bunt": 1,
    "grounded_into_double_play": 2,
    "strikeout_double_play": 2,
    "sac_fly_double_play": 2,
    "double_play": 2,
    "triple_play": 3,
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


def derive_stolen_bases_from_des(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Successful steals per batter/game from Statcast descriptions.

    Pitch-level Statcast often omits ``stolen_base*`` events; steals appear in
    ``des`` on the final pitch of the plate appearance. Attribute the runner
    from base state (on_1b -> steal of 2nd, on_2b -> 3rd, on_3b -> home).
    """
    required = {
        "game_date",
        "game_pk",
        "des",
        "on_1b",
        "on_2b",
        "on_3b",
    }
    if data.empty or not required.issubset(data.columns):
        return pd.DataFrame(
            columns=[
                "game_date",
                "game_pk",
                "batter",
                "stolen_bases",
            ]
        )

    des_rows = data.loc[
        data["des"].notna()
        & data["des"]
        .astype(str)
        .str.contains(" steals ", case=False, na=False)
        & ~data["des"]
        .astype(str)
        .str.contains("caught stealing", case=False, na=False)
    ].copy()

    if des_rows.empty:
        return pd.DataFrame(
            columns=[
                "game_date",
                "game_pk",
                "batter",
                "stolen_bases",
            ]
        )

    def _steal_runner(row) -> float | None:
        tail = str(row["des"]).lower().split(" steals ", 1)[-1]

        if re.search(r"\b2nd base\b", tail):
            return row["on_1b"]
        if re.search(r"\b3rd base\b", tail):
            return row["on_2b"]
        if re.search(r"\bhome\b", tail):
            return row["on_3b"]

        return None

    des_rows["steal_runner"] = des_rows.apply(
        _steal_runner,
        axis=1,
    )
    des_rows = des_rows[
        des_rows["steal_runner"].notna()
    ].copy()

    if des_rows.empty:
        return pd.DataFrame(
            columns=[
                "game_date",
                "game_pk",
                "batter",
                "stolen_bases",
            ]
        )

    des_rows["batter"] = (
        pd.to_numeric(
            des_rows["steal_runner"],
            errors="coerce",
        )
        .astype("Int64")
    )
    des_rows = des_rows[
        des_rows["batter"].notna()
    ].copy()

    return (
        des_rows.groupby(
            [
                "game_date",
                "game_pk",
                "batter",
            ],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size": "stolen_bases",
            }
        )
    )


def _fetch_boxscore_batting_rows(
    game_pk: int,
) -> list[dict]:
    import requests

    response = requests.get(
        f"{MLB_STATS_API}/game/{int(game_pk)}/boxscore",
        timeout=15,
    )
    response.raise_for_status()
    payload = response.json()

    rows: list[dict] = []
    for side in ("away", "home"):
        players = payload.get("teams", {}).get(side, {}).get("players") or {}
        for player_blob in players.values():
            person = player_blob.get("person") or {}
            batter = person.get("id")
            stats = (
                player_blob.get("stats") or {}
            ).get("batting") or {}
            if batter is None or not stats:
                continue

            rows.append(
                {
                    "game_pk": int(game_pk),
                    "batter": int(batter),
                    "stolen_bases": int(
                        stats.get("stolenBases") or 0
                    ),
                    "hit_by_pitch": int(
                        stats.get("hitByPitch") or 0
                    ),
                }
            )

    return rows


def load_mlb_boxscore_batting_cache() -> pd.DataFrame:
    if not MLB_BOXSCORE_BATTING_PATH.exists():
        return pd.DataFrame(
            columns=[
                "game_pk",
                "batter",
                "stolen_bases",
                "hit_by_pitch",
            ]
        )

    cache = pd.read_parquet(
        MLB_BOXSCORE_BATTING_PATH
    )
    return cache.drop_duplicates(
        subset=["game_pk", "batter"],
        keep="last",
    )


def build_mlb_boxscore_batting(
    game_pks: list[int],
) -> pd.DataFrame:
    """Official SB/HBP per batter/game from MLB boxscores (cached)."""
    if not game_pks:
        return pd.DataFrame(
            columns=[
                "game_pk",
                "batter",
                "stolen_bases",
                "hit_by_pitch",
            ]
        )

    cache = load_mlb_boxscore_batting_cache()
    needed = {
        int(game_pk)
        for game_pk in game_pks
        if pd.notna(game_pk)
    }
    cached_pks = set()
    if not cache.empty:
        cached_pks = set(
            cache["game_pk"]
            .astype(int)
            .unique()
        )

    missing = sorted(needed - cached_pks)
    fetched: list[dict] = []

    if missing and not live_fetch_disabled():
        for game_pk in missing:
            try:
                fetched.extend(
                    _fetch_boxscore_batting_rows(
                        game_pk
                    )
                )
            except Exception as exc:
                print(
                    f"Warning: MLB boxscore fetch failed "
                    f"for game_pk={game_pk}: {exc}"
                )

    if fetched:
        new_rows = pd.DataFrame(fetched)
        cache = pd.concat(
            [cache, new_rows],
            ignore_index=True,
        )
        cache = cache.drop_duplicates(
            subset=["game_pk", "batter"],
            keep="last",
        )
        MLB_BOXSCORE_BATTING_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        cache.to_parquet(
            MLB_BOXSCORE_BATTING_PATH,
            index=False,
        )

    if cache.empty:
        return cache

    return cache[
        cache["game_pk"]
        .astype(int)
        .isin(needed)
    ].copy()


def supplement_batter_games_from_mlb(
    result: pd.DataFrame,
) -> pd.DataFrame:
    """Replace SB/HBP with official MLB boxscore totals when available."""
    if result.empty or "game_pk" not in result.columns:
        return result

    supplement = build_mlb_boxscore_batting(
        result["game_pk"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    if supplement.empty:
        return result

    merged = result.merge(
        supplement,
        on=["game_pk", "batter"],
        how="left",
        suffixes=("", "_mlb"),
    )

    for stat in ("stolen_bases", "hit_by_pitch"):
        mlb_col = f"{stat}_mlb"
        if mlb_col not in merged.columns:
            continue

        merged[stat] = (
            merged[mlb_col]
            .fillna(merged[stat])
            .fillna(0)
            .astype(int)
        )
        merged = merged.drop(columns=[mlb_col])

    return merged


def derive_scoring_runners(
    row
):

    runs = int(
        row["post_bat_score"] -
        row["bat_score"]
    )

    if runs <= 0:

        return []

    event = row["events"]

    if pd.isna(event):

        return []

    batter = int(row["batter"])

    on1 = (
        int(row["on_1b"])
        if pd.notna(row["on_1b"])
        else None
    )

    on2 = (
        int(row["on_2b"])
        if pd.notna(row["on_2b"])
        else None
    )

    on3 = (
        int(row["on_3b"])
        if pd.notna(row["on_3b"])
        else None
    )

    runners = [
        runner
        for runner in [
            on3,
            on2,
            on1
        ]
        if runner is not None
    ]

    if event == "home_run":

        return (
            [batter] +
            [
                runner
                for runner in [
                    on1,
                    on2,
                    on3
                ]
                if runner is not None
            ]
        )

    if event in {
        "sac_fly",
        "sac_fly_double_play"
    }:

        return (
            [on3]
            if on3 is not None
            else []
        )

    if (
        event in {
            "walk",
            "intent_walk"
        }
        and on1
        and on2
        and on3
    ):

        return [on3]

    if event == "triple":

        return (
            [batter] +
            runners
        )[:runs]

    scored = []
    remaining = runs

    for runner in runners:

        if remaining <= 0:

            break

        scored.append(runner)
        remaining -= 1

    return scored


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

    data["hit_by_pitch"] = (
        data["events"]
        .eq("hit_by_pitch")
        .astype(int)
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
    # Runs scored by batter (from base-state heuristics)
    # -----------------------------------------------------

    data["runs_scored_on_play"] = (
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

    data["scoring_runners"] = (
        data.apply(
            derive_scoring_runners,
            axis=1
        )
    )

    scoring_rows = (

        data[
            data["scoring_runners"]
            .map(len)
            .gt(0)
        ]
        .explode(
            "scoring_runners"
        )
        .rename(
            columns={
                "scoring_runners":
                    "run_scorer"
            }
        )
    )

    runs_by_batter = (

        scoring_rows
        .groupby(
            [
                "game_date",
                "game_pk",
                "run_scorer"
            ],
            as_index=False
        )
        .size()
        .rename(
            columns={
                "run_scorer":
                    "batter",
                "size":
                    "runs"
            }
        )
    )

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

    data["home_team"] = data["home_team"]
    data["away_team"] = data["away_team"]

    # -----------------------------------------------------
    # Stolen bases (Statcast events + des/base-state fallback)
    # -----------------------------------------------------

    sb_mask = (
        data["events"]
        .astype(str)
        .str.startswith("stolen_base")
    )

    sb_rows = data[
        sb_mask
    ].copy()

    sb_frames = []

    if not sb_rows.empty:
        if (
            "runner" in sb_rows.columns
            and sb_rows["runner"].notna().any()
        ):
            sb_group_col = "runner"
        else:
            sb_group_col = "batter"

        sb_frames.append(
            sb_rows.groupby(
                [
                    "game_date",
                    "game_pk",
                    sb_group_col,
                ],
                as_index=False,
            )
            .size()
            .rename(
                columns={
                    sb_group_col: "batter",
                    "size": "stolen_bases",
                }
            )
        )

    des_steals = derive_stolen_bases_from_des(data)
    if not des_steals.empty:
        sb_frames.append(des_steals)

    if sb_frames:
        stolen_by_runner = (
            pd.concat(sb_frames, ignore_index=True)
            .groupby(
                [
                    "game_date",
                    "game_pk",
                    "batter",
                ],
                as_index=False,
            )["stolen_bases"]
            .sum()
        )
    else:
        stolen_by_runner = pd.DataFrame(
            columns=[
                "game_date",
                "game_pk",
                "batter",
                "stolen_bases",
            ]
        )

    # -----------------------------------------------------
    # Aggregate
    # -----------------------------------------------------

    result = (

        data        .groupby(
            [
                "game_date",
                "game_pk",
                "batter",
                "player_name",
                "team",
                "opponent",
                "is_home",
                "home_team",
                "away_team",
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

            walks=(
                "walk",
                "sum"
            ),

            hit_by_pitch=(
                "hit_by_pitch",
                "sum"
            ),

            plate_appearances=(
                "events",
                "count"
            )
        )
    )

    result = result.merge(
        runs_by_batter,
        on=[
            "game_date",
            "game_pk",
            "batter"
        ],
        how="left"
    )

    result["runs"] = (
        result["runs"]
        .fillna(0)
        .astype(int)
    )

    result["hits_runs_rbis"] = (
        result["hits"] +
        result["runs"] +
        result["rbi"]
    )

    result = result.merge(
        stolen_by_runner,
        on=[
            "game_date",
            "game_pk",
            "batter",
        ],
        how="left",
    )

    result["stolen_bases"] = (
        result["stolen_bases"]
        .fillna(0)
        .astype(int)
    )

    result["hit_by_pitch"] = (
        result["hit_by_pitch"]
        .fillna(0)
        .astype(int)
    )

    result = supplement_batter_games_from_mlb(result)

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

    data["home_run_allowed"] = (
        data["events"]
        .eq("home_run")
        .astype(int)
    )

    data["hit_by_pitch"] = (
        data["events"]
        .eq("hit_by_pitch")
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
    # Outs recorded
    # -----------------------------------------------------

    data["outs"] = (
        data["events"]
        .map(PITCHER_OUT_EVENTS)
        .fillna(0)
        .astype(int)
    )

    # -----------------------------------------------------
    # Earned runs allowed (approximation from play events)
    # -----------------------------------------------------

    data["runs_scored_on_play"] = (
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

    data["half_inning"] = (
        data["game_pk"].astype(str)
        + "_"
        + data["inning"].astype(str)
        + "_"
        + data["inning_topbot"].astype(str)
    )

    earned_parts = []

    for _, half in data.groupby(
        "half_inning",
        sort=False
    ):

        half = half.sort_values(
            [
                "at_bat_number",
                "pitch_number"
            ]
        ).copy()

        tainted = False

        earned_runs = []

        for _, row in half.iterrows():

            runs = int(
                row["runs_scored_on_play"]
            )

            if runs > 0:

                if (
                    tainted
                    or row["events"]
                    in INNING_TAINT_EVENTS
                ):

                    earned_runs.append(0)

                else:

                    earned_runs.append(runs)

            else:

                earned_runs.append(0)

            if (
                row["events"]
                in INNING_TAINT_EVENTS
            ):

                tainted = True

        half["earned_runs_on_play"] = (
            earned_runs
        )

        earned_parts.append(
            half
        )

    data = pd.concat(
        earned_parts,
        ignore_index=True
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

    data["home_team"] = data["home_team"]
    data["away_team"] = data["away_team"]

    data["player_name"] = (
        data["player_name"]
        .map(normalize_player_name)
    )

    # -----------------------------------------------------
    # Aggregate
    # -----------------------------------------------------

    result = (

        data        .groupby(
            [
                "game_date",
                "game_pk",
                "pitcher",
                "player_name",
                "team",
                "opponent",
                "is_home",
                "home_team",
                "away_team",
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

            home_runs_allowed=(
                "home_run_allowed",
                "sum"
            ),

            hit_by_pitch=(
                "hit_by_pitch",
                "sum"
            ),

            hits_allowed=(
                "hit_allowed",
                "sum"
            ),

            outs=(
                "outs",
                "sum"
            ),

            earned_runs=(
                "earned_runs_on_play",
                "sum"
            ),

            batters_faced=(
                "strikeout",
                "count",
            ),
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

    print(
        "Building pitcher stuff metrics..."
    )

    stuff_games = build_pitcher_stuff_games(
        statcast_df
    )

    pitchers = merge_stuff_into_pitcher_games(
        pitchers,
        stuff_games,
    )

    pitchers = add_pitcher_stuff_rolling_features(
        pitchers
    )

    # -----------------------------------------------------
    # Batter rolling stats (keep in sync with BATTER_FEATURES in train.py)
    # -----------------------------------------------------

    batter_stats = [
        "hits",
        "home_runs",
        "total_bases",
        "rbi",
        "runs",
        "walks",
        "hits_runs_rbis",
        "stolen_bases",
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
        "home_runs_allowed",
        "hit_by_pitch",
        "hits_allowed",
        "outs",
        "earned_runs",
        "batters_faced",
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

def write_feature_parquet(
    df,
    path,
    version,
):

    from train import (
        FEATURE_SCHEMA_VERSION,
        feature_schema_fingerprint,
    )

    fingerprint = feature_schema_fingerprint(
        version
    )

    table = pa.Table.from_pandas(
        df,
        preserve_index=False,
    )

    metadata = table.schema.metadata or {}
    metadata.update(
        {
            b"feature_schema_version": (
                FEATURE_SCHEMA_VERSION.encode()
            ),
            b"feature_schema_fingerprint": (
                fingerprint.encode()
            ),
        }
    )

    table = table.replace_schema_metadata(
        metadata
    )

    pq.write_table(
        table,
        path,
    )


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

    write_feature_parquet(
        batters,
        batter_path,
        version,
    )

    write_feature_parquet(
        pitchers,
        pitcher_path,
        version,
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
                raw,
                start_date=args.start,
                end_date=args.end,
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
