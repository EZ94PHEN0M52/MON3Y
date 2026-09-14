"""Build rolling player-game features for NFL prop models (V1 / V2)."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from features_v2 import (
    FEATURE_SCHEMA_VERSION_V2,
    apply_v2_features,
    v2_feature_column_names,
)
from utils import (
    injuries_raw_path,
    normalize_injury_status,
    normalize_version,
    player_features_path,
    player_stats_raw_path,
    schedules_raw_path,
)

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

FEATURE_SCHEMA_VERSION = "1"


def load_player_stats(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        path = player_stats_raw_path(season)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run:\n"
                f"  python fetch_data.py --stats --season {season}"
            )
        frames.append(pd.read_parquet(path))

    df = pd.concat(frames, ignore_index=True)
    if "season_type" in df.columns:
        df = df[df["season_type"].astype(str).str.upper().eq("REG")].copy()

    df["rush_reception_yards"] = (
        df.get("rushing_yards", 0).fillna(0)
        + df.get("receiving_yards", 0).fillna(0)
    )

    for col in STAT_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    if "target_share" in df.columns:
        df["target_share"] = pd.to_numeric(df["target_share"], errors="coerce")

    if "player_id" not in df.columns:
        raise ValueError("player_stats missing player_id")

    df["player_name"] = df.get(
        "player_display_name",
        df.get("player_name"),
    )
    df = df[df["player_id"].notna()].copy()
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)
    return df


def load_schedules(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        path = schedules_raw_path(season)
        if not path.exists():
            print(f"WARNING: missing schedule {path}; is_home will be NaN")
            continue
        frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_injuries(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        path = injuries_raw_path(season)
        if not path.exists():
            print(f"WARNING: missing injuries {path}")
            continue
        frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _attach_is_home(games: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    out = games.copy()
    out["is_home"] = np.nan
    if schedules.empty or "game_id" not in out.columns:
        return out

    sched = schedules.copy()
    if "game_id" not in sched.columns:
        return out

    home_map = (
        sched[["game_id", "home_team"]]
        .dropna()
        .drop_duplicates("game_id")
        .set_index("game_id")["home_team"]
        .to_dict()
    )
    out["home_team"] = out["game_id"].map(home_map)
    out["is_home"] = (
        out["team"].astype(str) == out["home_team"].astype(str)
    ).astype(float)
    out.loc[out["home_team"].isna(), "is_home"] = np.nan
    return out


def _attach_injury_features(
    games: pd.DataFrame,
    injuries: pd.DataFrame,
) -> pd.DataFrame:
    out = games.copy()
    out["injury_status_norm"] = "active"
    out["is_inactive"] = 0
    out["is_questionable"] = 0

    if injuries.empty:
        return out

    inj = injuries.copy()
    id_col = "gsis_id" if "gsis_id" in inj.columns else None
    if id_col is None:
        return out

    inj["season"] = pd.to_numeric(inj["season"], errors="coerce")
    inj["week"] = pd.to_numeric(inj["week"], errors="coerce")
    inj = inj.dropna(subset=["season", "week", id_col])
    inj["season"] = inj["season"].astype(int)
    inj["week"] = inj["week"].astype(int)
    inj["injury_status_norm"] = inj["report_status"].map(normalize_injury_status)
    inj["is_inactive"] = (
        inj["injury_status_norm"].eq("inactive").astype(int)
    )
    inj["is_questionable"] = (
        inj["injury_status_norm"].eq("questionable").astype(int)
    )

    slim = (
        inj[[id_col, "season", "week", "injury_status_norm", "is_inactive", "is_questionable"]]
        .drop_duplicates(subset=[id_col, "season", "week"], keep="last")
        .rename(columns={id_col: "player_id"})
    )

    out = out.merge(
        slim, on=["player_id", "season", "week"], how="left", suffixes=("", "_inj")
    )
    for col in ("injury_status_norm", "is_inactive", "is_questionable"):
        inj_col = f"{col}_inj"
        if inj_col in out.columns:
            out[col] = out[inj_col].combine_first(out[col])
            out = out.drop(columns=[inj_col])

    out["injury_status_norm"] = out["injury_status_norm"].fillna("active")
    out["is_inactive"] = out["is_inactive"].fillna(0).astype(int)
    out["is_questionable"] = out["is_questionable"].fillna(0).astype(int)
    return out


def add_rolling_features(games: pd.DataFrame) -> pd.DataFrame:
    frames = []
    games = games.sort_values(["player_id", "season", "week"]).copy()

    for _, group in games.groupby("player_id", sort=False):
        g = group.copy()
        for stat in STAT_COLUMNS:
            shifted = g[stat].shift(1)
            for window in ROLL_WINDOWS:
                g[f"{stat}_l{window}"] = shifted.rolling(
                    window, min_periods=1
                ).mean()
            season_means = []
            for _, sg in g.groupby("season", sort=False):
                prior = sg[stat].shift(1)
                season_means.append(prior.expanding(min_periods=1).mean())
            g[f"{stat}_season"] = pd.concat(season_means).sort_index()
        frames.append(g)

    return pd.concat(frames, ignore_index=True)


def build_features(seasons: list[int], version: str = "v1") -> pd.DataFrame:
    version = normalize_version(version)

    print()
    print("=" * 60)
    print(f"BUILDING PLAYER FEATURES {version} — seasons {seasons}")
    print("=" * 60)

    games = load_player_stats(seasons)
    schedules = load_schedules(seasons)
    injuries = load_injuries(seasons)

    games = _attach_is_home(games, schedules)
    games = add_rolling_features(games)
    games = _attach_injury_features(games, injuries)

    schema = FEATURE_SCHEMA_VERSION
    if version == "v2":
        games = apply_v2_features(games, seasons)
        schema = FEATURE_SCHEMA_VERSION_V2

    games["feature_schema_version"] = schema

    keep = [
        "player_id",
        "player_name",
        "position",
        "team",
        "opponent_team",
        "season",
        "week",
        "game_id",
        "is_home",
        "injury_status_norm",
        "is_inactive",
        "is_questionable",
        "feature_schema_version",
    ]

    for stat in STAT_COLUMNS:
        keep.append(stat)
        for window in ROLL_WINDOWS:
            keep.append(f"{stat}_l{window}")
        keep.append(f"{stat}_season")

    if version == "v2":
        keep.extend(["target_share", "rush_share", "offense_snap_pct"])
        keep.extend(v2_feature_column_names())

    keep = [c for c in dict.fromkeys(keep) if c in games.columns]
    out = games[keep].copy()

    path = player_features_path(version)
    out.to_parquet(path, index=False)
    print(f"Saved {len(out):,} feature rows → {path}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build NFL rolling feature tables."
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=[2024, 2025, 2026],
        help="Seasons to include (default: 2024 2025 2026).",
    )
    parser.add_argument(
        "--version",
        default="v1",
        choices=["v1", "v2"],
    )
    args = parser.parse_args()
    build_features(args.seasons, version=args.version)


if __name__ == "__main__":
    main()
