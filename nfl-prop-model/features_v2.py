"""V2 feature extras: usage, snap share, opponent defense allowed."""

from __future__ import annotations

import pandas as pd

from utils import (
    RAW_DIR,
    normalize_player_name,
)

FEATURE_SCHEMA_VERSION_V2 = "2"

# Columns added on top of V1 (rolling windows applied where noted).
USAGE_RAW = [
    "target_share",
    "rush_share",
    "offense_snap_pct",
]

OPP_DEFENSE_RAW = [
    "opp_pass_yards_allowed",
    "opp_rush_yards_allowed",
    "opp_rec_yards_allowed",
]

V2_EXTRA_BASE = USAGE_RAW + OPP_DEFENSE_RAW


def snap_counts_raw_path(season: int):
    return RAW_DIR / f"snap_counts_{season}.parquet"


def team_stats_raw_path(season: int):
    return RAW_DIR / f"team_stats_{season}.parquet"


def load_snap_counts(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        path = snap_counts_raw_path(season)
        if not path.exists():
            print(f"WARNING: missing snap counts {path}")
            continue
        frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "game_type" in df.columns:
        df = df[df["game_type"].astype(str).str.upper().eq("REG")].copy()
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)
    df["name_key"] = df["player"].map(normalize_player_name)
    return df


def load_team_stats(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        path = team_stats_raw_path(season)
        if not path.exists():
            print(f"WARNING: missing team stats {path}")
            continue
        frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "season_type" in df.columns:
        df = df[df["season_type"].astype(str).str.upper().eq("REG")].copy()
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)
    return df


def _add_rolling_for_columns(
    df: pd.DataFrame,
    columns: list[str],
    group_key: str,
) -> pd.DataFrame:
    """shift(1) L3/L5/season rolls within group_key, season-aware for season mean."""
    frames = []
    ordered = df.sort_values([group_key, "season", "week"]).copy()
    for _, group in ordered.groupby(group_key, sort=False):
        g = group.copy()
        for col in columns:
            if col not in g.columns:
                g[col] = pd.NA
            shifted = pd.to_numeric(g[col], errors="coerce").shift(1)
            g[f"{col}_l3"] = shifted.rolling(3, min_periods=1).mean()
            g[f"{col}_l5"] = shifted.rolling(5, min_periods=1).mean()
            season_parts = []
            for _, sg in g.groupby("season", sort=False):
                prior = pd.to_numeric(sg[col], errors="coerce").shift(1)
                season_parts.append(prior.expanding(min_periods=1).mean())
            g[f"{col}_season"] = pd.concat(season_parts).sort_index()
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


def attach_usage_features(games: pd.DataFrame) -> pd.DataFrame:
    """target_share, rush_share, offense_snap_pct + rolling forms."""
    out = games.copy()

    if "target_share" not in out.columns:
        out["target_share"] = pd.NA
    out["target_share"] = pd.to_numeric(out["target_share"], errors="coerce")

    # Rush share = player carries / team carries that game.
    if "carries" in out.columns:
        team_carries = (
            out.groupby(["season", "week", "team"], dropna=False)["carries"]
            .transform("sum")
            .replace(0, pd.NA)
        )
        out["rush_share"] = pd.to_numeric(out["carries"], errors="coerce") / team_carries
    else:
        out["rush_share"] = pd.NA

    out["offense_snap_pct"] = pd.NA
    return out


def merge_snap_pct(games: pd.DataFrame, snaps: pd.DataFrame) -> pd.DataFrame:
    out = games.copy()
    if snaps.empty:
        return out

    slim = snaps[
        ["season", "week", "team", "name_key", "offense_pct"]
    ].drop_duplicates(subset=["season", "week", "team", "name_key"], keep="last")
    slim = slim.rename(columns={"offense_pct": "offense_snap_pct"})

    out["name_key"] = out["player_name"].map(normalize_player_name)
    out = out.drop(columns=["offense_snap_pct"], errors="ignore")
    out = out.merge(
        slim,
        on=["season", "week", "team", "name_key"],
        how="left",
    )
    out = out.drop(columns=["name_key"], errors="ignore")
    return out


def build_opponent_defense_table(team_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Yards allowed by each defense, then prior-game rolling means.

    Offense yards posted against team D become D's allowed yards that week.
    """
    if team_stats.empty:
        return pd.DataFrame()

    needed = {"season", "week", "team", "opponent_team", "passing_yards", "rushing_yards"}
    if not needed.issubset(team_stats.columns):
        print("WARNING: team_stats missing columns for opponent defense")
        return pd.DataFrame()

    allowed = team_stats[
        ["season", "week", "opponent_team", "passing_yards", "rushing_yards"]
        + (["receiving_yards"] if "receiving_yards" in team_stats.columns else [])
    ].copy()

    rename = {
        "opponent_team": "team",
        "passing_yards": "opp_pass_yards_allowed",
        "rushing_yards": "opp_rush_yards_allowed",
    }
    if "receiving_yards" in allowed.columns:
        rename["receiving_yards"] = "opp_rec_yards_allowed"
    allowed = allowed.rename(columns=rename)

    if "opp_rec_yards_allowed" not in allowed.columns:
        allowed["opp_rec_yards_allowed"] = pd.NA

    allowed = allowed.drop_duplicates(subset=["season", "week", "team"], keep="last")
    allowed = _add_rolling_for_columns(
        allowed,
        [
            "opp_pass_yards_allowed",
            "opp_rush_yards_allowed",
            "opp_rec_yards_allowed",
        ],
        group_key="team",
    )
    return allowed


def merge_opponent_defense(
    games: pd.DataFrame,
    defense: pd.DataFrame,
) -> pd.DataFrame:
    out = games.copy()
    if defense.empty or "opponent_team" not in out.columns:
        for col in [
            "opp_pass_yards_allowed_l3",
            "opp_pass_yards_allowed_l5",
            "opp_pass_yards_allowed_season",
            "opp_rush_yards_allowed_l3",
            "opp_rush_yards_allowed_l5",
            "opp_rush_yards_allowed_season",
            "opp_rec_yards_allowed_l3",
            "opp_rec_yards_allowed_l5",
            "opp_rec_yards_allowed_season",
        ]:
            if col not in out.columns:
                out[col] = pd.NA
        return out

    keep = [
        "season",
        "week",
        "team",
        "opp_pass_yards_allowed_l3",
        "opp_pass_yards_allowed_l5",
        "opp_pass_yards_allowed_season",
        "opp_rush_yards_allowed_l3",
        "opp_rush_yards_allowed_l5",
        "opp_rush_yards_allowed_season",
        "opp_rec_yards_allowed_l3",
        "opp_rec_yards_allowed_l5",
        "opp_rec_yards_allowed_season",
    ]
    keep = [c for c in keep if c in defense.columns]
    slim = defense[keep].rename(columns={"team": "opponent_team"})
    out = out.merge(slim, on=["season", "week", "opponent_team"], how="left")
    return out


def apply_v2_features(
    games: pd.DataFrame,
    seasons: list[int],
) -> pd.DataFrame:
    """Attach V2 usage + opponent defense rolling features."""
    out = attach_usage_features(games)
    snaps = load_snap_counts(seasons)
    out = merge_snap_pct(out, snaps)

    # Rolling usage on the player grain.
    out = _add_rolling_for_columns(
        out,
        ["target_share", "rush_share", "offense_snap_pct"],
        group_key="player_id",
    )

    team_stats = load_team_stats(seasons)
    defense = build_opponent_defense_table(team_stats)
    out = merge_opponent_defense(out, defense)

    # Ensure is_home exists (V1 already sets it; keep explicit for V2 contract).
    if "is_home" not in out.columns:
        out["is_home"] = pd.NA

    return out


def v2_feature_column_names() -> list[str]:
    cols = []
    for base in V2_EXTRA_BASE:
        cols.extend([f"{base}_l3", f"{base}_l5", f"{base}_season"])
    return cols
