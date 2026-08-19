"""
Game-level market lines (totals, spreads) as context features for player props.

Consensus lines are joined to player feature rows by (game_date, team).
At inference time, live lines from current_game_lines.parquet are merged
using each prop's event (home_team, away_team, commence_time).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from training_odds import _book_weight, _normalize_side
from utils import (
    PROCESSED_DIR,
    american_to_implied_probability,
    devig_two_way,
    historical_game_lines_path,
)


GAME_LINE_FEATURES = [
    "game_total_line",
    "game_run_line",
    "game_implied_total_over_prob",
]


def _event_game_date(
    commence_time,
) -> str:
    return (
        pd.to_datetime(
            commence_time,
            utc=True,
        )
        .strftime("%Y-%m-%d")
    )


def _normalize_team(name) -> str:
    if not isinstance(name, str):
        return ""
    return name.strip().lower()


def build_consensus_game_lines(
    lines_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    One row per (game_date, home_team, away_team) with consensus total,
    per-team run line, and devigged Over probability on the game total.
    """

    if lines_df.empty:
        return pd.DataFrame(
            columns=[
                "game_date",
                "home_team",
                "away_team",
                "game_total_line",
                "game_run_line_home",
                "game_run_line_away",
                "game_implied_total_over_prob",
            ]
        )

    working = lines_df.copy()
    working["game_date"] = working["commence_time"].map(
        _event_game_date
    )

    if "event_id" not in working.columns:
        working["event_id"] = pd.NA

    rows = []

    group_keys = [
        "game_date",
        "home_team",
        "away_team",
    ]

    for keys, group in working.groupby(
        group_keys,
        dropna=False,
    ):
        game_date, home_team, away_team = keys

        totals = group[
            group["market"].eq("totals")
        ]

        total_line = np.nan
        total_over_prob = np.nan

        if not totals.empty:
            over_rows = totals[
                totals["side"].map(_normalize_side).eq("over")
            ]
            under_rows = totals[
                totals["side"].map(_normalize_side).eq("under")
            ]

            if not over_rows.empty:
                weights = over_rows.apply(
                    lambda row: _book_weight(
                        row.get("bookmaker_key"),
                        row.get("bookmaker"),
                    ),
                    axis=1,
                )
                total_line = float(
                    over_rows["line"].median()
                )

                devigged_probs = []

                for _, over_row in over_rows.iterrows():
                    book = over_row.get("bookmaker")
                    line_val = over_row["line"]
                    under_match = under_rows[
                        under_rows["bookmaker"].eq(book)
                        & under_rows["line"].eq(line_val)
                    ]
                    if under_match.empty:
                        devigged_probs.append(
                            american_to_implied_probability(
                                over_row["odds"]
                            )
                        )
                        continue

                    under_row = under_match.iloc[0]
                    fair_over, _ = devig_two_way(
                        over_row["odds"],
                        under_row["odds"],
                    )
                    devigged_probs.append(fair_over)

                if devigged_probs:
                    total_over_prob = float(
                        np.average(
                            devigged_probs,
                            weights=weights.iloc[
                                : len(devigged_probs)
                            ],
                        )
                    )

        spreads = group[
            group["market"].eq("spreads")
        ]

        home_spread = np.nan
        away_spread = np.nan

        if not spreads.empty:
            home_key = _normalize_team(home_team)
            away_key = _normalize_team(away_team)

            home_rows = spreads[
                spreads["side"].map(_normalize_team).eq(home_key)
            ]
            away_rows = spreads[
                spreads["side"].map(_normalize_team).eq(away_key)
            ]

            if not home_rows.empty:
                home_spread = float(
                    home_rows["line"].median()
                )

            if not away_rows.empty:
                away_spread = float(
                    away_rows["line"].median()
                )

        rows.append({
            "game_date": game_date,
            "home_team": home_team,
            "away_team": away_team,
            "game_total_line": total_line,
            "game_run_line_home": home_spread,
            "game_run_line_away": away_spread,
            "game_implied_total_over_prob": total_over_prob,
        })

    return pd.DataFrame(rows)


def _team_run_line(
    team: str,
    home_team: str,
    away_team: str,
    home_spread,
    away_spread,
):
    team_key = _normalize_team(team)

    if team_key == _normalize_team(home_team):
        return home_spread

    if team_key == _normalize_team(away_team):
        return away_spread

    return np.nan


def merge_game_lines_into_features(
    player_df: pd.DataFrame,
    game_lines_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach game line features to player game logs by (game_date, team).

    Expects player rows with team, home_team, away_team, game_date.
    """

    if player_df.empty:
        return player_df

    result = player_df.copy()

    if game_lines_df.empty:
        for col in GAME_LINE_FEATURES:
            result[col] = np.nan
        return result

    consensus = build_consensus_game_lines(
        game_lines_df
    )

    if consensus.empty:
        for col in GAME_LINE_FEATURES:
            result[col] = np.nan
        return result

    if "game_date" not in result.columns:
        for col in GAME_LINE_FEATURES:
            result[col] = np.nan
        return result

    result["game_date"] = (
        pd.to_datetime(
            result["game_date"]
        )
        .dt.strftime("%Y-%m-%d")
    )

    if "home_team" not in result.columns:
        result["home_team"] = np.where(
            result["is_home"].eq(1),
            result["team"],
            result["opponent"],
        )
        result["away_team"] = np.where(
            result["is_home"].eq(1),
            result["opponent"],
            result["team"],
        )

    merged = result.merge(
        consensus,
        on=[
            "game_date",
            "home_team",
            "away_team",
        ],
        how="left",
    )

    merged["game_run_line"] = merged.apply(
        lambda row: _team_run_line(
            row["team"],
            row["home_team"],
            row["away_team"],
            row.get("game_run_line_home"),
            row.get("game_run_line_away"),
        ),
        axis=1,
    )

    drop_cols = [
        "game_run_line_home",
        "game_run_line_away",
    ]

    merged = merged.drop(
        columns=drop_cols,
        errors="ignore",
    )

    for col in GAME_LINE_FEATURES:
        if col not in merged.columns:
            merged[col] = np.nan

    return merged


def load_historical_game_lines(
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    frames = []

    for snapshot_date in pd.date_range(
        pd.to_datetime(start_date),
        pd.to_datetime(end_date),
        freq="D",
    ):
        path = historical_game_lines_path(
            snapshot_date.strftime("%Y-%m-%d")
        )

        if not path.exists():
            continue

        frames.append(pd.read_parquet(path))

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def load_current_game_lines() -> pd.DataFrame:
    path = (
        PROCESSED_DIR /
        "current_game_lines.parquet"
    )

    if not path.exists():
        return pd.DataFrame()

    return pd.read_parquet(path)


def _consensus_lookup(
    consensus: pd.DataFrame,
) -> dict[tuple, pd.Series]:
    lookup = {}

    for _, row in consensus.iterrows():
        key = (
            row["game_date"],
            _normalize_team(row["home_team"]),
            _normalize_team(row["away_team"]),
        )
        lookup[key] = row

    return lookup


def enrich_feature_row_with_game_lines(
    feature_row: pd.Series,
    prop: pd.Series,
    consensus: pd.DataFrame | None = None,
) -> pd.Series:
    """
    Set game line features on a feature row using the prop's event context.

    Used at inference when feature rows reflect past games but props are
    for today's slate.
    """

    row = feature_row.copy()

    if consensus is None or consensus.empty:
        for col in GAME_LINE_FEATURES:
            row[col] = np.nan
        return row

    game_date = _event_game_date(
        prop.get("commence_time")
    )
    home = prop.get("home_team")
    away = prop.get("away_team")
    team = row.get("team")

    if pd.isna(team):
        if row.get("is_home") == 1:
            team = home
        else:
            team = away

    key = (
        game_date,
        _normalize_team(home),
        _normalize_team(away),
    )

    lookup = _consensus_lookup(consensus)
    match = lookup.get(key)

    if match is None:
        for col in GAME_LINE_FEATURES:
            row[col] = np.nan
        return row

    row["game_total_line"] = match.get(
        "game_total_line",
        np.nan,
    )
    row["game_implied_total_over_prob"] = match.get(
        "game_implied_total_over_prob",
        np.nan,
    )
    row["game_run_line"] = _team_run_line(
        team,
        home,
        away,
        match.get("game_run_line_home"),
        match.get("game_run_line_away"),
    )

    return row
