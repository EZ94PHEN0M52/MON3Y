import numpy as np
import pandas as pd

from odds_snapshots import (
    load_opening_snapshot,
    load_snapshots,
)
from utils import american_to_implied_probability


MOVEMENT_KEYS = [
    "player",
    "market",
    "event_id",
    "bookmaker",
    "side",
]


def _normalize_side(side):
    return str(side).strip().lower()


def _heavy_side_from_pair(
    over_odds,
    under_odds,
):
    if pd.isna(over_odds) or pd.isna(under_odds):
        return None

    over_prob = american_to_implied_probability(
        over_odds
    )
    under_prob = american_to_implied_probability(
        under_odds
    )

    if over_prob >= under_prob:
        return "over"

    return "under"


def _build_opening_side_pairs(opening_df):
    if opening_df.empty:
        return {}

    working = opening_df.copy()
    working["side_norm"] = working["side"].map(
        _normalize_side
    )

    pivot = working.pivot_table(
        index=[
            "player",
            "market",
            "event_id",
            "bookmaker",
            "opening_line",
        ],
        columns="side_norm",
        values="opening_odds",
        aggfunc="first",
    )

    lookup = {}

    for keys, row in pivot.iterrows():
        heavy = _heavy_side_from_pair(
            row.get("over"),
            row.get("under"),
        )

        if heavy is not None:
            lookup[keys] = heavy

    return lookup


def _steam_flag(
    line_delta,
    heavy_side,
):
    if pd.isna(line_delta) or heavy_side is None:
        return False

    if heavy_side == "over" and line_delta <= -0.5:
        return True

    if heavy_side == "under" and line_delta >= 0.5:
        return True

    return False


def compute_movement_features(
    current_props,
    snapshots_path=None,
    game_date=None,
):
    """
    Compare current props to the day's opening snapshot.

    Adds opening_line, opening_odds, line_delta, odds_delta, steam_flag.
    """

    if current_props.empty:
        return current_props.copy()

    result = current_props.copy()

    snapshots = load_snapshots(
        snapshots_path,
        game_date,
    )

    if snapshots.empty:
        for column in (
            "opening_line",
            "opening_odds",
            "line_delta",
            "odds_delta",
            "steam_flag",
        ):
            if column == "steam_flag":
                result[column] = False
            else:
                result[column] = np.nan
        return result

    if game_date is None:
        fetched = pd.to_datetime(
            result.get(
                "fetched_at",
                pd.Series(dtype="datetime64[ns, UTC]"),
            ),
            utc=True,
            errors="coerce",
        )

        if fetched.notna().any():
            game_date = fetched.max().date()
        else:
            game_date = (
                pd.Timestamp.now(tz="UTC")
                .date()
            )

    opening = load_opening_snapshot(
        snapshots,
        game_date,
    )

    if opening.empty:
        for column in (
            "opening_line",
            "opening_odds",
            "line_delta",
            "odds_delta",
            "steam_flag",
        ):
            if column == "steam_flag":
                result[column] = False
            else:
                result[column] = np.nan
        return result

    merged = result.merge(
        opening,
        on=MOVEMENT_KEYS,
        how="left",
    )

    merged["line_delta"] = (
        merged["line"]
        - merged["opening_line"]
    )

    merged["odds_delta"] = (
        merged["odds"]
        - merged["opening_odds"]
    )

    pair_lookup = _build_opening_side_pairs(
        opening
    )

    steam_flags = []

    for _, row in merged.iterrows():
        if pd.isna(row.get("opening_line")):
            steam_flags.append(False)
            continue

        keys = (
            row["player"],
            row["market"],
            row["event_id"],
            row["bookmaker"],
            row["opening_line"],
        )

        heavy_side = pair_lookup.get(keys)
        steam_flags.append(
            _steam_flag(
                row["line_delta"],
                heavy_side,
            )
        )

    merged["steam_flag"] = steam_flags

    return merged
