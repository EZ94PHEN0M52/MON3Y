"""
Join historical sportsbook props to Statcast feature rows for training.

Historical props live at:
  data/raw/odds/historical/date=YYYY-MM-DD/props.parquet
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from odds_aggregation import build_devigged_lookup
from prop_scoring import (
    MARKET_STAT_MAP,
    fuzzy_player_match,
)
from utils import (
    SHARP_BOOK_WEIGHTS,
    american_to_implied_probability,
    historical_odds_path,
)


DERIVED_LINE_FEATURES = [
    "market_implied_over_prob",
    "line_vs_season_avg",
]


def date_range(start: str, end: str) -> list[str]:
    dates = pd.date_range(
        pd.to_datetime(start),
        pd.to_datetime(end),
        freq="D",
    )
    return [d.strftime("%Y-%m-%d") for d in dates]


def load_historical_props(
    start_date: str,
    end_date: str,
    market_filter: str | None = None,
) -> pd.DataFrame:
    frames = []

    for snapshot_date in date_range(
        start_date,
        end_date,
    ):
        path = historical_odds_path(snapshot_date)

        if not path.exists():
            continue

        frames.append(pd.read_parquet(path))

    if not frames:
        return pd.DataFrame()

    props = pd.concat(frames, ignore_index=True)

    if market_filter:
        props = props[props["market"].eq(market_filter)]

    return props


def _book_weight(
    bookmaker_key,
    bookmaker,
) -> float:
    key = str(bookmaker_key or "").lower()
    name = str(bookmaker or "").lower()

    for sharp_key, weight in SHARP_BOOK_WEIGHTS.items():
        if key == sharp_key or name == sharp_key:
            return weight

    return 1.0


def _normalize_side(side) -> str:
    return str(side).strip().lower()


def add_game_date(props: pd.DataFrame) -> pd.DataFrame:
    if props.empty:
        return props

    result = props.copy()
    result["game_date"] = (
        pd.to_datetime(
            result["commence_time"],
            utc=True,
        )
        .dt.strftime("%Y-%m-%d")
    )
    return result


def devigged_over_probability(row, lookup) -> float:
    keys = (
        row["player"],
        row["market"],
        row.get("event_id"),
        row["line"],
        row["bookmaker"],
    )

    pair = lookup.get(keys)

    if pair is not None:
        fair_over, fair_under = pair
        side = _normalize_side(row["side"])

        if side == "over":
            return fair_over

        if side == "under":
            return fair_under

    side = _normalize_side(row["side"])
    implied = american_to_implied_probability(row["odds"])

    if side == "over":
        return implied

    if side == "under":
        return 1.0 - implied

    return np.nan


def build_consensus_lines(props: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (game_date, player, market) with consensus line and
    weighted devigged Over probability.
    """

    if props.empty:
        return pd.DataFrame(
            columns=[
                "game_date",
                "player",
                "market",
                "consensus_line",
                "market_implied_over_prob",
            ]
        )

    working = add_game_date(props)

    if "event_id" not in working.columns:
        working["event_id"] = pd.NA

    lookup = build_devigged_lookup(working)

    working["devigged_over_prob"] = working.apply(
        lambda row: devigged_over_probability(row, lookup),
        axis=1,
    )

    rows = []

    group_keys = [
        "game_date",
        "player",
        "market",
    ]

    for keys, group in working.groupby(
        group_keys,
        dropna=False,
    ):
        over_rows = group[
            group["side"].map(_normalize_side).eq("over")
        ]

        if over_rows.empty:
            over_rows = group

        weights = over_rows.apply(
            lambda row: _book_weight(
                row.get("bookmaker_key"),
                row.get("bookmaker"),
            ),
            axis=1,
        )

        devigged = over_rows["devigged_over_prob"]

        if devigged.notna().any():
            market_implied = np.average(
                devigged.fillna(devigged.median()),
                weights=weights,
            )
        else:
            market_implied = np.nan

        rows.append({
            "game_date": keys[0],
            "player": keys[1],
            "market": keys[2],
            "consensus_line": over_rows["line"].median(),
            "market_implied_over_prob": market_implied,
        })

    return pd.DataFrame(rows)


def stat_to_market_key(
    stat: str,
    role: str,
) -> str | None:
    prefix = (
        "batter_"
        if role == "batter"
        else "pitcher_"
    )

    for market, market_stat in MARKET_STAT_MAP.items():
        if (
            market_stat == stat
            and market.startswith(prefix)
        ):
            return market

    return None


def match_props_to_features(
    features_df: pd.DataFrame,
    props_df: pd.DataFrame,
    player_col: str,
    stat_col: str,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Join consensus historical lines to feature rows on (game_date, player).

    Target is actual_stat > consensus_line.
    """

    if props_df.empty or features_df.empty:
        return pd.DataFrame()

    role = "batter" if player_col == "batter" else "pitcher"
    market_key = stat_to_market_key(stat_col, role)

    if market_key is None:
        return pd.DataFrame()

    market_props = props_df[
        props_df["market"].eq(market_key)
    ]

    consensus = build_consensus_lines(market_props)

    if consensus.empty:
        return pd.DataFrame()

    usable = features_df.copy()
    usable = usable[
        usable[f"{stat_col}_season"].notna()
    ].copy()

    if usable.empty:
        return pd.DataFrame()

    usable["game_date"] = (
        pd.to_datetime(
            usable["game_date"]
        )
        .dt.strftime("%Y-%m-%d")
    )

    names_by_date = (
        usable.groupby("game_date")["player_name"]
        .apply(
            lambda series: (
                series.dropna().drop_duplicates()
            )
        )
    )

    lookup = (
        usable.set_index(
            ["game_date", "player_name"]
        )
        .sort_index()
    )

    rows = []

    for _, consensus_row in consensus.iterrows():
        game_date = consensus_row["game_date"]

        if game_date not in names_by_date.index:
            continue

        candidates = names_by_date.loc[game_date]

        match = fuzzy_player_match(
            consensus_row["player"],
            candidates,
        )

        if match is None:
            continue

        key = (game_date, match)

        if key not in lookup.index:
            continue

        feature_row = lookup.loc[key]

        if isinstance(feature_row, pd.DataFrame):
            feature_row = feature_row.iloc[0]

        actual_stat = feature_row.get(stat_col, np.nan)
        line = consensus_row["consensus_line"]

        if pd.isna(actual_stat) or pd.isna(line):
            continue

        season_avg = feature_row.get(
            f"{stat_col}_season",
            np.nan,
        )

        row = {
            "game_date": game_date,
            player_col: feature_row.get(player_col),
            "player_name": match,
            "opponent": feature_row.get("opponent"),
            "is_home": feature_row.get("is_home"),
            stat_col: actual_stat,
            "line": float(line),
            "target": int(float(actual_stat) > float(line)),
            "market": stat_col,
            "market_implied_over_prob": (
                consensus_row["market_implied_over_prob"]
            ),
            "line_vs_season_avg": (
                float(line) - float(season_avg)
                if pd.notna(season_avg)
                else np.nan
            ),
        }

        for col in feature_columns:
            row[col] = feature_row.get(col, np.nan)

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def attach_consensus_to_props(
    props: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add consensus_line and market_implied_over_prob to live/historical props.
    """

    if props.empty:
        return props

    consensus = build_consensus_lines(props)

    if consensus.empty:
        result = props.copy()
        result["consensus_line"] = np.nan
        result["market_implied_over_prob"] = np.nan
        return result

    working = add_game_date(props)

    return working.merge(
        consensus,
        on=["game_date", "player", "market"],
        how="left",
    )
