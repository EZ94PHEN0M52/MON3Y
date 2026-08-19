"""
Closing line value (CLV) for backtests (Phase 6).

Compares the devigged price at bet time to the closing devigged price for
the same prop key when multiple historical snapshots exist.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from odds_aggregation import (
    PAIR_KEYS,
    build_devigged_lookup,
)
from utils import american_to_implied_probability


PROP_ID_KEYS = [
    "player",
    "market",
    "event_id",
    "bookmaker",
    "side",
    "line",
]


def _devigged_prob_for_prop_row(
    row: pd.Series,
    lookup: dict,
) -> float:
    keys = tuple(
        row[key]
        for key in PAIR_KEYS
    )
    pair = lookup.get(keys)

    side = str(row["side"]).strip().lower()

    if pair is not None:
        fair_over, fair_under = pair

        if side == "over":
            return fair_over

        if side == "under":
            return fair_under

    return american_to_implied_probability(
        row["odds"]
    )


def build_closing_devigged_lookup(
    props: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each prop key, return devigged probability at the closing snapshot.

    Closing = latest ``snapshot_date`` when multiple snapshots exist.
    """

    if props.empty:
        return pd.DataFrame(
            columns=PROP_ID_KEYS + [
                "closing_devigged_prob",
                "opening_devigged_prob",
                "snapshot_count",
            ]
        )

    working = props.copy()

    if "snapshot_date" not in working.columns:
        working["snapshot_date"] = pd.NA

    lookup = build_devigged_lookup(working)

    working["devigged_prob"] = working.apply(
        lambda row: _devigged_prob_for_prop_row(
            row,
            lookup,
        ),
        axis=1,
    )

    working = working.sort_values(
        "snapshot_date",
    )

    grouped = working.groupby(
        PROP_ID_KEYS,
        dropna=False,
    )

    closing = (
        grouped
        .tail(1)
        .rename(
            columns={
                "devigged_prob": (
                    "closing_devigged_prob"
                ),
            }
        )
    )

    opening = (
        grouped
        .head(1)
        .rename(
            columns={
                "devigged_prob": (
                    "opening_devigged_prob"
                ),
            }
        )
    )

    counts = (
        grouped
        .size()
        .reset_index(name="snapshot_count")
    )

    result = closing[
        PROP_ID_KEYS + [
            "closing_devigged_prob",
            "snapshot_date",
        ]
    ].merge(
        opening[
            PROP_ID_KEYS + [
                "opening_devigged_prob",
            ]
        ],
        on=PROP_ID_KEYS,
        how="left",
    ).merge(
        counts,
        on=PROP_ID_KEYS,
        how="left",
    )

    return result


def attach_clv(
    detail: pd.DataFrame,
    props: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add CLV columns to backtest detail rows.

    ``clv`` = closing devigged prob − bet-time devigged prob (positive means
    the bet price beat the closing line). ``model_clv`` = model probability
    minus closing devigged prob (remaining edge vs close).
    """

    if detail.empty:
        return detail

    result = detail.copy()
    closing = build_closing_devigged_lookup(
        props,
    )

    if closing.empty:
        result["closing_devigged_prob"] = np.nan
        result["clv"] = np.nan
        result["model_clv"] = np.nan
        return result

    merge_keys = [
        key
        for key in PROP_ID_KEYS
        if key in result.columns
    ]

    result = result.merge(
        closing[
            merge_keys + [
                "closing_devigged_prob",
                "opening_devigged_prob",
                "snapshot_count",
            ]
        ],
        on=merge_keys,
        how="left",
    )

    bet_devigged = result.get(
        "devigged_market_prob",
        result.get("market_probability"),
    )

    result["clv"] = (
        result["closing_devigged_prob"]
        - bet_devigged
    )

    model_prob = result.get(
        "calibrated_probability",
        result.get("model_probability"),
    )

    result["model_clv"] = (
        model_prob
        - result["closing_devigged_prob"]
    )

    single_snapshot = result[
        "snapshot_count"
    ].fillna(1) <= 1

    result.loc[
        single_snapshot,
        "clv",
    ] = np.nan
    result.loc[
        single_snapshot,
        "model_clv",
    ] = np.nan

    return result
