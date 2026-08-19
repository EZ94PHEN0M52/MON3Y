import numpy as np
import pandas as pd

from utils import (
    SHARP_BOOK_WEIGHTS,
    american_to_implied_probability,
    devig_two_way,
    expected_value,
)


PAIR_KEYS = [
    "player",
    "market",
    "event_id",
    "line",
    "bookmaker",
]

CONSENSUS_KEYS = [
    "player",
    "market",
    "event_id",
]

BEST_PRICE_KEYS = [
    "player",
    "market",
    "event_id",
    "line",
    "side",
]

# One board row per player + market (best EV across books/lines/sides).
DEDUP_PROP_KEYS = [
    "player",
    "market",
]


def _book_weight(
    bookmaker_key,
    bookmaker,
):
    key = str(bookmaker_key or "").lower()
    name = str(bookmaker or "").lower()

    for sharp_key, weight in SHARP_BOOK_WEIGHTS.items():
        if key == sharp_key or name == sharp_key:
            return weight

    return 1.0


def _normalize_side(side):
    return str(side).strip().lower()


def _best_american_odds(series):
    return series.astype(float).max()


def build_devigged_lookup(df):
    """
    Map (player, market, event_id, line, bookmaker) → fair Over/Under probs.
    """

    if df.empty:
        return {}

    working = df.copy()
    working["side_norm"] = working["side"].map(
        _normalize_side
    )

    pivot = working.pivot_table(
        index=PAIR_KEYS,
        columns="side_norm",
        values="odds",
        aggfunc="first",
    )

    lookup = {}

    for keys, row in pivot.iterrows():
        over_odds = row.get("over")
        under_odds = row.get("under")

        if pd.isna(over_odds) or pd.isna(under_odds):
            continue

        fair_over, fair_under = devig_two_way(
            over_odds,
            under_odds,
        )

        lookup[keys] = (
            fair_over,
            fair_under,
        )

    return lookup


def _devigged_prob_for_row(
    row,
    lookup,
):
    keys = tuple(
        row[key]
        for key in PAIR_KEYS
    )

    pair = lookup.get(keys)

    if pair is not None:
        fair_over, fair_under = pair
        side = _normalize_side(row["side"])

        if side == "over":
            return fair_over

        if side == "under":
            return fair_under

    return american_to_implied_probability(
        row["odds"]
    )


def _consensus_probs(group):
    rows = []

    for _, row in group.iterrows():
        side = _normalize_side(row["side"])

        if side != "over":
            continue

        devigged = row.get("devigged_market_prob")

        if pd.isna(devigged):
            devigged = american_to_implied_probability(
                row["odds"]
            )

        rows.append({
            "line": row["line"],
            "devigged_over_prob": devigged,
            "weight": _book_weight(
                row.get("bookmaker_key"),
                row.get("bookmaker"),
            ),
        })

    if not rows:
        return pd.Series({
            "consensus_line": np.nan,
            "consensus_over_prob": np.nan,
        })

    frame = pd.DataFrame(rows)

    consensus_line = frame["line"].median()

    weighted = np.average(
        frame["devigged_over_prob"],
        weights=frame["weight"],
    )

    return pd.Series({
        "consensus_line": consensus_line,
        "consensus_over_prob": weighted,
    })


def _consensus_edge(row):
    side = _normalize_side(row["side"])
    consensus_over = row.get("consensus_over_prob")

    if pd.isna(consensus_over):
        return np.nan

    model_prob = (
        row["over_probability"]
        if side == "over"
        else row["under_probability"]
    )

    consensus_prob = (
        consensus_over
        if side == "over"
        else 1.0 - consensus_over
    )

    return model_prob - consensus_prob


def _mark_best_prices(df):
    result = df.copy()
    result["is_best_price"] = False
    result["best_book"] = pd.NA
    result["best_odds"] = np.nan
    result["best_ev"] = np.nan

    if result.empty:
        return result

    for _, group in result.groupby(
        BEST_PRICE_KEYS,
        dropna=False,
    ):
        best_idx = group["ev"].idxmax()
        best_row = group.loc[best_idx]

        result.loc[
            group.index,
            "best_book",
        ] = best_row["bookmaker"]
        result.loc[
            group.index,
            "best_odds",
        ] = best_row["odds"]
        result.loc[
            group.index,
            "best_ev",
        ] = best_row["ev"]
        result.loc[
            best_idx,
            "is_best_price",
        ] = True

    return result


def dedupe_best_prop(
    df,
    keys=None,
    sort_col="ev",
):
    """
    Keep a single row per prop identity with the highest EV.

    Default keys are (player, market) so the main board shows one best
    opportunity per player and market type.
    """
    if df.empty:
        return df.copy()

    group_keys = list(keys or DEDUP_PROP_KEYS)
    missing = [key for key in group_keys if key not in df.columns]
    if missing:
        raise KeyError(
            f"dedupe_best_prop missing columns: {missing}"
        )

    working = df.copy()
    if sort_col not in working.columns:
        raise KeyError(
            f"dedupe_best_prop missing sort column: {sort_col}"
        )

    working = working.sort_values(
        sort_col,
        ascending=False,
        na_position="last",
    )

    return (
        working.groupby(
            group_keys,
            dropna=False,
            as_index=False,
        )
        .head(1)
        .reset_index(drop=True)
    )


def enrich_predictions(df):
    """
    Add devigged market probability, consensus line/edge, and best-price flags.
    """

    if df.empty:
        return df

    result = df.copy()

    if "event_id" not in result.columns:
        result["event_id"] = pd.NA

    lookup = build_devigged_lookup(result)

    result["devigged_market_prob"] = result.apply(
        lambda row: _devigged_prob_for_row(
            row,
            lookup,
        ),
        axis=1,
    )

    model_prob = result.get(
        "calibrated_probability",
        result["model_probability"],
    )

    result["edge"] = (
        model_prob
        - result["devigged_market_prob"]
    )

    result["ev"] = result.apply(
        lambda row: expected_value(
            row.get(
                "calibrated_probability",
                row["model_probability"],
            ),
            row["odds"],
        ),
        axis=1,
    )

    consensus = (
        result.groupby(
            CONSENSUS_KEYS,
            dropna=False,
        )
        .apply(
            _consensus_probs,
            include_groups=False,
        )
        .reset_index()
    )

    result = result.merge(
        consensus,
        on=CONSENSUS_KEYS,
        how="left",
    )

    result["consensus_edge"] = result.apply(
        _consensus_edge,
        axis=1,
    )

    result = _mark_best_prices(result)

    return result
