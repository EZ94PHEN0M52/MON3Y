"""
Distributional count models for player props (Phase 6 / dual-head Phase 1).

Lightweight Poisson rate models predict expected counts; P(stat > line) is
derived at inference for any posted line. Markets: batter_hits,
pitcher_strikeouts, pitcher_walks. Pitcher K and walks also use the
classifier + regressor dual-head path (see DUAL_HEAD_MARKETS).
"""

from __future__ import annotations

import joblib
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import poisson

from prop_scoring import MARKET_STAT_MAP
from utils import (
    distributional_models_dir,
    normalize_version,
)


DISTRIBUTIONAL_MARKETS = {
    "batter_hits": {
        "stat": "hits",
        "role": "batter",
    },
    "pitcher_strikeouts": {
        "stat": "strikeouts",
        "role": "pitcher",
    },
    "pitcher_walks": {
        "stat": "walks",
        "role": "pitcher",
    },
}

# Phase 1 dual-head: classifier + Poisson regressor (edge stays on classifier).
DUAL_HEAD_MARKETS = frozenset({
    "pitcher_strikeouts",
    "pitcher_walks",
})


def distributional_model_path(
    market: str,
    version: str = "v2",
) -> Path:
    return (
        distributional_models_dir(version) /
        f"{market}.pkl"
    )


def count_threshold_for_line(line: float) -> int:
    """
    Minimum integer count for an Over hit at a half-point line.

    Example: line 1.5 → need 2+ → threshold 2.
    """

    return int(float(line) + 0.5)


def poisson_over_probability(
    rate: float,
    line: float,
) -> float:
    """P(stat > line) under Poisson(rate)."""

    if not np.isfinite(rate) or rate < 0:
        return np.nan

    rate = float(rate)
    threshold = count_threshold_for_line(line)

    if threshold <= 0:
        return 1.0

    return float(
        1.0 - poisson.cdf(
            threshold - 1,
            rate,
        )
    )


def poisson_under_probability(
    rate: float,
    line: float,
) -> float:
    over = poisson_over_probability(
        rate,
        line,
    )

    if pd.isna(over):
        return np.nan

    return 1.0 - over


def side_probability_from_rate(
    rate: float,
    line: float,
    side: str,
) -> float:
    side = str(side).strip().lower()

    if side == "over":
        return poisson_over_probability(
            rate,
            line,
        )

    if side == "under":
        return poisson_under_probability(
            rate,
            line,
        )

    return np.nan


def fit_rate_model(
    X: pd.DataFrame,
    y: pd.Series,
):
    """
    Train a LightGBM regressor to predict expected count (Poisson rate).
    """

    model = lgb.LGBMRegressor(
        objective="poisson",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbose=-1,
    )

    model.fit(
        X,
        y,
    )

    return model


def save_distributional_model(
    market: str,
    package: dict,
    version: str = "v2",
) -> Path:
    path = distributional_model_path(
        market,
        version,
    )
    joblib.dump(package, path)
    return path


def load_distributional_model(
    market: str,
    version: str = "v2",
):
    path = distributional_model_path(
        market,
        version,
    )

    if not path.exists():
        return None

    return joblib.load(path)


def predict_rate(
    package: dict,
    feature_row: pd.Series,
) -> float:
    features = package["features"]
    values = {}

    for feature in features:
        values[feature] = feature_row.get(
            feature,
            np.nan,
        )

    X = (
        pd.DataFrame([values])
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
    )

    rate = float(
        package["model"].predict(X)[0]
    )

    return max(rate, 0.0)


def score_distributional_prop(
    prop: dict | pd.Series,
    feature_row: pd.Series,
    package: dict,
):
    rate = predict_rate(
        package,
        feature_row,
    )

    line = float(prop["line"])
    side = str(prop["side"]).strip().lower()

    over_probability = poisson_over_probability(
        rate,
        line,
    )
    under_probability = poisson_under_probability(
        rate,
        line,
    )

    model_probability = (
        over_probability
        if side == "over"
        else under_probability
    )

    return {
        "predicted_count": rate,
        "predicted_rate": rate,
        "over_probability": over_probability,
        "under_probability": under_probability,
        "model_probability": model_probability,
        "distribution": "poisson",
    }


def market_supports_distributional(
    market: str,
) -> bool:
    return market in DISTRIBUTIONAL_MARKETS


def market_supports_dual_head(
    market: str,
) -> bool:
    return market in DUAL_HEAD_MARKETS
