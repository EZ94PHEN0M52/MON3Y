"""
Separate Statcast stuff → strikeout model (v2).

Predicts K% from SwStr%, chase%, and velocity rolling rates, converts to an
expected strikeout count via recent batters faced, then Poisson P(Over line).

Independent of the main LightGBM ``pitcher_strikeouts.pkl`` classifier.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import Ridge

from distributional import (
    count_threshold_for_line,
    poisson_over_probability,
    poisson_under_probability,
)
from utils import normalize_version, version_models_dir

STUFF_K_FEATURES = [
    "swstr_pct_l5",
    "chase_pct_l5",
    "avg_velocity_l5",
]

STUFF_BF_FEATURE = "batters_faced_l5"

MIN_TRAINING_PITCHES = 50


def stuff_strikeout_model_path(
    version: str = "v2",
) -> Path:
    return (
        version_models_dir(version) /
        "pitcher_strikeouts_stuff.pkl"
    )


def _feature_matrix(
    frame: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:

    return (
        frame[features]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .astype(float)
    )


def _training_frame(
    pitcher_games: pd.DataFrame,
    *,
    features: list[str] | None = None,
) -> pd.DataFrame:

    features = features or STUFF_K_FEATURES

    required = set(features) | {
        "strikeouts",
        "batters_faced",
        "pitches",
    }

    missing = required - set(
        pitcher_games.columns
    )

    if missing:
        raise ValueError(
            "Pitcher feature frame missing stuff-model columns: "
            f"{sorted(missing)}"
        )

    data = pitcher_games.copy()
    data["k_pct"] = np.where(
        pd.to_numeric(
            data["batters_faced"],
            errors="coerce",
        ).gt(0),
        pd.to_numeric(
            data["strikeouts"],
            errors="coerce",
        )
        / pd.to_numeric(
            data["batters_faced"],
            errors="coerce",
        ),
        np.nan,
    )

    feature_ok = (
        _feature_matrix(data, features)
        .notna()
        .all(axis=1)
    )

    return data[
        feature_ok
        & data["k_pct"].notna()
        & pd.to_numeric(
            data["pitches"],
            errors="coerce",
        ).ge(MIN_TRAINING_PITCHES)
    ].copy()


def fit_stuff_strikeout_model(
    pitcher_games: pd.DataFrame,
    *,
    features: list[str] | None = None,
    bf_feature: str = STUFF_BF_FEATURE,
    alpha: float = 1.0,
    train_start: str | None = None,
    train_end: str | None = None,
) -> dict:

    features = list(
        features or STUFF_K_FEATURES
    )
    train = _training_frame(
        pitcher_games,
        features=features,
    )

    if train.empty:
        raise ValueError(
            "No training rows with valid stuff metrics "
            f"(need {MIN_TRAINING_PITCHES}+ pitches)."
        )

    X = _feature_matrix(
        train,
        features,
    )
    y = train["k_pct"].astype(float)

    model = Ridge(
        alpha=alpha,
        fit_intercept=True,
    )
    model.fit(X, y)

    preds = model.predict(X)
    ss_res = float(
        np.sum((y - preds) ** 2)
    )
    ss_tot = float(
        np.sum((y - y.mean()) ** 2)
    )
    r2 = (
        1.0 - ss_res / ss_tot
        if ss_tot > 0
        else np.nan
    )

    swstr_only_r2 = np.nan
    swstr_col = next(
        (
            feature
            for feature in features
            if feature.startswith("swstr")
        ),
        None,
    )
    if swstr_col is not None:
        swstr = X[swstr_col].astype(float)
        if swstr.std() > 0 and y.std() > 0:
            swstr_r = np.corrcoef(
                swstr,
                y,
            )[0, 1]
            swstr_only_r2 = float(swstr_r ** 2)

    return {
        "model_type": "stuff_k_pct_ridge",
        "market": "pitcher_strikeouts",
        "features": features,
        "bf_feature": bf_feature,
        "model": model,
        "metrics": {
            "r2": r2,
            "swstr_only_r2": swstr_only_r2,
            "n_rows": len(train),
            "mean_k_pct": float(y.mean()),
        },
        "train_start": train_start,
        "train_end": train_end,
    }


def save_stuff_strikeout_model(
    package: dict,
    version: str = "v2",
) -> Path:

    path = stuff_strikeout_model_path(
        version
    )
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    joblib.dump(
        package,
        path,
    )
    return path


def load_stuff_strikeout_model(
    version: str = "v2",
):

    path = stuff_strikeout_model_path(
        version
    )

    if not path.exists():
        return None

    return joblib.load(path)


def predict_k_pct(
    package: dict,
    feature_row: pd.Series,
) -> float:

    features = package["features"]
    values = {
        feature: feature_row.get(
            feature,
            np.nan,
        )
        for feature in features
    }

    X = (
        pd.DataFrame([values])
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    if X.isna().any(axis=None):
        return np.nan

    return float(
        package["model"].predict(X)[0]
    )


def predict_expected_strikeouts(
    package: dict,
    feature_row: pd.Series,
) -> float:

    k_pct = predict_k_pct(
        package,
        feature_row,
    )

    if not np.isfinite(k_pct):
        return np.nan

    bf_feature = package.get(
        "bf_feature",
        STUFF_BF_FEATURE,
    )
    batters_faced = pd.to_numeric(
        feature_row.get(
            bf_feature,
            np.nan,
        ),
        errors="coerce",
    )

    if not np.isfinite(batters_faced) or batters_faced <= 0:
        batters_faced = pd.to_numeric(
            feature_row.get(
                "batters_faced_season",
                np.nan,
            ),
            errors="coerce",
        )

    if not np.isfinite(batters_faced) or batters_faced <= 0:
        return np.nan

    return max(
        float(k_pct * batters_faced),
        0.0,
    )


def score_stuff_strikeout_prop(
    prop: dict | pd.Series,
    feature_row: pd.Series,
    package: dict,
) -> dict:

    rate = predict_expected_strikeouts(
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

    k_pct = predict_k_pct(
        package,
        feature_row,
    )

    return {
        "stuff_predicted_count": rate,
        "stuff_k_pct": k_pct,
        "stuff_over_probability": over_probability,
        "stuff_under_probability": under_probability,
        "stuff_model_probability": model_probability,
        "distribution": "poisson",
    }


def format_stuff_strikeout_display(
    predicted_count: float,
    over_probability: float,
) -> str:
    """
    Compact board cell: expected K and Poisson Over %.
    """

    if not np.isfinite(predicted_count):
        return "—"

    over_text = (
        f"{over_probability * 100:.0f}%"
        if np.isfinite(over_probability)
        else "—"
    )

    return f"{predicted_count:.1f} K · {over_text} Over"
