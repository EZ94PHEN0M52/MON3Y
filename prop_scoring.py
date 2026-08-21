import joblib
import numpy as np
import pandas as pd

from calibration import (
    calibrate_over_under,
    load_calibrator,
)
from utils import (
    version_models_dir,
    american_to_implied_probability,
    expected_value,
    normalize_player_key,
)


MODEL_MAP = {
    "batter_hits":
        "batter_hits.pkl",
    "batter_home_runs":
        "batter_home_runs.pkl",
    "batter_total_bases":
        "batter_total_bases.pkl",
    "batter_rbis":
        "batter_rbi.pkl",
    "batter_runs_scored":
        "batter_runs.pkl",
    "batter_walks":
        "batter_walks.pkl",
    "batter_hits_runs_rbis":
        "batter_hits_runs_rbis.pkl",
    "batter_stolen_bases":
        "batter_stolen_bases.pkl",
    "pitcher_strikeouts":
        "pitcher_strikeouts.pkl",
    "pitcher_walks":
        "pitcher_walks.pkl",
    "pitcher_hits_allowed":
        "pitcher_hits_allowed.pkl",
    "pitcher_outs":
        "pitcher_outs.pkl",
    "pitcher_earned_runs":
        "pitcher_earned_runs.pkl",
}


MARKET_STAT_MAP = {
    "batter_hits": "hits",
    "batter_home_runs": "home_runs",
    "batter_total_bases": "total_bases",
    "batter_rbis": "rbi",
    "batter_runs_scored": "runs",
    "batter_walks": "walks",
    "batter_hits_runs_rbis": "hits_runs_rbis",
    "batter_stolen_bases": "stolen_bases",
    "pitcher_strikeouts": "strikeouts",
    "pitcher_walks": "walks",
    "pitcher_hits_allowed": "hits_allowed",
    "pitcher_outs": "outs",
    "pitcher_earned_runs": "earned_runs",
}


def load_model(
    filename,
    version="v2",
):
    path = (
        version_models_dir(version) /
        filename
    )

    if not path.exists():
        return None

    return joblib.load(path)


def _scalar_value(
    value,
    default=np.nan,
):
    if isinstance(value, pd.Series):
        if value.empty:
            return default
        return value.iloc[0]

    if isinstance(value, pd.DataFrame):
        if value.empty:
            return default
        return value.iloc[0, 0]

    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        return value.flat[0]

    return value


def _safe_notna(value) -> bool:
    value = _scalar_value(value, np.nan)
    try:
        return bool(pd.notna(value))
    except (ValueError, TypeError):
        return False


def _normalize_feature_row(feature_row):
    if isinstance(feature_row, pd.DataFrame):
        if feature_row.empty:
            return pd.Series(dtype=float)
        return feature_row.iloc[0]

    return feature_row


def fuzzy_player_match(
    player_name,
    candidates,
):
    if not isinstance(
        player_name,
        str,
    ):
        return None

    name = normalize_player_key(player_name)
    normalized = candidates.map(normalize_player_key)

    exact = candidates[normalized.eq(name)]

    if len(exact) > 0:
        return exact.iloc[0]

    parts = name.split()
    if len(parts) < 2:
        return None

    first_name, last_name = parts[0], parts[-1]
    last_names = normalized.str.split().str[-1]
    first_names = normalized.str.split().str[0]
    matches = candidates[
        last_names.eq(last_name)
        & first_names.eq(first_name)
    ]

    if len(matches) == 1:
        return matches.iloc[0]

    return None


def derived_line_features(
    prop,
    feature_row,
):
    feature_row = _normalize_feature_row(
        feature_row
    )
    market = _scalar_value(prop.get("market"), None)
    stat_col = MARKET_STAT_MAP.get(market)
    line = _scalar_value(prop.get("line"), np.nan)
    season_avg = np.nan

    if stat_col is not None:
        season_avg = _scalar_value(
            feature_row.get(
                f"{stat_col}_season",
                np.nan,
            ),
            np.nan,
        )

    line_vs_season_avg = np.nan

    if _safe_notna(line) and _safe_notna(season_avg):
        line_vs_season_avg = (
            float(line) - float(season_avg)
        )

    market_implied = prop.get(
        "market_implied_over_prob",
        np.nan,
    )

    return {
        "market_implied_over_prob": market_implied,
        "line_vs_season_avg": line_vs_season_avg,
    }


def build_feature_vector(
    feature_row,
    line,
    feature_names,
    prop=None,
):
    feature_row = _normalize_feature_row(
        feature_row
    )
    values = {}

    for feature in feature_names:
        if feature in (
            "market_implied_over_prob",
            "line_vs_season_avg",
        ):
            continue

        values[feature] = _scalar_value(
            feature_row.get(
                feature,
                np.nan,
            ),
            np.nan,
        )

    values["line"] = _scalar_value(line, np.nan)

    if prop is not None:
        values.update(
            derived_line_features(
                prop,
                feature_row,
            )
        )
    else:
        values["market_implied_over_prob"] = np.nan
        values["line_vs_season_avg"] = np.nan

    for feature in feature_names:
        if feature not in values:
            values[feature] = np.nan

    X = pd.DataFrame([values])

    return (
        X.replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
    )


def score_prop(
    prop,
    feature_row,
    package,
    version="v2",
    calibrator=None,
):
    X = build_feature_vector(
        feature_row,
        prop["line"],
        package["features"],
        prop=prop,
    )

    proba = package["model"].predict_proba(X)[0]
    raw_over_probability = float(proba[1])
    raw_under_probability = float(proba[0])

    if calibrator is None:
        calibrator = load_calibrator(
            prop["market"],
            version,
        )

    calibrated = calibrate_over_under(
        prop["market"],
        raw_over_probability,
        version=version,
        calibrator=calibrator,
    )

    over_probability = calibrated["over_probability"]
    under_probability = calibrated["under_probability"]

    side = str(prop["side"]).strip().lower()
    raw_model_probability = (
        raw_over_probability
        if side == "over"
        else raw_under_probability
    )
    calibrated_probability = (
        over_probability
        if side == "over"
        else under_probability
    )
    model_probability = calibrated_probability

    market_probability = (
        american_to_implied_probability(
            prop["odds"]
        )
    )

    edge = (
        model_probability -
        market_probability
    )

    ev = expected_value(
        model_probability,
        prop["odds"],
    )

    return {
        "over_probability": over_probability,
        "under_probability": under_probability,
        "raw_over_probability": raw_over_probability,
        "raw_under_probability": raw_under_probability,
        "model_probability": model_probability,
        "raw_model_probability": raw_model_probability,
        "calibrated_probability": calibrated_probability,
        "market_probability": market_probability,
        "edge": edge,
        "ev": ev,
    }
