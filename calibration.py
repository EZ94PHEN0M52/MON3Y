"""
Probability calibration for prop models (Phase 6).

Fits isotonic regression or Platt scaling on held-out backtest rows and
applies calibrators at inference time. Missing calibrators fall back to raw
model probabilities.
"""

from __future__ import annotations

import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from utils import calibrators_dir, normalize_version


MIN_CALIBRATION_SAMPLES = 100
DEFAULT_METHOD = "isotonic"


def calibrator_path(
    market: str,
    version: str = "v2",
) -> Path:
    return (
        calibrators_dir(version) /
        f"{market}.pkl"
    )


def fit_calibrator(
    probabilities: np.ndarray,
    labels: np.ndarray,
    method: str = DEFAULT_METHOD,
):
    """
    Fit a calibrator mapping raw P(Over) → calibrated P(Over).

    ``labels`` are binary: 1 if actual_stat > line, else 0.
    """

    probs = np.asarray(
        probabilities,
        dtype=float,
    ).reshape(-1)

    y = np.asarray(
        labels,
        dtype=int,
    ).reshape(-1)

    mask = (
        np.isfinite(probs)
        & np.isfinite(y)
    )

    probs = probs[mask]
    y = y[mask]

    if len(probs) < MIN_CALIBRATION_SAMPLES:
        return None

    probs = np.clip(
        probs,
        1e-6,
        1.0 - 1e-6,
    )

    method = str(method).lower().strip()

    if method == "platt":
        model = LogisticRegression(
            max_iter=1000,
        )
        model.fit(
            probs.reshape(-1, 1),
            y,
        )

        return {
            "method": "platt",
            "model": model,
            "n_samples": len(probs),
        }

    if method != "isotonic":
        raise ValueError(
            f"Unsupported calibration method {method!r}. "
            "Use 'isotonic' or 'platt'."
        )

    iso = IsotonicRegression(
        out_of_bounds="clip",
        y_min=0.0,
        y_max=1.0,
    )
    iso.fit(probs, y)

    return {
        "method": "isotonic",
        "model": iso,
        "n_samples": len(probs),
    }


def apply_calibration(
    package: dict | None,
    raw_over_probability: float,
) -> float:
    """
    Return calibrated P(Over). Falls back to raw when no calibrator exists.
    """

    raw = float(raw_over_probability)

    if package is None:
        return raw

    model = package.get("model")

    if model is None:
        return raw

    method = package.get("method", DEFAULT_METHOD)
    clipped = float(
        np.clip(raw, 1e-6, 1.0 - 1e-6)
    )

    if method == "platt":
        calibrated = model.predict_proba(
            np.array([[clipped]])
        )[0, 1]
    else:
        calibrated = float(
            model.predict([clipped])[0]
        )

    return float(
        np.clip(calibrated, 0.0, 1.0)
    )


def save_calibrator(
    market: str,
    package: dict,
    version: str = "v2",
) -> Path:
    path = calibrator_path(
        market,
        version,
    )
    joblib.dump(package, path)
    return path


def load_calibrator(
    market: str,
    version: str = "v2",
):
    path = calibrator_path(
        market,
        version,
    )

    if not path.exists():
        return None

    return joblib.load(path)


def load_all_calibrators(
    version: str = "v2",
) -> dict[str, dict]:
    version = normalize_version(version)
    directory = calibrators_dir(version)
    calibrators = {}

    for path in directory.glob("*.pkl"):
        market = path.stem
        calibrators[market] = joblib.load(path)

    return calibrators


def calibrate_over_under(
    market: str,
    raw_over_probability: float,
    version: str = "v2",
    calibrator: dict | None = None,
):
    """
    Calibrate Over/Under probabilities for a market.

    Returns (over, under, raw_side_prob, calibrated_side_prob) where side
    probabilities are Over-specific; callers derive side-specific values.
    """

    if calibrator is None:
        calibrator = load_calibrator(
            market,
            version,
        )

    raw_over = float(raw_over_probability)
    calibrated_over = apply_calibration(
        calibrator,
        raw_over,
    )
    raw_under = 1.0 - raw_over
    calibrated_under = 1.0 - calibrated_over

    return {
        "over_probability": calibrated_over,
        "under_probability": calibrated_under,
        "raw_over_probability": raw_over,
        "raw_under_probability": raw_under,
    }


def fit_calibrators_from_frame(
    frame: pd.DataFrame,
    version: str = "v2",
    method: str = DEFAULT_METHOD,
    min_samples: int = MIN_CALIBRATION_SAMPLES,
) -> dict[str, Path]:
    """
    Fit one calibrator per market from a backtest DataFrame.

    Expects columns ``market``, ``actual_over``, and either
    ``raw_over_probability`` or ``over_probability`` (raw preferred).
    """

    saved = {}

    if frame.empty:
        return saved

    prob_col = (
        "raw_over_probability"
        if "raw_over_probability" in frame.columns
        else "over_probability"
    )

    for market, group in frame.groupby("market"):
        package = fit_calibrator(
            group[prob_col].values,
            group["actual_over"].values,
            method=method,
        )

        if package is None:
            continue

        if package["n_samples"] < min_samples:
            continue

        path = save_calibrator(
            market,
            package,
            version=version,
        )
        saved[market] = path

    return saved
