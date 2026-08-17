from pathlib import Path
import os

import numpy as np
from dotenv import load_dotenv


# ---------------------------------------------------------
# PROJECT DIRECTORIES
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"
MODELS_DIR = BASE_DIR / "models"

SUPPORTED_VERSIONS = ("v1", "v2")


for directory in [
    DATA_DIR,
    RAW_DIR,
    PROCESSED_DIR,
    PREDICTIONS_DIR,
    MODELS_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ---------------------------------------------------------
# ENVIRONMENT
# ---------------------------------------------------------

load_dotenv()

ODDS_API_KEY = os.getenv(
    "ODDS_API_KEY"
)

ODDS_API_BASE = (
    "https://api.the-odds-api.com/v4"
)

MLB_SPORT = "baseball_mlb"


# ---------------------------------------------------------
# VERSION PATHS
# ---------------------------------------------------------

def normalize_version(
    version
):

    version = (
        str(version)
        .lower()
        .strip()
    )

    if version not in SUPPORTED_VERSIONS:

        raise ValueError(
            f"Unsupported version {version!r}. "
            f"Use one of {SUPPORTED_VERSIONS}."
        )

    return version


def version_models_dir(
    version="v2"
):

    version = normalize_version(
        version
    )

    path = MODELS_DIR / version

    path.mkdir(
        parents=True,
        exist_ok=True
    )

    return path


def batter_features_path(
    start_date,
    end_date,
    version="v2"
):

    version = normalize_version(
        version
    )

    if version == "v1":

        return (
            PROCESSED_DIR /
            f"batter_features_{start_date}_{end_date}.parquet"
        )

    return (
        PROCESSED_DIR /
        f"batter_features_v2_{start_date}_{end_date}.parquet"
    )


def pitcher_features_path(
    start_date,
    end_date,
    version="v2"
):

    version = normalize_version(
        version
    )

    if version == "v1":

        return (
            PROCESSED_DIR /
            f"pitcher_features_{start_date}_{end_date}.parquet"
        )

    return (
        PROCESSED_DIR /
        f"pitcher_features_v2_{start_date}_{end_date}.parquet"
    )


def predictions_path(
    version="v2"
):

    version = normalize_version(
        version
    )

    if version == "v1":

        return (
            PREDICTIONS_DIR /
            "predictions.csv"
        )

    return (
        PREDICTIONS_DIR /
        "predictions_v2.csv"
    )


# ---------------------------------------------------------
# ODDS CONVERSION
# ---------------------------------------------------------

def american_to_implied_probability(
    odds
):
    """
    Convert American odds to raw implied probability.

    Does NOT remove sportsbook vig.
    """

    odds = float(odds)

    if odds > 0:
        return 100 / (odds + 100)

    return abs(odds) / (
        abs(odds) + 100
    )


def american_to_decimal(
    odds
):
    odds = float(odds)

    if odds > 0:
        return 1 + odds / 100

    return 1 + 100 / abs(odds)


def expected_value(
    probability,
    american_odds
):
    """
    Expected return per $1 wager.

    Example:
        probability = 0.60
        odds = -110
    """

    decimal_odds = american_to_decimal(
        american_odds
    )

    return (
        probability * decimal_odds
    ) - 1


# ---------------------------------------------------------
# NUMERIC HELPERS
# ---------------------------------------------------------

def safe_float(value):

    try:
        return float(value)

    except (
        TypeError,
        ValueError
    ):
        return np.nan
