from datetime import date
from pathlib import Path
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from dotenv import load_dotenv


# ---------------------------------------------------------
# PROJECT DIRECTORIES
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"
BACKTEST_DIR = DATA_DIR / "backtest"
MODELS_DIR = BASE_DIR / "models"

SUPPORTED_VERSIONS = ("v1", "v2")


for directory in [
    DATA_DIR,
    RAW_DIR,
    PROCESSED_DIR,
    PREDICTIONS_DIR,
    BACKTEST_DIR,
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

# Statcast / feature parquet abbreviations → Odds API / probables team names
TEAM_ABBR_TO_ODDS = {
    "ATH": "Athletics",
    "ATL": "Atlanta Braves",
    "AZ": "Arizona Diamondbacks",
    "BAL": "Baltimore Orioles",
    "BOS": "Boston Red Sox",
    "CHC": "Chicago Cubs",
    "CIN": "Cincinnati Reds",
    "CLE": "Cleveland Guardians",
    "COL": "Colorado Rockies",
    "CWS": "Chicago White Sox",
    "DET": "Detroit Tigers",
    "HOU": "Houston Astros",
    "KC": "Kansas City Royals",
    "LAA": "Los Angeles Angels",
    "LAD": "Los Angeles Dodgers",
    "MIA": "Miami Marlins",
    "MIL": "Milwaukee Brewers",
    "MIN": "Minnesota Twins",
    "NYM": "New York Mets",
    "NYY": "New York Yankees",
    "PHI": "Philadelphia Phillies",
    "PIT": "Pittsburgh Pirates",
    "SD": "San Diego Padres",
    "SEA": "Seattle Mariners",
    "SF": "San Francisco Giants",
    "STL": "St. Louis Cardinals",
    "TB": "Tampa Bay Rays",
    "TEX": "Texas Rangers",
    "TOR": "Toronto Blue Jays",
    "WSH": "Washington Nationals",
}

# MLB Stats API names that differ from Odds API props
MLB_TO_ODDS_TEAM = {
    "Oakland Athletics": "Athletics",
}


def canonical_odds_team_key(name_or_abbr) -> str:
    """Normalize team abbr or full name to lowercase Odds API key."""
    if not isinstance(name_or_abbr, str) or not name_or_abbr.strip():
        return ""

    cleaned = name_or_abbr.strip()
    mapped = TEAM_ABBR_TO_ODDS.get(
        cleaned.upper(),
        cleaned,
    )
    mapped = MLB_TO_ODDS_TEAM.get(mapped, mapped)
    return mapped.strip().lower()


def coerce_mlb_id(value) -> int | None:
    """Return int MLB id or None for missing/NaN/TBD."""
    if value is None:
        return None

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None

    if numeric != numeric:  # NaN
        return None

    return int(numeric)


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


def calibrators_dir(
    version="v2",
):
    path = (
        version_models_dir(version) /
        "calibrators"
    )

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def distributional_models_dir(
    version="v2",
):
    path = (
        version_models_dir(version) /
        "dist"
    )

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def statcast_raw_path(
    start_date: str,
    end_date: str,
) -> Path:
    return RAW_DIR / f"statcast_{start_date}_{end_date}.parquet"


def parquet_max_game_date(
    path: Path,
    column: str = "game_date",
) -> date | None:
    """Return the latest game_date in a feature or Statcast parquet."""
    if not path.exists():
        return None

    schema = pq.read_schema(path)
    if column not in schema.names:
        return None

    table = pq.read_table(path, columns=[column])
    if table.num_rows == 0:
        return None

    series = table[column].to_pandas()
    return pd.to_datetime(series).max().date()


def statcast_needs_refresh(
    start_date: str,
    end_date: str,
) -> bool:
    """True when raw Statcast is missing or lacks games through end_date."""
    path = statcast_raw_path(start_date, end_date)
    if not path.exists():
        return True

    max_date = parquet_max_game_date(path)
    if max_date is None:
        return True

    return max_date < pd.Timestamp(end_date).date()


def feature_parquet_needs_refresh(
    path: Path,
    end_date: str,
) -> bool:
    """True when a feature parquet exists but stops before end_date."""
    if not path.exists():
        return False

    max_date = parquet_max_game_date(path)
    if max_date is None:
        return False

    return max_date < pd.Timestamp(end_date).date()


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


def historical_odds_path(
    snapshot_date,
):
    return (
        RAW_DIR /
        "odds" /
        "historical" /
        f"date={snapshot_date}" /
        "props.parquet"
    )


def historical_game_lines_path(
    snapshot_date,
):
    return (
        RAW_DIR /
        "odds" /
        "historical" /
        f"date={snapshot_date}" /
        "game_lines.parquet"
    )


def current_game_lines_path():
    return (
        PROCESSED_DIR /
        "current_game_lines.parquet"
    )


def odds_snapshots_dir():
    path = (
        RAW_DIR /
        "odds" /
        "snapshots"
    )

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def backtest_output_path(
    start_date,
    end_date,
):
    return (
        BACKTEST_DIR /
        f"backtest_{start_date}_{end_date}.csv"
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


def predictions_best_path(
    version="v2",
):
    version = normalize_version(
        version
    )

    if version == "v1":
        return (
            PREDICTIONS_DIR /
            "predictions_best.csv"
        )

    return (
        PREDICTIONS_DIR /
        "predictions_v2_best.csv"
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


def devig_two_way(
    over_odds,
    under_odds,
):
    """
    Remove two-way vig: normalize implied Over/Under probs to sum to 1.

    Returns (fair_over_probability, fair_under_probability).
    """

    over_implied = american_to_implied_probability(
        over_odds
    )
    under_implied = american_to_implied_probability(
        under_odds
    )

    total = over_implied + under_implied

    if total <= 0:
        return np.nan, np.nan

    return (
        over_implied / total,
        under_implied / total,
    )


# Optional sharp-book weights for consensus (equal weight when omitted).
SHARP_BOOK_WEIGHTS = {
    "pinnacle": 2.0,
    "circa": 2.0,
    "bookmaker": 1.5,
}


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
