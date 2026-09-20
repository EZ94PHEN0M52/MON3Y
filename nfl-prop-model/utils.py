"""Shared paths, env, NFL calendar, and injury status helpers."""

from __future__ import annotations

import os
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"
BACKTEST_DIR = DATA_DIR / "backtest"
MODELS_DIR = BASE_DIR / "models"

SUPPORTED_VERSIONS = ("v1", "v2")

NFL_SCHEDULE_TZ = ZoneInfo("America/New_York")

NAME_SUFFIXES = frozenset({
    "jr",
    "sr",
    "ii",
    "iii",
    "iv",
})

# Board / predict gates — players in these buckets are inactive by default.
INACTIVE_REPORT_STATUSES = frozenset({
    "out",
    "doubtful",
    "injured reserve",
    "ir",
    "pup",
    "physically unable to perform",
    "suspended",
})

QUESTIONABLE_REPORT_STATUSES = frozenset({
    "questionable",
})

for directory in [
    DATA_DIR,
    RAW_DIR,
    PROCESSED_DIR,
    PREDICTIONS_DIR,
    BACKTEST_DIR,
    MODELS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

load_dotenv(BASE_DIR / ".env")

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
NFL_SPORT = "americanfootball_nfl"


class LiveFetchDisabled(RuntimeError):
    """Raised when DISABLE_LIVE_FETCH blocks a network download."""


def live_fetch_disabled() -> bool:
    value = os.getenv("DISABLE_LIVE_FETCH", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def require_live_fetch(operation: str) -> None:
    if live_fetch_disabled():
        raise LiveFetchDisabled(
            f"{operation} blocked: DISABLE_LIVE_FETCH=1 is set — "
            "use cached parquets or unset the env var."
        )


def normalize_version(version: str | None) -> str:
    if version is None:
        return "v1"
    value = version.strip().lower()
    if value not in SUPPORTED_VERSIONS:
        raise ValueError(f"Unsupported version {version!r}; use v1 or v2")
    return value


def version_models_dir(version: str) -> Path:
    path = MODELS_DIR / normalize_version(version)
    path.mkdir(parents=True, exist_ok=True)
    return path


def predictions_path(version: str) -> Path:
    v = normalize_version(version)
    name = "predictions_v1.csv" if v == "v1" else "predictions_v2.csv"
    return PREDICTIONS_DIR / name


def player_stats_raw_path(season: int) -> Path:
    return RAW_DIR / f"player_stats_{season}.parquet"


def injuries_raw_path(season: int) -> Path:
    return RAW_DIR / f"injuries_{season}.parquet"


def schedules_raw_path(season: int) -> Path:
    return RAW_DIR / f"schedules_{season}.parquet"


def rosters_weekly_raw_path(season: int) -> Path:
    return RAW_DIR / f"rosters_weekly_{season}.parquet"


def current_props_path() -> Path:
    return PROCESSED_DIR / "current_props.parquet"


def current_injuries_path() -> Path:
    """Latest injury + roster status snapshot for board / predict gates."""
    return PROCESSED_DIR / "current_injuries.parquet"


def sleeper_props_path() -> Path:
    return PROCESSED_DIR / "sleeper_props.parquet"


def parse_commence_datetime(commence_time):
    """Parse ISO strings or Unix epoch seconds/ms to a UTC pandas Timestamp."""
    if commence_time is None or (isinstance(commence_time, float) and pd.isna(commence_time)):
        return None
    if isinstance(commence_time, (int, float, np.integer, np.floating)):
        value = float(commence_time)
        if value <= 0:
            return None
        if value >= 1e11:
            return pd.to_datetime(value, unit="ms", utc=True)
        if value >= 1e9:
            return pd.to_datetime(value, unit="s", utc=True)
    try:
        return pd.to_datetime(commence_time, utc=True)
    except (TypeError, ValueError):
        return None


def nfl_slate_now(when: datetime | None = None) -> datetime:
    if when is None:
        when = datetime.now(timezone.utc)
    elif when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when.astimezone(NFL_SCHEDULE_TZ)


def normalize_injury_status(status: str | None) -> str:
    """Map free-text report_status → active | questionable | inactive."""
    if status is None:
        return "active"
    text = str(status).strip().lower()
    if not text or text in {"nan", "none", "null"}:
        return "active"
    if text in INACTIVE_REPORT_STATUSES or text.startswith("out"):
        return "inactive"
    if text in QUESTIONABLE_REPORT_STATUSES:
        return "questionable"
    # Probable / full participation / empty report → treat as active.
    return "active"


def is_inactive_status(status: str | None) -> bool:
    return normalize_injury_status(status) == "inactive"


def is_questionable_status(status: str | None) -> bool:
    return normalize_injury_status(status) == "questionable"


def american_to_implied_probability(odds) -> float:
    """Convert American odds to raw implied probability (includes vig)."""
    odds = float(odds)
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def american_to_decimal(odds) -> float:
    odds = float(odds)
    if odds > 0:
        return 1 + odds / 100
    return 1 + 100 / abs(odds)


def expected_value(probability, american_odds) -> float:
    """Expected return per $1 stake at American odds."""
    decimal_odds = american_to_decimal(american_odds)
    return probability * decimal_odds - 1


def normalize_player_name(name: str | None) -> str:
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    text = str(name).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    parts = [p for p in text.split() if p and p not in NAME_SUFFIXES]
    return " ".join(parts)


def player_features_path(version: str = "v1") -> Path:
    v = normalize_version(version)
    return PROCESSED_DIR / f"player_features_{v}.parquet"
