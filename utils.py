from datetime import date, timedelta, datetime, timezone
from pathlib import Path
import os
import unicodedata
from zoneinfo import ZoneInfo

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


NAME_SUFFIXES = frozenset({
    "jr",
    "sr",
    "ii",
    "iii",
    "iv",
})


MLB_SCHEDULE_TZ = ZoneInfo("America/New_York")


def mlb_schedule_date(when: datetime | None = None) -> str:
    """MLB slate calendar date in US Eastern (matches prop commence → game_date)."""
    if when is None:
        when = datetime.now(timezone.utc)
    elif when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)

    return when.astimezone(MLB_SCHEDULE_TZ).strftime("%Y-%m-%d")


def game_date_from_commence(commence_time) -> str | None:
    """Map Odds API commence_time to the MLB Eastern schedule date."""
    if commence_time is None or pd.isna(commence_time):
        return None

    try:
        return (
            pd.to_datetime(commence_time, utc=True)
            .tz_convert(MLB_SCHEDULE_TZ)
            .strftime("%Y-%m-%d")
        )
    except (TypeError, ValueError):
        return None


def normalize_player_key(name) -> str:
    """Lowercase player name with accents stripped for cross-source matching."""
    text = str(name).strip().lower()
    text = (
        unicodedata
        .normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode()
    )
    # Odds API often drops periods in initials (JT Ginn vs J.T. Ginn in Statcast).
    text = text.replace(".", "")
    return " ".join(text.split())


def strip_name_suffix(key: str) -> str:
    """Drop trailing generational suffixes (Jr, Sr, II, etc.) from a normalized key."""
    parts = key.split()
    while parts and parts[-1] in NAME_SUFFIXES:
        parts.pop()
    return " ".join(parts)


PITCHER_SP_PROP_MARKETS = frozenset({
    "pitcher_strikeouts",
    "pitcher_walks",
    "pitcher_hits_allowed",
    "pitcher_outs",
})

DEFAULT_SP_LAG_THRESHOLD = 2

DAILY_PROBABLES_PATH = PROCESSED_DIR / "daily_probables.parquet"
CURRENT_PROPS_PATH = PROCESSED_DIR / "current_props.parquet"


def slate_dates_from_props(props: pd.DataFrame) -> set[str]:
    """Eastern schedule dates present in a props dataframe."""
    if props is None or props.empty or "commence_time" not in props.columns:
        return set()

    dates = props["commence_time"].map(game_date_from_commence)
    return {str(value) for value in dates if value}


def slate_games_from_props(props: pd.DataFrame) -> pd.DataFrame:
    """Unique games in props with Eastern schedule date and team names."""
    columns = ["game_date", "home_team", "away_team", "game"]
    if props is None or props.empty:
        return pd.DataFrame(columns=columns)

    required = {"commence_time", "home_team", "away_team"}
    if not required.issubset(props.columns):
        return pd.DataFrame(columns=columns)

    working = props.dropna(
        subset=["home_team", "away_team", "commence_time"],
        how="any",
    ).copy()
    if working.empty:
        return pd.DataFrame(columns=columns)

    working["game_date"] = working["commence_time"].map(
        game_date_from_commence
    )
    working = working[working["game_date"].notna()].copy()
    if working.empty:
        return pd.DataFrame(columns=columns)

    if "game" not in working.columns:
        working["game"] = (
            working["away_team"].astype(str)
            + " @ "
            + working["home_team"].astype(str)
        )

    return (
        working.drop_duplicates(
            subset=["game_date", "home_team", "away_team"],
        )[columns]
        .reset_index(drop=True)
    )


def _props_game_key(row) -> tuple:
    event_id = row.get("event_id")
    if pd.notna(event_id) and str(event_id).strip():
        return ("event", str(event_id))

    home = row.get("home_team")
    away = row.get("away_team")
    if pd.notna(home) and pd.notna(away):
        return (
            "teams",
            canonical_odds_team_key(home),
            canonical_odds_team_key(away),
        )

    return ("unknown",)


def _game_label(home_team, away_team) -> str:
    if pd.notna(home_team) and pd.notna(away_team):
        return f"{away_team} @ {home_team}"
    return "unknown game"


def _is_known_game_key(key) -> bool:
    """True when props row maps to an identifiable game."""
    return (
        isinstance(key, tuple)
        and len(key) > 0
        and key[0] != "unknown"
    )


def analyze_sp_prop_coverage(
    props: pd.DataFrame,
    *,
    probables: pd.DataFrame | None = None,
    lag_threshold: int = DEFAULT_SP_LAG_THRESHOLD,
) -> dict:
    """
    Compare posted pitcher props to expected starting-pitcher coverage.

    Expect ~2 SPs with props per game (one per team). Returns a summary dict;
    does not print or abort.
    """
    empty = {
        "ok": True,
        "game_count": 0,
        "team_count": 0,
        "expected_min_pitchers": 0,
        "pitcher_count": 0,
        "warnings": [],
        "games_missing_sp": [],
    }

    if props is None or props.empty:
        return empty

    if probables is None and DAILY_PROBABLES_PATH.exists():
        try:
            probables = pd.read_parquet(DAILY_PROBABLES_PATH)
        except Exception:
            probables = None

    games = (
        props.dropna(subset=["home_team", "away_team"], how="any")
        if {"home_team", "away_team"}.issubset(props.columns)
        else pd.DataFrame()
    )

    if games.empty and "event_id" in props.columns:
        games = props.dropna(subset=["event_id"])

    if games.empty:
        return empty

    games = games.copy()
    games["_game_key"] = games.apply(_props_game_key, axis=1)
    games = games[
        games["_game_key"].map(_is_known_game_key)
    ]

    game_count = games["_game_key"].nunique()
    if game_count == 0:
        return empty

    team_count = game_count * 2
    expected_min_pitchers = max(
        0,
        team_count - max(0, int(lag_threshold)),
    )

    sp_props = props[
        props["market"].isin(PITCHER_SP_PROP_MARKETS)
        & props["player"].notna()
    ].copy()

    if not sp_props.empty:
        sp_props["_game_key"] = sp_props.apply(_props_game_key, axis=1)
        sp_props["_player_key"] = sp_props["player"].map(normalize_player_key)
    else:
        sp_props["_game_key"] = pd.Series(dtype=object)
        sp_props["_player_key"] = pd.Series(dtype=object)

    pitcher_count = (
        sp_props["_player_key"].nunique()
        if not sp_props.empty
        else 0
    )

    games_missing_sp = []
    game_meta = (
        games.groupby("_game_key", as_index=False)
        .agg(
            home_team=("home_team", "first"),
            away_team=("away_team", "first"),
        )
    )

    prob_lookup = {}
    if probables is not None and not probables.empty:
        for _, row in probables.iterrows():
            key = (
                "teams",
                canonical_odds_team_key(row.get("home_team")),
                canonical_odds_team_key(row.get("away_team")),
            )
            prob_lookup[key] = row

    for _, game in game_meta.iterrows():
        game_key = game["_game_key"]
        game_sp = sp_props[
            sp_props["_game_key"] == game_key
        ]
        posted_keys = set(game_sp["_player_key"].dropna().unique())
        posted_count = len(posted_keys)

        if posted_count >= 2:
            continue

        label = _game_label(game["home_team"], game["away_team"])
        missing_sides = []

        prob_row = prob_lookup.get(game_key)
        if prob_row is not None:
            for side, name_col in (
                ("away", "away_sp_name"),
                ("home", "home_sp_name"),
            ):
                sp_name = prob_row.get(name_col)
                if not isinstance(sp_name, str) or not sp_name.strip():
                    missing_sides.append(f"{side} SP TBD (no probable)")
                    continue

                if normalize_player_key(sp_name) not in posted_keys:
                    missing_sides.append(f"{side} SP {sp_name}")
        elif posted_count == 0:
            missing_sides.append("no SP props posted")
        else:
            missing_sides.append(
                f"only {posted_count}/2 SP(s) with props"
            )

        games_missing_sp.append({
            "game": label,
            "posted_sp_count": posted_count,
            "missing": missing_sides,
        })

    warnings = []
    if pitcher_count < expected_min_pitchers:
        warnings.append(
            "Only "
            f"{pitcher_count} unique pitcher(s) with SP props "
            f"(expected at least {expected_min_pitchers} for "
            f"{game_count} game(s), ~2 per game minus lag "
            f"threshold {lag_threshold})."
        )
    elif pitcher_count < team_count:
        warnings.append(
            f"{pitcher_count} unique pitcher(s) with SP props "
            f"for {team_count} teams playing "
            f"({game_count} game(s))."
        )

    if games_missing_sp:
        detail_lines = []
        for item in games_missing_sp:
            missing_text = "; ".join(item["missing"])
            detail_lines.append(
                f"  - {item['game']}: {missing_text}"
            )
        warnings.append(
            "Games missing SP prop coverage:\n"
            + "\n".join(detail_lines)
        )

    ok = pitcher_count >= expected_min_pitchers

    return {
        "ok": ok,
        "game_count": int(game_count),
        "team_count": int(team_count),
        "expected_min_pitchers": int(expected_min_pitchers),
        "pitcher_count": int(pitcher_count),
        "warnings": warnings,
        "games_missing_sp": games_missing_sp,
    }


def warn_sp_prop_coverage(
    props: pd.DataFrame,
    *,
    probables: pd.DataFrame | None = None,
    lag_threshold: int = DEFAULT_SP_LAG_THRESHOLD,
    context: str = "",
) -> dict:
    """
    Print non-fatal warnings when SP props look incomplete.

    Advisory only — never blocks predict or Streamlit.
    """
    result = analyze_sp_prop_coverage(
        props,
        probables=probables,
        lag_threshold=lag_threshold,
    )

    if result["game_count"] == 0:
        return result

    prefix = "WARNING: SP prop coverage"
    if context:
        prefix = f"{prefix} ({context})"

    print()
    print("=" * 60)
    print(f"{prefix}")
    print("=" * 60)
    print(
        f"Games: {result['game_count']} | "
        f"Teams: {result['team_count']} | "
        f"Pitchers with SP props: {result['pitcher_count']} | "
        f"Expected min: {result['expected_min_pitchers']}"
    )

    if result["ok"]:
        print("OK — SP prop coverage looks complete.")
    else:
        for message in result["warnings"]:
            print(message)

    if result["games_missing_sp"] and result["ok"]:
        print(
            "Per-game gaps (within aggregate lag threshold):"
        )
        for item in result["games_missing_sp"]:
            missing_text = "; ".join(item["missing"])
            print(f"  - {item['game']}: {missing_text}")

    print()

    return result


class LiveFetchDisabledError(RuntimeError):
    """Raised when DISABLE_LIVE_FETCH blocks a network/API download."""


def live_fetch_disabled() -> bool:
    """True when live Statcast/Odds/MLB API calls must not run (backtests, offline)."""
    value = os.getenv("DISABLE_LIVE_FETCH", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def require_live_fetch(operation: str) -> None:
    """
    Abort if DISABLE_LIVE_FETCH=1.

    Set DISABLE_LIVE_FETCH=1 for backtests and any job that should read only
    data/raw/ and data/processed/ parquets.
    """
    if live_fetch_disabled():
        raise LiveFetchDisabledError(
            f"Live fetch blocked ({operation}). "
            "DISABLE_LIVE_FETCH=1 is set — use cached parquets or unset the env var."
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


MLB_STATS_API = "https://statsapi.mlb.com/api/v1"


def mlb_games_on_date(game_date: str) -> bool | None:
    """True if MLB games are scheduled; None when schedule is unavailable."""
    if live_fetch_disabled():
        return None

    try:
        import requests

        response = requests.get(
            f"{MLB_STATS_API}/schedule",
            params={"sportId": 1, "date": game_date},
            timeout=10,
        )
        response.raise_for_status()
        dates = response.json().get("dates") or []
        if not dates:
            return False
        return bool(dates[0].get("games"))
    except Exception:
        return None


def last_mlb_game_date_on_or_before(
    end_date: str,
    max_lookback: int = 10,
) -> date | None:
    """Latest date <= end_date with scheduled MLB games, or None if unknown."""
    end = pd.Timestamp(end_date).date()
    for offset in range(max_lookback + 1):
        candidate = end - timedelta(days=offset)
        games = mlb_games_on_date(candidate.isoformat())
        if games is True:
            return candidate
        if games is None:
            return None
    return None


def required_max_game_date(
    end_date: str,
    statcast_path: Path | None = None,
) -> date:
    """Latest game_date data should cover through end_date.

    Uses the MLB schedule so an off-day end_date (no games) does not require
    rows on that calendar date. Falls back to Statcast max when offline.
    """
    last_game = last_mlb_game_date_on_or_before(end_date)
    if last_game is not None:
        return last_game

    if statcast_path is not None:
        statcast_max = parquet_max_game_date(statcast_path)
        if statcast_max is not None:
            return statcast_max

    return pd.Timestamp(end_date).date()


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

    required = required_max_game_date(end_date, path)
    return max_date < required


def feature_parquet_needs_refresh(
    path: Path,
    start_date: str,
    end_date: str,
) -> bool:
    """True when a feature parquet is behind Statcast or the required game date."""
    if not path.exists():
        return False

    max_date = parquet_max_game_date(path)
    if max_date is None:
        return False

    statcast_path = statcast_raw_path(start_date, end_date)
    statcast_max = parquet_max_game_date(statcast_path)
    if statcast_max is not None:
        return max_date < statcast_max

    required = required_max_game_date(end_date, statcast_path)
    return max_date < required


def _feature_prefix(version: str, role: str) -> str:
    version = normalize_version(version)
    if version == "v1":
        return (
            "batter_features_"
            if role == "batter"
            else "pitcher_features_"
        )
    return (
        "batter_features_v2_"
        if role == "batter"
        else "pitcher_features_v2_"
    )


def find_covering_feature_path(
    start_date,
    end_date,
    version="v2",
    role="batter",
):
    """Smallest on-disk feature parquet that covers [start_date, end_date]."""
    version = normalize_version(version)
    prefix = _feature_prefix(version, role)
    candidates = []

    for path in PROCESSED_DIR.glob(f"{prefix}*.parquet"):
        stem = path.stem.replace(prefix, "", 1)
        if "_" not in stem:
            continue

        file_start, file_end = stem.split("_", 1)
        if file_start <= start_date and file_end >= end_date:
            span = (
                pd.to_datetime(file_end)
                - pd.to_datetime(file_start)
            ).days
            candidates.append((span, path))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def resolve_feature_path(
    start_date,
    end_date,
    version="v2",
    role="batter",
):
    """Exact feature path if present, else smallest covering parquet."""
    exact = (
        batter_features_path(start_date, end_date, version)
        if role == "batter"
        else pitcher_features_path(start_date, end_date, version)
    )
    if exact.exists():
        return exact

    covering = find_covering_feature_path(
        start_date,
        end_date,
        version,
        role,
    )
    if covering is not None:
        return covering

    return exact


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


BATTER_SCORE_VALIDATION_PATH = (
    BACKTEST_DIR / "batter_score_validation.json"
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
# VERSION COMPARE (multi-generation board)
# ---------------------------------------------------------

VERSION_COMPARE_SLOTS = (
    {
        "key": "v1",
        "label": "V1",
        "description": "Rolling player form (frozen baseline)",
        "model_version": "v1",
    },
    {
        "key": "v2",
        "label": "V2",
        "description": "Opponent, handedness, and park features",
        "model_version": "v2",
    },
    {
        "key": "v3",
        "label": "V3",
        "description": "Phases 1–6, calibration, distributional models (tag v3)",
        "model_version": "v2",
    },
    {
        "key": "main",
        "label": "Main",
        "description": "Active development workspace (current daily board)",
        "model_version": "v2",
    },
)


def compare_predictions_path(slot_key):
    """
    CSV path for a version-compare column.

    v1 → predictions.csv; v2/main → predictions_v2.csv;
    v3 → predictions_v3.csv (copy from frozen mlb-prop-model-v3/ if needed).
    """
    slot_key = str(slot_key).lower().strip()

    if slot_key == "v1":
        return predictions_path("v1")

    if slot_key == "v2":
        return PREDICTIONS_DIR / "predictions_v2.csv"

    if slot_key == "v3":
        return PREDICTIONS_DIR / "predictions_v3.csv"

    if slot_key == "main":
        return predictions_path("v2")

    raise ValueError(
        f"Unknown compare slot {slot_key!r}. "
        f"Use one of: {[s['key'] for s in VERSION_COMPARE_SLOTS]}."
    )


def version_has_models(version="v2"):
    """True when at least one market model .pkl exists for the version."""
    version = normalize_version(version)
    models_dir = version_models_dir(version)
    return any(models_dir.glob("*.pkl"))


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
