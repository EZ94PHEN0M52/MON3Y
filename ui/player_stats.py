"""Load per-game stat history and rolling over-rates for the UI."""

from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd

from ui.market_filters import EXCLUDED_UI_MARKETS
from utils import PROCESSED_DIR, normalize_version

MARKET_STAT_MAP = {
    "batter_hits": ("batter", "hits"),
    "batter_home_runs": ("batter", "home_runs"),
    "batter_total_bases": ("batter", "total_bases"),
    "batter_rbis": ("batter", "rbi"),
    "batter_runs_scored": ("batter", "runs"),
    "batter_walks": ("batter", "walks"),
    "batter_hits_runs_rbis": ("batter", "hits_runs_rbis"),
    "batter_stolen_bases": ("batter", "stolen_bases"),
    "pitcher_strikeouts": ("pitcher", "strikeouts"),
    "pitcher_walks": ("pitcher", "walks"),
    "pitcher_hits_allowed": ("pitcher", "hits_allowed"),
    "pitcher_outs": ("pitcher", "outs"),
    "pitcher_earned_runs": ("pitcher", "earned_runs"),
}

BATTER_MARKETS = {
    market
    for market, (kind, _) in MARKET_STAT_MAP.items()
    if kind == "batter"
}
PITCHER_MARKETS = {
    market
    for market, (kind, _) in MARKET_STAT_MAP.items()
    if kind == "pitcher"
}


def _feature_prefix(kind, version):
    version = normalize_version(version)
    if version == "v2":
        return f"{kind}_features_v2_"
    return f"{kind}_features_"


def find_latest_feature_path(kind, version="v2"):
    """Return the feature parquet with the latest end date in its filename."""
    prefix = _feature_prefix(kind, version)
    candidates = list(PROCESSED_DIR.glob(f"{prefix}*.parquet"))

    if normalize_version(version) == "v1":
        candidates = [
            path
            for path in candidates
            if "_v2_" not in path.name
        ]

    if not candidates:
        return None

    return max(candidates, key=_filename_end_date)


def _filename_end_date(path):
    return path.stem.rsplit("_", 1)[-1]


def _feature_cache_key(path):
    resolved = Path(path)
    return (str(resolved), resolved.stat().st_mtime_ns)


@lru_cache(maxsize=8)
def _load_features(path_key):
    path_str, _mtime = path_key
    return pd.read_parquet(path_str)


def get_last_n_games(player_name, market, version="v2", n=10):
    """
    Return a dataframe indexed by game label with one stat column for the
    player's last *n* games (oldest to newest), or None if unavailable.
    """
    if market not in MARKET_STAT_MAP:
        return None

    kind, stat_col = MARKET_STAT_MAP[market]
    path = find_latest_feature_path(kind, version)
    if path is None:
        return None

    features = _load_features(_feature_cache_key(path))
    if "player_name" not in features.columns or stat_col not in features.columns:
        return None

    name = str(player_name).strip().lower()
    player_rows = features[
        features["player_name"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq(name)
    ]

    if player_rows.empty:
        return None

    recent = (
        player_rows.sort_values("game_date")
        .tail(n)
        .copy()
    )

    recent["game_label"] = (
        pd.to_datetime(recent["game_date"])
        .dt.strftime("%m/%d")
    )

    chart = recent.set_index("game_label")[[stat_col]]
    chart.columns = [stat_col]
    return chart


def infer_player_kind(markets) -> str:
    """Return 'pitcher' or 'batter' from the player's prop markets."""
    market_set = set(markets)
    has_pitcher = bool(market_set & PITCHER_MARKETS)
    has_batter = bool(market_set & BATTER_MARKETS)

    if has_pitcher and not has_batter:
        return "pitcher"

    return "batter"


def markets_for_kind(kind: str):
    if kind == "pitcher":
        markets = PITCHER_MARKETS
    else:
        markets = BATTER_MARKETS
    return sorted(markets - EXCLUDED_UI_MARKETS)


def window_average(game_log, stat_col, window):
    if game_log is None or game_log.empty:
        return np.nan

    values = pd.to_numeric(
        game_log[stat_col],
        errors="coerce",
    ).dropna()

    if values.empty:
        return np.nan

    recent = values.tail(window)
    if recent.empty:
        return np.nan

    return float(recent.mean())


def get_stat_history(
    player_name,
    market,
    version="v2",
    n=10,
):
    """Last *n* games for any supported market (alias for get_last_n_games)."""
    return get_last_n_games(
        player_name,
        market,
        version=version,
        n=n,
    )


def _player_key(name):
    return str(name).strip().lower()


def _fuzzy_player_key(player_name, candidate_keys):
    """Return a cache key for *player_name*, or None if no match."""
    if not isinstance(player_name, str):
        return None

    key = _player_key(player_name)
    if key in candidate_keys:
        return key

    last_name = key.split()[-1]
    matches = [
        candidate
        for candidate in candidate_keys
        if last_name in candidate.split()
    ]

    if len(matches) == 1:
        return matches[0]

    return None


def _over_rate(values, line, window):
    """
    Share of the last *window* games where the stat strictly exceeded *line*.
    """
    if not values:
        return np.nan

    recent = values[-window:]
    if not recent:
        return np.nan

    return float(np.mean([value > line for value in recent]))


def _format_l5_l10_pct(l5_pct, l10_pct):
    def _fmt(value):
        if pd.isna(value):
            return "—"
        return f"{value * 100:.0f}%"

    return f"{_fmt(l5_pct)} / {_fmt(l10_pct)}"


def _build_player_game_cache(features):
    """Map normalized player name to game rows sorted by date."""
    if features is None or features.empty:
        return {}

    if "player_name" not in features.columns or "game_date" not in features.columns:
        return {}

    sorted_features = features.sort_values("game_date")
    cache = {}

    for name, group in sorted_features.groupby("player_name", sort=False):
        cache[_player_key(name)] = group

    return cache


def _kind_player_cache_key(kind, version):
    path = find_latest_feature_path(kind, version)
    if path is None:
        return (kind, version, None, 0)

    path_str, mtime = _feature_cache_key(path)
    return (kind, version, path_str, mtime)


@lru_cache(maxsize=4)
def _kind_player_game_cache(cache_key):
    _kind, _version, path_str, mtime = cache_key
    if path_str is None:
        return {}

    features = _load_features((path_str, mtime))
    return _build_player_game_cache(features)


def get_features_max_game_date(kind="batter", version="v2"):
    """Return YYYY-MM-DD for the latest game_date in the feature parquet."""
    path = find_latest_feature_path(kind, version)
    if path is None:
        return None

    features = _load_features(_feature_cache_key(path))
    if features.empty or "game_date" not in features.columns:
        return None

    return (
        pd.to_datetime(features["game_date"])
        .max()
        .strftime("%Y-%m-%d")
    )


def rolling_over_rates(player_name, market, line, version="v2"):
    """
    Return (l5_pct, l10_pct) over-rates for a player/market/line, or (nan, nan).
    """
    if market not in MARKET_STAT_MAP:
        return np.nan, np.nan

    kind, stat_col = MARKET_STAT_MAP[market]
    cache = _kind_player_game_cache(_kind_player_cache_key(kind, version))
    if not cache:
        return np.nan, np.nan

    player_key = _fuzzy_player_key(player_name, cache.keys())
    if player_key is None:
        return np.nan, np.nan

    player_games = cache[player_key]
    if stat_col not in player_games.columns:
        return np.nan, np.nan

    values = pd.to_numeric(player_games[stat_col], errors="coerce").dropna().tolist()
    line = float(line)

    return (
        _over_rate(values, line, 5),
        _over_rate(values, line, 10),
    )


def enrich_with_l5_l10_pct(df, version="v2"):
    """
    Add l5_pct, l10_pct, and l5_l10_pct columns by joining feature parquets.

    L5/L10 % is the share of the player's last 5 / 10 completed games in the
    feature dataset where the market stat strictly exceeded the posted line.
    """
    empty_columns = {
        "l5_pct": pd.Series(dtype=float),
        "l10_pct": pd.Series(dtype=float),
        "l5_l10_pct": pd.Series(dtype=object),
    }

    if df.empty:
        result = df.copy()
        result = result.assign(**empty_columns)
        return result

    rate_map = {}
    for player, market, line in (
        df[["player", "market", "line"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    ):
        rate_map[(player, market, line)] = rolling_over_rates(
            player,
            market,
            line,
            version=version,
        )

    l5_values = []
    l10_values = []

    for player, market, line in zip(df["player"], df["market"], df["line"]):
        l5_pct, l10_pct = rate_map[(player, market, line)]
        l5_values.append(l5_pct)
        l10_values.append(l10_pct)

    result = df.copy()
    result["l5_pct"] = l5_values
    result["l10_pct"] = l10_values
    result["l5_l10_pct"] = [
        _format_l5_l10_pct(l5, l10)
        for l5, l10 in zip(l5_values, l10_values)
    ]

    return result
