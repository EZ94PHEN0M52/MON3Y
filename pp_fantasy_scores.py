"""
PrizePicks MLB hitter fantasy scores — official scoring chart + game archive.

Built once from batter feature parquets (Statcast game logs). The archive is
the source of truth for L5/L10 over-rates vs the posted PP fantasy line.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from utils import PROCESSED_DIR, normalize_player_key

PP_FANTASY_GAME_SCORES_PATH = (
    PROCESSED_DIR / "pp_fantasy_game_scores.parquet"
)

_ARCHIVE_COLUMNS = (
    "game_date",
    "game_pk",
    "batter",
    "player_name",
    "hits",
    "home_runs",
    "total_bases",
    "runs",
    "rbi",
    "walks",
    "stolen_bases",
    "hit_by_pitch",
    "pp_fantasy_score",
)


def compute_pp_fantasy_score_from_stats(
    hits: int,
    home_runs: int,
    total_bases: float,
    *,
    runs: int = 0,
    rbi: int = 0,
    walks: int = 0,
    hit_by_pitch: int = 0,
    stolen_bases: int = 0,
    triples: int = 0,
) -> float:
    """
    PrizePicks hitter fantasy score (official chart).

    Single 3 · Double 5 · Triple 8 · HR 10 · Run/RBI/BB/HBP 2 · SB 5.
    Assumes no triples when *triples* is omitted (typical from feature parquets).
    """
    hits = int(hits or 0)
    home_runs = int(home_runs or 0)
    triples = int(triples or 0)
    total_bases = float(total_bases or 0)

    remaining_hits = hits - home_runs - triples
    remaining_tb = total_bases - (4 * home_runs) - (3 * triples)
    doubles = max(0, int(round(remaining_tb - remaining_hits)))
    singles = max(0, remaining_hits - doubles)

    return float(
        (3 * singles)
        + (5 * doubles)
        + (8 * triples)
        + (10 * home_runs)
        + (2 * (int(runs or 0) + int(rbi or 0) + int(walks or 0) + int(hit_by_pitch or 0)))
        + (5 * int(stolen_bases or 0))
    )


def compute_pp_fantasy_score_from_row(row) -> float:
    """PP fantasy score from a batter feature row or archive row."""
    return compute_pp_fantasy_score_from_stats(
        row.get("hits", 0),
        row.get("home_runs", 0),
        row.get("total_bases", 0),
        runs=row.get("runs", 0),
        rbi=row.get("rbi", 0),
        walks=row.get("walks", 0),
        hit_by_pitch=row.get("hit_by_pitch", 0),
        stolen_bases=row.get("stolen_bases", 0),
    )


def _vectorized_pp_fantasy_scores(frame: pd.DataFrame) -> pd.Series:
    hits = pd.to_numeric(frame["hits"], errors="coerce").fillna(0).astype(int)
    home_runs = (
        pd.to_numeric(frame["home_runs"], errors="coerce").fillna(0).astype(int)
    )
    total_bases = pd.to_numeric(frame["total_bases"], errors="coerce").fillna(0)
    runs = pd.to_numeric(frame["runs"], errors="coerce").fillna(0).astype(int)
    rbi = pd.to_numeric(frame["rbi"], errors="coerce").fillna(0).astype(int)
    walks = pd.to_numeric(frame["walks"], errors="coerce").fillna(0).astype(int)
    hit_by_pitch = (
        pd.to_numeric(frame["hit_by_pitch"], errors="coerce").fillna(0).astype(int)
        if "hit_by_pitch" in frame.columns
        else pd.Series(0, index=frame.index, dtype=int)
    )
    stolen_bases = (
        pd.to_numeric(frame["stolen_bases"], errors="coerce").fillna(0).astype(int)
    )

    remaining_hits = hits - home_runs
    remaining_tb = total_bases - (4 * home_runs)
    doubles = (remaining_tb - remaining_hits).clip(lower=0).astype(int)
    singles = (remaining_hits - doubles).clip(lower=0)

    return (
        (3 * singles)
        + (5 * doubles)
        + (10 * home_runs)
        + (2 * (runs + rbi + walks + hit_by_pitch))
        + (5 * stolen_bases)
    ).astype(float)


def build_pp_fantasy_game_scores(version: str = "v2") -> pd.DataFrame:
    """Build the full-season PP fantasy score archive from batter features."""
    from ui.player_stats import (
        _feature_cache_key,
        _load_features,
        find_latest_feature_path,
    )

    path = find_latest_feature_path("batter", version)
    if path is None:
        return pd.DataFrame(columns=list(_ARCHIVE_COLUMNS))

    features = _load_features(_feature_cache_key(path))
    required = {
        "game_date",
        "player_name",
        "hits",
        "home_runs",
        "total_bases",
        "runs",
        "rbi",
        "walks",
        "stolen_bases",
        "hit_by_pitch",
    }
    if features.empty or not required.issubset(features.columns):
        return pd.DataFrame(columns=list(_ARCHIVE_COLUMNS))

    work = features.sort_values(["player_name", "game_date"]).copy()
    work["pp_fantasy_score"] = _vectorized_pp_fantasy_scores(work)
    work["player_key"] = work["player_name"].map(normalize_player_key)

    if "game_pk" not in work.columns:
        work["game_pk"] = pd.NA
    if "batter" not in work.columns:
        work["batter"] = pd.NA

    archive = work[list(_ARCHIVE_COLUMNS)].copy()
    archive["game_date"] = pd.to_datetime(archive["game_date"]).dt.strftime("%Y-%m-%d")
    return archive.reset_index(drop=True)


def save_pp_fantasy_game_scores(
    archive: pd.DataFrame,
    output_path: Path = PP_FANTASY_GAME_SCORES_PATH,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    archive.to_parquet(output_path, index=False)
    return output_path


def rebuild_pp_fantasy_game_scores(version: str = "v2") -> Path | None:
    """Rebuild and persist the archive; returns output path or None if empty."""
    archive = build_pp_fantasy_game_scores(version=version)
    if archive.empty:
        return None

    path = save_pp_fantasy_game_scores(archive)
    _pp_fantasy_archive_cache.cache_clear()
    _player_pp_fantasy_scores_cache.cache_clear()
    return path


def archive_is_stale(version: str = "v2") -> bool:
    """True when features are newer than the fantasy score archive."""
    from ui.player_stats import find_latest_feature_path

    feature_path = find_latest_feature_path("batter", version)
    if feature_path is None:
        return False

    if not PP_FANTASY_GAME_SCORES_PATH.exists():
        return True

    return feature_path.stat().st_mtime_ns > PP_FANTASY_GAME_SCORES_PATH.stat().st_mtime_ns


def _archive_cache_key(version: str) -> tuple:
    path = PP_FANTASY_GAME_SCORES_PATH
    if not path.exists():
        return (version, None, 0)
    return (version, str(path), path.stat().st_mtime_ns)


@lru_cache(maxsize=4)
def _pp_fantasy_archive_cache(cache_key: tuple) -> pd.DataFrame:
    _version, path_str, _mtime = cache_key
    if path_str is None:
        return pd.DataFrame(columns=list(_ARCHIVE_COLUMNS))

    archive = pd.read_parquet(path_str)
    if archive.empty:
        return archive

    archive = archive.copy()
    archive["player_key"] = archive["player_name"].map(normalize_player_key)
    archive["game_date"] = pd.to_datetime(archive["game_date"])
    return archive.sort_values(["player_key", "game_date"]).reset_index(drop=True)


def load_pp_fantasy_game_scores(version: str = "v2") -> pd.DataFrame:
    return _pp_fantasy_archive_cache(_archive_cache_key(version))


@lru_cache(maxsize=512)
def _player_pp_fantasy_scores_cache(
    player_key: str,
    version: str,
    cache_key: tuple,
) -> tuple[float, ...]:
    archive = _pp_fantasy_archive_cache(cache_key)
    if archive.empty or player_key not in archive["player_key"].values:
        return tuple()

    scores = (
        archive.loc[archive["player_key"].eq(player_key), "pp_fantasy_score"]
        .astype(float)
        .tolist()
    )
    return tuple(scores)


def player_pp_fantasy_score_values(
    player_name: str,
    version: str = "v2",
) -> list[float]:
    """Chronological PP fantasy scores for *player_name* from the archive."""
    from ui.player_stats import _fuzzy_player_key, _player_key

    cache_key = _archive_cache_key(version)
    if cache_key[1] is None:
        rebuild_pp_fantasy_game_scores(version=version)
        cache_key = _archive_cache_key(version)

    archive = _pp_fantasy_archive_cache(cache_key)
    if archive.empty:
        return []

    player_key = _fuzzy_player_key(
        player_name,
        archive["player_key"].unique(),
    )
    if player_key is None:
        player_key = _player_key(player_name)
        if player_key not in archive["player_key"].values:
            return []

    return list(
        _player_pp_fantasy_scores_cache(player_key, version, cache_key)
    )


def archive_max_game_date(version: str = "v2") -> str | None:
    archive = load_pp_fantasy_game_scores(version=version)
    if archive.empty:
        return None
    return archive["game_date"].max().strftime("%Y-%m-%d")
