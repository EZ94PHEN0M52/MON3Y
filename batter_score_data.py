"""
Load batter game logs from feature parquets and compute Batter Score.

Phase B adds opposing SP ERA (L5), optional H2H blend, and game-context
gating via daily_probables.parquet.

Orthogonal to LightGBM prop models — used for UI ranking and player context.

Cache-first policy (no redundant live API calls):
  - Game logs, rolling stats: data/processed/*_features_*.parquet only
  - Pitch arsenal v1 (five buckets): primary season statcast shard (@lru_cache)
  - Pitch arsenal v2 (Savant types) + board H2H: merged statcast shards
  - Scoring H2H vs SP: primary season statcast shard unless statcast= is passed
  - Probable SP lookup: data/processed/daily_probables.parquet
  - Never calls pybaseball.statcast(), Odds API, or MLB Stats API directly.
  - Set DISABLE_LIVE_FETCH=1 to block accidental fetches in shared helpers
    (odds_api, fetch_data, fetch_probables) during backtests/offline runs.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from batter_score import (
    AVG_THRESHOLDS,
    BatterInputs,
    BatterScoreResult,
    GameLine,
    MAX_GRADE_POINTS,
    PHASE_A_GATES,
    PHASE_B_GATES,
    PHASE_D_GATES,
    USE_TEAM_PITCHING_PROXY,
    WEIGHTS_V1,
    WEIGHTS_V2,
    Weights,
    compute_batter_score_partial,
    compute_batter_score_phase_b,
    compute_batter_score_phase_d,
    grade_min_threshold,
)
from build_features import EXTRA_BASES, HITS
from fetch_probables import PROBABLES_PATH, lookup_opposing_sp
from pitch_matchup import (
    AB_EVENTS,
    arsenal_ready,
    build_opponent_pitcher_arsenal,
    build_opponent_pitcher_arsenal_detailed,
)
from ui.player_stats import (
    _feature_cache_key,
    _fuzzy_player_key,
    _kind_player_cache_key,
    _kind_player_game_cache,
    _player_key,
    find_latest_feature_path,
)
from utils import (
    BATTER_SCORE_VALIDATION_PATH,
    RAW_DIR,
    coerce_mlb_id,
    game_date_from_commence,
)


BATTER_SCORE_COLUMNS = [
    "batter_score",
    "batter_score_season_baseline",
    "batter_score_recent_form",
    "batter_score_matchup_grade",
    "batter_score_pitcher_form",
    "batter_score_partial",
    "batter_score_label",
]


def _raw_points_row(row) -> float:
    hits = float(row.get("hits", 0) or 0)
    total_bases = float(row.get("total_bases", 0) or 0)
    walks = float(row.get("walks", 0) or 0)
    return hits + total_bases + walks


def _rows_to_game_log(player_rows: pd.DataFrame) -> List[GameLine]:
    """Build game log newest-first from sorted feature rows."""
    sorted_rows = player_rows.sort_values(
        "game_date",
        ascending=False,
    )

    games = []
    for _, row in sorted_rows.iterrows():
        opponent = row.get("opponent")
        if pd.isna(opponent):
            opponent = ""

        games.append(
            GameLine(
                date=str(row["game_date"])[:10],
                opponent=str(opponent),
                hits=int(row.get("hits", 0) or 0),
                total_bases=int(row.get("total_bases", 0) or 0),
                walks=int(row.get("walks", 0) or 0),
            )
        )

    return games


def _parse_game_teams(game: str) -> Tuple[Optional[str], Optional[str]]:
    if not isinstance(game, str) or " @ " not in game:
        return None, None

    away, home = game.split(" @ ", 1)
    return away.strip(), home.strip()


def _game_date_from_commence(commence_time) -> Optional[str]:
    return game_date_from_commence(commence_time)


def _probables_cache_key():
    if not PROBABLES_PATH.exists():
        return (None, 0)
    return _feature_cache_key(PROBABLES_PATH)


@lru_cache(maxsize=2)
def _load_probables(path_key) -> pd.DataFrame:
    path_str, _mtime = path_key
    if path_str is None:
        return pd.DataFrame()

    return pd.read_parquet(path_str)


def _parse_statcast_shard_dates(path: Path) -> tuple[str, str] | None:
    """Return (start_date, end_date) from ``statcast_{start}_{end}.parquet``."""
    stem = path.stem
    if not stem.startswith("statcast_"):
        return None

    body = stem.removeprefix("statcast_")
    if "_" not in body:
        return None

    start_date, end_date = body.split("_", 1)
    if not start_date or not end_date:
        return None

    return start_date, end_date


def _statcast_cache_key():
    """
    Primary season Statcast shard for scoring lookups (v1 arsenal, scoring H2H).

    Prefers cumulative season windows (``statcast_{open}_{latest}.parquet``)
    over single-day incrementals (``statcast_{day}_{day}.parquet``), which can
    sort later by filename but contain almost no pitch history.
    """
    candidates = list(RAW_DIR.glob("statcast_*.parquet"))
    if not candidates:
        return (None, 0)

    parsed: list[tuple[Path, str, str]] = []
    for path in candidates:
        dates = _parse_statcast_shard_dates(path)
        if dates is None:
            continue
        parsed.append((path, dates[0], dates[1]))

    if not parsed:
        return _feature_cache_key(max(candidates, key=lambda path: path.stem))

    cumulative = [
        item for item in parsed if item[1] != item[2]
    ]
    pool = cumulative or parsed

    def _rank(item: tuple[Path, str, str]) -> tuple[str, int]:
        path, _start, end = item
        return end, path.stat().st_size

    best_path = max(pool, key=_rank)[0]
    return _feature_cache_key(best_path)


@lru_cache(maxsize=2)
def _load_latest_statcast(path_key) -> Optional[pd.DataFrame]:
    path_str, _mtime = path_key
    if path_str is None:
        return None

    return pd.read_parquet(path_str)


def _merged_statcast_cache_key():
    candidates = sorted(RAW_DIR.glob("statcast_*.parquet"), key=lambda path: path.stem)
    if not candidates:
        return ((), ())

    return (
        tuple(str(path) for path in candidates),
        tuple(_feature_cache_key(path)[1] for path in candidates),
    )


@lru_cache(maxsize=1)
def _load_merged_statcast(cache_key) -> Optional[pd.DataFrame]:
    """All statcast shards merged and deduped — career H2H / wOBA / pitch-type stats.

    Includes every ``data/raw/statcast_*.parquet`` (current season from
    ``run_daily.sh`` plus optional history from ``fetch_statcast_history.py``).
    """
    paths, _mtimes = cache_key
    if not paths:
        return None

    frames = [pd.read_parquet(path) for path in paths]
    merged = pd.concat(frames, ignore_index=True)

    dedup_cols = ["game_pk", "at_bat_number", "pitch_number"]
    if all(column in merged.columns for column in dedup_cols):
        merged = merged.drop_duplicates(subset=dedup_cols, keep="last")

    return merged


def _batter_rows(
    player_name: str,
    version: str = "v2",
) -> Optional[pd.DataFrame]:
    cache = _kind_player_game_cache(_kind_player_cache_key("batter", version))
    if not cache:
        return None

    player_key = _fuzzy_player_key(player_name, cache.keys())
    if player_key is None:
        return None

    return cache[player_key]


def _pitcher_rows_by_sp(
    sp_id: Optional[int],
    sp_name: Optional[str],
    version: str = "v2",
) -> Optional[pd.DataFrame]:
    cache = _kind_player_game_cache(_kind_player_cache_key("pitcher", version))
    if not cache:
        return None

    sp_id = coerce_mlb_id(sp_id)

    if sp_id is not None:
        for rows in cache.values():
            if "pitcher" not in rows.columns:
                continue
            pitcher_ids = pd.to_numeric(rows["pitcher"], errors="coerce")
            if pitcher_ids.eq(sp_id).any():
                return rows

    if sp_name:
        player_key = _fuzzy_player_key(sp_name, cache.keys())
        if player_key is not None:
            return cache[player_key]

    return None


def _compute_sp_era_l5(
    sp_id: Optional[int],
    sp_name: Optional[str],
    version: str = "v2",
) -> Optional[float]:
    """ERA over the pitcher's last five starts in feature game logs."""
    rows = _pitcher_rows_by_sp(sp_id, sp_name, version=version)
    if rows is None or rows.empty:
        return None

    required = {"earned_runs", "outs", "game_date"}
    if not required.issubset(rows.columns):
        return None

    recent = rows.sort_values("game_date").tail(5)
    total_er = pd.to_numeric(
        recent["earned_runs"],
        errors="coerce",
    ).fillna(0).sum()
    total_outs = pd.to_numeric(
        recent["outs"],
        errors="coerce",
    ).fillna(0).sum()

    if total_outs <= 0:
        return None

    innings = total_outs / 3.0
    return float(total_er / innings * 9.0)


FIP_FALLBACK_CONSTANT = 3.10
FIP_L5_COLUMNS = (
    "home_runs_allowed",
    "hit_by_pitch",
    "walks",
    "strikeouts",
    "outs",
    "game_date",
)
FIP_TOTAL_COLUMNS = (
    "strikeouts",
    "walks",
    "home_runs_allowed",
    "hit_by_pitch",
    "outs",
    "earned_runs",
)


def _fip_constant_from_totals(totals: pd.Series) -> Optional[float]:
    innings = float(totals["outs"]) / 3.0
    if innings <= 0:
        return None

    league_era = float(totals["earned_runs"]) / innings * 9.0
    fip_core = (
        13.0 * float(totals["home_runs_allowed"])
        + 3.0 * (float(totals["walks"]) + float(totals["hit_by_pitch"]))
        - 2.0 * float(totals["strikeouts"])
    ) / innings
    return float(league_era - fip_core)


def _pitcher_league_fip_totals(version: str = "v2") -> Optional[pd.Series]:
    """Sum FIP inputs across all cached pitcher game-log rows."""
    cache = _kind_player_game_cache(_kind_player_cache_key("pitcher", version))
    if not cache:
        return None

    frames = [
        frame
        for frame in cache.values()
        if frame is not None and not frame.empty
    ]
    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True)
    if not set(FIP_TOTAL_COLUMNS).issubset(combined.columns):
        return None

    return combined[list(FIP_TOTAL_COLUMNS)].apply(
        pd.to_numeric,
        errors="coerce",
    ).fillna(0).sum()


def _fast_fip_totals_from_statcast(statcast: pd.DataFrame) -> Optional[pd.Series]:
    """
    Vectorized league FIP totals from Statcast (no half-inning ER loop).

    Used only when pitcher feature parquets are missing HR/HBP columns.
    Earned runs fall back to runs scored on the pitcher's pitches.
    """
    if statcast is None or statcast.empty:
        return None

    from build_features import PITCHER_OUT_EVENTS

    required = {
        "events",
        "game_pk",
        "pitcher",
        "post_bat_score",
        "bat_score",
    }
    if not required.issubset(statcast.columns):
        return None

    data = statcast[statcast["events"].notna()].copy()
    if data.empty:
        return None

    data["strikeouts"] = data["events"].isin(
        {"strikeout", "strikeout_double_play"},
    ).astype(int)
    data["walks"] = data["events"].isin(
        {"walk", "intent_walk"},
    ).astype(int)
    data["home_runs_allowed"] = data["events"].eq("home_run").astype(int)
    data["hit_by_pitch"] = data["events"].eq("hit_by_pitch").astype(int)
    data["outs"] = (
        data["events"]
        .map(PITCHER_OUT_EVENTS)
        .fillna(0)
        .astype(int)
    )
    data["earned_runs"] = (
        pd.to_numeric(data["post_bat_score"], errors="coerce")
        - pd.to_numeric(data["bat_score"], errors="coerce")
    ).fillna(0).clip(lower=0)

    grouped = data.groupby(["game_pk", "pitcher"], as_index=False).agg(
        strikeouts=("strikeouts", "sum"),
        walks=("walks", "sum"),
        home_runs_allowed=("home_runs_allowed", "sum"),
        hit_by_pitch=("hit_by_pitch", "sum"),
        outs=("outs", "sum"),
        earned_runs=("earned_runs", "sum"),
    )
    return grouped[list(FIP_TOTAL_COLUMNS)].sum()


@lru_cache(maxsize=4)
def _league_fip_constant(version: str = "v2") -> float:
    """
    League FIP constant from cached pitcher game logs (Statcast-derived).

    ``constant = league_ERA - (13*HR + 3*(BB+HBP) - 2*K) / IP`` so FIP scales
    like ERA across the cached sample. Falls back to a fast Statcast aggregate
    while pitcher parquets are rebuilding.
    """
    totals = _pitcher_league_fip_totals(version)
    if totals is None:
        statcast = _load_merged_statcast(_merged_statcast_cache_key())
        totals = _fast_fip_totals_from_statcast(statcast)

    if totals is None:
        return FIP_FALLBACK_CONSTANT

    constant = _fip_constant_from_totals(totals)
    if constant is None:
        return FIP_FALLBACK_CONSTANT
    return constant


def _compute_sp_fip_l5(
    sp_id: Optional[int],
    sp_name: Optional[str],
    version: str = "v2",
) -> Optional[float]:
    """FIP over the pitcher's last five starts (Batter Score v2 pitcher form)."""
    rows = _pitcher_rows_by_sp(sp_id, sp_name, version=version)
    if rows is None or rows.empty:
        return None

    if not set(FIP_L5_COLUMNS).issubset(rows.columns):
        return None

    recent = rows.sort_values("game_date").tail(5)
    home_runs = pd.to_numeric(
        recent["home_runs_allowed"],
        errors="coerce",
    ).fillna(0).sum()
    hit_by_pitch = pd.to_numeric(
        recent["hit_by_pitch"],
        errors="coerce",
    ).fillna(0).sum()
    walks = pd.to_numeric(recent["walks"], errors="coerce").fillna(0).sum()
    strikeouts = pd.to_numeric(
        recent["strikeouts"],
        errors="coerce",
    ).fillna(0).sum()
    total_outs = pd.to_numeric(recent["outs"], errors="coerce").fillna(0).sum()

    innings = float(total_outs) / 3.0
    if innings <= 0:
        return None

    constant = _league_fip_constant(version)
    fip_core = (
        13.0 * float(home_runs)
        + 3.0 * (float(walks) + float(hit_by_pitch))
        - 2.0 * float(strikeouts)
    ) / innings
    return float(fip_core + constant)


def _h2h_raw_points(events: pd.Series) -> Tuple[int, float]:
    hits = events.isin(HITS).astype(int)
    total_bases = events.map(EXTRA_BASES).fillna(0)
    walks = events.isin({"walk", "intent_walk"}).astype(int)
    pa_count = int(events.notna().sum())
    raw_points = float((hits + total_bases + walks).sum())
    return pa_count, raw_points


def _h2h_hits_ab(events: pd.Series) -> Tuple[int, int]:
    """Career hits and at-bats vs one SP from Statcast terminal events."""
    hits = int(events.isin(HITS).sum())
    ab = int(events.isin(AB_EVENTS).sum())
    return hits, ab


def _compute_h2h_stats(
    batter_id: Optional[int],
    sp_id: Optional[int],
    *,
    statcast: Optional[pd.DataFrame] = None,
) -> Tuple[Optional[int], Optional[float], Optional[int], Optional[int]]:
    """
    Batter vs SP career H2H from Statcast raw.

    Returns (pa_count, avg H+TB+BB per game vs that SP, hits, ab).
    Scoring still omits H2H when PA < MIN_PA_H2H; hits/ab are for board display.
    """
    if batter_id is None or sp_id is None:
        return None, None, None, None

    batter_id = coerce_mlb_id(batter_id)
    sp_id = coerce_mlb_id(sp_id)
    if batter_id is None or sp_id is None:
        return None, None, None, None

    if statcast is None:
        statcast = _load_latest_statcast(_statcast_cache_key())
    if statcast is None or statcast.empty:
        return None, None, None, None

    required = {"batter", "pitcher", "events", "game_date"}
    if not required.issubset(statcast.columns):
        return None, None, None, None

    matchups = statcast[
        statcast["batter"].astype(int).eq(batter_id)
        & statcast["pitcher"].astype(int).eq(sp_id)
        & statcast["events"].notna()
    ].copy()

    if matchups.empty:
        return 0, None, 0, 0

    pa_count = int(len(matchups))
    h2h_hits, h2h_ab = _h2h_hits_ab(matchups["events"])

    game_stats = []
    for _, group in matchups.groupby("game_date", sort=False):
        _, raw_points = _h2h_raw_points(group["events"])
        game_stats.append(raw_points)

    if not game_stats:
        return pa_count, None, h2h_hits, h2h_ab

    avg_raw_points = float(np.mean(game_stats))
    return pa_count, avg_raw_points, h2h_hits, h2h_ab


def _team_opp_earned_runs_proxy(
    player_rows: pd.DataFrame,
) -> Optional[float]:
    """Latest opp_team_earned_runs_season from batter feature row."""
    if player_rows is None or player_rows.empty:
        return None

    if "opp_team_earned_runs_season" not in player_rows.columns:
        return None

    latest = player_rows.sort_values("game_date").iloc[-1]
    value = latest.get("opp_team_earned_runs_season")
    if value is None or pd.isna(value):
        return None

    return float(value)


def _lookup_opposing_sp_for_context(
    game_context: dict,
    batter_team: str,
) -> Tuple[Optional[str], Optional[int]]:
    probables = _load_probables(_probables_cache_key())
    if probables.empty:
        return None, None

    game_date = game_context.get("game_date")
    home_team = game_context.get("home_team")
    away_team = game_context.get("away_team")

    if not all([game_date, home_team, away_team, batter_team]):
        return None, None

    return lookup_opposing_sp(
        probables,
        str(game_date)[:10],
        home_team,
        away_team,
        batter_team,
    )


def build_game_context(
    *,
    game: Optional[str] = None,
    commence_time=None,
    home_team: Optional[str] = None,
    away_team: Optional[str] = None,
) -> Optional[dict]:
    if home_team and away_team:
        parsed_home = home_team
        parsed_away = away_team
    else:
        parsed_away, parsed_home = _parse_game_teams(game or "")

    game_date = _game_date_from_commence(commence_time)
    if not all([parsed_home, parsed_away, game_date]):
        return None

    return {
        "game_date": game_date,
        "home_team": parsed_home,
        "away_team": parsed_away,
    }



def parse_h2h_fraction(text: str) -> Optional[Tuple[int, int]]:
    """
    Parse career H2H batting average input like ``3/8`` or ``12 / 40``.

    Returns ``(hits, ab)`` or None when invalid.
    """
    if text is None or not str(text).strip():
        return None

    match = re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*$", str(text).strip())
    if not match:
        return None

    hits = int(match.group(1))
    ab = int(match.group(2))
    if ab <= 0 or hits < 0 or hits > ab:
        return None

    return hits, ab


def estimate_h2h_avg_raw_points_from_hits_ab(
    hits: int,
    ab: int,
    *,
    max_raw_points_for_100: float = 6.0,
) -> float:
    """
    Map career H/AB to per-game raw points for manual H2H scoring.

    Uses the same AVG letter grades as matchup grade so a strong career line
    (e.g. 3/8 = .375) maps to a high H2H index instead of a muted raw-points
    estimate.
    """
    if ab <= 0:
        raise ValueError("ab must be positive")
    if hits < 0 or hits > ab:
        raise ValueError("hits must be between 0 and ab")

    _, grade_points = grade_min_threshold(hits / ab, AVG_THRESHOLDS)
    return float(grade_points / MAX_GRADE_POINTS * max_raw_points_for_100)


def apply_manual_h2h_override(
    batter: BatterInputs,
    hits: int,
    ab: int,
) -> BatterInputs:
    """Replace Statcast H2H fields with user-supplied career H/AB."""
    avg_raw = estimate_h2h_avg_raw_points_from_hits_ab(hits, ab)
    return replace(
        batter,
        h2h_pa=ab,
        h2h_hits=hits,
        h2h_ab=ab,
        h2h_avg_raw_points=avg_raw,
        h2h_manual_override=True,
    )


def _score_batter_inputs(
    batter: BatterInputs,
    *,
    game_context: Optional[dict] = None,
    weights: Weights = WEIGHTS_V1,
    pitcher_form_use_fip: bool = False,
) -> Optional[BatterScoreResult]:
    """Shared Phase A/B/D selection for Batter Score v1 paths."""
    sp_named = bool(batter.opposing_sp_name and game_context is not None)
    if pitcher_form_use_fip:
        sp_ready = sp_named and batter.opponent_pitcher_fip_l5 is not None
    else:
        sp_ready = sp_named and batter.opponent_pitcher_era_l5 is not None
    matchup_ready = (
        sp_ready
        and arsenal_ready(batter.opponent_pitcher_arsenal)
    )

    try:
        if matchup_ready:
            return compute_batter_score_phase_d(
                batter,
                gates=PHASE_D_GATES,
                weights=weights,
                pitcher_form_use_fip=pitcher_form_use_fip,
            )

        if sp_ready:
            return compute_batter_score_phase_b(
                batter,
                gates=PHASE_B_GATES,
                weights=weights,
                pitcher_form_use_fip=pitcher_form_use_fip,
            )

        if (
            USE_TEAM_PITCHING_PROXY
            and batter.team_opp_earned_runs_proxy is not None
            and game_context is not None
            and not sp_named
        ):
            return compute_batter_score_phase_b(
                batter,
                gates=PHASE_B_GATES,
                sp_tbd=True,
                team_proxy=True,
                weights=weights,
                pitcher_form_use_fip=pitcher_form_use_fip,
            )

        return compute_batter_score_partial(
            batter,
            gates=PHASE_A_GATES,
            weights=weights,
            sp_tbd=game_context is not None and not sp_named,
            pitcher_form_use_fip=pitcher_form_use_fip,
        )
    except ValueError:
        return None


def _score_batter_inputs_v2(
    batter: BatterInputs,
    *,
    game_context: Optional[dict] = None,
) -> Optional[BatterScoreResult]:
    """Batter Score v2: v2 weights, Savant pitch types, FIP L5 pitcher form."""
    sp_named = bool(batter.opposing_sp_name and game_context is not None)
    sp_ready = sp_named and batter.opponent_pitcher_fip_l5 is not None
    matchup_ready_v2 = (
        sp_ready
        and arsenal_ready(batter.opponent_pitcher_arsenal_v2)
    )

    try:
        if matchup_ready_v2:
            batter_v2 = replace(
                batter,
                opponent_pitcher_arsenal=batter.opponent_pitcher_arsenal_v2,
            )
            return compute_batter_score_phase_d(
                batter_v2,
                gates=PHASE_D_GATES,
                weights=WEIGHTS_V2,
                pitcher_form_use_fip=True,
            )

        if sp_ready:
            return compute_batter_score_phase_b(
                batter,
                gates=PHASE_B_GATES,
                weights=WEIGHTS_V2,
                pitcher_form_use_fip=True,
            )

        if (
            USE_TEAM_PITCHING_PROXY
            and batter.team_opp_earned_runs_proxy is not None
            and game_context is not None
            and not sp_named
        ):
            return compute_batter_score_phase_b(
                batter,
                gates=PHASE_B_GATES,
                sp_tbd=True,
                team_proxy=True,
                weights=WEIGHTS_V2,
                pitcher_form_use_fip=True,
            )

        return compute_batter_score_partial(
            batter,
            gates=PHASE_A_GATES,
            weights=WEIGHTS_V2,
            sp_tbd=game_context is not None and not sp_named,
            pitcher_form_use_fip=True,
        )
    except ValueError:
        return None


def _normalize_game_date(value) -> str:
    return str(pd.to_datetime(value).strftime("%Y-%m-%d"))


def build_batter_inputs_from_rows(
    player_rows: pd.DataFrame,
    *,
    display_name: Optional[str] = None,
    game_context: Optional[dict] = None,
    version: str = "v2",
) -> Optional[BatterInputs]:
    """Construct BatterInputs from a pre-filtered game-log dataframe."""
    if player_rows is None or player_rows.empty:
        return None

    required = {"hits", "total_bases", "walks", "game_date"}
    if not required.issubset(player_rows.columns):
        return None

    if len(player_rows) < 10:
        return None

    raw_points = player_rows.apply(_raw_points_row, axis=1)
    season_avg = float(raw_points.mean())
    game_log = _rows_to_game_log(player_rows)

    if display_name is None:
        if "player_name" in player_rows.columns:
            display_name = str(
                player_rows["player_name"].iloc[-1]
            ).strip()
        else:
            display_name = ""

    latest = player_rows.sort_values("game_date").iloc[-1]
    batter_team = latest.get("team")
    batter_id = latest.get("batter")
    if batter_id is not None and not pd.isna(batter_id):
        batter_id = int(batter_id)
    else:
        batter_id = None

    sp_name = None
    sp_id = None
    sp_era_l5 = None
    sp_fip_l5 = None
    h2h_pa = None
    h2h_avg_raw_points = None
    h2h_hits = None
    h2h_ab = None
    team_proxy = None
    opponent_arsenal = []
    opponent_arsenal_v2 = []

    if game_context and isinstance(batter_team, str) and batter_team.strip():
        sp_name, sp_id = _lookup_opposing_sp_for_context(
            game_context,
            batter_team,
        )
        sp_id = coerce_mlb_id(sp_id)

        if sp_name:
            sp_era_l5 = _compute_sp_era_l5(
                sp_id,
                sp_name,
                version=version,
            )
            sp_fip_l5 = _compute_sp_fip_l5(
                sp_id,
                sp_name,
                version=version,
            )
            if sp_id is not None:
                h2h_pa, h2h_avg_raw_points, h2h_hits, h2h_ab = _compute_h2h_stats(
                    batter_id,
                    sp_id,
                )
                statcast_latest = _load_latest_statcast(
                    _statcast_cache_key()
                )
                statcast_merged = _load_merged_statcast(
                    _merged_statcast_cache_key()
                )
                opponent_arsenal = build_opponent_pitcher_arsenal(
                    statcast_latest,
                    batter_id,
                    sp_id,
                )
                statcast_for_v2 = statcast_merged
                if statcast_for_v2 is None or statcast_for_v2.empty:
                    statcast_for_v2 = statcast_latest
                opponent_arsenal_v2 = build_opponent_pitcher_arsenal_detailed(
                    statcast_for_v2,
                    batter_id,
                    sp_id,
                )
        elif USE_TEAM_PITCHING_PROXY:
            team_proxy = _team_opp_earned_runs_proxy(player_rows)

    return BatterInputs(
        name=display_name or "Unknown",
        season_avg_raw_points=season_avg,
        game_log=game_log,
        opponent_pitcher_arsenal=opponent_arsenal,
        opponent_pitcher_arsenal_v2=opponent_arsenal_v2,
        opponent_pitcher_era_l5=sp_era_l5,
        opponent_pitcher_fip_l5=sp_fip_l5,
        opposing_sp_name=sp_name,
        h2h_pa=h2h_pa,
        h2h_avg_raw_points=h2h_avg_raw_points,
        h2h_hits=h2h_hits,
        h2h_ab=h2h_ab,
        team_opp_earned_runs_proxy=team_proxy,
    )


def build_batter_inputs_as_of(
    player_rows: pd.DataFrame,
    target_date: str,
    *,
    game_context: Optional[dict] = None,
    version: str = "v2",
) -> Optional[BatterInputs]:
    """
    Point-in-time BatterInputs using only games strictly before *target_date*.

    Used by the validation backtest to avoid lookahead in season/form features.
    """
    if player_rows is None or player_rows.empty:
        return None

    target = _normalize_game_date(target_date)
    dated = player_rows.copy()
    dated["game_date"] = dated["game_date"].map(_normalize_game_date)
    prior = dated[dated["game_date"].lt(target)].copy()

    display_name = None
    if "player_name" in player_rows.columns:
        display_name = str(player_rows["player_name"].iloc[-1]).strip()

    return build_batter_inputs_from_rows(
        prior,
        display_name=display_name,
        game_context=game_context,
        version=version,
    )


def actual_raw_points_on_date(
    player_rows: pd.DataFrame,
    target_date: str,
) -> Optional[float]:
    """H+TB+BB composite outcome for *target_date* (validation target stat)."""
    if player_rows is None or player_rows.empty:
        return None

    target = _normalize_game_date(target_date)
    dated = player_rows.copy()
    dated["game_date"] = dated["game_date"].map(_normalize_game_date)
    day_rows = dated[dated["game_date"].eq(target)]

    if day_rows.empty:
        return None

    return float(_raw_points_row(day_rows.iloc[0]))


def score_batter_as_of(
    player_rows: pd.DataFrame,
    target_date: str,
    *,
    game_context: Optional[dict] = None,
    version: str = "v2",
) -> Optional[BatterScoreResult]:
    """Compute Batter Score from pre-game history only (backtest-safe)."""
    batter = build_batter_inputs_as_of(
        player_rows,
        target_date,
        game_context=game_context,
        version=version,
    )
    if batter is None:
        return None

    sp_named = bool(batter.opposing_sp_name and game_context is not None)
    sp_ready = sp_named and batter.opponent_pitcher_era_l5 is not None
    matchup_ready = (
        sp_ready
        and arsenal_ready(batter.opponent_pitcher_arsenal)
    )

    try:
        if matchup_ready:
            return compute_batter_score_phase_d(
                batter,
                gates=PHASE_D_GATES,
                weights=WEIGHTS_V1,
            )

        if sp_ready:
            return compute_batter_score_phase_b(
                batter,
                gates=PHASE_B_GATES,
                weights=WEIGHTS_V1,
            )

        if (
            USE_TEAM_PITCHING_PROXY
            and batter.team_opp_earned_runs_proxy is not None
            and game_context is not None
            and not sp_named
        ):
            return compute_batter_score_phase_b(
                batter,
                gates=PHASE_B_GATES,
                sp_tbd=True,
                team_proxy=True,
                weights=WEIGHTS_V1,
            )

        return compute_batter_score_partial(
            batter,
            gates=PHASE_A_GATES,
            weights=WEIGHTS_V1,
            sp_tbd=game_context is not None and not sp_named,
        )
    except ValueError:
        return None


def build_batter_inputs(
    player_name: str,
    version: str = "v2",
    game_context: Optional[dict] = None,
) -> Optional[BatterInputs]:
    """
    Construct BatterInputs from the latest batter feature parquet.

    When *game_context* is supplied, Phase B SP / H2H fields are populated.
    Returns None when the player is missing or has fewer than 10 games.
    """
    player_rows = _batter_rows(player_name, version=version)
    if player_rows is None:
        return None

    return build_batter_inputs_from_rows(
        player_rows,
        display_name=str(player_name).strip(),
        game_context=game_context,
        version=version,
    )


def score_batter(
    player_name: str,
    version: str = "v2",
    game_context: Optional[dict] = None,
) -> Optional[BatterScoreResult]:
    """Compute batter score for one player (Phase D when matchup data exists)."""
    batter = build_batter_inputs(
        player_name,
        version=version,
        game_context=game_context,
    )
    if batter is None:
        return None

    return _score_batter_inputs(batter, game_context=game_context)


def score_batter_with_manual_h2h(
    player_name: str,
    hits: int,
    ab: int,
    *,
    version: str = "v2",
    game_context: Optional[dict] = None,
) -> Optional[BatterScoreResult]:
    """
    Batter Score v1 with user-supplied career H2H H/AB overriding Statcast H2H.

    All other inputs (season form, SP ERA, pitch arsenal) still come from cache.
    """
    batter = build_batter_inputs(
        player_name,
        version=version,
        game_context=game_context,
    )
    if batter is None:
        return None

    batter = apply_manual_h2h_override(batter, hits, ab)
    return _score_batter_inputs(batter, game_context=game_context)


def score_batter_v2(
    player_name: str,
    version: str = "v2",
    game_context: Optional[dict] = None,
) -> Optional[BatterScoreResult]:
    """
    Batter Score v2: v2 weights, Savant pitch-type matchup, FIP L5 pitcher form.
    """
    batter = build_batter_inputs(
        player_name,
        version=version,
        game_context=game_context,
    )
    if batter is None:
        return None

    return _score_batter_inputs_v2(batter, game_context=game_context)


def _score_cache_key(
    player_name: str,
    game_context: Optional[dict],
) -> str:
    if not game_context:
        return _player_key(player_name)

    return "|".join(
        [
            _player_key(player_name),
            str(game_context.get("game_date", "")),
            str(game_context.get("home_team", "")).lower(),
            str(game_context.get("away_team", "")).lower(),
        ]
    )


@lru_cache(maxsize=256)
def _cached_score(
    cache_key: str,
    player_name: str,
    version: str,
    game_context_json: Optional[str],
) -> Optional[BatterScoreResult]:
    game_context = None
    if game_context_json:
        import json

        game_context = json.loads(game_context_json)

    return score_batter(
        player_name,
        version=version,
        game_context=game_context,
    )


def lookup_h2h_board_stats(
    player_name: str,
    version: str = "v2",
    game_context: Optional[dict] = None,
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Career H2H (pa, hits, ab) for the board vs-pitcher column.

    Uses merged statcast shards so prior-season matchups are included.
    Scoring still reads only the latest statcast file via build_batter_inputs().
    """
    player_rows = _batter_rows(player_name, version=version)
    if player_rows is None or player_rows.empty or not game_context:
        return None, None, None

    latest = player_rows.sort_values("game_date").iloc[-1]
    batter_id = coerce_mlb_id(latest.get("batter"))
    batter_team = latest.get("team")
    if batter_id is None or not isinstance(batter_team, str) or not batter_team.strip():
        return None, None, None

    _, sp_id = _lookup_opposing_sp_for_context(game_context, batter_team)
    sp_id = coerce_mlb_id(sp_id)
    if sp_id is None:
        return None, None, None

    statcast = _load_merged_statcast(_merged_statcast_cache_key())
    pa, _, hits, ab = _compute_h2h_stats(
        batter_id,
        sp_id,
        statcast=statcast,
    )
    return pa, hits, ab


def lookup_batter_score(
    player_name: str,
    version: str = "v2",
    game_context: Optional[dict] = None,
) -> Optional[BatterScoreResult]:
    import json

    context_json = (
        json.dumps(game_context, sort_keys=True)
        if game_context
        else None
    )
    cache_key = _score_cache_key(player_name, game_context)

    return _cached_score(
        cache_key,
        player_name,
        version,
        context_json,
    )


@lru_cache(maxsize=256)
def _cached_score_v2(
    cache_key: str,
    player_name: str,
    version: str,
    game_context_json: Optional[str],
) -> Optional[BatterScoreResult]:
    game_context = None
    if game_context_json:
        import json

        game_context = json.loads(game_context_json)

    return score_batter_v2(
        player_name,
        version=version,
        game_context=game_context,
    )


def lookup_batter_score_v2(
    player_name: str,
    version: str = "v2",
    game_context: Optional[dict] = None,
) -> Optional[BatterScoreResult]:
    import json

    context_json = (
        json.dumps(game_context, sort_keys=True)
        if game_context
        else None
    )
    cache_key = _score_cache_key(player_name, game_context) + "|v2"

    return _cached_score_v2(
        cache_key,
        player_name,
        version,
        context_json,
    )


def _result_to_row(result: BatterScoreResult) -> dict:
    return {
        "batter_score": result.batter_score,
        "batter_score_season_baseline": result.season_baseline,
        "batter_score_recent_form": result.recent_form,
        "batter_score_matchup_grade": result.matchup_grade,
        "batter_score_pitcher_form": result.pitcher_form,
        "batter_score_partial": result.is_partial,
        "batter_score_label": result.partial_label or "",
    }


def _row_game_context(row) -> Optional[dict]:
    return build_game_context(
        game=row.get("game"),
        commence_time=row.get("commence_time"),
        home_team=row.get("home_team"),
        away_team=row.get("away_team"),
    )


def analyze_batter_score_sp_coverage(
    df: pd.DataFrame,
) -> dict:
    """Summarize how many batter-score rows resolved an opposing SP."""
    from ui.player_stats import BATTER_MARKETS

    empty = {
        "ok": True,
        "player_games": 0,
        "with_sp": 0,
        "sp_tbd": 0,
        "warnings": [],
    }

    if df is None or df.empty or "batter_score_label" not in df.columns:
        return empty

    batters = df[df["market"].isin(BATTER_MARKETS)].copy()
    if batters.empty:
        return empty

    if "game" in batters.columns:
        keys = batters.dropna(subset=["player", "game"]).drop_duplicates(
            subset=["player", "game"]
        )
    else:
        keys = batters.dropna(subset=["player"]).drop_duplicates(
            subset=["player"]
        )

    labels = keys["batter_score_label"].fillna("").astype(str)
    sp_tbd = labels.str.contains("SP TBD", case=False, na=False).sum()
    player_games = len(keys)
    with_sp = player_games - sp_tbd

    warnings = []
    if player_games and sp_tbd == player_games:
        warnings.append(
            "All batter-score player-games are Partial · SP TBD — probables "
            "likely misaligned with the slate (timezone or stale fetch)."
        )
    elif player_games and sp_tbd / player_games >= 0.5:
        warnings.append(
            f"{sp_tbd}/{player_games} batter-score player-games are "
            "Partial · SP TBD — check daily_probables.parquet."
        )

    return {
        "ok": bool(sp_tbd == 0),
        "player_games": int(player_games),
        "with_sp": int(with_sp),
        "sp_tbd": int(sp_tbd),
        "warnings": warnings,
    }


def warn_batter_score_sp_coverage(
    df: pd.DataFrame,
    *,
    context: str = "",
) -> dict:
    """
    Print non-fatal warnings when Batter Score could not resolve SPs.

    Advisory only — enrichment still returns Partial · SP TBD rows.
    """
    result = analyze_batter_score_sp_coverage(df)

    if result["player_games"] == 0:
        return result

    prefix = "WARNING: Batter Score SP resolution"
    if context:
        prefix = f"{prefix} ({context})"

    if result["warnings"]:
        print()
        print("=" * 60)
        print(prefix)
        print("=" * 60)
        print(
            f"Player-games scored: {result['player_games']} | "
            f"With opposing SP: {result['with_sp']} | "
            f"SP TBD: {result['sp_tbd']}"
        )
        for message in result["warnings"]:
            print(message)
        print(
            "Fix: python fetch_data.py --probables, then restart Streamlit."
        )
        print()

    return result


def enrich_with_batter_score(
    df: pd.DataFrame,
    version: str = "v2",
) -> pd.DataFrame:
    """
    Add Batter Score columns to a predictions dataframe.

    Scores are computed per player and game when commence_time / game are
    available (Phase B SP lookup). Batter markets only.
    """
    from ui.player_stats import BATTER_MARKETS

    empty = {
        column: pd.Series(dtype=float if column == "batter_score" else object)
        for column in BATTER_SCORE_COLUMNS
    }
    empty["batter_score_partial"] = pd.Series(dtype=bool)

    if df.empty:
        result = df.copy()
        return result.assign(**empty)

    result = df.copy()

    score_values = []
    season_values = []
    recent_values = []
    matchup_values = []
    pitcher_values = []
    partial_values = []
    label_values = []

    score_cache: Dict[str, BatterScoreResult] = {}

    for _, row in result.iterrows():
        player = row["player"]
        market = row["market"]

        if market not in BATTER_MARKETS:
            score_values.append(np.nan)
            season_values.append(np.nan)
            recent_values.append(np.nan)
            matchup_values.append(np.nan)
            pitcher_values.append(np.nan)
            partial_values.append(False)
            label_values.append("")
            continue

        game_context = _row_game_context(row)
        cache_key = _score_cache_key(player, game_context)

        if cache_key not in score_cache:
            score_cache[cache_key] = lookup_batter_score(
                player,
                version=version,
                game_context=game_context,
            )

        scored = score_cache[cache_key]
        if scored is None:
            score_values.append(np.nan)
            season_values.append(np.nan)
            recent_values.append(np.nan)
            matchup_values.append(np.nan)
            pitcher_values.append(np.nan)
            partial_values.append(False)
            label_values.append("")
            continue

        row_data = _result_to_row(scored)
        score_values.append(row_data["batter_score"])
        season_values.append(row_data["batter_score_season_baseline"])
        recent_values.append(row_data["batter_score_recent_form"])
        matchup_values.append(row_data["batter_score_matchup_grade"])
        pitcher_values.append(row_data["batter_score_pitcher_form"])
        partial_values.append(row_data["batter_score_partial"])
        label_values.append(row_data["batter_score_label"])

    result["batter_score"] = score_values
    result["batter_score_season_baseline"] = season_values
    result["batter_score_recent_form"] = recent_values
    result["batter_score_matchup_grade"] = matchup_values
    result["batter_score_pitcher_form"] = pitcher_values
    result["batter_score_partial"] = partial_values
    result["batter_score_label"] = label_values

    return result


def get_batter_score_game_log(
    player_name: str,
    version: str = "v2",
    n: int = 10,
) -> Optional[pd.DataFrame]:
    """
    Last *n* games of H+TB+BB raw points for player-page charts.

    Reuses feature parquet game logs (same source as ui/player_stats.py).
    """
    player_rows = _batter_rows(player_name, version=version)
    if player_rows is None:
        return None

    recent = (
        player_rows.sort_values("game_date")
        .tail(n)
        .copy()
    )

    recent["raw_points"] = recent.apply(_raw_points_row, axis=1)
    recent["game_label"] = (
        pd.to_datetime(recent["game_date"])
        .dt.strftime("%m/%d")
    )

    return recent.set_index("game_label")[["raw_points"]]


_EMPTY_VALIDATION: Dict[str, Any] = {
    "validated": False,
    "sample_size": 0,
    "pearson_correlation": None,
    "spearman_correlation": None,
    "mae_implied_raw_points": None,
    "date_range": None,
    "criteria_used": {},
    "thresholds": {},
    "timestamp": None,
}


@lru_cache(maxsize=1)
def load_batter_score_validation() -> Dict[str, Any]:
    """Read cached Batter Score validation JSON (empty dict if missing)."""
    return _read_batter_score_validation_file(
        BATTER_SCORE_VALIDATION_PATH,
    )


def _read_batter_score_validation_file(
    path: Path,
) -> Dict[str, Any]:
    if not path.exists():
        return dict(_EMPTY_VALIDATION)

    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return dict(_EMPTY_VALIDATION)

    if not isinstance(payload, dict):
        return dict(_EMPTY_VALIDATION)

    merged = dict(_EMPTY_VALIDATION)
    merged.update(payload)
    return merged


def clear_batter_score_validation_cache() -> None:
    """Invalidate cached validation payload (for tests)."""
    load_batter_score_validation.cache_clear()


def is_batter_score_validated() -> bool:
    """True when the latest backtest passed configured validation gates."""
    return bool(
        load_batter_score_validation().get("validated", False)
    )


def write_batter_score_validation(payload: Dict[str, Any]) -> Path:
    """Persist validation results and refresh the loader cache."""
    BATTER_SCORE_VALIDATION_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with BATTER_SCORE_VALIDATION_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    clear_batter_score_validation_cache()
    return BATTER_SCORE_VALIDATION_PATH
