"""
Load batter game logs from feature parquets and compute Batter Score.

Phase B adds opposing SP ERA (L5), optional H2H blend, and game-context
gating via daily_probables.parquet.

Orthogonal to LightGBM prop models — used for UI ranking and player context.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from batter_score import (
    BatterInputs,
    BatterScoreResult,
    GameLine,
    PHASE_A_GATES,
    PHASE_B_GATES,
    PHASE_D_GATES,
    USE_TEAM_PITCHING_PROXY,
    compute_batter_score_partial,
    compute_batter_score_phase_b,
    compute_batter_score_phase_d,
)
from build_features import EXTRA_BASES, HITS
from fetch_probables import PROBABLES_PATH, lookup_opposing_sp
from pitch_matchup import arsenal_ready, build_opponent_pitcher_arsenal
from ui.player_stats import (
    _feature_cache_key,
    _fuzzy_player_key,
    _kind_player_cache_key,
    _kind_player_game_cache,
    _player_key,
    find_latest_feature_path,
)
from utils import RAW_DIR, coerce_mlb_id


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
    if commence_time is None or pd.isna(commence_time):
        return None

    try:
        return (
            pd.to_datetime(commence_time, utc=True)
            .strftime("%Y-%m-%d")
        )
    except (TypeError, ValueError):
        return None


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


def _statcast_cache_key():
    candidates = list(RAW_DIR.glob("statcast_*.parquet"))
    if not candidates:
        return (None, 0)

    latest = max(candidates, key=lambda path: path.stem)
    return _feature_cache_key(latest)


@lru_cache(maxsize=2)
def _load_latest_statcast(path_key) -> Optional[pd.DataFrame]:
    path_str, _mtime = path_key
    if path_str is None:
        return None

    return pd.read_parquet(path_str)


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


def _h2h_raw_points(events: pd.Series) -> Tuple[int, float]:
    hits = events.isin(HITS).astype(int)
    total_bases = events.map(EXTRA_BASES).fillna(0)
    walks = events.isin({"walk", "intent_walk"}).astype(int)
    pa_count = int(events.notna().sum())
    raw_points = float((hits + total_bases + walks).sum())
    return pa_count, raw_points


def _compute_h2h_stats(
    batter_id: Optional[int],
    sp_id: Optional[int],
) -> Tuple[Optional[int], Optional[float]]:
    """
    Batter vs SP career H2H from Statcast raw.

    Returns (pa_count, avg H+TB+BB per game vs that SP). H2H is omitted
    (None, None) when PA < MIN_PA_H2H — never zeroed out in scoring.
    """
    if batter_id is None or sp_id is None:
        return None, None

    batter_id = coerce_mlb_id(batter_id)
    sp_id = coerce_mlb_id(sp_id)
    if batter_id is None or sp_id is None:
        return None, None

    statcast = _load_latest_statcast(_statcast_cache_key())
    if statcast is None or statcast.empty:
        return None, None

    required = {"batter", "pitcher", "events", "game_date"}
    if not required.issubset(statcast.columns):
        return None, None

    matchups = statcast[
        statcast["batter"].astype(int).eq(batter_id)
        & statcast["pitcher"].astype(int).eq(sp_id)
        & statcast["events"].notna()
    ].copy()

    if matchups.empty:
        return 0, None

    pa_count = int(len(matchups))

    game_stats = []
    for _, group in matchups.groupby("game_date", sort=False):
        _, raw_points = _h2h_raw_points(group["events"])
        game_stats.append(raw_points)

    if not game_stats:
        return pa_count, None

    avg_raw_points = float(np.mean(game_stats))
    return pa_count, avg_raw_points


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

    required = {"hits", "total_bases", "walks", "game_date"}
    if not required.issubset(player_rows.columns):
        return None

    if len(player_rows) < 10:
        return None

    raw_points = player_rows.apply(_raw_points_row, axis=1)
    season_avg = float(raw_points.mean())
    game_log = _rows_to_game_log(player_rows)

    display_name = str(
        player_rows["player_name"].iloc[-1]
    ).strip()

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
    h2h_pa = None
    h2h_avg_raw_points = None
    team_proxy = None
    opponent_arsenal = []

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
            if sp_id is not None:
                h2h_pa, h2h_avg_raw_points = _compute_h2h_stats(
                    batter_id,
                    sp_id,
                )
                statcast = _load_latest_statcast(_statcast_cache_key())
                opponent_arsenal = build_opponent_pitcher_arsenal(
                    statcast,
                    batter_id,
                    sp_id,
                )
        elif USE_TEAM_PITCHING_PROXY:
            team_proxy = _team_opp_earned_runs_proxy(player_rows)

    return BatterInputs(
        name=display_name or str(player_name).strip(),
        season_avg_raw_points=season_avg,
        game_log=game_log,
        opponent_pitcher_arsenal=opponent_arsenal,
        opponent_pitcher_era_l5=sp_era_l5,
        opposing_sp_name=sp_name,
        h2h_pa=h2h_pa,
        h2h_avg_raw_points=h2h_avg_raw_points,
        team_opp_earned_runs_proxy=team_proxy,
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
            )

        if sp_ready:
            return compute_batter_score_phase_b(
                batter,
                gates=PHASE_B_GATES,
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
            )

        return compute_batter_score_partial(
            batter,
            gates=PHASE_A_GATES,
            sp_tbd=game_context is not None and not sp_named,
        )
    except ValueError:
        return None


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
