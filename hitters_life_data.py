"""
Data helpers for the Hitter's Life batting-average board.

Reads feature parquets and cached Statcast only (no live API calls except
Rotowire lineups via fetch_rotowire_lineups).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import pandas as pd

from batter_score_data import (
    _batter_rows,
    _load_merged_statcast,
    _merged_statcast_cache_key,
    build_game_context,
    lookup_h2h_board_stats,
)
from pitch_matchup import (
    PITCH_BUCKETS,
    aggregate_batter_pitch_stats,
    aggregate_pitcher_arsenal_usage_detailed,
    build_opponent_pitcher_arsenal,
)
from ui.player_stats import (
    _fuzzy_player_key,
    _player_key,
    get_last_n_games,
    lookup_pitcher_hand,
    BATTER_MARKETS,
)
from utils import TEAM_ABBR_TO_ODDS, coerce_mlb_id


def _format_avg_rate(hits: int, ab: int) -> str:
    if ab <= 0:
        return "—"
    avg_text = f"{hits / ab:.3f}".removeprefix("0")
    return avg_text


def _format_avg_cell(hits: int, ab: int) -> str:
    if ab <= 0:
        return "—"
    return f"{hits}/{ab} {_format_avg_rate(hits, ab)}"


def _format_avg_value(value) -> str:
    if value is None or pd.isna(value):
        return "—"
    avg_text = f"{float(value):.3f}".removeprefix("0")
    return avg_text


@lru_cache(maxsize=512)
def _cached_batter_game_hits_ab(
    batter_id: int,
    cache_key: tuple,
) -> tuple[tuple[str, int, int], ...]:
    statcast = _load_merged_statcast(cache_key)
    if statcast is None or statcast.empty:
        return tuple()

    from batter_score_data import _h2h_hits_ab

    rows = statcast[
        statcast["batter"].astype(int).eq(int(batter_id))
        & statcast["events"].notna()
    ]
    if rows.empty or "game_date" not in rows.columns:
        return tuple()

    games: list[tuple[str, int, int]] = []
    for game_date, group in rows.groupby("game_date", sort=True):
        hits, ab = _h2h_hits_ab(group["events"])
        if ab > 0:
            games.append((str(game_date)[:10], hits, ab))

    return tuple(games)


def _avg_from_games(
    games: tuple[tuple[str, int, int], ...],
    *,
    window: int | None = None,
) -> Optional[float]:
    if not games:
        return None

    subset = games[-window:] if window else games
    hits = sum(item[1] for item in subset)
    ab = sum(item[2] for item in subset)
    if ab <= 0:
        return None
    return hits / ab


def lookup_batting_average_windows(
    player_name: str,
    version: str = "v2",
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Season, L5, and L10 batting average from merged Statcast."""
    player_rows = _batter_rows(player_name, version=version)
    if player_rows is None or player_rows.empty:
        return None, None, None

    latest = player_rows.sort_values("game_date").iloc[-1]
    batter_id = coerce_mlb_id(latest.get("batter"))
    if batter_id is None:
        return None, None, None

    games = _cached_batter_game_hits_ab(
        batter_id,
        _merged_statcast_cache_key(),
    )
    return (
        _avg_from_games(games),
        _avg_from_games(games, window=5),
        _avg_from_games(games, window=10),
    )


def hitters_life_player_link(player_name: str) -> str:
    """Player profile link; fragment holds display name for LinkColumn."""
    from urllib.parse import urlencode

    name = str(player_name).strip()
    return "/?" + urlencode({"player": name}) + f"#{name}"


def format_batting_average_column(
    player_name: str,
    version: str,
) -> str:
    """Season, L5, and L10 batting average (no H2H — that lives in Vs pitcher)."""
    season, l5, l10 = lookup_batting_average_windows(player_name, version)
    parts = [
        f"Szn {_format_avg_value(season)}",
        f"L5 {_format_avg_value(l5)}",
        f"L10 {_format_avg_value(l10)}",
    ]
    return " · ".join(parts)


def lookup_pitch_bucket_woba(
    player_name: str,
    pitch_bucket: str,
    version: str = "v2",
) -> Optional[float]:
    """Career wOBA vs a pitch bucket from Statcast."""
    if pitch_bucket not in PITCH_BUCKETS:
        return None

    player_rows = _batter_rows(player_name, version=version)
    if player_rows is None or player_rows.empty:
        return None

    latest = player_rows.sort_values("game_date").iloc[-1]
    batter_id = coerce_mlb_id(latest.get("batter"))
    if batter_id is None:
        return None

    statcast = _load_merged_statcast(_merged_statcast_cache_key())
    if statcast is None or statcast.empty:
        return None

    stats = aggregate_batter_pitch_stats(statcast, batter_id)
    bucket_stats = stats.get(pitch_bucket, {})
    woba = bucket_stats.get("woba")
    if woba is None or pd.isna(woba):
        return None
    return float(woba)


def lookup_arsenal_weighted_woba(
    player_name: str,
    version: str,
    game_context: dict | None,
) -> Optional[float]:
    """
    Usage-weighted career wOBA vs the opposing SP's pitch arsenal (Phase D).

    Each pitch bucket is weighted by the SP's usage over their last five starts.
    """
    if not game_context:
        return None

    from batter_score_data import _lookup_opposing_sp_for_context

    player_rows = _batter_rows(player_name, version=version)
    if player_rows is None or player_rows.empty:
        return None

    latest = player_rows.sort_values("game_date").iloc[-1]
    batter_id = coerce_mlb_id(latest.get("batter"))
    batter_team = latest.get("team")
    if batter_id is None or batter_team is None or pd.isna(batter_team):
        return None

    _, sp_id = _lookup_opposing_sp_for_context(
        game_context,
        str(batter_team),
    )
    sp_id = coerce_mlb_id(sp_id)
    if sp_id is None:
        return None

    statcast = _load_merged_statcast(_merged_statcast_cache_key())
    if statcast is None or statcast.empty:
        return None

    arsenal = build_opponent_pitcher_arsenal(statcast, batter_id, sp_id)
    if not arsenal:
        return None

    weighted = sum(item.usage_pct * item.batter_woba for item in arsenal)
    usage_total = sum(item.usage_pct for item in arsenal)
    if usage_total <= 0:
        return None
    return float(weighted / usage_total)


def format_pitch_woba(value) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.3f}".removeprefix("0")


def format_total_bases_game_log(
    player_name: str,
    version: str = "v2",
    *,
    n: int = 5,
) -> str:
    """Last *n* total-base outcomes as a space-separated string."""
    log = get_last_n_games(
        player_name,
        "batter_total_bases",
        version=version,
        n=n,
    )
    if log is None or log.empty:
        return "—"

    values = (
        pd.to_numeric(log["total_bases"], errors="coerce")
        .fillna(0)
        .astype(int)
        .tolist()
    )
    if not values:
        return "—"
    return " ".join(str(value) for value in values)


def lookup_sp_arsenal_usage(
    player_name: str,
    version: str,
    game_context: dict | None,
) -> dict[str, float]:
    """Opposing SP pitch-bucket usage (last 5 starts) for the SP arsenal column."""
    if not game_context:
        return {}

    from batter_score_data import _lookup_opposing_sp_for_context

    player_rows = _batter_rows(player_name, version=version)
    if player_rows is None or player_rows.empty:
        return {}

    latest = player_rows.sort_values("game_date").iloc[-1]
    batter_team = latest.get("team")
    if batter_team is None or pd.isna(batter_team):
        return {}

    _, sp_id = _lookup_opposing_sp_for_context(
        game_context,
        str(batter_team),
    )
    sp_id = coerce_mlb_id(sp_id)
    if sp_id is None:
        return {}

    statcast = _load_merged_statcast(_merged_statcast_cache_key())
    if statcast is None or statcast.empty:
        return {}

    return aggregate_pitcher_arsenal_usage_detailed(statcast, sp_id)


def format_sp_arsenal_column(usage: dict[str, float]) -> str:
    """Savant pitch types from last 5 starts, sorted by usage."""
    if not usage:
        return "—"

    ordered = sorted(
        usage.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    parts = [
        name
        for name, pct in ordered
        if pct and pct > 0
    ]
    return " · ".join(parts) if parts else "—"


def format_vs_pitcher_cell(
    player_name: str,
    version: str,
    game_context: dict | None,
    result,
    *,
    sp_display: str,
) -> str:
    """Vs pitcher: SP name + H2H batting average or SP ERA L5."""
    from batter_score_data import lookup_h2h_board_stats

    MIN_PA_H2H_BOARD = 3
    label = sp_display if sp_display and sp_display != "TBD" else "SP TBD"

    pa = hits = ab = None
    if game_context:
        pa, hits, ab = lookup_h2h_board_stats(
            player_name,
            version=version,
            game_context=game_context,
        )

    if pa is not None and pa >= MIN_PA_H2H_BOARD and ab and ab > 0:
        hit_count = hits if hits is not None else 0
        return f"{label} · {_format_avg_cell(hit_count, ab)}"

    if result is not None and result.opposing_sp_era_l5 is not None:
        return f"{label} · SP ERA L5 {result.opposing_sp_era_l5:.2f}"

    return label if label != "SP TBD" else "—"


def _lookup_batter_team_abbr(
    player_name: str,
    version: str,
) -> str | None:
    player_rows = _batter_rows(player_name, version=version)
    if player_rows is None or player_rows.empty:
        return None

    latest = player_rows.sort_values("game_date").iloc[-1]
    team = latest.get("team")
    if team is None or pd.isna(team):
        return None

    team_abbr = str(team).strip().upper()
    if team_abbr in TEAM_ABBR_TO_ODDS:
        return team_abbr

    from fetch_rotowire_lineups import odds_team_to_abbr

    return odds_team_to_abbr(str(team))


def _prepare_hitters_life_slate(
    df: pd.DataFrame,
    *,
    markets=None,
) -> pd.DataFrame:
    batters = df[df["market"].isin(BATTER_MARKETS)].copy()
    if markets:
        batters = batters[batters["market"].isin(markets)]
    if batters.empty:
        return batters

    hits_rows = batters[batters["market"] == "batter_hits"]
    if not hits_rows.empty:
        source = hits_rows
    else:
        source = batters

    def _best_row_index(group):
        if "edge" in group.columns and group["edge"].notna().any():
            return group["edge"].idxmax()
        return group.index[0]

    best_idx = source.groupby("player").apply(_best_row_index)
    return (
        source.loc[best_idx.tolist()]
        .sort_values("player")
        .reset_index(drop=True)
    )


def build_hitters_life_row(
    row,
    version: str,
    *,
    pitch_bucket: str,
) -> dict:
    from batter_score_data import lookup_batter_score
    from ui.formatting import format_game_time, format_name_with_hand

    game_context = build_game_context(
        game=row.get("game"),
        commence_time=row.get("commence_time"),
        home_team=row.get("home_team"),
        away_team=row.get("away_team"),
    )
    result = lookup_batter_score(
        row["player"],
        version=version,
        game_context=game_context,
    )

    opposing = "TBD"
    if result and result.opposing_sp_name:
        sp_hand = lookup_pitcher_hand(
            result.opposing_sp_name,
            version=version,
        )
        opposing = format_name_with_hand(
            result.opposing_sp_name,
            sp_hand,
        )

    woba = lookup_pitch_bucket_woba(
        row["player"],
        pitch_bucket,
        version=version,
    )
    arsenal_woba = lookup_arsenal_weighted_woba(
        row["player"],
        version,
        game_context,
    )
    arsenal_usage = lookup_sp_arsenal_usage(
        row["player"],
        version,
        game_context,
    )

    return {
        "player": row["player"],
        "player_link": hitters_life_player_link(row["player"]),
        "game_time": format_game_time(row.get("game"), row.get("commence_time")),
        "vs_pitcher": format_vs_pitcher_cell(
            row["player"],
            version,
            game_context,
            result,
            sp_display=opposing,
        ),
        "arsenal_woba": format_pitch_woba(arsenal_woba),
        "batting_average": format_batting_average_column(
            row["player"],
            version,
        ),
        "pitch_woba": format_pitch_woba(woba),
        "sp_arsenal": format_sp_arsenal_column(arsenal_usage),
        "total_bases_log": format_total_bases_game_log(
            row["player"],
            version=version,
        ),
        "_game": row.get("game") or "",
        "_team_abbr": _lookup_batter_team_abbr(row["player"], version),
        "_opposing_sp_hand": (
            lookup_pitcher_hand(result.opposing_sp_name, version=version)
            if result and result.opposing_sp_name
            else None
        ),
    }


def build_hitters_life_df(
    df: pd.DataFrame,
    version: str,
    *,
    pitch_bucket: str,
    markets=None,
) -> pd.DataFrame:
    slate = _prepare_hitters_life_slate(df, markets=markets)
    if slate.empty:
        return pd.DataFrame()

    rows = [
        build_hitters_life_row(
            row,
            version,
            pitch_bucket=pitch_bucket,
        )
        for _, row in slate.iterrows()
    ]
    return pd.DataFrame(rows)


def match_player_to_lineup(
    player_name: str,
    lineup_names: list[str],
) -> bool:
    if not lineup_names:
        return False

    keys = {_player_key(name) for name in lineup_names}
    return _fuzzy_player_key(player_name, keys) is not None


def lineup_sort_key(
    player_name: str,
    lineup_names: list[str],
) -> tuple[int, str]:
    """Sort key: lineup slot when matched, else tail alphabetically."""
    if not lineup_names:
        return (999, _player_key(player_name))

    keyed = {_player_key(name): idx for idx, name in enumerate(lineup_names, start=1)}
    match = _fuzzy_player_key(player_name, keyed.keys())
    if match is None:
        return (999, _player_key(player_name))
    return (keyed[match], _player_key(player_name))
