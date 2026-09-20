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


def format_h2h_avg_display(
    h2h_avg: float | None,
    *,
    hits: int | None = None,
    ab: int | None = None,
    career_override: bool = False,
) -> str:
    """Display H2H average as ``hits/AB .AVG`` (e.g. ``4/10 .400``)."""
    if h2h_avg is None or (isinstance(h2h_avg, float) and pd.isna(h2h_avg)):
        return "—"

    if ab and ab > 0 and hits is not None:
        text = _format_avg_cell(int(hits), int(ab))
    else:
        avg_text = f"{float(h2h_avg):.3f}".removeprefix("0")
        text = avg_text if avg_text.startswith(".") else avg_text

    if career_override and text != "—":
        return f"{text} · career"
    return text


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


@lru_cache(maxsize=512)
def _cached_batter_game_total_bases(
    batter_id: int,
    cache_key: tuple,
) -> tuple[tuple[str, int], ...]:
    """Per-game total bases from merged Statcast (matches build_features TB logic)."""
    statcast = _load_merged_statcast(cache_key)
    if statcast is None or statcast.empty:
        return tuple()

    from build_features import EXTRA_BASES

    rows = statcast[
        statcast["batter"].astype(int).eq(int(batter_id))
        & statcast["events"].notna()
    ]
    if rows.empty or "game_date" not in rows.columns:
        return tuple()

    games: list[tuple[str, int]] = []
    for game_date, group in rows.groupby("game_date", sort=True):
        total_bases = int(
            group["events"]
            .map(EXTRA_BASES)
            .fillna(0)
            .sum()
        )
        games.append((str(game_date)[:10], total_bases))

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


@lru_cache(maxsize=512)
def _cached_batter_game_hits_ab_by_hand(
    batter_id: int,
    cache_key: tuple,
) -> tuple[tuple[str, str, int, int], ...]:
    """
    Per-game hits/AB split by opposing pitcher hand (``p_throws``).

    Each tuple is ``(game_date, hand, hits, ab)`` with hand ``R`` or ``L``.
    Games with no ABs vs that hand are omitted.
    """
    statcast = _load_merged_statcast(cache_key)
    if statcast is None or statcast.empty:
        return tuple()

    required = {"batter", "events", "game_date", "p_throws"}
    if not required.issubset(statcast.columns):
        return tuple()

    from batter_score_data import _h2h_hits_ab

    rows = statcast[
        pd.to_numeric(statcast["batter"], errors="coerce").eq(int(batter_id))
        & statcast["events"].notna()
        & statcast["p_throws"].astype(str).str.upper().isin(["R", "L"])
    ].copy()
    if rows.empty:
        return tuple()

    rows["game_date_key"] = rows["game_date"].map(lambda value: str(value)[:10])
    rows["hand"] = rows["p_throws"].astype(str).str.upper()

    games: list[tuple[str, str, int, int]] = []
    for (game_date, hand), group in rows.groupby(
        ["game_date_key", "hand"],
        sort=True,
    ):
        hits, ab = _h2h_hits_ab(group["events"])
        if ab > 0:
            games.append((str(game_date), str(hand), int(hits), int(ab)))

    return tuple(games)


def _games_for_hand(
    games: tuple[tuple[str, str, int, int], ...],
    hand: str,
) -> tuple[tuple[str, int, int], ...]:
    target = str(hand).upper()
    return tuple(
        (game_date, hits, ab)
        for game_date, game_hand, hits, ab in games
        if game_hand == target
    )


def lookup_batting_average_vs_hand_windows(
    player_name: str,
    version: str = "v2",
) -> dict[str, Optional[float]]:
    """
    L5 / L10 batting average vs RHP and LHP.

    Windows are the last 5 / 10 *games dates* with at least one AB vs that hand
    (Statcast ``p_throws``), not calendar days.
    """
    empty = {
        "l5_vs_r": None,
        "l10_vs_r": None,
        "l5_vs_l": None,
        "l10_vs_l": None,
    }
    player_rows = _batter_rows(player_name, version=version)
    if player_rows is None or player_rows.empty:
        return empty

    latest = player_rows.sort_values("game_date").iloc[-1]
    batter_id = coerce_mlb_id(latest.get("batter"))
    if batter_id is None:
        return empty

    games = _cached_batter_game_hits_ab_by_hand(
        batter_id,
        _merged_statcast_cache_key(),
    )
    vs_r = _games_for_hand(games, "R")
    vs_l = _games_for_hand(games, "L")
    return {
        "l5_vs_r": _avg_from_games(vs_r, window=5),
        "l10_vs_r": _avg_from_games(vs_r, window=10),
        "l5_vs_l": _avg_from_games(vs_l, window=5),
        "l10_vs_l": _avg_from_games(vs_l, window=10),
    }


def format_batting_average_vs_hand_column(
    player_name: str,
    version: str,
    *,
    hand: str,
) -> str:
    """Format ``L5 .xxx · L10 .xxx`` for AVG vs RHP or LHP."""
    windows = lookup_batting_average_vs_hand_windows(player_name, version)
    key = "r" if str(hand).upper().startswith("R") else "l"
    l5 = windows.get(f"l5_vs_{key}")
    l10 = windows.get(f"l10_vs_{key}")
    return f"L5 {_format_avg_value(l5)} · L10 {_format_avg_value(l10)}"


@lru_cache(maxsize=512)
def _cached_pitcher_season_baa_vs_stand(
    pitcher_id: int,
    cache_key: tuple,
) -> tuple[Optional[float], Optional[float]]:
    """Season BAA allowed vs RHB / LHB (batter ``stand``). Returns (vs_R, vs_L)."""
    statcast = _load_merged_statcast(cache_key)
    if statcast is None or statcast.empty:
        return None, None

    required = {"pitcher", "events", "stand"}
    if not required.issubset(statcast.columns):
        return None, None

    from batter_score_data import _h2h_hits_ab

    rows = statcast[
        pd.to_numeric(statcast["pitcher"], errors="coerce").eq(int(pitcher_id))
        & statcast["events"].notna()
        & statcast["stand"].astype(str).str.upper().isin(["R", "L"])
    ]
    if rows.empty:
        return None, None

    result: dict[str, Optional[float]] = {"R": None, "L": None}
    for hand in ("R", "L"):
        subset = rows[rows["stand"].astype(str).str.upper().eq(hand)]
        if subset.empty:
            continue
        hits, ab = _h2h_hits_ab(subset["events"])
        if ab > 0:
            result[hand] = hits / ab

    return result["R"], result["L"]


def lookup_opposing_sp_baa_vs_hands(
    player_name: str,
    version: str = "v2",
    game_context: Optional[dict] = None,
) -> tuple[Optional[float], Optional[float]]:
    """Opposing SP season BAA vs RHB / LHB for *player_name*'s game."""
    if not game_context:
        return None, None

    from batter_score_data import _lookup_opposing_sp_for_context

    player_rows = _batter_rows(player_name, version=version)
    if player_rows is None or player_rows.empty:
        return None, None

    latest = player_rows.sort_values("game_date").iloc[-1]
    batter_team = latest.get("team")
    if not isinstance(batter_team, str) or not batter_team.strip():
        return None, None

    _, sp_id = _lookup_opposing_sp_for_context(game_context, batter_team)
    sp_id = coerce_mlb_id(sp_id)
    if sp_id is None:
        return None, None

    return _cached_pitcher_season_baa_vs_stand(
        sp_id,
        _merged_statcast_cache_key(),
    )


def format_opposing_sp_baa_column(
    player_name: str,
    version: str,
    game_context: Optional[dict] = None,
) -> str:
    """Format opposing SP season BAA as ``vs R .xxx · vs L .xxx``."""
    vs_r, vs_l = lookup_opposing_sp_baa_vs_hands(
        player_name,
        version=version,
        game_context=game_context,
    )
    return (
        f"vs R {_format_avg_value(vs_r)} · vs L {_format_avg_value(vs_l)}"
    )


# Savant xwOBA: contact uses estimated_woba_using_speedangle; walks/HBP/K use
# woba_value weights (same pitch-level fields as Baseball Savant CSV).
_PA_XWOBA_ZERO = frozenset({"strikeout", "strikeout_double_play"})
_PA_XWOBA_USE_WOBA_VALUE = frozenset(
    {"walk", "intent_walk", "hit_by_pitch", "catcher_interf", "truncated_pa"}
)

# FanGraphs wOBA scale + lg R/PA by season (lgwOBA computed from local Statcast).
_LEAGUE_WRC_FIXED: dict[int, tuple[float, float]] = {
    2024: (1.243, 0.118),
    2025: (1.247, 0.117),
    2026: (1.247, 0.117),
}


def _pa_woba_xwoba_parts(
    event,
    woba_value,
    woba_denom,
    estimated_xwoba,
) -> tuple[float, float, float] | None:
    """Return (wOBA num, xwOBA num, denom) for one plate appearance."""
    if event is None or (isinstance(event, float) and pd.isna(event)):
        return None

    event = str(event)
    denom = pd.to_numeric(woba_denom, errors="coerce")
    if pd.isna(denom) or float(denom) <= 0:
        denom = 1.0
    else:
        denom = float(denom)

    woba = pd.to_numeric(woba_value, errors="coerce")
    w_num = float(woba) if pd.notna(woba) else 0.0

    if event in _PA_XWOBA_ZERO:
        x_num = 0.0
    elif event in _PA_XWOBA_USE_WOBA_VALUE:
        x_num = w_num
    else:
        est = pd.to_numeric(estimated_xwoba, errors="coerce")
        if pd.notna(est):
            x_num = float(est)
        else:
            x_num = w_num

    return w_num * denom, x_num * denom, denom


@lru_cache(maxsize=512)
def _cached_batter_game_woba_xwoba(
    batter_id: int,
    cache_key: tuple,
) -> tuple[tuple[str, float, float, float], ...]:
    """Per-game (wOBA sum, xwOBA sum, wOBA denom) from merged Statcast."""
    statcast = _load_merged_statcast(cache_key)
    if statcast is None or statcast.empty:
        return tuple()

    rows = statcast[
        statcast["batter"].astype(int).eq(int(batter_id))
        & statcast["events"].notna()
    ]
    if rows.empty or "game_date" not in rows.columns:
        return tuple()

    games: list[tuple[str, float, float, float]] = []
    for game_date, group in rows.groupby("game_date", sort=True):
        w_sum = 0.0
        x_sum = 0.0
        d_sum = 0.0
        for _, row in group.iterrows():
            parts = _pa_woba_xwoba_parts(
                row.get("events"),
                row.get("woba_value"),
                row.get("woba_denom"),
                row.get("estimated_woba_using_speedangle"),
            )
            if parts is None:
                continue
            w_part, x_part, denom = parts
            w_sum += w_part
            x_sum += x_part
            d_sum += denom

        if d_sum > 0:
            games.append((str(game_date)[:10], w_sum, x_sum, d_sum))

    return tuple(games)


def _woba_rate_from_games(
    games: tuple[tuple[str, float, float, float], ...],
    *,
    window: int | None = None,
    use_xwoba: bool = False,
) -> Optional[float]:
    if not games:
        return None

    subset = games[-window:] if window else games
    if use_xwoba:
        numerator = sum(item[2] for item in subset)
    else:
        numerator = sum(item[1] for item in subset)
    denominator = sum(item[3] for item in subset)
    if denominator <= 0:
        return None
    return numerator / denominator


@lru_cache(maxsize=4)
def _league_wrc_constants(
    cache_key: tuple,
) -> tuple[float, float, float]:
    """Return (lgwOBA, wOBA scale, lg R/PA) for wRC+ (Savant/FanGraphs formula)."""
    statcast = _load_merged_statcast(cache_key)
    fallback_scale, fallback_r_pa = _LEAGUE_WRC_FIXED.get(2026, (1.247, 0.117))
    if statcast is None or statcast.empty:
        return 0.310, fallback_scale, fallback_r_pa

    pa = statcast[statcast["events"].notna()]
    if pa.empty:
        return 0.310, fallback_scale, fallback_r_pa

    denom = pd.to_numeric(pa["woba_denom"], errors="coerce").fillna(0)
    woba_val = pd.to_numeric(pa["woba_value"], errors="coerce").fillna(0)
    total_denom = float(denom.sum())
    if total_denom <= 0:
        lgwoba = 0.310
    else:
        lgwoba = float((woba_val * denom).sum() / total_denom)

    years = pd.to_datetime(pa["game_date"], errors="coerce").dt.year.dropna()
    season = int(years.max()) if not years.empty else 2026
    scale, lg_r_pa = _LEAGUE_WRC_FIXED.get(season, (fallback_scale, fallback_r_pa))
    return lgwoba, scale, lg_r_pa


def _wrc_plus_from_woba(
    woba: float | None,
    lgwoba: float,
    woba_scale: float,
    lg_r_pa: float,
) -> Optional[float]:
    """wRC+ = 100 * (wRC/PA) / (lg R/PA) using FanGraphs/Savant definition."""
    if woba is None or pd.isna(woba):
        return None
    if woba_scale <= 0 or lg_r_pa <= 0:
        return None

    wrc_per_pa = (float(woba) - lgwoba) / woba_scale + lg_r_pa
    return 100.0 * wrc_per_pa / lg_r_pa


def _wrc_plus_from_games(
    games: tuple[tuple[str, float, float, float], ...],
    constants: tuple[float, float, float],
    *,
    window: int | None = None,
) -> Optional[float]:
    woba = _woba_rate_from_games(games, window=window, use_xwoba=False)
    return _wrc_plus_from_woba(woba, *constants)


def _batter_game_woba_xwoba(
    player_name: str,
    version: str,
) -> tuple[tuple[str, float, float, float], ...] | None:
    player_rows = _batter_rows(player_name, version=version)
    if player_rows is None or player_rows.empty:
        return None

    latest = player_rows.sort_values("game_date").iloc[-1]
    batter_id = coerce_mlb_id(latest.get("batter"))
    if batter_id is None:
        return None

    return _cached_batter_game_woba_xwoba(
        batter_id,
        _merged_statcast_cache_key(),
    )


def lookup_xwoba_windows(
    player_name: str,
    version: str = "v2",
) -> tuple[Optional[float], Optional[float]]:
    """L5 and L10 xwOBA from merged Statcast (Savant methodology)."""
    games = _batter_game_woba_xwoba(player_name, version)
    if not games:
        return None, None

    return (
        _woba_rate_from_games(games, window=5, use_xwoba=True),
        _woba_rate_from_games(games, window=10, use_xwoba=True),
    )


def lookup_wrc_plus_windows(
    player_name: str,
    version: str = "v2",
) -> tuple[Optional[float], Optional[float]]:
    """Last-30-game and last-10-game wRC+ (pooled PA per window)."""
    games = _batter_game_woba_xwoba(player_name, version)
    if not games:
        return None, None

    constants = _league_wrc_constants(_merged_statcast_cache_key())
    return (
        _wrc_plus_from_games(games, constants, window=30),
        _wrc_plus_from_games(games, constants, window=10),
    )


def format_xwoba_column(player_name: str, version: str) -> str:
    """L5 and L10 xwOBA (Statcast expected wOBA); L5 first for sorting."""
    l5, l10 = lookup_xwoba_windows(player_name, version)
    parts = [
        f"L5 {format_pitch_woba(l5)}",
        f"L10 {format_pitch_woba(l10)}",
    ]
    return " · ".join(parts)


def _format_wrc_plus_value(value) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{round(float(value))}"


def format_wrc_plus_column(player_name: str, version: str) -> str:
    """L30 and L10 wRC+ (100 = league average); L30 first for sorting."""
    l30, l10 = lookup_wrc_plus_windows(player_name, version)
    parts = [
        f"L30 {_format_wrc_plus_value(l30)}",
        f"L10 {_format_wrc_plus_value(l10)}",
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
    """
    Last *n* total-base outcomes as space-separated integers.

    Leftmost value is the most recent game. Uses merged Statcast (same source
    as season/L5/L10 AVG). Falls back to feature parquets when Statcast is
    unavailable.
    """
    player_rows = _batter_rows(player_name, version=version)
    if player_rows is not None and not player_rows.empty:
        latest = player_rows.sort_values("game_date").iloc[-1]
        batter_id = coerce_mlb_id(latest.get("batter"))
        if batter_id is not None:
            games = _cached_batter_game_total_bases(
                batter_id,
                _merged_statcast_cache_key(),
            )
            if games:
                recent = list(reversed(games[-n:]))
                return " ".join(str(total_bases) for _, total_bases in recent)

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
        .dropna()
        .astype(int)
        .tolist()
    )
    if not values:
        return "—"
    return " ".join(str(value) for value in reversed(values))


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
    """Legacy combined cell (batter score pick cards). Prefer ``build_vs_pitcher_fields``."""
    fields = build_vs_pitcher_fields(
        player_name,
        version,
        game_context,
        result,
        sp_display=sp_display,
    )
    label = fields["opposing_sp"]
    if label in ("—", "SP TBD"):
        return "—" if label == "—" else "SP TBD"
    if fields["h2h_avg"] is not None and fields.get("h2h_ab"):
        hits, ab = fields["h2h_hits"], fields["h2h_ab"]
        return f"{label} · {_format_avg_cell(hits, ab)}"
    if fields.get("sp_era_l5") is not None:
        return f"{label} · SP ERA L5 {fields['sp_era_l5']:.2f}"
    return label


MIN_PA_H2H_BOARD = 3


def build_vs_pitcher_fields(
    player_name: str,
    version: str,
    game_context: dict | None,
    result,
    *,
    sp_display: str,
) -> dict:
    """
    Split opposing-SP display for Hitter's Life.

    Returns ``opposing_sp`` (name), ``h2h_avg`` (float or None for sorting),
    and optional ``h2h_hits`` / ``h2h_ab`` / ``sp_era_l5``.
    """
    label = sp_display if sp_display and sp_display != "TBD" else "SP TBD"
    opposing_sp = label if label != "SP TBD" else "—"

    pa = hits = ab = None
    if game_context:
        pa, hits, ab = lookup_h2h_board_stats(
            player_name,
            version=version,
            game_context=game_context,
        )

    h2h_avg = None
    career_override = False
    if pa is not None and pa >= MIN_PA_H2H_BOARD and ab and ab > 0:
        hit_count = hits if hits is not None else 0
        h2h_avg = hit_count / ab
        if result is not None and result.opposing_sp_name:
            from h2h_career_overrides import lookup_career_h2h

            career_override = (
                lookup_career_h2h(player_name, result.opposing_sp_name)
                is not None
            )

    sp_era_l5 = None
    if h2h_avg is None and result is not None and result.opposing_sp_era_l5 is not None:
        sp_era_l5 = float(result.opposing_sp_era_l5)

    return {
        "opposing_sp": opposing_sp,
        "h2h_avg": h2h_avg,
        "h2h_hits": hits if hits is not None else None,
        "h2h_ab": ab,
        "h2h_career_override": career_override,
        "sp_era_l5": sp_era_l5,
    }


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
    from batter_score_data import lookup_batter_score, lookup_batter_score_v2, lookup_batter_score_v3
    from ui.formatting import format_name_with_hand

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
    result_v2 = lookup_batter_score_v2(
        row["player"],
        version=version,
        game_context=game_context,
    )
    result_v3 = lookup_batter_score_v3(
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

    vs_pitcher = build_vs_pitcher_fields(
        row["player"],
        version,
        game_context,
        result,
        sp_display=opposing,
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

    from ui.batter_score import format_batter_score_display
    from ui.player_stats import (
        format_prizepicks_fantasy_line,
        format_underdog_fantasy_line,
        lookup_prizepicks_fantasy_line,
        lookup_underdog_fantasy_line,
    )

    pp_line = lookup_prizepicks_fantasy_line(row["player"])
    ud_line = lookup_underdog_fantasy_line(row["player"])

    batter_score_v2 = result_v2.batter_score if result_v2 else None
    batter_score_v2_label = (result_v2.partial_label if result_v2 else "") or ""
    batter_score_v3 = result_v3.batter_score if result_v3 else None
    batter_score_v3_label = (result_v3.partial_label if result_v3 else "") or ""

    return {
        "player": row["player"],
        "player_link": hitters_life_player_link(row["player"]),
        "opposing_sp": vs_pitcher["opposing_sp"],
        "h2h_avg": format_h2h_avg_display(
            vs_pitcher["h2h_avg"],
            hits=vs_pitcher["h2h_hits"],
            ab=vs_pitcher["h2h_ab"],
            career_override=bool(vs_pitcher.get("h2h_career_override")),
        ),
        "_h2h_avg": vs_pitcher["h2h_avg"],
        "arsenal_woba": format_pitch_woba(arsenal_woba),
        "batting_average": format_batting_average_column(
            row["player"],
            version,
        ),
        "avg_vs_rhp": format_batting_average_vs_hand_column(
            row["player"],
            version,
            hand="R",
        ),
        "avg_vs_lhp": format_batting_average_vs_hand_column(
            row["player"],
            version,
            hand="L",
        ),
        "sp_baa": format_opposing_sp_baa_column(
            row["player"],
            version,
            game_context,
        ),
        "xwoba": format_xwoba_column(
            row["player"],
            version,
        ),
        "wrc_plus": format_wrc_plus_column(
            row["player"],
            version,
        ),
        "pp_fantasy_line": format_prizepicks_fantasy_line(row["player"]),
        "ud_fantasy_line": format_underdog_fantasy_line(row["player"]),
        "_pp_line": pp_line,
        "_ud_line": ud_line,
        "batter_score_v2_display": format_batter_score_display(
            batter_score_v2,
            batter_score_v2_label,
        ),
        "batter_score_v3_display": format_batter_score_display(
            batter_score_v3,
            batter_score_v3_label,
        ),
        "pitch_woba": format_pitch_woba(woba),
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
