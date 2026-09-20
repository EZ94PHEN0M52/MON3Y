"""
Opportunity layer for Batter Score: batting order, expected PA, and volume.

Batter Score rates *quality when a player is in the lineup*. Platoon / low-TB-per-week
bats still need a separate opportunity read so volume isn't confused with talent.

Uses Rotowire 1–9 order (official Today's Lineup when cached, else default vs SP hand)
plus a short lookback of games played for start-rate / TB-per-week context.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Sequence

import pandas as pd

# Approximate MLB expected PA by batting-order slot (full game).
EXPECTED_PA_BY_SLOT: dict[int, float] = {
    1: 4.55,
    2: 4.45,
    3: 4.35,
    4: 4.25,
    5: 4.15,
    6: 4.05,
    7: 3.95,
    8: 3.85,
    9: 3.75,
}
LEAGUE_AVG_EXPECTED_PA = 4.15  # roughly slot 5
TEAM_GAMES_L14_DEFAULT = 12.0  # ~6/week × 2 weeks
LOW_VOLUME_START_RATE = 0.55
LOW_VOLUME_TB_PER_WEEK = 6.0
VOLUME_LOOKBACK_DAYS = 14


@dataclass(frozen=True)
class OpportunitySnapshot:
    batting_order: Optional[int]
    expected_pa: Optional[float]
    lineup_source: str  # official | default | none
    games_l14: int
    tb_l14: float
    tb_per_week: float
    start_rate_l14: float
    is_low_volume: bool
    is_platoon_candidate: bool
    opportunity_index: float  # 0–100 tonight's PA opportunity
    volume_label: str  # Full | Platoon | Bench | —
    order_display: str
    volume_display: str
    opportunity_display: str


def expected_pa_for_slot(slot: Optional[int]) -> Optional[float]:
    if slot is None:
        return None
    try:
        key = int(slot)
    except (TypeError, ValueError):
        return None
    return EXPECTED_PA_BY_SLOT.get(key)


def batting_order_from_lineup(
    player_name: str,
    lineup_names: Sequence[str],
) -> Optional[int]:
    """1–9 slot from an ordered Rotowire lineup list, or None if not matched."""
    if not lineup_names or not player_name:
        return None

    from hitters_life_data import lineup_sort_key

    slot, _ = lineup_sort_key(player_name, list(lineup_names))
    if slot >= 999:
        return None
    return int(slot)


def in_lineup(player_name: str, lineup_names: Sequence[str]) -> bool:
    if not lineup_names or not player_name:
        return False
    from hitters_life_data import match_player_to_lineup

    return match_player_to_lineup(player_name, list(lineup_names))


def opportunity_index_from_pa(expected_pa: Optional[float]) -> float:
    """Map expected PA to 0–100 (slot 1 ≈ 100, slot 9 ≈ ~82)."""
    if expected_pa is None:
        return 0.0
    top = EXPECTED_PA_BY_SLOT[1]
    bottom = EXPECTED_PA_BY_SLOT[9]
    if top <= bottom:
        return 0.0
    scaled = (float(expected_pa) - bottom) / (top - bottom)
    return max(0.0, min(100.0, scaled * 100.0))


def volume_adjusted_score(
    quality_score: Optional[float],
    expected_pa: Optional[float],
) -> Optional[float]:
    """
    Scale a quality Batter Score by tonight's expected PA vs league-average PA.

    Platoon / lower-order bats keep their rate score but show a lower volume-adjusted
    figure so weekly TB opportunity is visible without rewriting the composite.
    """
    if quality_score is None or expected_pa is None:
        return None
    return float(quality_score) * (float(expected_pa) / LEAGUE_AVG_EXPECTED_PA)


def _parse_game_dates(game_dates) -> list[pd.Timestamp]:
    if game_dates is None:
        return []
    series = pd.to_datetime(pd.Series(list(game_dates)), errors="coerce")
    return [ts for ts in series if not pd.isna(ts)]


def volume_from_game_log(
    game_dates,
    total_bases,
    *,
    as_of: Optional[datetime] = None,
    lookback_days: int = VOLUME_LOOKBACK_DAYS,
    team_games_in_window: float = TEAM_GAMES_L14_DEFAULT,
) -> tuple[int, float, float, float]:
    """
    Return (games_l14, tb_l14, tb_per_week, start_rate_l14).

    *game_dates* / *total_bases* are parallel sequences (one entry per game played).
    """
    dates = _parse_game_dates(game_dates)
    tbs = list(total_bases) if total_bases is not None else []
    if len(tbs) < len(dates):
        tbs = tbs + [0.0] * (len(dates) - len(tbs))
    elif len(tbs) > len(dates):
        tbs = tbs[: len(dates)]

    if as_of is None:
        as_of = datetime.utcnow()
    as_of_ts = pd.Timestamp(as_of)
    cutoff = as_of_ts - timedelta(days=lookback_days)

    games = 0
    tb_sum = 0.0
    for date, tb in zip(dates, tbs):
        if date < cutoff or date > as_of_ts:
            continue
        games += 1
        try:
            tb_sum += float(tb or 0)
        except (TypeError, ValueError):
            pass

    weeks = max(lookback_days / 7.0, 1e-6)
    tb_per_week = tb_sum / weeks
    start_rate = games / max(float(team_games_in_window), 1.0)
    start_rate = max(0.0, min(1.0, start_rate))
    return games, tb_sum, tb_per_week, start_rate


def classify_volume(
    *,
    start_rate_l14: float,
    tb_per_week: float,
    is_platoon_candidate: bool,
    batting_order: Optional[int],
) -> tuple[str, bool]:
    """Return (volume_label, is_low_volume)."""
    low = (
        start_rate_l14 < LOW_VOLUME_START_RATE
        or tb_per_week < LOW_VOLUME_TB_PER_WEEK
        or is_platoon_candidate
    )
    if batting_order is None and start_rate_l14 <= 0:
        return "—", True
    if is_platoon_candidate or (
        start_rate_l14 < LOW_VOLUME_START_RATE and start_rate_l14 > 0
    ):
        return "Platoon", True
    if start_rate_l14 < 0.35:
        return "Bench", True
    if low:
        return "Low vol", True
    return "Full", False


def build_opportunity_snapshot(
    player_name: str,
    *,
    lineup_names: Sequence[str] | None = None,
    lineup_source: str = "none",
    opposite_lineup_names: Sequence[str] | None = None,
    game_dates=None,
    total_bases=None,
    as_of: Optional[datetime] = None,
) -> OpportunitySnapshot:
    lineup = list(lineup_names or [])
    opposite = list(opposite_lineup_names or [])
    slot = batting_order_from_lineup(player_name, lineup) if lineup else None
    exp_pa = expected_pa_for_slot(slot)

    games_l14, tb_l14, tb_per_week, start_rate = volume_from_game_log(
        game_dates,
        total_bases,
        as_of=as_of,
    )

    platoon = False
    if lineup and opposite:
        in_today = in_lineup(player_name, lineup)
        in_other = in_lineup(player_name, opposite)
        # In one hand's default order but not the other → classic platoon.
        if in_today != in_other:
            platoon = True
    if start_rate < LOW_VOLUME_START_RATE and start_rate > 0:
        platoon = True

    volume_label, is_low_volume = classify_volume(
        start_rate_l14=start_rate,
        tb_per_week=tb_per_week,
        is_platoon_candidate=platoon,
        batting_order=slot,
    )

    opp_idx = opportunity_index_from_pa(exp_pa)
    order_display = str(slot) if slot is not None else "—"
    if exp_pa is not None:
        order_display = f"{slot} ({exp_pa:.2f} PA)"

    volume_display = (
        f"{volume_label} · {games_l14}g/{VOLUME_LOOKBACK_DAYS}d · "
        f"{tb_per_week:.1f} TB/wk"
    )
    opportunity_display = f"{opp_idx:.0f}" if exp_pa is not None else "—"

    return OpportunitySnapshot(
        batting_order=slot,
        expected_pa=exp_pa,
        lineup_source=lineup_source or "none",
        games_l14=games_l14,
        tb_l14=tb_l14,
        tb_per_week=tb_per_week,
        start_rate_l14=start_rate,
        is_low_volume=is_low_volume,
        is_platoon_candidate=platoon,
        opportunity_index=opp_idx,
        volume_label=volume_label,
        order_display=order_display,
        volume_display=volume_display,
        opportunity_display=opportunity_display,
    )


def empty_opportunity_snapshot() -> OpportunitySnapshot:
    return OpportunitySnapshot(
        batting_order=None,
        expected_pa=None,
        lineup_source="none",
        games_l14=0,
        tb_l14=0.0,
        tb_per_week=0.0,
        start_rate_l14=0.0,
        is_low_volume=True,
        is_platoon_candidate=False,
        opportunity_index=0.0,
        volume_label="—",
        order_display="—",
        volume_display="—",
        opportunity_display="—",
    )


def player_volume_series(
    player_name: str,
    version: str = "v2",
) -> tuple[list, list]:
    """Return (game_dates, total_bases) for volume lookback."""
    from batter_score_data import _batter_rows

    rows = _batter_rows(player_name, version=version)
    if rows is None or rows.empty or "game_date" not in rows.columns:
        return [], []

    working = rows.sort_values("game_date")
    dates = working["game_date"].tolist()
    if "total_bases" in working.columns:
        tbs = (
            pd.to_numeric(working["total_bases"], errors="coerce")
            .fillna(0)
            .tolist()
        )
    else:
        tbs = [0.0] * len(dates)
    return dates, tbs


def snapshot_for_player(
    player_name: str,
    *,
    version: str = "v2",
    lineup_names: Sequence[str] | None = None,
    lineup_source: str = "none",
    opposite_lineup_names: Sequence[str] | None = None,
) -> OpportunitySnapshot:
    dates, tbs = player_volume_series(player_name, version=version)
    return build_opportunity_snapshot(
        player_name,
        lineup_names=lineup_names,
        lineup_source=lineup_source,
        opposite_lineup_names=opposite_lineup_names,
        game_dates=dates,
        total_bases=tbs,
    )


def resolve_game_lineup_context(
    game: str | None,
    *,
    home_team: str | None = None,
    away_team: str | None = None,
    commence_time=None,
    version: str = "v2",
) -> dict:
    """
    Build away/home lineups + opposite-hand defaults for platoon detection.

    Mirrors Hitter's Life lineup resolution (official → default vs SP hand).
    """
    from batter_score_data import (
        build_game_context,
        _lookup_opposing_sp_for_context,
    )
    from fetch_rotowire_lineups import (
        ensure_rotowire_lineups,
        lineup_for_team,
        lineup_for_team_hand,
        odds_team_to_abbr,
    )
    from ui.player_stats import lookup_pitcher_hand

    away = away_team
    home = home_team
    if (not away or not home) and isinstance(game, str) and " @ " in game:
        parts = game.split(" @ ", 1)
        away = away or parts[0].strip()
        home = home or parts[1].strip()

    away_abbr = odds_team_to_abbr(away) if away else None
    home_abbr = odds_team_to_abbr(home) if home else None
    team_abbrs = [a for a in (away_abbr, home_abbr) if a]

    empty = {
        "away_abbr": away_abbr,
        "home_abbr": home_abbr,
        "away_lineup": [],
        "home_lineup": [],
        "away_opposite": [],
        "home_opposite": [],
        "away_lineup_source": "none",
        "home_lineup_source": "none",
    }
    if not team_abbrs:
        return empty

    lineups_df = ensure_rotowire_lineups(team_abbrs)

    def _sp_hand_for_batter_team(batter_team_full: str | None) -> str | None:
        if not batter_team_full or not home or not away:
            return None
        ctx = build_game_context(
            game=game,
            commence_time=commence_time,
            home_team=home,
            away_team=away,
        )
        if not ctx:
            return None
        sp_name, _ = _lookup_opposing_sp_for_context(ctx, batter_team_full)
        if not sp_name:
            return None
        return lookup_pitcher_hand(sp_name, version=version)

    # Away batters face home SP; home batters face away SP.
    home_sp_hand = _sp_hand_for_batter_team(away) or "R"
    away_sp_hand = _sp_hand_for_batter_team(home) or "R"

    away_lineup, away_source = (
        lineup_for_team(lineups_df, away_abbr, home_sp_hand)
        if away_abbr
        else ([], "none")
    )
    home_lineup, home_source = (
        lineup_for_team(lineups_df, home_abbr, away_sp_hand)
        if home_abbr
        else ([], "none")
    )

    # Opposite-hand default for platoon flag (ignore official for this check).
    opp_home_sp = "L" if str(home_sp_hand).upper().startswith("R") else "R"
    opp_away_sp = "L" if str(away_sp_hand).upper().startswith("R") else "R"
    away_opposite = (
        lineup_for_team_hand(lineups_df, away_abbr, opp_home_sp)
        if away_abbr
        else []
    )
    home_opposite = (
        lineup_for_team_hand(lineups_df, home_abbr, opp_away_sp)
        if home_abbr
        else []
    )

    return {
        "away_abbr": away_abbr,
        "home_abbr": home_abbr,
        "away_lineup": away_lineup,
        "home_lineup": home_lineup,
        "away_opposite": away_opposite,
        "home_opposite": home_opposite,
        "away_lineup_source": away_source,
        "home_lineup_source": home_source,
    }


def opportunity_for_player_in_game(
    player_name: str,
    team_abbr: str | None,
    lineup_context: dict,
    *,
    version: str = "v2",
) -> OpportunitySnapshot:
    if not team_abbr or not lineup_context:
        return snapshot_for_player(player_name, version=version)

    away_abbr = lineup_context.get("away_abbr")
    home_abbr = lineup_context.get("home_abbr")
    if team_abbr == away_abbr:
        return snapshot_for_player(
            player_name,
            version=version,
            lineup_names=lineup_context.get("away_lineup") or [],
            lineup_source=lineup_context.get("away_lineup_source") or "none",
            opposite_lineup_names=lineup_context.get("away_opposite") or [],
        )
    if team_abbr == home_abbr:
        return snapshot_for_player(
            player_name,
            version=version,
            lineup_names=lineup_context.get("home_lineup") or [],
            lineup_source=lineup_context.get("home_lineup_source") or "none",
            opposite_lineup_names=lineup_context.get("home_opposite") or [],
        )
    return snapshot_for_player(player_name, version=version)
