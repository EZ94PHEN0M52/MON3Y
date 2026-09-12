"""Lineup resolution for the Sleeper Prop Analyzer (read-only Rotowire / probables)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fetch_rotowire_lineups import ensure_rotowire_lineups, lineup_for_team
from hitters_life_data import match_player_to_lineup
from ui.player_stats import _fuzzy_player_key, _player_key, lookup_pitcher_hand
from utils import DAILY_PROBABLES_PATH, TEAM_ABBR_TO_ODDS


@dataclass(frozen=True)
class LineupPlayer:
    player: str
    role: str
    order: int | None
    position: str | None
    image_url: str | None


@dataclass(frozen=True)
class TeamLineupView:
    team_abbr: str
    team_name: str
    source: str
    source_label: str
    sp_name: str | None
    lineup_players: list[LineupPlayer]
    fallback_players: list[LineupPlayer]


def _abbr_to_odds_name(abbr: str) -> str | None:
    return TEAM_ABBR_TO_ODDS.get(str(abbr or "").strip().upper())


def _lookup_probable_sp(
    away_abbr: str,
    home_abbr: str,
    team_abbr: str,
    schedule_date: str | None,
) -> str | None:
    if not DAILY_PROBABLES_PATH.exists() or not schedule_date:
        return None

    probables = pd.read_parquet(DAILY_PROBABLES_PATH)
    if probables.empty:
        return None

    away_name = _abbr_to_odds_name(away_abbr)
    home_name = _abbr_to_odds_name(home_abbr)
    if not away_name or not home_name:
        return None

    subset = probables[
        probables["game_date"].astype(str).eq(str(schedule_date))
        & probables["away_team"].astype(str).eq(away_name)
        & probables["home_team"].astype(str).eq(home_name)
    ]
    if subset.empty:
        return None

    row = subset.iloc[0]
    if str(team_abbr).upper() == str(home_abbr).upper():
        sp = row.get("home_sp_name")
    else:
        sp = row.get("away_sp_name")

    if pd.isna(sp) or not str(sp).strip():
        return None
    return str(sp).strip()


def _opposing_sp_hand(
    away_abbr: str,
    home_abbr: str,
    team_abbr: str,
    schedule_date: str | None,
    version: str,
) -> str | None:
    sp_name = _lookup_probable_sp(away_abbr, home_abbr, team_abbr, schedule_date)
    if not sp_name:
        return None

    if str(team_abbr).upper() == str(home_abbr).upper():
        opposing_sp = _lookup_probable_sp(away_abbr, home_abbr, away_abbr, schedule_date)
    else:
        opposing_sp = _lookup_probable_sp(away_abbr, home_abbr, home_abbr, schedule_date)

    if not opposing_sp:
        return "R"

    hand = lookup_pitcher_hand(opposing_sp, version=version)
    if hand in ("L", "R"):
        return hand
    return "R"


def _source_label(source: str) -> str:
    if source == "official":
        return "Confirmed"
    if source == "default":
        return "Projected"
    return "Unavailable"


def _player_row(
    name: str,
    *,
    role: str,
    order: int | None,
    meta: pd.DataFrame,
) -> LineupPlayer:
    position = None
    image_url = None
    if not meta.empty:
        match = meta[meta["player"].astype(str).eq(name)]
        if match.empty:
            keys = meta["player"].astype(str).tolist()
            target = _fuzzy_player_key(
                name,
                {_player_key(p) for p in keys},
            )
            if target:
                match = meta[
                    meta["player"].map(
                        lambda p: _player_key(p) == target
                    )
                ]
        if not match.empty:
            position = match.iloc[0].get("player_position")
            image_url = match.iloc[0].get("player_image")

    pos_text = str(position).strip() if pd.notna(position) and str(position).strip() else None
    img = str(image_url).strip() if pd.notna(image_url) and str(image_url).strip() else None
    return LineupPlayer(
        player=name,
        role=role,
        order=order,
        position=pos_text,
        image_url=img,
    )


def build_team_lineup_view(
    *,
    team_abbr: str,
    team_name: str,
    away_abbr: str,
    home_abbr: str,
    schedule_date: str | None,
    team_props: pd.DataFrame,
    player_meta: pd.DataFrame,
    version: str,
) -> TeamLineupView:
    team_abbr = str(team_abbr or "").strip().upper()
    lineups_df = ensure_rotowire_lineups([away_abbr, home_abbr])
    sp_hand = _opposing_sp_hand(
        away_abbr,
        home_abbr,
        team_abbr,
        schedule_date,
        version,
    )
    ordered_names, source = lineup_for_team(
        lineups_df,
        team_abbr,
        sp_hand or "R",
    )

    sp_name = _lookup_probable_sp(away_abbr, home_abbr, team_abbr, schedule_date)

    props_players = sorted(team_props["player"].dropna().astype(str).unique())
    in_lineup = [name for name in ordered_names if name]
    fallback_names = [
        name for name in props_players
        if not match_player_to_lineup(name, in_lineup)
    ]

    lineup_players: list[LineupPlayer] = []
    if sp_name:
        lineup_players.append(
            _player_row(
                sp_name,
                role="SP",
                order=None,
                meta=player_meta,
            )
        )

    for idx, name in enumerate(in_lineup, start=1):
        lineup_players.append(
            _player_row(
                name,
                role="BAT",
                order=idx,
                meta=player_meta,
            )
        )

    fallback_players = [
        _player_row(
            name,
            role="EXTRA",
            order=None,
            meta=player_meta,
        )
        for name in sorted(fallback_names)
    ]

    return TeamLineupView(
        team_abbr=team_abbr,
        team_name=team_name,
        source=source,
        source_label=_source_label(source),
        sp_name=sp_name,
        lineup_players=lineup_players,
        fallback_players=fallback_players,
    )
