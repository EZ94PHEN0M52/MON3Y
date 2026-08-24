"""Hitter's Life batting-average board UI."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from fetch_rotowire_lineups import (
    ensure_rotowire_lineups,
    lineup_for_team_hand,
    odds_team_to_abbr,
)
from hitters_life_data import (
    build_hitters_life_df,
    lineup_sort_key,
    match_player_to_lineup,
)
from pitch_matchup import PITCH_BUCKETS
from ui.glossary import GLOSSARY
from ui.hitters_life_highlights import style_hitters_life_board
from ui.player_stats import lookup_pitcher_hand


def _parse_game_teams(game: str) -> tuple[str | None, str | None]:
    if not isinstance(game, str) or " @ " not in game:
        return None, None
    away, home = game.split(" @ ", 1)
    return away.strip(), home.strip()


def _sp_hand_for_team(
    game: str,
    commence_time,
    home_team: str,
    away_team: str,
    batter_team_full: str,
    version: str,
) -> str | None:
    from batter_score_data import build_game_context, _lookup_opposing_sp_for_context

    game_context = build_game_context(
        game=game,
        commence_time=commence_time,
        home_team=home_team,
        away_team=away_team,
    )
    if not game_context:
        return None

    sp_name, _ = _lookup_opposing_sp_for_context(
        game_context,
        batter_team_full,
    )
    if not sp_name:
        return None
    return lookup_pitcher_hand(sp_name, version=version)


def _lineup_context_for_game(
    game: str,
    sample_row,
    version: str,
) -> dict:
    away_team, home_team = _parse_game_teams(game)
    away_abbr = odds_team_to_abbr(away_team) if away_team else None
    home_abbr = odds_team_to_abbr(home_team) if home_team else None
    team_abbrs = [abbr for abbr in (away_abbr, home_abbr) if abbr]

    home_sp_hand = _sp_hand_for_team(
        game,
        sample_row.get("commence_time"),
        home_team,
        away_team,
        away_team,
        version,
    )
    away_sp_hand = _sp_hand_for_team(
        game,
        sample_row.get("commence_time"),
        home_team,
        away_team,
        home_team,
        version,
    )

    lineups_df = ensure_rotowire_lineups(team_abbrs)
    away_lineup = (
        lineup_for_team_hand(lineups_df, away_abbr, home_sp_hand or "R")
        if away_abbr
        else []
    )
    home_lineup = (
        lineup_for_team_hand(lineups_df, home_abbr, away_sp_hand or "R")
        if home_abbr
        else []
    )

    return {
        "away_abbr": away_abbr,
        "home_abbr": home_abbr,
        "away_lineup": away_lineup,
        "home_lineup": home_lineup,
        "home_sp_hand": home_sp_hand,
        "away_sp_hand": away_sp_hand,
    }


def _apply_lineup_filter(
    display_df: pd.DataFrame,
    context: dict,
) -> pd.DataFrame:
    away_abbr = context.get("away_abbr")
    home_abbr = context.get("home_abbr")
    away_lineup = context.get("away_lineup") or []
    home_lineup = context.get("home_lineup") or []

    if not away_lineup and not home_lineup:
        return display_df

    kept = []
    for _, row in display_df.iterrows():
        team_abbr = row.get("_team_abbr")
        player = row.get("player")
        if team_abbr == away_abbr and match_player_to_lineup(player, away_lineup):
            kept.append(row)
        elif team_abbr == home_abbr and match_player_to_lineup(player, home_lineup):
            kept.append(row)

    if not kept:
        return display_df.iloc[0:0].copy()

    filtered = pd.DataFrame(kept)

    def _sort_key(row):
        team_abbr = row.get("_team_abbr")
        if team_abbr == away_abbr:
            return (0, *lineup_sort_key(row["player"], away_lineup))
        if team_abbr == home_abbr:
            return (1, *lineup_sort_key(row["player"], home_lineup))
        return (2, 999, str(row.get("player", "")))

    filtered["_sort"] = filtered.apply(_sort_key, axis=1)
    filtered = (
        filtered.sort_values("_sort")
        .drop(columns=["_sort"])
        .reset_index(drop=True)
    )
    return filtered


def _hitters_life_column_config(pitch_bucket: str):
    return {
        "player_link": st.column_config.LinkColumn(
            "Player",
            help="Open player profile.",
            display_text=r"#(.*)$",
        ),
        "game_time": st.column_config.TextColumn(
            "Game & time",
            help="Matchup and scheduled first pitch (ET).",
        ),
        "vs_pitcher": st.column_config.TextColumn(
            "Vs pitcher",
            help=(
                "Probable opposing starter and career batting average vs that "
                "SP (hits/AB) when PA ≥ 3; otherwise SP ERA over last 5 starts. "
                "Light green when H2H average is above .300."
            ),
        ),
        "arsenal_woba": st.column_config.TextColumn(
            "Arsenal wOBA",
            help=(
                "Usage-weighted career wOBA vs the opposing SP's pitch mix "
                "(last 5 starts), by pitch bucket."
            ),
        ),
        "batting_average": st.column_config.TextColumn(
            "Batting average",
            help=(
                "Full-season AVG and rolling AVG over the last 5 and 10 games "
                "(Statcast). Light green when season AVG is above .300."
            ),
        ),
        "pitch_woba": st.column_config.TextColumn(
            f"wOBA vs {pitch_bucket}",
            help=(
                f"Career wOBA vs {pitch_bucket} pitches (Statcast balls in play). "
                "Change pitch type with the selector above the table."
            ),
        ),
        "sp_arsenal": st.column_config.TextColumn(
            "SP arsenal",
            help=(
                "Individual pitch types in the opposing starter's arsenal over "
                "their last 5 starts (Baseball Savant names from Statcast), "
                "sorted by usage."
            ),
        ),
        "total_bases_log": st.column_config.TextColumn(
            "TB per game",
            help=(
                "Last 5 games total bases, oldest to newest (space-separated). "
                "Light green when every game in the log is non-zero."
            ),
        ),
    }


def _render_hitters_life_dataframe(
    df: pd.DataFrame,
    *,
    pitch_bucket: str,
    height: int,
):
    st.dataframe(
        style_hitters_life_board(df),
        hide_index=True,
        height=height,
        column_config=_hitters_life_column_config(pitch_bucket),
    )


@st.cache_data(show_spinner=False)
def _cached_hitters_life_df(
    df: pd.DataFrame,
    version: str,
    pitch_bucket: str,
    markets: tuple[str, ...],
) -> pd.DataFrame:
    return build_hitters_life_df(
        df,
        version,
        pitch_bucket=pitch_bucket,
        markets=list(markets) or None,
    )


def render_hitters_life_board(
    df: pd.DataFrame,
    key_prefix: str,
    *,
    version: str = "v2",
):
    """Render the Hitter's Life batting board with game and lineup filters."""
    markets = st.session_state.get(f"{key_prefix}_markets", [])
    pitch_bucket = st.selectbox(
        "Pitch type wOBA",
        PITCH_BUCKETS,
        key=f"{key_prefix}_hitters_life_pitch_bucket",
        help="Show each batter's career wOBA vs the selected pitch bucket.",
    )

    all_df = _cached_hitters_life_df(
        df,
        version,
        pitch_bucket,
        tuple(markets or ()),
    )

    if all_df.empty:
        st.caption("No batters on today's slate for the current filters.")
        return

    games = sorted(
        game for game in all_df["_game"].dropna().unique() if str(game).strip()
    )
    selected_game = st.selectbox(
        "Game",
        ["All games", *games],
        key=f"{key_prefix}_hitters_life_game_filter",
        help="Filter batters to one matchup (away @ home).",
    )

    display_df = all_df
    if selected_game != "All games":
        display_df = all_df[all_df["_game"] == selected_game].copy()
        display_df = display_df.reset_index(drop=True)

    lineup_filter = False
    lineup_context = None
    if selected_game != "All games" and not display_df.empty:
        sample = df[
            df["game"].astype(str).eq(selected_game)
        ].iloc[0]
        lineup_context = _lineup_context_for_game(
            selected_game,
            sample,
            version,
        )
        with st.popover(":material/filter_list: Lineup filter"):
            st.markdown(GLOSSARY.get(
                "hitters_life_lineup_filter",
                "Order batters using Rotowire default lineups vs the "
                "opposing starter's hand.",
            ))
            away = lineup_context.get("away_abbr") or "—"
            home = lineup_context.get("home_abbr") or "—"
            st.caption(
                f"**Away ({away})** vs {lineup_context.get('home_sp_hand') or '?'} SP · "
                f"**Home ({home})** vs {lineup_context.get('away_sp_hand') or '?'} SP"
            )
            lineup_filter = st.toggle(
                "Use Rotowire lineup order",
                value=True,
                key=f"{key_prefix}_hitters_life_lineup_filter",
                help=(
                    "Keep batters in each team's default lineup vs the opposing "
                    "starter's throwing hand and sort 1–9."
                ),
            )

        if lineup_filter:
            filtered = _apply_lineup_filter(display_df, lineup_context)
            if filtered.empty:
                st.warning(
                    "No slate batters matched the Rotowire lineups for this game. "
                    "Showing all batters in the game instead."
                )
            else:
                display_df = filtered

    _render_hitters_life_dataframe(
        display_df,
        pitch_bucket=pitch_bucket,
        height=min(42 * len(display_df) + 38, 720),
    )
