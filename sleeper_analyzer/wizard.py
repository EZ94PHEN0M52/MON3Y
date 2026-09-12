"""Streamlit wizard for the Sleeper Prop Analyzer (Sleeper Picks page only)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from fetch_sleeper_props import SLEEPER_PROPS_PATH
from sleeper_analyzer.data import (
    available_schedule_dates,
    default_schedule_date,
    list_games,
    player_metadata,
    props_for_game,
    props_for_team,
)
from sleeper_analyzer.hit_rates import HIT_RATE_RULE, best_l5_props_by_player, build_player_prop_rows
from sleeper_analyzer.lineups import LineupPlayer, build_team_lineup_view
from sleeper_analyzer.styles import ANALYZER_SHELL_KEY, inject_analyzer_styles
from ui.formatting import format_commence_time


def _state_prefix(version: str) -> str:
    return f"sa_{version}"


def _init_state(prefix: str, parquet_mtime: float) -> None:
    defaults = {
        f"{prefix}_step": "matchups",
        f"{prefix}_date": default_schedule_date(parquet_mtime),
        f"{prefix}_game_key": None,
        f"{prefix}_team": None,
        f"{prefix}_player": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _set_step(prefix: str, step: str) -> None:
    st.session_state[f"{prefix}_step"] = step


def _reset_from_matchups(prefix: str) -> None:
    st.session_state[f"{prefix}_game_key"] = None
    st.session_state[f"{prefix}_team"] = None
    st.session_state[f"{prefix}_player"] = None
    _set_step(prefix, "matchups")


def _reset_from_lineup(prefix: str) -> None:
    st.session_state[f"{prefix}_player"] = None
    _set_step(prefix, "lineup")


def _pct_bar_label(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value * 100:.0f}%"


def _render_context_bar(
    *,
    game_label: str,
    game_time: str,
    team_name: str | None = None,
    player_name: str | None = None,
) -> None:
    parts = [game_label, game_time]
    if team_name:
        parts.append(team_name)
    if player_name:
        parts.append(player_name)
    line = " · ".join(p for p in parts if p)
    st.markdown(f'<p class="sa-context-line">{line}</p>', unsafe_allow_html=True)


def _format_player_label(player: LineupPlayer) -> str:
    if player.role == "SP":
        return f"SP - {player.player}"
    order_text = f"#{player.order} - " if player.order else ""
    pos_text = f" ({player.position})" if player.position else ""
    return f"{order_text}{player.player}{pos_text}"


def _format_top_l5_prop(prop: dict | None) -> str:
    if not prop:
        return "—"
    stat = str(prop.get("stat_display") or "—")
    line = prop.get("line")
    l5 = prop.get("l5_pct")
    line_text = f"{float(line):.1f}" if line is not None and pd.notna(line) else "—"
    l5_text = _pct_bar_label(l5)
    return f"{stat} O{line_text} · L5 {l5_text}"


def _render_player_button(
    player: LineupPlayer,
    *,
    prefix: str,
    key_suffix: str,
    top_l5_prop: dict | None = None,
) -> None:
    with st.container(border=True, key=f"sleeper_analyzer_player_{key_suffix}"):
        col_name, col_prop, col_go = st.columns(
            [6, 3, 1],
            vertical_alignment="center",
        )
        with col_name:
            st.markdown(
                f'<p class="sa-player-name">{_format_player_label(player)}</p>',
                unsafe_allow_html=True,
            )
        with col_prop:
            st.markdown(
                f'<p class="sa-player-prop">{_format_top_l5_prop(top_l5_prop)}</p>',
                unsafe_allow_html=True,
            )
        with col_go:
            if st.button("Open", key=f"{prefix}_pick_{key_suffix}"):
                st.session_state[f"{prefix}_player"] = player.player
                _set_step(prefix, "player")
                st.rerun()


def _render_matchups(prefix: str, parquet_mtime: float) -> None:
    dates = available_schedule_dates(parquet_mtime)
    if not dates:
        st.info("No Sleeper games in the cache. Run `python fetch_data.py --sleeper-props`.")
        return

    date_key = f"{prefix}_date"
    if st.session_state.get(date_key) not in dates:
        st.session_state[date_key] = default_schedule_date(parquet_mtime)

    picked_date = st.date_input(
        "Date",
        value=pd.to_datetime(st.session_state[date_key]).date(),
        min_value=pd.to_datetime(dates[0]).date(),
        max_value=pd.to_datetime(dates[-1]).date(),
        key=f"{prefix}_date_picker",
        format="YYYY-MM-DD",
    )
    st.session_state[date_key] = picked_date.strftime("%Y-%m-%d")
    selected_date = st.session_state[date_key]
    games = list_games(parquet_mtime)
    day_games = games[games["schedule_date"].astype(str).eq(selected_date)]

    if day_games.empty:
        st.info(f"No Sleeper games for {selected_date}.")
        return

    st.markdown(
        f'<p class="sa-body-text">{len(day_games)} game(s) on {selected_date}</p>',
        unsafe_allow_html=True,
    )

    for _, game in day_games.iterrows():
        game_key_safe = str(game["game_key"]).replace("@", "_").replace(":", "_")
        with st.container(border=True, key=f"sleeper_analyzer_game_{game_key_safe}"):
            header = st.columns([3, 2])
            with header[0]:
                st.markdown(
                    f'<p class="sa-game-title">{game["game"]}</p>',
                    unsafe_allow_html=True,
                )
                venue_bits = [
                    str(game.get("venue_name") or "").strip(),
                    str(game.get("game_status") or "").strip().replace("_", " "),
                ]
                venue_bits = [bit for bit in venue_bits if bit]
                if venue_bits:
                    st.markdown(
                        f'<p class="sa-caption">{" · ".join(venue_bits)}</p>',
                        unsafe_allow_html=True,
                    )
            with header[1]:
                st.markdown(
                    f'<p class="sa-caption">{format_commence_time(game.get("game_start"))}</p>',
                    unsafe_allow_html=True,
                )

            btn_cols = st.columns(2)
            away = str(game["away_team"])
            home = str(game["home_team"])
            with btn_cols[0]:
                if st.button(
                    game["away_team_name"],
                    key=f"{prefix}_away_{game['game_key']}",
                ):
                    st.session_state[f"{prefix}_game_key"] = game["game_key"]
                    st.session_state[f"{prefix}_team"] = away
                    st.session_state[f"{prefix}_player"] = None
                    _set_step(prefix, "lineup")
                    st.rerun()
            with btn_cols[1]:
                if st.button(
                    game["home_team_name"],
                    key=f"{prefix}_home_{game['game_key']}",
                ):
                    st.session_state[f"{prefix}_game_key"] = game["game_key"]
                    st.session_state[f"{prefix}_team"] = home
                    st.session_state[f"{prefix}_player"] = None
                    _set_step(prefix, "lineup")
                    st.rerun()


def _render_lineup(prefix: str, parquet_mtime: float, version: str) -> None:
    game_key = st.session_state.get(f"{prefix}_game_key")
    team_abbr = st.session_state.get(f"{prefix}_team")
    if not game_key or not team_abbr:
        _reset_from_matchups(prefix)
        st.rerun()
        return

    game_props = props_for_game(parquet_mtime, game_key)
    if game_props.empty:
        st.warning("Game props missing from cache.")
        if st.button("Back to matchups", key=f"{prefix}_back_matchups_empty"):
            _reset_from_matchups(prefix)
            st.rerun()
        return

    sample = game_props.iloc[0]
    from sleeper_analyzer.data import team_display_name

    team_name = team_display_name(str(team_abbr))

    _render_context_bar(
        game_label=str(sample.get("game") or ""),
        game_time=format_commence_time(sample.get("game_start")),
        team_name=str(team_name),
    )

    with st.container(key="sleeper_analyzer_back_matchups"):
        if st.button("← Back to matchups", key=f"{prefix}_back_matchups"):
            _reset_from_matchups(prefix)
            st.rerun()

    team_props = props_for_team(game_props, str(team_abbr))
    meta = player_metadata(game_props)
    from utils import game_date_from_commence

    lineup_view = build_team_lineup_view(
        team_abbr=str(team_abbr),
        team_name=str(team_name),
        away_abbr=str(sample.get("away_team") or ""),
        home_abbr=str(sample.get("home_team") or ""),
        schedule_date=game_date_from_commence(sample.get("game_start")),
        team_props=team_props,
        player_meta=meta,
        version=version,
    )

    st.markdown(
        f'<p class="sa-page-title">{lineup_view.team_name} lineup</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p class="sa-page-subtitle">{lineup_view.source_label} batting order</p>',
        unsafe_allow_html=True,
    )

    top_l5_by_player = best_l5_props_by_player(team_props, version=version)

    if not lineup_view.lineup_players:
        st.warning(
            "No lineup available yet. Run `./run_official_lineups.sh` pre-game "
            "for confirmed orders, or use the fallback list below."
        )
    else:
        for idx, player in enumerate(lineup_view.lineup_players):
            if player.role == "EXTRA":
                continue
            _render_player_button(
                player,
                prefix=prefix,
                key_suffix=f"lineup_{idx}",
                top_l5_prop=top_l5_by_player.get(player.player),
            )

    if lineup_view.fallback_players:
        st.divider()
        st.markdown(
            '<p class="sa-section-title">More props (not in lineup)</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="sa-page-subtitle">'
            "Sleeper posted props for these players outside the current lineup."
            "</p>",
            unsafe_allow_html=True,
        )
        for idx, player in enumerate(lineup_view.fallback_players):
            _render_player_button(
                player,
                prefix=prefix,
                key_suffix=f"fallback_{idx}",
                top_l5_prop=top_l5_by_player.get(player.player),
            )


def _render_player_props(prefix: str, parquet_mtime: float, version: str) -> None:
    game_key = st.session_state.get(f"{prefix}_game_key")
    team_abbr = st.session_state.get(f"{prefix}_team")
    player_name = st.session_state.get(f"{prefix}_player")
    if not game_key or not team_abbr or not player_name:
        _reset_from_lineup(prefix)
        st.rerun()
        return

    game_props = props_for_game(parquet_mtime, game_key)
    team_props = props_for_team(game_props, str(team_abbr))
    player_props = team_props[
        team_props["player"].astype(str).eq(str(player_name))
    ]
    if player_props.empty:
        st.warning("No Sleeper props found for this player.")
        if st.button("← Back to lineup", key=f"{prefix}_back_lineup_missing"):
            _reset_from_lineup(prefix)
            st.rerun()
        return

    sample = game_props.iloc[0]
    meta = player_metadata(player_props)
    position = None
    image_url = None
    if not meta.empty:
        position = meta.iloc[0].get("player_position")
        image_url = meta.iloc[0].get("player_image")

    _render_context_bar(
        game_label=str(sample.get("game") or ""),
        game_time=format_commence_time(sample.get("game_start")),
        team_name=str(team_abbr),
        player_name=str(player_name),
    )

    nav = st.columns([2, 8])
    with nav[0]:
        with st.container(key="sleeper_analyzer_back_lineup"):
            if st.button("← Back to lineup", key=f"{prefix}_back_lineup"):
                _reset_from_lineup(prefix)
                st.rerun()

    hero = st.columns([1, 4])
    with hero[0]:
        if image_url and str(image_url).strip():
            st.image(str(image_url), width=72)
    with hero[1]:
        pos_text = f" ({position})" if pd.notna(position) and str(position).strip() else ""
        st.markdown(
            f'<p class="sa-page-title">{player_name}{pos_text}</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<p class="sa-page-subtitle">{HIT_RATE_RULE}</p>',
            unsafe_allow_html=True,
        )

    table = build_player_prop_rows(player_props, version=version)
    if table.empty:
        st.info("No prop rows for this player.")
        return

    header = st.columns([3, 1, 1, 1, 2, 2])
    headers = ["Stat", "Line", "L5", "L10", "Over mult", "Under mult"]
    for col, label in zip(header, headers):
        with col:
            st.markdown(f'<p class="sa-table-header">{label}</p>', unsafe_allow_html=True)

    for idx, row in table.iterrows():
        with st.container(border=True, key=f"sleeper_analyzer_prop_{idx}"):
            cols = st.columns([3, 1, 1, 1, 2, 2])
            with cols[0]:
                st.write(str(row.get("stat_display") or "—"))
            with cols[1]:
                st.write(f"{float(row['line']):.1f}" if pd.notna(row.get("line")) else "—")
            with cols[2]:
                l5 = row.get("l5_pct")
                st.write(_pct_bar_label(l5))
                if l5 is not None and not pd.isna(l5):
                    st.progress(float(l5))
            with cols[3]:
                l10 = row.get("l10_pct")
                st.write(_pct_bar_label(l10))
                if l10 is not None and not pd.isna(l10):
                    st.progress(float(l10))
            with cols[4]:
                om = row.get("over_multiplier")
                st.write(f"{float(om):.2f}x" if pd.notna(om) else "—")
            with cols[5]:
                um = row.get("under_multiplier")
                st.write(f"{float(um):.2f}x" if pd.notna(um) else "—")


def render_sleeper_prop_analyzer(version: str, parquet_mtime: float) -> None:
    prefix = _state_prefix(version)
    _init_state(prefix, parquet_mtime)

    st.divider()

    with st.container(border=False, key=ANALYZER_SHELL_KEY):
        inject_analyzer_styles()
        st.markdown(
            '<p class="sa-flow-caption">'
            "Matchup → lineup → player props with Sleeper lines and L5/L10 hit rates."
            "</p>",
            unsafe_allow_html=True,
        )

        step = st.session_state.get(f"{prefix}_step", "matchups")
        if step == "matchups":
            st.markdown(
                '<p class="sa-page-title">Sleeper analyzer</p>',
                unsafe_allow_html=True,
            )
            _render_matchups(prefix, parquet_mtime)
        elif step == "lineup":
            _render_lineup(prefix, parquet_mtime, version)
        elif step == "player":
            _render_player_props(prefix, parquet_mtime, version)
        else:
            _reset_from_matchups(prefix)
            st.rerun()
