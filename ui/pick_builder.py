"""Session-state Pick Builder (favorites slip) for the MLB Prop Model UI."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ui.formatting import (
    format_batter_score_cell,
    format_commence_time,
    format_odds,
    format_pct,
    market_label,
)
from ui.glossary import GLOSSARY
from ui.batter_score_highlights import format_batter_score_pick_details_html
from ui.market_filters import EXCLUDED_UI_MARKETS
from ui.player_stats import BATTER_MARKETS

PICKS_STATE_KEY = "pick_builder_picks"


def pick_key(player, market, side, line, bookmaker) -> str:
    """Unique key for duplicate prevention: (player, market, side, line, book)."""
    line_val = round(float(line), 1) if pd.notna(line) else "na"
    return "|".join(
        (
            str(player).strip(),
            str(market).strip(),
            str(side).strip(),
            str(line_val),
            str(bookmaker).strip(),
        )
    )


def _init_picks():
    if PICKS_STATE_KEY not in st.session_state:
        st.session_state[PICKS_STATE_KEY] = {}


def get_picks() -> list[dict]:
    _init_picks()
    return [
        pick
        for pick in st.session_state[PICKS_STATE_KEY].values()
        if pick.get("market") not in EXCLUDED_UI_MARKETS
    ]


def pick_count() -> int:
    return len(get_picks())


def row_to_pick(row: pd.Series) -> dict:
    """Build a pick dict from a predictions dataframe row."""
    pick = {
        "key": pick_key(
            row["player"],
            row["market"],
            row["side"],
            row["line"],
            row["bookmaker"],
        ),
        "player": row["player"],
        "market": row["market"],
        "side": row["side"],
        "line": float(row["line"]) if pd.notna(row.get("line")) else None,
        "bookmaker": row["bookmaker"],
        "over_probability": row.get("over_probability"),
        "under_probability": row.get("under_probability"),
        "edge": row.get("edge"),
        "game": row.get("game"),
        "commence_time": row.get("commence_time"),
        "home_team": row.get("home_team"),
        "away_team": row.get("away_team"),
        "odds": row.get("odds"),
        "model_probability": row.get("model_probability"),
        "ev": row.get("ev"),
    }

    if row.get("market") in BATTER_MARKETS:
        pick["batter_score"] = row.get("batter_score")
        pick["batter_score_label"] = row.get("batter_score_label")

    return pick


def format_pick_option(row: pd.Series, *, include_player: bool = True) -> str:
    """Human-readable label for selectbox / multiselect options."""
    market = market_label(row["market"])
    odds = format_odds(row.get("odds"))
    edge = format_pct(row.get("edge"))
    ev = format_pct(row.get("ev"))
    line = f"{float(row['line']):.1f}" if pd.notna(row.get("line")) else "—"
    base = (
        f"{row['side']} {line} @ {row['bookmaker']} ({odds})"
        f" · Edge {edge} · EV {ev}"
    )
    if include_player:
        return f"{row['player']} · {market} · {base}"
    return f"{market} · {base}"


def add_pick(pick: dict) -> tuple[bool, str]:
    """Add a pick; returns (added, message). Skips duplicates."""
    _init_picks()
    key = pick.get("key") or pick_key(
        pick["player"],
        pick["market"],
        pick["side"],
        pick["line"],
        pick["bookmaker"],
    )
    pick = {**pick, "key": key}

    if key in st.session_state[PICKS_STATE_KEY]:
        return False, "Already on your slip."

    st.session_state[PICKS_STATE_KEY][key] = pick
    return True, "Added to Pick Builder."


def add_pick_from_row(row: pd.Series) -> tuple[bool, str]:
    return add_pick(row_to_pick(row))


def remove_pick(key: str) -> None:
    _init_picks()
    st.session_state[PICKS_STATE_KEY].pop(key, None)


def clear_picks() -> None:
    _init_picks()
    st.session_state[PICKS_STATE_KEY] = {}


def _format_pick_line(pick: dict) -> str:
    if pick.get("line") is not None and pd.notna(pick.get("line")):
        return f"{float(pick['line']):.1f}"
    return "—"


def _format_pick_game(pick: dict) -> str:
    game = pick.get("game")
    if (not game or pd.isna(game)) and pick.get("home_team") and pick.get("away_team"):
        game = f"{pick['away_team']} @ {pick['home_team']}"
    time_str = format_commence_time(pick.get("commence_time"))
    if game and pd.notna(game) and time_str != "—":
        return f"{game} · {time_str}"
    if game and pd.notna(game):
        return str(game)
    return time_str


def _format_pick_probabilities(pick: dict) -> str:
    over = pick.get("over_probability")
    under = pick.get("under_probability")
    side = str(pick.get("side", "")).strip().lower()
    over_fmt = format_pct(over)
    under_fmt = format_pct(under)

    has_over = over is not None and pd.notna(over)
    has_under = under is not None and pd.notna(under)

    if has_over and has_under:
        if side == "over":
            return f"**Over {over_fmt}** · Under {under_fmt}"
        if side == "under":
            return f"Over {over_fmt} · **Under {under_fmt}**"
        return f"Over {over_fmt} · Under {under_fmt}"

    if side == "over" and has_over:
        return f"Over {over_fmt}"
    if side == "under" and has_under:
        return f"Under {under_fmt}"
    if has_over:
        return f"Over {over_fmt}"
    if has_under:
        return f"Under {under_fmt}"
    return "—"


def _format_pick_batter_score(pick: dict) -> str | None:
    if pick.get("market") not in BATTER_MARKETS:
        return None
    label = pick.get("batter_score_label") or ""
    return format_batter_score_cell(pick.get("batter_score"), label)


def render_sidebar_pick_builder():
    """Always-visible sidebar panel listing current picks."""
    _init_picks()
    count = pick_count()
    title = ":material/bookmark: Pick Builder"
    if count:
        title += f" ({count})"

    st.sidebar.markdown(f"### {title}")
    with st.sidebar.popover(":material/help: What's this?"):
        st.markdown(GLOSSARY["pick_builder"])

    picks = get_picks()

    if not picks:
        st.sidebar.caption("No picks yet. Add props from the board or a player page.")
        return

    if st.sidebar.button(
        "Clear all picks",
        key="pick_builder_clear_all",
        width="stretch",
    ):
        clear_picks()
        st.rerun()

    for pick in picks:
        market = market_label(pick["market"])
        line = _format_pick_line(pick)
        edge = format_pct(pick.get("edge"))
        probs = _format_pick_probabilities(pick)
        batter_score = _format_pick_batter_score(pick)
        game = _format_pick_game(pick)

        with st.sidebar.container(border=True):
            st.markdown(f"**{pick['player']}**")
            st.caption(f"{market} · {pick['side']} {line}")
            st.markdown(probs)
            st.caption(f"Edge {edge}")
            if batter_score is not None:
                st.caption(f"Batter Score {batter_score}")
            st.caption(game)
            if st.button(
                "Remove",
                key=f"pick_builder_remove_{pick['key']}",
                width="stretch",
            ):
                remove_pick(pick["key"])
                st.rerun()


def _add_rows_by_index(df: pd.DataFrame, indices: list, *, include_player: bool):
    if not indices:
        st.warning("Select at least one prop to add.")
        return

    added = 0
    skipped = 0
    for idx in indices:
        row = df.loc[idx]
        success, _ = add_pick_from_row(row)
        if success:
            added += 1
        else:
            skipped += 1

    if added and skipped:
        st.success(f"Added {added} pick(s). {skipped} already on your slip.")
    elif added:
        st.success(f"Added {added} pick(s) to Pick Builder.")
    elif skipped:
        st.info("Selected props are already on your slip.")
    st.rerun()


def render_board_add_controls(filtered_df: pd.DataFrame, key_prefix: str):
    """Expandable multiselect + Add Selected for the main board table."""
    if filtered_df.empty:
        return

    with st.expander(":material/bookmark_add: Add to Pick Builder", expanded=False):
        st.caption(GLOSSARY["pick_builder_add"])

        option_indices = list(filtered_df.index)
        labels = {
            idx: format_pick_option(filtered_df.loc[idx], include_player=True)
            for idx in option_indices
        }

        selected = st.multiselect(
            "Props in current table",
            options=option_indices,
            format_func=lambda idx: labels[idx],
            key=f"{key_prefix}_pick_builder_multiselect",
            placeholder="Choose one or more props…",
        )

        add_row = st.container(horizontal=True)
        with add_row:
            if st.button(
                "Add selected",
                key=f"{key_prefix}_pick_builder_add_selected",
                type="primary",
            ):
                _add_rows_by_index(filtered_df, selected, include_player=True)

            if st.button(
                "Add top EV",
                key=f"{key_prefix}_pick_builder_add_top_ev",
                help="Add the highest-EV row currently visible.",
            ):
                top_idx = filtered_df["ev"].idxmax()
                _add_rows_by_index(filtered_df, [top_idx], include_player=True)


def render_player_market_add_controls(
    market_rows: pd.DataFrame,
    market: str,
    player_name: str,
):
    """Per-market selectbox + add buttons on the player page."""
    if market_rows.empty:
        return

    option_indices = list(market_rows.index)
    labels = {
        idx: format_pick_option(market_rows.loc[idx], include_player=False)
        for idx in option_indices
    }

    control_row = st.container(horizontal=True, vertical_alignment="bottom")
    with control_row:
        selected_idx = st.selectbox(
            "Add to Pick Builder",
            options=option_indices,
            format_func=lambda idx: labels[idx],
            key=f"player_pick_{player_name}_{market}",
            label_visibility="collapsed",
        )
        if st.button(
            "Add",
            key=f"player_pick_add_{player_name}_{market}",
            type="secondary",
        ):
            _add_rows_by_index(market_rows, [selected_idx], include_player=False)

        best_ev_idx = market_rows["ev"].idxmax()
        if st.button(
            "Add best EV",
            key=f"player_pick_best_ev_{player_name}_{market}",
            help="Add the highest-EV line in this market.",
        ):
            _add_rows_by_index(market_rows, [best_ev_idx], include_player=False)


# ---------------------------------------------------------------------------
# Batter Score Pick Builder (separate slip — DFS / batter score board rows)
# ---------------------------------------------------------------------------

BATTER_SCORE_PICKS_STATE_KEY = "batter_score_pick_builder_picks"


def batter_score_pick_key(player, game) -> str:
    """Unique key: (player, game) — one batter row per matchup on the slate."""
    game_text = str(game or "").strip() or "slate"
    return "|".join((str(player).strip(), game_text))


def _init_batter_score_picks():
    if BATTER_SCORE_PICKS_STATE_KEY not in st.session_state:
        st.session_state[BATTER_SCORE_PICKS_STATE_KEY] = {}


def get_batter_score_picks() -> list[dict]:
    _init_batter_score_picks()
    return list(st.session_state[BATTER_SCORE_PICKS_STATE_KEY].values())


def batter_score_pick_count() -> int:
    return len(get_batter_score_picks())


def batter_score_row_to_pick(row: pd.Series) -> dict:
    """Build a batter-score pick dict from a batter score board row."""
    player = row.get("player")
    game = row.get("_game") or ""
    pick = {
        "key": batter_score_pick_key(player, game),
        "player": player,
        "game": game,
        "game_time": row.get("game_time"),
        "opposing_sp": row.get("opposing_sp"),
        "vs_pitcher": row.get("vs_pitcher"),
        "pp_fantasy_line": row.get("pp_fantasy_line"),
        "ud_fantasy_line": row.get("ud_fantasy_line"),
        "l5_l10_pct": row.get("l5_l10_pct"),
        "pp_line": row.get("_pp_line"),
        "ud_line": row.get("_ud_line"),
        "l5_pct": row.get("_l5_pct"),
        "l10_pct": row.get("_l10_pct"),
        "batter_score_display": row.get("batter_score_display"),
        "batter_score": row.get("_batter_score"),
        "batter_score_label": row.get("batter_score_label") or "",
    }
    return pick


def format_batter_score_pick_option(row: pd.Series) -> str:
    """Human-readable label for batter score board multiselect."""
    score = row.get("batter_score_display") or "—"
    game_time = row.get("game_time") or "—"
    pp = row.get("pp_fantasy_line") or "—"
    ud = row.get("ud_fantasy_line") or "—"
    return (
        f"{row['player']} · BS {score} · {game_time} · "
        f"PP {pp} / UD {ud}"
    )


def add_batter_score_pick(pick: dict) -> tuple[bool, str]:
    _init_batter_score_picks()
    key = pick.get("key") or batter_score_pick_key(
        pick["player"],
        pick.get("game"),
    )
    pick = {**pick, "key": key}

    if key in st.session_state[BATTER_SCORE_PICKS_STATE_KEY]:
        return False, "Already on your batter score slip."

    st.session_state[BATTER_SCORE_PICKS_STATE_KEY][key] = pick
    return True, "Added to Batter Score Pick Builder."


def add_batter_score_pick_from_row(row: pd.Series) -> tuple[bool, str]:
    return add_batter_score_pick(batter_score_row_to_pick(row))


def remove_batter_score_pick(key: str) -> None:
    _init_batter_score_picks()
    st.session_state[BATTER_SCORE_PICKS_STATE_KEY].pop(key, None)


def clear_batter_score_picks() -> None:
    _init_batter_score_picks()
    st.session_state[BATTER_SCORE_PICKS_STATE_KEY] = {}


def _add_batter_score_rows_by_index(df: pd.DataFrame, indices: list):
    if not indices:
        st.warning("Select at least one batter to add.")
        return

    added = 0
    skipped = 0
    for idx in indices:
        row = df.loc[idx]
        success, _ = add_batter_score_pick_from_row(row)
        if success:
            added += 1
        else:
            skipped += 1

    if added and skipped:
        st.success(
            f"Added {added} batter(s). {skipped} already on your slip."
        )
    elif added:
        st.success(f"Added {added} batter(s) to Batter Score Pick Builder.")
    elif skipped:
        st.info("Selected batters are already on your slip.")
    st.rerun()


def render_sidebar_batter_score_pick_builder():
    """Sidebar panel for batter score board favorites."""
    _init_batter_score_picks()
    count = batter_score_pick_count()
    title = ":material/groups: Batter Score Picks"
    if count:
        title += f" ({count})"

    st.sidebar.markdown(f"### {title}")
    with st.sidebar.popover(":material/help: What's this?"):
        st.markdown(GLOSSARY["batter_score_pick_builder"])

    picks = get_batter_score_picks()

    if not picks:
        st.sidebar.caption(
            "No batters yet. Add from **Batter score by game** below the board."
        )
        return

    if st.sidebar.button(
        "Clear all batter picks",
        key="batter_score_pick_builder_clear_all",
        width="stretch",
    ):
        clear_batter_score_picks()
        st.rerun()

    for pick in picks:
        with st.sidebar.container(border=True):
            st.markdown(f"**{pick['player']}**")
            st.caption(pick.get("batter_score_display") or "—")
            st.caption(pick.get("game_time") or pick.get("game") or "—")
            st.caption(f"vs {pick.get('opposing_sp') or 'TBD'}")
            st.markdown(
                format_batter_score_pick_details_html(pick),
                unsafe_allow_html=True,
            )
            if st.button(
                "Remove",
                key=f"batter_score_pick_builder_remove_{pick['key']}",
                width="stretch",
            ):
                remove_batter_score_pick(pick["key"])
                st.rerun()


def render_batter_score_add_controls(filtered_df: pd.DataFrame, key_prefix: str):
    """Multiselect + add buttons for the batter score by game table."""
    if filtered_df.empty:
        return

    board_df = filtered_df.reset_index(drop=True)

    with st.expander(
        ":material/bookmark_add: Add to Batter Score Pick Builder",
        expanded=False,
    ):
        st.caption(GLOSSARY["batter_score_pick_builder_add"])

        option_indices = list(board_df.index)
        labels = {
            idx: format_batter_score_pick_option(board_df.loc[idx])
            for idx in option_indices
        }

        selected = st.multiselect(
            "Batters in current table",
            options=option_indices,
            format_func=lambda idx: labels[idx],
            key=f"{key_prefix}_batter_score_pick_multiselect",
            placeholder="Choose one or more batters…",
        )

        add_row = st.container(horizontal=True)
        with add_row:
            if st.button(
                "Add selected",
                key=f"{key_prefix}_batter_score_pick_add_selected",
                type="primary",
            ):
                _add_batter_score_rows_by_index(board_df, selected)

            if st.button(
                "Add top batter score",
                key=f"{key_prefix}_batter_score_pick_add_top",
                help="Add the highest Batter Score row currently visible.",
            ):
                scores = pd.to_numeric(
                    board_df["_batter_score"],
                    errors="coerce",
                )
                if scores.notna().any():
                    top_idx = scores.idxmax()
                    _add_batter_score_rows_by_index(board_df, [top_idx])
                else:
                    st.warning("No batter scores in the current table.")
