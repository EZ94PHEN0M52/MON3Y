"""
Fetch default Rotowire batting orders (vs RHP / vs LHP) per team.

Source: https://www.rotowire.com/baseball/batting-orders.php?team={ABBR}
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd
import requests

from utils import (
    PROCESSED_DIR,
    TEAM_ABBR_TO_ODDS,
    canonical_odds_team_key,
    require_live_fetch,
)

ROTOWIRE_LINEUPS_PATH = PROCESSED_DIR / "rotowire_lineups.parquet"
ROTOWIRE_BASE_URL = (
    "https://www.rotowire.com/baseball/batting-orders.php"
)
ROTOWIRE_HANDS = ("RHP", "LHP")
_LINEUP_BLOCK_RE = re.compile(
    r"Default vs\. (RHP|LHP)</div>\s*"
    r'<ol class="list is-rankings[^"]*">(.*?)</ol>',
    re.DOTALL | re.IGNORECASE,
)
_PLAYER_NAME_RE = re.compile(
    r'/baseball/player/[^"]+">([^<]+)</a>',
)


def odds_team_to_abbr(team_name: str) -> Optional[str]:
    """Map Odds API / feature team name to Rotowire team code."""
    if not isinstance(team_name, str) or not team_name.strip():
        return None

    target = canonical_odds_team_key(team_name)
    if not target:
        return None

    for abbr, odds_name in TEAM_ABBR_TO_ODDS.items():
        if canonical_odds_team_key(odds_name) == target:
            return abbr

    return None


def parse_rotowire_lineups_html(html: str) -> Dict[str, List[str]]:
    """
    Parse Rotowire team page HTML into ``{"RHP": [...], "LHP": [...]}`` lists.
    """
    lineups: Dict[str, List[str]] = {hand: [] for hand in ROTOWIRE_HANDS}

    if not html:
        return lineups

    for match in _LINEUP_BLOCK_RE.finditer(html):
        hand = match.group(1).upper()
        names = [
            name.strip()
            for name in _PLAYER_NAME_RE.findall(match.group(2))
            if name.strip()
        ]
        lineups[hand] = names

    return lineups


def _fetch_team_html(team_abbr: str) -> str:
    require_live_fetch(f"Rotowire lineups for {team_abbr}")
    response = requests.get(
        ROTOWIRE_BASE_URL,
        params={"team": team_abbr.upper()},
        headers={"User-Agent": "mlb-prop-model/1.0"},
        timeout=30,
    )
    response.raise_for_status()
    return response.text


def fetch_team_lineups(
    team_abbr: str,
    *,
    html: str | None = None,
) -> Dict[str, List[str]]:
    """Return default vs RHP / vs LHP batting orders for one team."""
    page = html if html is not None else _fetch_team_html(team_abbr)
    return parse_rotowire_lineups_html(page)


def lineups_to_rows(
    team_abbr: str,
    lineups: Dict[str, List[str]],
    *,
    fetched_at: str | None = None,
) -> list[dict]:
    fetched_at = fetched_at or (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )
    rows: list[dict] = []

    for hand in ROTOWIRE_HANDS:
        for slot, player_name in enumerate(lineups.get(hand, []), start=1):
            rows.append(
                {
                    "team_abbr": team_abbr.upper(),
                    "vs_hand": hand,
                    "slot": slot,
                    "player_name": player_name,
                    "fetched_at": fetched_at,
                }
            )

    return rows


def fetch_lineups_for_teams(team_abbrs: list[str]) -> pd.DataFrame:
    """Fetch and combine Rotowire lineups for multiple team abbreviations."""
    unique = sorted({abbr.upper() for abbr in team_abbrs if abbr})
    if not unique:
        return pd.DataFrame(
            columns=[
                "team_abbr",
                "vs_hand",
                "slot",
                "player_name",
                "fetched_at",
            ]
        )

    fetched_at = (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )
    rows: list[dict] = []

    for team_abbr in unique:
        lineups = fetch_team_lineups(team_abbr)
        rows.extend(
            lineups_to_rows(
                team_abbr,
                lineups,
                fetched_at=fetched_at,
            )
        )

    return pd.DataFrame(rows)


def load_rotowire_lineups() -> pd.DataFrame:
    if not ROTOWIRE_LINEUPS_PATH.exists():
        return pd.DataFrame()

    return pd.read_parquet(ROTOWIRE_LINEUPS_PATH)


def save_rotowire_lineups(df: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ROTOWIRE_LINEUPS_PATH, index=False)


def ensure_rotowire_lineups(team_abbrs: list[str]) -> pd.DataFrame:
    """
    Load cached lineups or fetch missing teams.

    When ``DISABLE_LIVE_FETCH=1``, returns cached data only (may be empty).
    """
    cached = load_rotowire_lineups()
    unique = sorted({abbr.upper() for abbr in team_abbrs if abbr})
    if not unique:
        return cached

    if cached.empty:
        missing = unique
    else:
        cached_teams = set(cached["team_abbr"].astype(str).str.upper())
        missing = [abbr for abbr in unique if abbr not in cached_teams]

    if not missing:
        return cached

    try:
        fresh = fetch_lineups_for_teams(missing)
    except Exception:
        return cached

    if fresh.empty:
        return cached

    if cached.empty:
        combined = fresh
    else:
        keep = cached[
            ~cached["team_abbr"]
            .astype(str)
            .str.upper()
            .isin(missing)
        ]
        combined = pd.concat([keep, fresh], ignore_index=True)

    save_rotowire_lineups(combined)
    return combined


def lineup_for_team_hand(
    lineups_df: pd.DataFrame,
    team_abbr: str,
    sp_hand: str,
) -> list[str]:
    """
    Return ordered player names for *team_abbr* vs *sp_hand* (R/L throwing hand).

    Rotowire labels lineups as ``Default vs. RHP`` / ``Default vs. LHP``.
    """
    if lineups_df is None or lineups_df.empty or not team_abbr:
        return []

    vs_hand = "RHP" if str(sp_hand).upper().startswith("R") else "LHP"
    subset = lineups_df[
        lineups_df["team_abbr"].astype(str).str.upper().eq(team_abbr.upper())
        & lineups_df["vs_hand"].astype(str).str.upper().eq(vs_hand)
    ].sort_values("slot")

    if subset.empty:
        return []

    return subset["player_name"].astype(str).tolist()
