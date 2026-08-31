"""
Fetch Rotowire batting orders per team.

Sources:
- Default vs RHP / vs LHP: https://www.rotowire.com/baseball/batting-orders.php?team={ABBR}
- Official **Today's Lineup** (same page, when posted close to first pitch)
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd
import requests

from utils import (
    CURRENT_PROPS_PATH,
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
OFFICIAL_LINEUP_HAND = "OFFICIAL"
DEFAULT_MIN_OFFICIAL_PLAYERS = 8
DEFAULT_MAX_OFFICIAL_PLAYERS = 10
DEFAULT_MIN_SLATE_OVERLAP = 3

_LINEUP_BLOCK_RE = re.compile(
    r"Default vs\. (RHP|LHP)</div>\s*"
    r'<ol class="list is-rankings[^"]*">(.*?)</ol>',
    re.DOTALL | re.IGNORECASE,
)
_TODAY_LINEUP_RE = re.compile(
    r"Today's Lineup</div>\s*"
    r'<ol class="list is-rankings[^"]*">(.*?)</ol>',
    re.DOTALL | re.IGNORECASE,
)
_LINEUP_NOT_ANNOUNCED_RE = re.compile(
    r"lineup has yet to be announced|do not have a game today",
    re.IGNORECASE,
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
    Parse Rotowire team page HTML into lineup lists.

    Keys: ``RHP``, ``LHP`` (default orders), and ``OFFICIAL`` (Today's Lineup).
    """
    lineups: Dict[str, List[str]] = {
        hand: [] for hand in ROTOWIRE_HANDS
    }
    lineups[OFFICIAL_LINEUP_HAND] = []

    if not html:
        return lineups

    for match in _LINEUP_BLOCK_RE.finditer(html):
        hand = match.group(1).upper()
        names = _extract_player_names(match.group(2))
        lineups[hand] = names

    today_match = _TODAY_LINEUP_RE.search(html)
    if today_match:
        names = _extract_player_names(today_match.group(1))
        if names:
            lineups[OFFICIAL_LINEUP_HAND] = names
        elif _LINEUP_NOT_ANNOUNCED_RE.search(
            html[today_match.start() : today_match.end() + 400]
        ):
            lineups[OFFICIAL_LINEUP_HAND] = []

    return lineups


def _extract_player_names(block_html: str) -> List[str]:
    return [
        name.strip()
        for name in _PLAYER_NAME_RE.findall(block_html)
        if name.strip()
    ]


def parse_official_lineup_html(html: str) -> List[str]:
    """Return Today's Lineup player names, or empty if not posted."""
    return parse_rotowire_lineups_html(html).get(OFFICIAL_LINEUP_HAND, [])


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
    hands: tuple[str, ...] | None = None,
) -> list[dict]:
    fetched_at = fetched_at or (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )
    rows: list[dict] = []
    hand_keys = hands if hands is not None else ROTOWIRE_HANDS

    for hand in hand_keys:
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


def backup_rotowire_lineups() -> Optional[str]:
    """Copy the cached lineups parquet to a timestamped backup. Returns path or None."""
    if not ROTOWIRE_LINEUPS_PATH.exists():
        return None

    stamp = (
        datetime.now(timezone.utc)
        .strftime("%Y%m%dT%H%M%SZ")
    )
    backup_path = ROTOWIRE_LINEUPS_PATH.with_suffix(
        f".{stamp}.bak.parquet"
    )
    shutil.copy2(ROTOWIRE_LINEUPS_PATH, backup_path)
    return str(backup_path)


def save_rotowire_lineups_atomic(df: pd.DataFrame) -> None:
    """Write lineups parquet via temp file + replace."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = ROTOWIRE_LINEUPS_PATH.with_suffix(".tmp.parquet")
    df.to_parquet(temp_path, index=False)
    temp_path.replace(ROTOWIRE_LINEUPS_PATH)


def slate_team_abbrs_from_props(
    props_path=CURRENT_PROPS_PATH,
) -> list[str]:
    """Unique Rotowire team codes for today's prop slate."""
    if not props_path.exists():
        return []

    props = pd.read_parquet(props_path)
    if props.empty:
        return []

    abbrs: set[str] = set()
    for col in ("home_team", "away_team"):
        if col not in props.columns:
            continue
        for team_name in props[col].dropna().astype(str).unique():
            abbr = odds_team_to_abbr(team_name)
            if abbr:
                abbrs.add(abbr.upper())

    return sorted(abbrs)


def slate_players_by_team_abbr(
    props_path=CURRENT_PROPS_PATH,
) -> dict[str, set[str]]:
    """
    Map Rotowire team abbr → normalized slate player names.

    Requires a ``team`` / ``player_team`` column on props, or pass an explicit
    mapping from ``scripts/update_official_lineups.py`` (built from features).
    """
    if not props_path.exists():
        return {}

    props = pd.read_parquet(props_path)
    if props.empty or "player" not in props.columns:
        return {}

    team_col = None
    for candidate in ("team", "player_team"):
        if candidate in props.columns:
            team_col = candidate
            break

    if team_col is None:
        return {}

    grouped: dict[str, set[str]] = {}
    for _, row in props.iterrows():
        player = row.get("player")
        team_name = row.get(team_col)
        if not isinstance(player, str) or not player.strip():
            continue
        if not isinstance(team_name, str) or not team_name.strip():
            continue
        abbr = odds_team_to_abbr(team_name)
        if not abbr:
            continue
        grouped.setdefault(abbr.upper(), set()).add(_normalize_player_key(player))

    return grouped


def _normalize_player_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def validate_official_lineup(
    names: list[str],
    *,
    min_players: int = DEFAULT_MIN_OFFICIAL_PLAYERS,
    max_players: int = DEFAULT_MAX_OFFICIAL_PLAYERS,
    slate_players: set[str] | None = None,
    min_slate_overlap: int = DEFAULT_MIN_SLATE_OVERLAP,
) -> tuple[bool, str]:
    """Redundancy checks before accepting Rotowire Today's Lineup."""
    if not names:
        return False, "empty lineup (not posted on Rotowire yet)"

    if len(names) < min_players:
        return False, f"only {len(names)} batters (need at least {min_players})"

    if len(names) > max_players:
        return False, f"{len(names)} batters exceeds max {max_players}"

    cleaned = [name.strip() for name in names if str(name).strip()]
    if len(cleaned) != len(names):
        return False, "blank player name in lineup"

    keys = [_normalize_player_key(name) for name in cleaned]
    if len(set(keys)) != len(keys):
        return False, "duplicate player names in lineup"

    if slate_players:
        overlap = sum(1 for key in keys if key in slate_players)
        if overlap < min_slate_overlap:
            return (
                False,
                f"only {overlap} lineup batters match today's prop slate "
                f"(need at least {min_slate_overlap})",
            )

    return True, "ok"


def official_lineup_from_df(
    lineups_df: pd.DataFrame,
    team_abbr: str,
) -> list[str]:
    if lineups_df is None or lineups_df.empty or not team_abbr:
        return []

    subset = lineups_df[
        lineups_df["team_abbr"].astype(str).str.upper().eq(team_abbr.upper())
        & lineups_df["vs_hand"]
        .astype(str)
        .str.upper()
        .eq(OFFICIAL_LINEUP_HAND)
    ].sort_values("slot")

    if subset.empty:
        return []

    return subset["player_name"].astype(str).tolist()


@dataclass
class OfficialLineupUpdateResult:
    team_abbr: str
    status: str
    message: str
    players: list[str] = field(default_factory=list)


def fetch_official_lineup(
    team_abbr: str,
    *,
    html: str | None = None,
) -> list[str]:
    """Fetch Today's Lineup for one team."""
    page = html if html is not None else _fetch_team_html(team_abbr)
    return parse_official_lineup_html(page)


def merge_official_lineups(
    cached: pd.DataFrame,
    official_by_team: dict[str, list[str]],
    *,
    fetched_at: str | None = None,
) -> pd.DataFrame:
    """Replace OFFICIAL rows for updated teams; keep default RHP/LHP rows."""
    fetched_at = fetched_at or (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )

    if cached is None or cached.empty:
        base = pd.DataFrame(
            columns=[
                "team_abbr",
                "vs_hand",
                "slot",
                "player_name",
                "fetched_at",
            ]
        )
    else:
        teams = {abbr.upper() for abbr in official_by_team}
        base = cached[
            ~(
                cached["team_abbr"].astype(str).str.upper().isin(teams)
                & cached["vs_hand"]
                .astype(str)
                .str.upper()
                .eq(OFFICIAL_LINEUP_HAND)
            )
        ].copy()

    new_rows: list[dict] = []
    for team_abbr, names in official_by_team.items():
        new_rows.extend(
            lineups_to_rows(
                team_abbr,
                {OFFICIAL_LINEUP_HAND: names},
                fetched_at=fetched_at,
                hands=(OFFICIAL_LINEUP_HAND,),
            )
        )

    if not new_rows:
        return base

    return pd.concat([base, pd.DataFrame(new_rows)], ignore_index=True)


def update_official_lineups(
    team_abbrs: list[str],
    *,
    min_players: int = DEFAULT_MIN_OFFICIAL_PLAYERS,
    max_players: int = DEFAULT_MAX_OFFICIAL_PLAYERS,
    min_slate_overlap: int = DEFAULT_MIN_SLATE_OVERLAP,
    slate_players_by_team: dict[str, set[str]] | None = None,
    dry_run: bool = False,
    backup: bool = True,
) -> list[OfficialLineupUpdateResult]:
    """
    Fetch Rotowire **Today's Lineup** for each team and merge into cache.

    Skips teams when the lineup is not posted or fails validation.
    Never removes existing OFFICIAL rows unless a new valid lineup replaces them.
    """
    unique = sorted({abbr.upper() for abbr in team_abbrs if abbr})
    if not unique:
        return []

    require_live_fetch("Rotowire official lineups")
    cached = load_rotowire_lineups()
    fetched_at = (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )
    results: list[OfficialLineupUpdateResult] = []
    accepted: dict[str, list[str]] = {}

    for team_abbr in unique:
        try:
            names = fetch_official_lineup(team_abbr)
        except Exception as exc:
            results.append(
                OfficialLineupUpdateResult(
                    team_abbr=team_abbr,
                    status="failed",
                    message=f"fetch error: {exc}",
                )
            )
            continue

        slate_players = None
        if slate_players_by_team is not None:
            slate_players = slate_players_by_team.get(team_abbr)

        ok, reason = validate_official_lineup(
            names,
            min_players=min_players,
            max_players=max_players,
            slate_players=slate_players,
            min_slate_overlap=min_slate_overlap,
        )
        if not ok:
            results.append(
                OfficialLineupUpdateResult(
                    team_abbr=team_abbr,
                    status="skipped",
                    message=reason,
                    players=names,
                )
            )
            continue

        existing = official_lineup_from_df(cached, team_abbr)
        if existing == names:
            results.append(
                OfficialLineupUpdateResult(
                    team_abbr=team_abbr,
                    status="unchanged",
                    message="already cached",
                    players=names,
                )
            )
            continue

        accepted[team_abbr] = names
        results.append(
            OfficialLineupUpdateResult(
                team_abbr=team_abbr,
                status="updated",
                message=f"{len(names)} batters",
                players=names,
            )
        )

    if not accepted or dry_run:
        return results

    merged = merge_official_lineups(cached, accepted, fetched_at=fetched_at)
    if backup:
        backup_rotowire_lineups()
    save_rotowire_lineups_atomic(merged)
    return results


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
        # Refresh default RHP/LHP only — never drop cached OFFICIAL rows.
        missing_set = set(missing)
        keep = cached[
            ~(
                cached["team_abbr"].astype(str).str.upper().isin(missing_set)
                & cached["vs_hand"]
                .astype(str)
                .str.upper()
                .isin(ROTOWIRE_HANDS)
            )
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


def lineup_for_team(
    lineups_df: pd.DataFrame,
    team_abbr: str,
    sp_hand: str | None = None,
    *,
    min_official_players: int = DEFAULT_MIN_OFFICIAL_PLAYERS,
) -> tuple[list[str], str]:
    """
    Prefer Rotowire **Today's Lineup** when cached; else default vs SP hand.

    Returns ``(player_names, source)`` where *source* is ``official``,
    ``default``, or ``none``.
    """
    if not team_abbr:
        return [], "none"

    official = official_lineup_from_df(lineups_df, team_abbr)
    if len(official) >= min_official_players:
        return official, "official"

    if sp_hand is None:
        return [], "none"

    default = lineup_for_team_hand(lineups_df, team_abbr, sp_hand)
    if default:
        return default, "default"

    return [], "none"
