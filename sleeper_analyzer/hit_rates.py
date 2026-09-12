"""L5/L10 hit-rate helpers for Sleeper Prop Analyzer."""

from __future__ import annotations

import pandas as pd

from ui.player_stats import MARKET_STAT_MAP, rolling_over_rates

HIT_RATE_RULE = (
    "L5/L10 count games where the stat strictly exceeded the line "
    "(stat > line). Exact line = miss, same as the main board."
)


def supports_hit_rate(market: str) -> bool:
    return market in MARKET_STAT_MAP


def compute_hit_rates(
    player: str,
    market: str,
    line: float,
    *,
    version: str = "v2",
) -> tuple[float | None, float | None]:
    l5, l10 = rolling_over_rates(player, market, float(line), version=version)
    l5_out = None if pd.isna(l5) else float(l5)
    l10_out = None if pd.isna(l10) else float(l10)
    return l5_out, l10_out


def build_player_prop_rows(
    player_props: pd.DataFrame,
    *,
    version: str = "v2",
) -> pd.DataFrame:
    if player_props.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    deduped = player_props.sort_values("stat_display").drop_duplicates(
        subset=["sleeper_stat", "line"],
        keep="first",
    )

    for _, row in deduped.iterrows():
        market = row.get("market")
        line = row.get("line")
        l5_pct = None
        l10_pct = None
        if market and line is not None and supports_hit_rate(str(market)):
            l5_pct, l10_pct = compute_hit_rates(
                str(row["player"]),
                str(market),
                float(line),
                version=version,
            )

        rows.append({
            "stat_display": row.get("stat_display"),
            "line": line,
            "l5_pct": l5_pct,
            "l10_pct": l10_pct,
            "over_multiplier": row.get("over_multiplier"),
            "under_multiplier": row.get("under_multiplier"),
            "market": market,
            "sleeper_stat": row.get("sleeper_stat"),
        })

    preferred = [
        "hits",
        "total_bases",
        "runs",
        "rbis",
        "home_runs",
        "hits_runs_rbis",
        "singles",
        "walks",
        "bat_walks",
        "strike_outs",
    ]

    def _sort_key(item: dict) -> tuple[int, str]:
        stat = str(item.get("sleeper_stat") or "")
        if stat in preferred:
            return (preferred.index(stat), stat)
        return (len(preferred), stat)

    rows.sort(key=_sort_key)
    return pd.DataFrame(rows)


def best_l5_prop_for_player(
    player_props: pd.DataFrame,
    *,
    version: str = "v2",
) -> dict | None:
    """Return the prop row with the highest L5 over rate for one player."""
    if player_props.empty:
        return None

    rows = build_player_prop_rows(player_props, version=version)
    if rows.empty:
        return None

    with_l5 = rows[rows["l5_pct"].notna()]
    if with_l5.empty:
        return None

    best_idx = with_l5["l5_pct"].idxmax()
    return with_l5.loc[best_idx].to_dict()


def best_l5_props_by_player(
    team_props: pd.DataFrame,
    *,
    version: str = "v2",
) -> dict[str, dict]:
    """Map player name → best L5 prop row for a team's Sleeper props."""
    if team_props.empty:
        return {}

    lookup: dict[str, dict] = {}
    for player in team_props["player"].dropna().astype(str).unique():
        player_props = team_props[team_props["player"].astype(str).eq(player)]
        best = best_l5_prop_for_player(player_props, version=version)
        if best is not None:
            lookup[player] = best
    return lookup
