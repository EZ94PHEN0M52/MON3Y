"""
Pitch-type matchup helpers for Batter Score Phase D.

Aggregates batter wOBA/AVG and opposing SP arsenal usage from Statcast raw,
maps pitch codes to Fastball / Slider / Curveball / Changeup / Other buckets,
and builds usage-weighted PitchTypeMatchup rows for matchup_grade_index().
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils import coerce_mlb_id

from batter_score import PitchTypeMatchup
from build_features import HITS


PITCH_BUCKETS = (
    "Fastball",
    "Slider",
    "Curveball",
    "Changeup",
    "Other",
)

# Statcast pitch_type codes → matchup bucket (per Batter Score spec).
PITCH_CODE_TO_BUCKET = {
    "FF": "Fastball",
    "FA": "Fastball",
    "FT": "Fastball",
    "SI": "Fastball",
    "FC": "Fastball",
    "SL": "Slider",
    "ST": "Slider",
    "SV": "Curveball",
    "CU": "Curveball",
    "KC": "Curveball",
    "CS": "Curveball",
    "CH": "Changeup",
    "FS": "Changeup",
    "FO": "Other",
    "EP": "Other",
    "KN": "Other",
    "SC": "Other",
    "PO": "Other",
    "UN": "Other",
}

AB_EVENTS = HITS | {
    "field_out",
    "strikeout",
    "force_out",
    "grounded_into_double_play",
    "fielders_choice",
    "fielders_choice_out",
    "double_play",
    "triple_play",
    "sac_bunt",
    "sac_bunt_double_play",
    "field_error",
    "strikeout_double_play",
}

DEFAULT_WOBA = 0.320
DEFAULT_AVG = 0.250

SP_ARSENAL_LAST_N_STARTS = 5


def pitch_code_to_bucket(code) -> str:
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return "Other"

    normalized = str(code).strip().upper()
    if not normalized:
        return "Other"

    return PITCH_CODE_TO_BUCKET.get(normalized, "Other")


def _add_pitch_bucket(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "pitch_type" not in result.columns:
        result["pitch_bucket"] = "Other"
        return result

    result["pitch_bucket"] = result["pitch_type"].map(pitch_code_to_bucket)
    return result


def _in_play_mask(df: pd.DataFrame) -> pd.Series:
    if "type" in df.columns:
        return df["type"].astype(str).str.upper().eq("X")

    if "launch_speed" in df.columns:
        return pd.to_numeric(
            df["launch_speed"],
            errors="coerce",
        ).notna()

    return df["events"].isin(AB_EVENTS)


def aggregate_batter_pitch_stats(
    statcast: pd.DataFrame,
    batter_id: int,
) -> Dict[str, Dict[str, float]]:
    """
    Batter wOBA and AVG by pitch bucket from Statcast raw.

    wOBA uses mean ``woba_value`` on balls in play; AVG is hits / AB events
    on balls in play for that bucket.
    """
    empty = {
        bucket: {"woba": DEFAULT_WOBA, "avg": DEFAULT_AVG}
        for bucket in PITCH_BUCKETS
    }

    if statcast is None or statcast.empty or batter_id is None:
        return empty

    required = {"batter", "pitch_type"}
    if not required.issubset(statcast.columns):
        return empty

    batter_rows = statcast[
        statcast["batter"].astype(int).eq(int(batter_id))
    ].copy()

    if batter_rows.empty:
        return empty

    batter_rows = _add_pitch_bucket(batter_rows)
    in_play = _in_play_mask(batter_rows)
    contact = batter_rows[in_play].copy()

    stats = dict(empty)

    for bucket in PITCH_BUCKETS:
        bucket_rows = contact[contact["pitch_bucket"] == bucket]
        if bucket_rows.empty:
            continue

        if "woba_value" in bucket_rows.columns:
            woba_series = pd.to_numeric(
                bucket_rows["woba_value"],
                errors="coerce",
            ).dropna()
            if not woba_series.empty:
                stats[bucket]["woba"] = float(woba_series.mean())

        if "events" in bucket_rows.columns:
            events = bucket_rows["events"].dropna()
            ab_count = int(events.isin(AB_EVENTS).sum())
            hit_count = int(events.isin(HITS).sum())
            if ab_count > 0:
                stats[bucket]["avg"] = hit_count / ab_count

    return stats


def _pitcher_recent_game_dates(
    statcast: pd.DataFrame,
    pitcher_id: int,
    last_n_starts: int,
) -> List:
    pitcher_rows = statcast[
        statcast["pitcher"].astype(int).eq(pitcher_id)
    ]

    if pitcher_rows.empty or "game_date" not in pitcher_rows.columns:
        return []

    dates = (
        pitcher_rows["game_date"]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    return dates[-last_n_starts:]


def aggregate_pitcher_arsenal_usage(
    statcast: pd.DataFrame,
    pitcher_id: int,
    *,
    last_n_starts: int = SP_ARSENAL_LAST_N_STARTS,
) -> Dict[str, float]:
    """Opposing SP pitch usage % by bucket (last N starts)."""
    pitcher_id = coerce_mlb_id(pitcher_id)
    if statcast is None or statcast.empty or pitcher_id is None:
        return {}

    required = {"pitcher", "pitch_type"}
    if not required.issubset(statcast.columns):
        return {}

    pitcher_rows = statcast[
        statcast["pitcher"].astype(int).eq(pitcher_id)
    ].copy()

    if pitcher_rows.empty:
        return {}

    if "game_date" in pitcher_rows.columns and last_n_starts > 0:
        recent_dates = _pitcher_recent_game_dates(
            statcast,
            pitcher_id,
            last_n_starts,
        )
        if recent_dates:
            pitcher_rows = pitcher_rows[
                pitcher_rows["game_date"].isin(recent_dates)
            ]

    pitcher_rows = _add_pitch_bucket(pitcher_rows)
    counts = (
        pitcher_rows["pitch_bucket"]
        .value_counts(normalize=True)
        .to_dict()
    )

    return {
        bucket: float(counts.get(bucket, 0.0))
        for bucket in PITCH_BUCKETS
        if counts.get(bucket, 0.0) > 0
    }


def build_opponent_pitcher_arsenal(
    statcast: pd.DataFrame,
    batter_id: Optional[int],
    pitcher_id: Optional[int],
) -> List[PitchTypeMatchup]:
    """
    Usage-weighted pitch buckets for matchup_grade_index().

    Returns an empty list when SP usage is unavailable. Buckets with SP usage
    but no batter sample fall back to the batter's overall wOBA/AVG.
    """
    if statcast is None or statcast.empty or pitcher_id is None:
        return []

    pitcher_id = coerce_mlb_id(pitcher_id)
    if pitcher_id is None:
        return []

    usage = aggregate_pitcher_arsenal_usage(
        statcast,
        pitcher_id,
    )
    if not usage:
        return []

    batter_stats = aggregate_batter_pitch_stats(
        statcast,
        batter_id,
    )

    overall_woba = DEFAULT_WOBA
    overall_avg = DEFAULT_AVG
    if batter_id is not None:
        all_stats = aggregate_batter_pitch_stats(statcast, batter_id)
        woba_values = [row["woba"] for row in all_stats.values()]
        avg_values = [row["avg"] for row in all_stats.values()]
        overall_woba = float(np.mean(woba_values))
        overall_avg = float(np.mean(avg_values))

    arsenal: List[PitchTypeMatchup] = []
    for bucket, usage_pct in usage.items():
        bucket_stats = batter_stats.get(bucket, {})
        batter_woba = bucket_stats.get("woba", overall_woba)
        batter_avg = bucket_stats.get("avg", overall_avg)

        arsenal.append(
            PitchTypeMatchup(
                pitch_type=bucket,
                usage_pct=usage_pct,
                batter_woba=batter_woba,
                batter_avg=batter_avg,
            )
        )

    usage_total = sum(item.usage_pct for item in arsenal)
    if usage_total <= 0:
        return []

    if abs(usage_total - 1.0) > 0.01:
        arsenal = [
            PitchTypeMatchup(
                pitch_type=item.pitch_type,
                usage_pct=item.usage_pct / usage_total,
                batter_woba=item.batter_woba,
                batter_avg=item.batter_avg,
            )
            for item in arsenal
        ]

    return arsenal


def arsenal_ready(arsenal: List[PitchTypeMatchup]) -> bool:
    if not arsenal:
        return False

    usage_total = sum(item.usage_pct for item in arsenal)
    return usage_total > 0 and abs(usage_total - 1.0) <= 0.01
