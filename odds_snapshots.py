from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from utils import RAW_DIR


SNAPSHOT_PREFIX = "props_"


def snapshots_dir():
    path = (
        RAW_DIR /
        "odds" /
        "snapshots"
    )

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def _ensure_fetched_at(df):
    result = df.copy()

    now = (
        datetime.now(timezone.utc)
        .isoformat()
    )

    if "fetched_at" not in result.columns:
        result["fetched_at"] = now
    else:
        result["fetched_at"] = (
            result["fetched_at"]
            .fillna(now)
        )

    return result


def save_live_snapshot(props_df):
    """
    Append-only snapshot of live props.

    Writes data/raw/odds/snapshots/props_{YYYYMMDD_HHMMSS}.parquet
    without touching current_props.parquet.
    """

    if props_df is None or props_df.empty:
        return None

    df = _ensure_fetched_at(props_df)

    timestamp = (
        datetime.now(timezone.utc)
        .strftime("%Y%m%d_%H%M%S")
    )

    base_name = f"{SNAPSHOT_PREFIX}{timestamp}.parquet"
    output = snapshots_dir() / base_name

    counter = 2

    while output.exists():
        output = (
            snapshots_dir() /
            f"{SNAPSHOT_PREFIX}{timestamp}_{counter}.parquet"
        )
        counter += 1

    df.to_parquet(
        output,
        index=False,
    )

    return output


def _parse_snapshot_timestamp(path):
    stem = path.stem

    if not stem.startswith(SNAPSHOT_PREFIX):
        return None

    raw = stem[len(SNAPSHOT_PREFIX):]

    try:
        return datetime.strptime(
            raw,
            "%Y%m%d_%H%M%S",
        ).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def list_snapshot_files(
    snapshots_path=None,
    snapshot_date=None,
):
    directory = (
        snapshots_path
        if snapshots_path is not None
        else snapshots_dir()
    )

    if not directory.exists():
        return []

    files = sorted(
        directory.glob(
            f"{SNAPSHOT_PREFIX}*.parquet"
        )
    )

    if snapshot_date is None:
        return files

    target = pd.Timestamp(
        snapshot_date
    ).date()

    matched = []

    for path in files:
        parsed = _parse_snapshot_timestamp(path)

        if parsed is not None:
            if parsed.date() == target:
                matched.append(path)
            continue

        try:
            sample = pd.read_parquet(
                path,
                columns=["fetched_at"],
            )

            if sample.empty:
                continue

            fetched = pd.to_datetime(
                sample["fetched_at"].iloc[0],
                utc=True,
            )

            if fetched.date() == target:
                matched.append(path)

        except Exception:
            continue

    return matched


def load_snapshots(
    snapshots_path=None,
    snapshot_date=None,
):
    files = list_snapshot_files(
        snapshots_path,
        snapshot_date,
    )

    if not files:
        return pd.DataFrame()

    frames = []

    for path in files:
        frame = pd.read_parquet(path)
        frame["_snapshot_file"] = path.name
        frames.append(frame)

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    combined["fetched_at"] = pd.to_datetime(
        combined["fetched_at"],
        utc=True,
        errors="coerce",
    )

    return combined


OPENING_KEYS = [
    "player",
    "market",
    "event_id",
    "bookmaker",
    "side",
]


def load_opening_snapshot(
    snapshots_df,
    game_date=None,
):
    """
    Earliest snapshot row per player/market/book/side for the game day,
    preferring rows fetched before commence_time when available.
    """

    if snapshots_df.empty:
        return pd.DataFrame()

    working = snapshots_df.copy()

    working["fetched_at"] = pd.to_datetime(
        working["fetched_at"],
        utc=True,
        errors="coerce",
    )

    working["commence_time"] = pd.to_datetime(
        working["commence_time"],
        utc=True,
        errors="coerce",
    )

    if game_date is not None:
        day = pd.Timestamp(game_date).date()
        working = working[
            working["fetched_at"]
            .dt.date == day
        ]

    if working.empty:
        return pd.DataFrame()

    working = working.sort_values(
        "fetched_at",
    )

    pregame = working[
        working["fetched_at"]
        < working["commence_time"]
    ]

    source = (
        pregame
        if not pregame.empty
        else working
    )

    opening = (
        source
        .groupby(
            OPENING_KEYS,
            dropna=False,
        )
        .first()
        .reset_index()
    )

    opening = opening.rename(
        columns={
            "line": "opening_line",
            "odds": "opening_odds",
        }
    )

    keep = OPENING_KEYS + [
        "opening_line",
        "opening_odds",
        "fetched_at",
    ]

    return opening[
        [
            column
            for column in keep
            if column in opening.columns
        ]
    ]


def load_latest_snapshot(
    snapshots_df,
    game_date=None,
):
    """
    Latest snapshot row per player/market/book/side for the game day.
    """

    if snapshots_df.empty:
        return pd.DataFrame()

    working = snapshots_df.copy()

    working["fetched_at"] = pd.to_datetime(
        working["fetched_at"],
        utc=True,
        errors="coerce",
    )

    if game_date is not None:
        day = pd.Timestamp(game_date).date()
        working = working[
            working["fetched_at"]
            .dt.date == day
        ]

    if working.empty:
        return pd.DataFrame()

    latest = (
        working
        .sort_values("fetched_at")
        .groupby(
            OPENING_KEYS,
            dropna=False,
        )
        .last()
        .reset_index()
    )

    return latest
