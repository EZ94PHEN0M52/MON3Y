"""
Append-only learning logs for single-market retrain loops (Track 1: pitcher_outs).

Writes under data/learning/ — does not change predictions CSV, board columns,
edge ranking, or Streamlit UI unless a retrained model is deployed separately.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from prop_scoring import MARKET_STAT_MAP, fuzzy_player_match
from utils import (
    OUTCOMES_LOG_PATH,
    PREDICTIONS_LOG_PATH,
    game_date_from_commence,
    normalize_version,
    resolve_feature_path,
)

# Track 1 v1 scope — extend when other markets get a learning loop.
LEARNING_MARKETS = frozenset({"pitcher_outs"})

PREDICTION_LOG_COLUMNS = [
    "logged_at",
    "feature_start",
    "feature_end",
    "version",
    "game_date",
    "event_id",
    "game",
    "player",
    "market",
    "side",
    "line",
    "odds",
    "bookmaker",
    "bookmaker_key",
    "commence_time",
    "over_probability",
    "under_probability",
    "model_probability",
    "raw_model_probability",
    "calibrated_probability",
    "market_probability",
    "devigged_market_prob",
    "edge",
    "ev",
    "consensus_line",
    "predicted_count",
    "dist_over_probability",
]

OUTCOME_LOG_COLUMNS = [
    "joined_at",
    "game_date",
    "player",
    "market",
    "line",
    "side",
    "actual_stat",
    "over_hit",
    "model_probability",
    "calibrated_probability",
    "edge",
    "ev",
    "prediction_error",
    "logged_at",
    "version",
]


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )


def _read_parquet(path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _write_parquet(path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def build_prediction_log_rows(
    predictions: pd.DataFrame,
    *,
    version: str,
    feature_start: str,
    feature_end: str,
    markets: frozenset[str] | None = None,
    logged_at: str | None = None,
) -> pd.DataFrame:
    """Select and normalize prediction rows for the learning log."""

    if predictions.empty:
        return pd.DataFrame(columns=PREDICTION_LOG_COLUMNS)

    version = normalize_version(version)
    markets = markets or LEARNING_MARKETS
    logged_at = logged_at or _utc_now_iso()

    subset = predictions[
        predictions["market"].isin(markets)
    ].copy()

    if subset.empty:
        return pd.DataFrame(columns=PREDICTION_LOG_COLUMNS)

    subset["logged_at"] = logged_at
    subset["feature_start"] = feature_start
    subset["feature_end"] = feature_end
    subset["version"] = version
    subset["game_date"] = subset["commence_time"].map(
        game_date_from_commence
    )

    for column in PREDICTION_LOG_COLUMNS:
        if column not in subset.columns:
            subset[column] = np.nan

    return subset[PREDICTION_LOG_COLUMNS].copy()


def append_predictions_log(
    predictions: pd.DataFrame,
    *,
    version: str,
    feature_start: str,
    feature_end: str,
    markets: frozenset[str] | None = None,
    path=PREDICTIONS_LOG_PATH,
) -> int:
    """
    Append pitcher_outs (or other learning-market) rows to predictions_log.

    Returns number of rows appended. Failures should be caught by callers so
    predict.py never aborts because of logging.
    """

    new_rows = build_prediction_log_rows(
        predictions,
        version=version,
        feature_start=feature_start,
        feature_end=feature_end,
        markets=markets,
    )

    if new_rows.empty:
        return 0

    existing = _read_parquet(path)
    combined = pd.concat(
        [existing, new_rows],
        ignore_index=True,
    )
    _write_parquet(path, combined)
    return len(new_rows)


def load_predictions_log(
    *,
    market: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    path=PREDICTIONS_LOG_PATH,
) -> pd.DataFrame:
    frame = _read_parquet(path)

    if frame.empty:
        return frame

    if market is not None:
        frame = frame[frame["market"] == market]

    if start_date is not None:
        frame = frame[frame["game_date"] >= start_date]

    if end_date is not None:
        frame = frame[frame["game_date"] <= end_date]

    return frame.reset_index(drop=True)


def _actual_stat_lookup(
    pitchers: pd.DataFrame,
    *,
    game_date: str,
    player: str,
    stat: str,
) -> float | None:
    day_rows = pitchers[
        pitchers["game_date"].astype(str) == str(game_date)
    ]

    if day_rows.empty:
        return None

    names = (
        day_rows["player_name"]
        .dropna()
        .drop_duplicates()
    )
    match = fuzzy_player_match(player, names)

    if match is None:
        return None

    row = day_rows[
        day_rows["player_name"] == match
    ].iloc[0]

    if stat not in row.index:
        return None

    value = row[stat]

    if pd.isna(value):
        return None

    return float(value)


def join_outcomes_for_market(
    market: str,
    start_date: str,
    end_date: str,
    *,
    version: str = "v2",
    predictions: pd.DataFrame | None = None,
    feature_end: str | None = None,
) -> pd.DataFrame:
    """
    Match logged predictions to realized stat values from pitcher features.

    Uses one row per (logged_at, game_date, player, side, line) from the
    predictions log (latest log wins when duplicates exist for the same key
    without logged_at in the dedupe key — callers dedupe before join).
    """

    if market not in LEARNING_MARKETS:
        raise ValueError(
            f"Market {market!r} is not enabled for the learning loop. "
            f"Supported: {sorted(LEARNING_MARKETS)}"
        )

    stat = MARKET_STAT_MAP.get(market)
    if stat is None:
        raise ValueError(f"No stat mapping for market {market!r}")

    version = normalize_version(version)
    preds = predictions

    if preds is None:
        preds = load_predictions_log(
            market=market,
            start_date=start_date,
            end_date=end_date,
        )

    if preds.empty:
        return pd.DataFrame(columns=OUTCOME_LOG_COLUMNS)

    preds = preds.copy()
    preds = preds.dropna(subset=["game_date"])
    preds = preds[
        (preds["game_date"] >= start_date)
        & (preds["game_date"] <= end_date)
    ]

    if preds.empty:
        return pd.DataFrame(columns=OUTCOME_LOG_COLUMNS)

    feature_end = feature_end or end_date
    pitcher_path = resolve_feature_path(
        start_date,
        feature_end,
        version,
        role="pitcher",
    )

    if not pitcher_path.exists():
        raise FileNotFoundError(
            f"No pitcher feature parquet for {start_date} → {feature_end}. "
            "Run ensure_features.py --fix first."
        )

    pitchers = pd.read_parquet(pitcher_path)
    pitchers["game_date"] = (
        pd.to_datetime(pitchers["game_date"])
        .dt.strftime("%Y-%m-%d")
    )

    joined_at = _utc_now_iso()
    rows: list[dict] = []

    deduped = (
        preds.sort_values("logged_at")
        .drop_duplicates(
            subset=[
                "game_date",
                "player",
                "market",
                "side",
                "line",
                "bookmaker_key",
            ],
            keep="last",
        )
    )

    for _, pred in deduped.iterrows():
        actual = _actual_stat_lookup(
            pitchers,
            game_date=str(pred["game_date"]),
            player=str(pred["player"]),
            stat=stat,
        )

        if actual is None:
            continue

        line = float(pred["line"])
        side = str(pred["side"]).strip().lower()
        over_hit = int(actual > line)

        model_prob = pred.get("calibrated_probability")
        if pd.isna(model_prob):
            model_prob = pred.get("model_probability")

        model_prob = float(model_prob) if pd.notna(model_prob) else np.nan
        prediction_error = (
            model_prob - over_hit
            if np.isfinite(model_prob)
            else np.nan
        )

        rows.append({
            "joined_at": joined_at,
            "game_date": pred["game_date"],
            "player": pred["player"],
            "market": market,
            "line": line,
            "side": side,
            "actual_stat": actual,
            "over_hit": over_hit,
            "model_probability": pred.get("model_probability"),
            "calibrated_probability": pred.get("calibrated_probability"),
            "edge": pred.get("edge"),
            "ev": pred.get("ev"),
            "prediction_error": prediction_error,
            "logged_at": pred.get("logged_at"),
            "version": pred.get("version", version),
        })

    if not rows:
        return pd.DataFrame(columns=OUTCOME_LOG_COLUMNS)

    return pd.DataFrame(rows)[OUTCOME_LOG_COLUMNS]


def append_outcomes_log(
    outcomes: pd.DataFrame,
    *,
    path=OUTCOMES_LOG_PATH,
) -> int:
    if outcomes.empty:
        return 0

    existing = _read_parquet(path)

    if existing.empty:
        combined = outcomes.copy()
    else:
        keys = [
            "game_date",
            "player",
            "market",
            "side",
            "line",
        ]
        existing_index = existing.set_index(keys)
        new_index = outcomes.set_index(keys)
        merged = existing_index[
            ~existing_index.index.isin(new_index.index)
        ]
        combined = pd.concat(
            [merged.reset_index(), outcomes],
            ignore_index=True,
        )

    _write_parquet(path, combined)
    return len(outcomes)


def summarize_learning_log(
    market: str = "pitcher_outs",
) -> dict:
    preds = load_predictions_log(market=market)
    outcomes = _read_parquet(OUTCOMES_LOG_PATH)

    if not outcomes.empty:
        outcomes = outcomes[outcomes["market"] == market]

    return {
        "market": market,
        "predictions_logged": int(len(preds)),
        "outcomes_joined": int(len(outcomes)),
        "prediction_dates": (
            sorted(preds["game_date"].dropna().unique().tolist())
            if not preds.empty
            else []
        ),
        "outcome_dates": (
            sorted(outcomes["game_date"].dropna().unique().tolist())
            if not outcomes.empty
            else []
        ),
    }
