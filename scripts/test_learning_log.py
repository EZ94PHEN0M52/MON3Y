#!/usr/bin/env python3
"""Unit tests for Track 1 learning log (pitcher_outs)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import learning_log  # noqa: E402
from learning_log import (  # noqa: E402
    append_outcomes_log,
    append_predictions_log,
    build_prediction_log_rows,
    join_outcomes_for_market,
)


def test_build_prediction_log_rows_filters_market() -> None:
    preds = pd.DataFrame([
        {
            "commence_time": "2026-08-22T23:05:00Z",
            "player": "Gerrit Cole",
            "market": "pitcher_outs",
            "side": "over",
            "line": 17.5,
            "odds": -110,
            "bookmaker": "DraftKings",
            "bookmaker_key": "draftkings",
            "over_probability": 0.55,
            "under_probability": 0.45,
            "model_probability": 0.58,
            "raw_model_probability": 0.58,
            "calibrated_probability": 0.57,
            "market_probability": 0.52,
            "devigged_market_prob": 0.51,
            "edge": 0.06,
            "ev": 0.04,
            "consensus_line": 17.5,
            "event_id": "evt1",
            "game": "NYY @ BOS",
        },
        {
            "commence_time": "2026-08-22T23:05:00Z",
            "player": "Aaron Judge",
            "market": "batter_hits",
            "side": "over",
            "line": 1.5,
            "odds": -120,
            "bookmaker": "FanDuel",
            "bookmaker_key": "fanduel",
            "over_probability": 0.50,
            "under_probability": 0.50,
            "model_probability": 0.52,
            "raw_model_probability": 0.52,
            "calibrated_probability": 0.52,
            "market_probability": 0.50,
            "devigged_market_prob": 0.50,
            "edge": 0.02,
            "ev": 0.01,
            "consensus_line": 1.5,
            "event_id": "evt1",
            "game": "NYY @ BOS",
        },
    ])

    rows = build_prediction_log_rows(
        preds,
        version="v2",
        feature_start="2026-03-25",
        feature_end="2026-08-21",
        logged_at="2026-08-22T12:00:00+00:00",
    )

    assert len(rows) == 1
    assert rows.iloc[0]["market"] == "pitcher_outs"
    assert rows.iloc[0]["game_date"] == "2026-08-22"
    assert rows.iloc[0]["player"] == "Gerrit Cole"


def test_join_and_append_roundtrip(tmp_path: Path) -> None:
    feature_path = tmp_path / "pitcher_features_v2_2026-03-25_2026-08-21.parquet"
    pd.DataFrame([
        {
            "game_date": "2026-08-19",
            "player_name": "Gerrit Cole",
            "pitcher": 543037,
            "outs": 18,
        },
    ]).to_parquet(feature_path, index=False)

    original_resolve = learning_log.resolve_feature_path

    def fake_resolve(start, end, version, role):
        assert role == "pitcher"
        return feature_path

    learning_log.resolve_feature_path = fake_resolve

    try:
        preds = pd.DataFrame([
            {
                "logged_at": "2026-08-20T12:00:00+00:00",
                "feature_start": "2026-03-25",
                "feature_end": "2026-08-21",
                "version": "v2",
                "game_date": "2026-08-19",
                "event_id": "evt1",
                "game": "NYY @ BOS",
                "player": "Gerrit Cole",
                "market": "pitcher_outs",
                "side": "over",
                "line": 17.5,
                "odds": -110,
                "bookmaker": "DraftKings",
                "bookmaker_key": "draftkings",
                "commence_time": "2026-08-19T23:05:00Z",
                "over_probability": 0.55,
                "under_probability": 0.45,
                "model_probability": 0.58,
                "raw_model_probability": 0.58,
                "calibrated_probability": 0.57,
                "market_probability": 0.52,
                "devigged_market_prob": 0.51,
                "edge": 0.06,
                "ev": 0.04,
                "consensus_line": 17.5,
                "predicted_count": pd.NA,
                "dist_over_probability": pd.NA,
            },
        ])

        outcomes = join_outcomes_for_market(
            "pitcher_outs",
            "2026-08-19",
            "2026-08-19",
            predictions=preds,
        )
        assert len(outcomes) == 1
        assert outcomes.iloc[0]["actual_stat"] == 18.0
        assert int(outcomes.iloc[0]["over_hit"]) == 1
        err = float(outcomes.iloc[0]["prediction_error"])
        assert abs(err - (-0.43)) < 0.001

        pred_path = tmp_path / "predictions_log.parquet"
        append_preds = pd.DataFrame([
            {
                "commence_time": "2026-08-22T23:05:00Z",
                "player": "Gerrit Cole",
                "market": "pitcher_outs",
                "side": "over",
                "line": 17.5,
                "odds": -110,
                "bookmaker": "DraftKings",
                "bookmaker_key": "draftkings",
                "over_probability": 0.55,
                "under_probability": 0.45,
                "model_probability": 0.58,
                "raw_model_probability": 0.58,
                "calibrated_probability": 0.57,
                "market_probability": 0.52,
                "devigged_market_prob": 0.51,
                "edge": 0.06,
                "ev": 0.04,
                "consensus_line": 17.5,
                "event_id": "evt1",
                "game": "NYY @ BOS",
            },
        ])
        assert append_predictions_log(
            append_preds,
            version="v2",
            feature_start="2026-03-25",
            feature_end="2026-08-21",
            path=pred_path,
        ) == 1

        outcome_path = tmp_path / "outcomes_log.parquet"
        append_outcomes_log(outcomes, path=outcome_path)
        updated = outcomes.copy()
        updated.loc[0, "actual_stat"] = 19.0
        append_outcomes_log(updated, path=outcome_path)
        reread = pd.read_parquet(outcome_path)
        assert len(reread) == 1
        assert reread.iloc[0]["actual_stat"] == 19.0
    finally:
        learning_log.resolve_feature_path = original_resolve


def main() -> None:
    test_build_prediction_log_rows_filters_market()
    print("test_build_prediction_log_rows_filters_market: ok")

    with tempfile.TemporaryDirectory() as tmp:
        test_join_and_append_roundtrip(Path(tmp))
    print("test_join_and_append_roundtrip: ok")
    print("All learning_log tests passed.")


if __name__ == "__main__":
    main()
