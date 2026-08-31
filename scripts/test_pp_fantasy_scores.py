#!/usr/bin/env python3
"""Tests for PP fantasy score archive and official scoring."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pp_fantasy_scores import (  # noqa: E402
    PP_FANTASY_GAME_SCORES_PATH,
    build_pp_fantasy_game_scores,
    compute_pp_fantasy_score_from_stats,
    player_pp_fantasy_score_values,
    rebuild_pp_fantasy_game_scores,
    save_pp_fantasy_game_scores,
)
from ui.player_stats import rolling_pp_fantasy_over_rates  # noqa: E402


def test_official_pp_scoring_chart() -> None:
    # Two singles, no counting stats
    assert compute_pp_fantasy_score_from_stats(2, 0, 2.0) == 6.0
    # One HR + one single (TB=5)
    assert compute_pp_fantasy_score_from_stats(2, 1, 5.0) == 13.0
    # Single + walk + run
    assert compute_pp_fantasy_score_from_stats(
        1, 0, 1.0, runs=1, walks=1
    ) == 7.0


def test_archive_build_and_l5_lookup(tmp_path, monkeypatch) -> None:
    import pp_fantasy_scores as pfs
    import ui.player_stats as ps

    feature_path = tmp_path / "batter_features_v2_2026-03-25_2026-08-21.parquet"
    archive_path = tmp_path / "pp_fantasy_game_scores.parquet"

    rows = []
    for i, (hits, tb, hr) in enumerate([(2, 2, 0), (1, 4, 1), (0, 0, 0)], start=1):
        rows.append(
            {
                "game_date": f"2026-08-{i:02d}",
                "game_pk": i,
                "batter": 999,
                "player_name": "Test Hitter",
                "hits": hits,
                "home_runs": hr,
                "total_bases": tb,
                "runs": 0,
                "rbi": 0,
                "walks": 0,
                "stolen_bases": 0,
            }
        )

    pd.DataFrame(rows).to_parquet(feature_path, index=False)

    monkeypatch.setattr(ps, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(pfs, "PP_FANTASY_GAME_SCORES_PATH", archive_path)
    ps._load_features.cache_clear()
    ps._kind_player_game_cache.cache_clear()
    pfs._pp_fantasy_archive_cache.cache_clear()
    pfs._player_pp_fantasy_scores_cache.cache_clear()

    def fake_find(kind, version):
        return feature_path if kind == "batter" else None

    monkeypatch.setattr(ps, "find_latest_feature_path", fake_find)

    archive = build_pp_fantasy_game_scores(version="v2")
    assert len(archive) == 3
    assert "pp_fantasy_score" in archive.columns

    save_pp_fantasy_game_scores(archive, archive_path)
    scores = player_pp_fantasy_score_values("Test Hitter", version="v2")
    assert scores == archive["pp_fantasy_score"].astype(float).tolist()

    l5, l10 = rolling_pp_fantasy_over_rates("Test Hitter", 6.5, version="v2")
    # last 3 games scores: 6, 13, 0 vs line 6.5 → one strict over (13)
    assert l5 == 1 / 3
    assert l10 == 1 / 3


def test_rebuild_clears_caches(tmp_path, monkeypatch) -> None:
    import pp_fantasy_scores as pfs

    out = tmp_path / "pp_fantasy_game_scores.parquet"
    monkeypatch.setattr(pfs, "PP_FANTASY_GAME_SCORES_PATH", out)

    empty = pd.DataFrame(columns=list(pfs._ARCHIVE_COLUMNS))
    save_pp_fantasy_game_scores(empty, out)
    assert rebuild_pp_fantasy_game_scores(version="v2") is None


if __name__ == "__main__":
    test_official_pp_scoring_chart()
    print("OK")
