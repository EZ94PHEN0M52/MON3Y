# MLB Prop Model — Architecture Roadmap

Confirmed architecture for the next modeling iteration. This document supersedes ad-hoc planning notes and is the source of truth for **what we build now** vs **what we defer**.

The completed Phases 1–6 roadmap (historical odds, multi-book intelligence, calibration, etc.) remains documented in [README.md](../README.md#roadmap-phases-16). This file covers the **dual-head count-market upgrade** and Batter Score validation track.

---

## Confirmed architecture (summary)

| Decision | Choice |
|----------|--------|
| **Model shape** | Lightweight **50/50 dual-head** per market: **classifier** (beat the line?) + **regressor** (expected count) |
| **Markets v1** | **Pitcher strikeouts** and **pitcher walks** only — no earned runs in v1 |
| **Batter Score** | Parallel **validation track**; excluded from board rankings/edge until validated; show `batter_score_validated` flag on player page once criteria are met |
| **Heavier upgrades** | Deferred to Phase 2+ (see below) |

### Canonical market keys (pitcher v1)

| Layer | Strikeouts | Walks |
|-------|------------|-------|
| Odds API / board / scoring (`odds_api.py`, `prop_scoring.py`, `ui/market_filters.py`) | `pitcher_strikeouts` | `pitcher_walks` |
| Training stat columns (`train.py` `PITCHER_MARKETS`) | `strikeouts` → saves `pitcher_strikeouts.pkl` | `walks` → saves `pitcher_walks.pkl` |
| Distributional Poisson (`distributional.py`) | `pitcher_strikeouts` | `pitcher_walks` |

**Out of v1 scope:** `pitcher_earned_runs` (and ER with projected innings) — deferred until count markets are stable.

---

## Phase 1 — Now (lightweight dual-head)

**Goal:** Ship a minimal dual-head pipeline for pitcher K and walks without the heavier feedback-loop or nightly retrain machinery.

**Status (Aug 2026):** Core dual-head inference shipped for `pitcher_strikeouts` and `pitcher_walks`. Classifier remains the source of truth for edge/EV; regressor outputs (`predicted_count`, `dist_over_probability`) are visible on the board. **50/50 probability blend deferred** to Phase 2+ (see composite retrain item below).

### Modeling

| Item | Status |
|------|--------|
| Classifier head (P(actual > line)) — unchanged | **Done** |
| Poisson regressor head (expected count μ) — `fit_distributional.py` → `models/v2/dist/{market}.pkl` | **Done** |
| Dual-head inference in `predict.py` (`DUAL_HEAD_MARKETS`) | **Done** |
| 50/50 blend of classifier + regressor over probability for edge | **Deferred** (Phase 2+) |
| Training on existing feature parquets / real-book lines | **Done** (no outcome log or nightly retrain) |

### Board & UI

| Item | Status |
|------|--------|
| `model_probability` / `calibrated_probability` (classifier path) | **Done** (unchanged) |
| `predicted_count` (regressor μ, 1 decimal) | **Done** |
| `dist_over_probability` (Poisson P(over) from μ + line) | **Done** |
| Edge / EV from classifier path only (Phase 1) | **Done** |
| **Markets in scope:** `pitcher_strikeouts`, `pitcher_walks`; other markets unchanged | **Done** |

### Batter Score (validation track)

Batter Score remains **orthogonal** to LightGBM prop models but is **not** wired into board edge or top-list ranking until validation passes.

| Item | Phase 1 behavior |
|------|------------------|
| Board `batter_score_display` / ranking | Unchanged or hidden from edge sort until validated |
| Player page | Show Batter Score summary as today; add **`batter_score_validated`** flag once criteria met |
| Validation | Run in parallel — does not block dual-head pitcher work |

#### Batter Score validation criteria

Run the dedicated backtest (parallel to prop-model `scripts/backtest.py`):

```bash
python scripts/backtest_batter_score.py --start YYYY-MM-DD --end YYYY-MM-DD
```

**Outcome target:** same-game **H + TB + BB** raw points (the stat Batter Score is built from).

**Scoring method:** point-in-time — season/form features use only games **strictly before** each evaluation date (no lookahead). Historical SP/matchup context is omitted unless probables exist for that date; most backtest rows are **Form only** (Phase A).

**Primary metric — Spearman ρ:** Batter Score is a ranked 0–100 composite with letter-grade thresholds; Spearman captures monotonic ordering vs outcomes without assuming linearity. Pearson *r* is reported for reference.

| Criterion | Default | CLI flag |
|-----------|---------|----------|
| Minimum sample size | ≥ 100 batter-game rows | `--min-sample` |
| Spearman correlation | ≥ 0.15 | `--min-spearman` |

When both gates pass, `data/backtest/batter_score_validation.json` sets `"validated": true`. The player page shows **✓ Batter Score validated** via `is_batter_score_validated()` — board edge/ranking remain unchanged until a separate UX decision.

Optional per-row detail: `--write-detail` → `batter_score_validation_detail.parquet`.

| Item | Status |
|------|--------|
| Backtest script | `scripts/backtest_batter_score.py` |
| Validation loader | `load_batter_score_validation()` / `is_batter_score_validated()` in `batter_score_data.py` |
| Player-page flag | `ui/batter_score.py` |
| Board edge / ranking | **Excluded** until validated + explicit wiring |

---

## Phase 2+ — Later (deferred / wishlist)

Heavier items intentionally **not** in Phase 1:

| Item | Notes |
|------|-------|
| **65/35 composite retrain metric** | Replace 50/50 inference blend with a training objective weighted **65% count (MAE/MSE)** / **35% classification (log loss)** for joint head tuning |
| **`outcomes_log.parquet` + MAE/MSE feedback loop** | Append post-game actuals; feed regressor error back into features and retrain cadence |
| **Nightly batch retrain** | `run_retrain.sh` — scheduled full or incremental retrains |
| **`prev_model_prediction` / `prediction_error_l5` features** | Rolling model error as features for drift correction |
| **Ridge / ElasticNet baselines, SGD online learning** | Linear baselines and optional online updates |
| **Expand count markets** | Additional batter/pitcher count props beyond K + walks |
| **Earned runs with projected innings** | Removed from v1; revisit when IP projection is reliable |
| **FastAPI** | Optional serving layer if Streamlit-only inference becomes a bottleneck |

---

## Implementation order (suggested)

1. ~~Dual-head train/infer for `pitcher_strikeouts` and `pitcher_walks`~~ **Done** (classifier + regressor; blend deferred)
2. ~~Board columns for dual-head outputs~~ **Done** (`predicted_count`, `dist_over_probability`)
3. ~~Batter Score backtest harness + validation criteria doc update~~ **Done** (`scripts/backtest_batter_score.py`)
4. Flip `batter_score_validated` when gates pass (run backtest; JSON flag drives player page)
5. Phase 2+ items as capacity allows (50/50 blend, 65/35 retrain, outcomes log, etc.)

---

## Related docs

- [README — Roadmap Phases 1–6](../README.md#roadmap-phases-16) — completed pipeline phases (odds history, multi-book, calibration, CLV, etc.)
- [README — Markets table](../README.md) — full `PROP_MARKETS` list including out-of-v1 pitcher props
