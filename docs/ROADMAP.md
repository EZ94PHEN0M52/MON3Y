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
| Distributional Poisson (`distributional.py`) | `pitcher_strikeouts` (existing POC) | *not yet* — dual-head regressor replaces for v1 |

**Out of v1 scope:** `pitcher_earned_runs` (and ER with projected innings) — deferred until count markets are stable.

---

## Phase 1 — Now (lightweight dual-head)

**Goal:** Ship a minimal dual-head pipeline for pitcher K and walks without the heavier feedback-loop or nightly retrain machinery.

### Modeling

- Per market (`pitcher_strikeouts`, `pitcher_walks`):
  - **Classifier head** — P(actual > line) for the posted book line
  - **Regressor head** — expected count (μ) for the same feature row
- **Composite inference (v1):** equal **50/50** blend of classifier probability and regressor-derived over probability (Poisson or distributional mapping from μ to P(over))
- Training stays on existing feature parquets and real-book lines (Phase 3 path); no new outcome log or nightly retrain yet

### Board & UI

- Extend board columns to surface dual-head outputs where useful, e.g.:
  - `model_probability` / `calibrated_probability` (classifier path)
  - Expected count (regressor μ)
  - Blended over probability used for edge (50/50 v1)
- **Markets in scope for edge/ranking:** `pitcher_strikeouts`, `pitcher_walks` under the new heads; other markets unchanged until expanded in Phase 2+

### Batter Score (validation track)

Batter Score remains **orthogonal** to LightGBM prop models but is **not** wired into board edge or top-list ranking until validation passes.

| Item | Phase 1 behavior |
|------|------------------|
| Board `batter_score_display` / ranking | Unchanged or hidden from edge sort until validated |
| Player page | Show Batter Score summary as today; add **`batter_score_validated`** flag once criteria met |
| Validation | Run in parallel — does not block dual-head pitcher work |

#### Batter Score validation criteria (TBD)

Placeholder gates — numbers and scripts to be finalized before flipping the flag:

| Criterion | Placeholder | Status |
|-----------|-------------|--------|
| Backtest script | `scripts/backtest.py` extended or dedicated Batter Score backtest | TBD |
| Minimum sample size | e.g. ≥ N batter-game rows with score + outcome | TBD |
| Correlation / calibration | e.g. score decile vs hit-rate monotonicity; correlation threshold | TBD |
| Holdout period | e.g. last K weeks out-of-sample | TBD |

When all criteria pass, set `batter_score_validated = true` in config or feature metadata and expose on the player page; only then consider Batter Score in edge/ranking UX.

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

1. Dual-head train/infer for `pitcher_strikeouts` and `pitcher_walks` (50/50 blend)
2. Board columns + edge from blended probability
3. Batter Score backtest harness + validation criteria doc update
4. Flip `batter_score_validated` when gates pass
5. Phase 2+ items as capacity allows

---

## Related docs

- [README — Roadmap Phases 1–6](../README.md#roadmap-phases-16) — completed pipeline phases (odds history, multi-book, calibration, CLV, etc.)
- [README — Markets table](../README.md) — full `PROP_MARKETS` list including out-of-v1 pitcher props
