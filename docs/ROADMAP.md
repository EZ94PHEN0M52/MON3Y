# MLB Prop Model — Architecture Roadmap

Confirmed architecture for the next modeling iteration. This document supersedes ad-hoc planning notes and is the source of truth for **what we build now** vs **what we defer**.

The completed Phases 1–6 roadmap (historical odds, multi-book intelligence, calibration, etc.) remains documented in [README.md](../README.md#roadmap-phases-16). This file covers the **dual-head count-market upgrade**, **Batter Score validation**, and **Track 1 pitcher outs learning**.

---

## Confirmed architecture (summary)

| Decision | Choice |
|----------|--------|
| **Model shape** | Lightweight **50/50 dual-head** per market: **classifier** (beat the line?) + **regressor** (expected count) |
| **Markets v1** | **Pitcher strikeouts**, **pitcher walks**, and **pitcher outs** — no earned runs in v1 |
| **Batter Score** | Parallel **validation track**; excluded from board rankings/edge until validated; show `batter_score_validated` flag on player page once criteria are met |
| **Heavier upgrades** | Deferred to Phase 2+ (see below) |

### Canonical market keys (pitcher v1)

| Layer | Strikeouts | Walks | Outs |
|-------|------------|-------|------|
| Odds API / board / scoring | `pitcher_strikeouts` | `pitcher_walks` | `pitcher_outs` |
| Training stat columns | `strikeouts` | `walks` | `outs` |
| Distributional Poisson | `pitcher_strikeouts` | `pitcher_walks` | `pitcher_outs` |

**Out of v1 scope:** `pitcher_earned_runs` (and ER with projected innings) — deferred until count markets are stable.

---

## Phase 1 — Now (lightweight dual-head)

**Goal:** Ship a minimal dual-head pipeline for pitcher K, walks, and outs without the heavier feedback-loop or nightly retrain machinery.

**Status (Aug 2026):** Core dual-head inference shipped for `pitcher_strikeouts`, `pitcher_walks`, and `pitcher_outs`. Classifier remains the source of truth for edge/EV; regressor outputs (`predicted_count`, `dist_over_probability`) are visible on the board. **50/50 probability blend deferred** to Phase 2+ (see composite retrain item below).

### Modeling

| Item | Status |
|------|--------|
| Classifier head (P(actual > line)) — unchanged | **Done** |
| Poisson regressor head (expected count μ) — `fit_distributional.py` → `models/v2/dist/{market}.pkl` | **Done** (K, walks, outs) |
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
| **Markets in scope:** `pitcher_strikeouts`, `pitcher_walks`, `pitcher_outs`; other markets unchanged | **Done** |

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
| **Validation run (2026-08-22)** | **PASS** — Spearman 0.161, n=33,148 |

---

## Track 1 — Pitcher count-market learning loop ✅ v1 shipped

Self-learning for **`pitcher_strikeouts`**, **`pitcher_walks`**, and **`pitcher_outs`** — log predictions, join outcomes, single-market retrain. Full docs: [README → Pitcher count-market learning loop](../README.md#pitcher-outs-learning-loop-track-1).

| Component | Path | Status |
|-----------|------|--------|
| Prediction logging | `learning_log.py` + `predict.py` hook | **Done** (all three count markets) |
| Outcome join | `scripts/log_outcomes.py` | **Done** |
| Single-market retrain | `scripts/retrain_market.py` | **Done** (K / walks / outs) |
| Orchestration script | `run_pitcher_outs_learning.sh` | **Done** (loops all three by default) |
| Poisson dual-head | `distributional.py` + `--fit-distributional` | **Done** |

**Board impact:**

| Action | UI change |
|--------|-----------|
| Log / join only | None |
| Retrain classifier + re-predict | Pitcher K / Walks / Outs Over % / Edge / EV may change |
| `--fit-distributional` | **Pred #** and **Dist Over %** populate for those markets (existing columns) |

**Deferred (Track 1 Phase 2):**

- Feed `prediction_error_l5` from `outcomes_log` back into features
- Scheduled nightly `run_retrain.sh`
- Auto-calibrator refresh for outs from logged outcomes

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
| **Expand count markets** | Additional batter/pitcher count props beyond K + walks + outs |
| **Earned runs with projected innings** | Removed from v1; revisit when IP projection is reliable |
| **FastAPI** | Optional serving layer if Streamlit-only inference becomes a bottleneck |
| **Batter Score ML weights** | Learn component weights vs H+TB+BB or PP fantasy (separate track) |

---

## Implementation order (suggested)

1. ~~Dual-head train/infer for `pitcher_strikeouts` and `pitcher_walks`~~ **Done**
2. ~~Board columns for dual-head outputs~~ **Done**
3. ~~Batter Score backtest harness + validation criteria doc update~~ **Done**
4. ~~Flip `batter_score_validated` when gates pass~~ **Done** (2026-08-22)
5. ~~Track 1 pitcher outs learning loop (log → join → retrain)~~ **Done** (2026-08-22)
6. Phase 2+ items as capacity allows (50/50 blend, 65/35 retrain, error features, nightly retrain)

---

## Related docs

- [README — Roadmap Phases 1–6](../README.md#roadmap-phases-16) — completed pipeline phases (odds history, multi-book, calibration, CLV, etc.)
- [README — Pitcher outs learning](../README.md#pitcher-outs-learning-loop-track-1) — workflow and board impact
- [README — Markets table](../README.md) — full `PROP_MARKETS` list including out-of-v1 pitcher props
