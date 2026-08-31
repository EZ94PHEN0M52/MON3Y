# MLB Prop Model (Active Development)

**Active workspace** for the MLB player-prop research pipeline. Compare model **Over %** and **Under %** against sportsbook lines, browse props in Streamlit, and track how the project evolved from **V1 → V2 → V3 → Main**.

- **GitHub:** [EZ94PHEN0M52/MON3Y](https://github.com/EZ94PHEN0M52/MON3Y) — tags **`v1`**, **`v2`**, **`v3`** mark frozen baselines; active development is on **`main`**
- **Frozen local copies:** [`mlb-prop-model-v1/`](../mlb-prop-model-v1), [`mlb-prop-model-v2/`](../mlb-prop-model-v2), [`mlb-prop-model-v3/`](../mlb-prop-model-v3/)

**Table of contents:** [Quick notes](#quick-notes) · [Quick start](#quick-start-for-beginners) · [Shell scripts walkthrough](#shell-scripts--quick-walkthrough) · [Spin up V1 / V2](#spin-up-v1-or-v2-action-paths) · [Version compare](#version-compare-v1--v2--v3--main) · [Version snapshots](#version-snapshots) · [Cache-first policy](#cache-first-data-policy-no-redundant-api-calls) · [Daily workflow](#daily-workflow-v2) · [Official lineups (pre-game)](#official-rotowire-lineups-pre-game) · [Stuff strikeout model (v2)](#stuff-strikeout-model-v2) · [Pitcher outs learning](#pitcher-outs-learning-loop-track-1) · [Command reference](#command-reference) · [Streamlit UI](#streamlit-ui) · [Changelog](#changelog)

> **📌 Latest (main) note:** This folder (`mlb-prop-model/`) is the **active development workspace** on branch **`main`**. Use **`./run_daily.sh`** for the modern V2+ pipeline (Phases 1–6, Batter Score, Pick Builder, PP/Underdog fantasy boards, **Hitter's Life**). **Stuff K (v2)** (Statcast SwStr/chase/velocity strikeout column) needs a one-time **`./run_pitcher_strikeout_stuff.sh`** — then daily predict picks it up automatically. Close to first pitch, run **`./run_official_lineups.sh`** so the [Hitter's Life](#hitters-life-board) lineup filter uses Rotowire **Today's Lineup** (official) instead of default vs RHP/LHP. For the **V1 rolling-form baseline**, use a frozen copy, git tag **`v1`**, or `predict.py --version v1` here — **not** `./run_daily.sh`. Frozen snapshots live in sibling folders and on GitHub tags **`v1`**, **`v2`**, **`v3`**.

---

## Quick notes

### `run_daily.sh` vs `run_pitcher_outs_learning.sh`

- **`./run_pitcher_outs_learning.sh` already runs `./run_daily.sh` as Step 1** (features + predict), then joins outcomes, retrains pitcher K / walks / outs models, and runs `predict.py` again. You usually do **not** need both scripts in one session unless you want a specific order (see below).
- **Props fetch / Odds API credits:** the learning script’s embedded daily pass passes **`--skip-props` by default** — it uses cached `data/processed/current_props.parquet` and does **not** spend Odds API tokens on props. To refresh props inside the learning loop, run **`./run_pitcher_outs_learning.sh --fetch-props`** (or run `./run_daily.sh` first, then learning with `--skip-daily`).
- **What the default learning run still fetches:** game lines and probables (MLB Stats API / non-Odds-API paths) unless you pass `--skip-game-lines` or `--skip-probables` to the learning script.
- **Do not run both scripts at the same time** — they share feature parquets, `predictions_v2.csv`, and learning logs with no file locking; parallel runs can corrupt or overwrite outputs.
- **Recommended patterns:**
  - **Learning only (usual):** `./run_pitcher_outs_learning.sh`
  - **Fresh props, then retrain:** `./run_daily.sh` → wait for finish → `./run_pitcher_outs_learning.sh --skip-daily`
  - **Daily board refresh only:** `./run_daily.sh`

See also [Shell scripts walkthrough](#shell-scripts--quick-walkthrough) and [Pitcher outs learning loop (Track 1)](#pitcher-outs-learning-loop-track-1).

### Stuff K (v2) vs main pitcher strikeouts model

- **Main board Model % / Edge / EV** still come from **`pitcher_strikeouts.pkl`** (LightGBM on rolling box-score stats). That path is unchanged by Stuff K v2.
- **Stuff K (v2)** is a **separate** Statcast process model (`models/v2/pitcher_strikeouts_stuff.pkl`): SwStr%, chase%, velocity → K% → Poisson expected K and Over %. Shown in the **Stuff K (v2)** column on strikeout rows only — same idea as **Batter Score v2** sitting beside v1.
- **`./run_daily.sh`** scores Stuff K v2 **only if** the stuff model file already exists. It does **not** train that model. First-time or after a big Statcast refresh, run **`./run_pitcher_strikeout_stuff.sh`** once (features → fit → predict). See [Stuff strikeout model (v2)](#stuff-strikeout-model-v2).

### Historical Statcast for H2H / wOBA

Career **H2H vs SP**, **pitch-type wOBA**, and **Arsenal wOBA** on Hitter's Life merge **every** `data/raw/statcast_*.parquet` shard (not just the current-season file from `./run_daily.sh`). To backfill a prior season:

```bash
python scripts/fetch_statcast_history.py --season 2024
```

That saves `statcast_2024-03-28_2024-09-29.parquet` (~full 2024 regular season). Restart Streamlit after download. One-time fetch; not part of the daily pipeline.

---

## Quick start for beginners

This section assumes you have **never opened this repo before**. Follow the steps in order. Each command explains what it does, what files it creates, how long it usually takes, and what to do when something is missing.

### What this project is

The MLB Prop Model is a **research tool** for Major League Baseball **player props** — bets like “Will this pitcher get over 5.5 strikeouts?” or “Will this batter get over 1.5 hits?”

The pipeline does four things:

1. **Downloads data** — Statcast stats, today’s sportsbook lines (The Odds API), probable starting pitchers, and game totals/spreads.
2. **Builds features** — rolling averages, opponent strength, park effects, and similar inputs for each player-game.
3. **Trains or loads ML models** — LightGBM classifiers that estimate **P(actual stat > posted line)** for each prop.
4. **Shows results in Streamlit** — a sortable **board** with **Over %**, **Under %**, **edge** (model vs market), and **EV** (expected value after vig).

**Versions** are snapshots of how the project grew:

| Label | What it means |
|-------|----------------|
| **V1** | Rolling form only (simple baseline) |
| **V2** | + opponent, handedness, park |
| **V3** | + Phases 1–6 (historical odds, multi-book, calibration, Batter Score, Pick Builder) — frozen at git tag **`v3`** |
| **Main** | Today’s active code in this folder — same models as V3 plus post-v3 upgrades (dual-head pitcher K/walks/outs, Track 1 K/walks/outs learning, **validated** Batter Score + **Batter Score v2** (Savant pitch-type matchup), **Stuff K (v2)** strikeout model (SwStr/chase/velocity), **Hitter's Life** batting board, **official Rotowire lineups** (`./run_official_lineups.sh`), **Batter Score Pick Builder**, PrizePicks/Underdog fantasy lines on batter score boards, Rotowire lineup filter, Top Over/Under **Game & time** + L5/L10 %, player stat **H2H** filter, bat/throw **(L)/(R)** labels) |

**Version compare** puts all four generations side-by-side on one table so you can see how **Over %** and **Under %** differ for the same player and market.

This is **not** a betting service or a guarantee of profit. It is a statistical research dashboard.

### Spin up V1 or V2 (action paths)

Pick **one** path below. Both can coexist on the same machine (separate folders or `--version` flags). For side-by-side **Version compare**, see [Prepare all versions](#prepare-all-versions-for-version-compare).

#### V1 — rolling-form baseline

| Item | Value |
|------|-------|
| **Models** | `models/v1/*.pkl` |
| **Predictions** | `data/predictions/predictions.csv` |
| **Features** | `batter_features_v1_*`, `pitcher_features_v1_*` |

**How to spin up (choose one):**

| Path | When to use | First-time setup |
|------|-------------|------------------|
| **Frozen folder** | Self-contained V1, never touch active code | [`mlb-prop-model-v1/`](../mlb-prop-model-v1/) — `./run_daily.sh` there; see [that README](../mlb-prop-model-v1/README.md#spin-up-v1-self-contained) |
| **Git tag `v1`** | Exact repo snapshot from GitHub | `git checkout v1` in a clone of [MON3Y](https://github.com/EZ94PHEN0M52/MON3Y) |
| **Main repo + `--version v1`** | Compare V1 next to V2/V3 in this folder | [First-time setup](#first-time-setup-from-zero), then V1 predict below |

**Daily commands (main repo or tag checkout — manual pipeline, no `run_daily.sh`):**

```bash
cd /Users/edosaona-enagbare/pfinder_v1/mlb-prop-model
source .venv/bin/activate

SEASON_START=2026-03-25
YESTERDAY=$(date -v-1d +%Y-%m-%d)

python fetch_data.py --props
python fetch_data.py --statcast --start $SEASON_START --end $YESTERDAY
python build_features.py --start $SEASON_START --end $YESTERDAY --version v1
python predict.py --start $SEASON_START --end $YESTERDAY --version v1
streamlit run app.py
```

**What NOT to run for V1 in this folder:** `./run_daily.sh` here — it always targets **V2** (`predict.py --version v2`, V2 feature parquets). For the frozen V1 copy, use [`mlb-prop-model-v1/`](../mlb-prop-model-v1/) and **that** repo's `./run_daily.sh`. V1 in this repo is a **compare column** and sidebar option, not the daily default.

**Streamlit:** Sidebar **Model version** → select **`v1`** (board reads `predictions.csv`). Open **Version compare** for V1 vs V2 vs V3 vs Main.

**Offline refresh (cached data only):**

```bash
DISABLE_LIVE_FETCH=1 python predict.py --start 2026-03-25 --end 2026-08-16 --version v1
```

---

#### V2+ — modern pipeline (default)

| Item | Value |
|------|-------|
| **Models** | `models/v2/*.pkl` (+ calibrators / distributional on main) |
| **Predictions** | `data/predictions/predictions_v2.csv`, `predictions_v2_best.csv` |
| **Features** | `batter_features_v2_*`, `pitcher_features_v2_*` |

**How to spin up (choose one):**

| Path | When to use | First-time setup |
|------|-------------|------------------|
| **This folder (main)** | Active dev — Batter Score, Pick Builder, dual-head K/walks, **Stuff K v2** | [First-time setup](#first-time-setup-from-zero) |
| **Frozen folder `mlb-prop-model-v2/`** | Pre–Phases 1–6 snapshot (git tag **`v2`**) | [Standalone v2 setup](#standalone-v2-setup-legacy-baseline) |
| **Frozen folder `mlb-prop-model-v3/`** | Phases 1–6 frozen at tag **`v3`** | [Standalone v3 setup](#standalone-v3-setup-recommended-frozen-copy) |

**Daily commands (main — preferred):**

```bash
cd /Users/edosaona-enagbare/pfinder_v1/mlb-prop-model
./run_daily.sh --streamlit          # evening (~8pm PT): fresh props + board for tomorrow
./run_daily.sh --skip-props --streamlit   # morning: refresh rolling stats, keep pre-game lines
./run_daily.sh --skip-props         # reuse cached current_props.parquet only
```

**First-time Stuff K (v2) column** (after clone or before first board with Statcast stuff metrics):

```bash
./run_pitcher_strikeout_stuff.sh --streamlit   # build stuff features → train stuff model → predict → board
```

Later daily runs only need `./run_daily.sh` — it re-scores Stuff K v2 when `models/v2/pitcher_strikeouts_stuff.pkl` exists. Re-run the stuff script after a long Statcast gap or when [`pitcher_stuff.py`](pitcher_stuff.py) changes. See [Stuff strikeout model (v2)](#stuff-strikeout-model-v2).

**Recommended rhythm:** evening fetch when props post, morning `--skip-props` for fresh Statcast, **pre-game** official lineups for Hitter's Life — see [Step A → Recommended daily rhythm](#step-a--v2--main-daily-pipeline), [Official lineups (pre-game)](#official-rotowire-lineups-pre-game), and [Shell scripts walkthrough](#shell-scripts--quick-walkthrough).

**Pre-game lineups (Hitter's Life only — optional, ~1–2 hours before first pitch):**

Rotowire posts confirmed batting orders under **Today's Lineup** on each team page. Default orders (vs RHP/LHP) load automatically the first time you use the [Hitter's Life](#hitters-life-board) lineup filter; run this when you want **official** 1–9 orders instead:

```bash
./run_official_lineups.sh              # fetch Today's Lineup for all slate teams → rotowire_lineups.parquet
./run_official_lineups.sh --dry-run    # validate only, no cache write
./run_official_lineups.sh --watch 300  # poll every 5 min until lineups post (Ctrl+C to stop)
```

Then **reload Streamlit** (or refresh the page). On Hitter's Life, pick one **Game**, open **Lineup filter** — the caption shows **Today's Lineup** per team when cached, otherwise **default vs {SP hand}**. Does **not** change prop edge/EV or Batter Score math.

**Streamlit:** Sidebar **Model version** defaults to **`v2`**. Board, Pick Builder, Batter Score, and market filters use `predictions_v2.csv`. **Version compare** adds V3 (manual CSV copy) and Main (same file as V2 today).

**Full step breakdown:** [Step A — V2 + Main](#step-a--v2--main-daily-pipeline) · [Daily workflow (V2)](#daily-workflow-v2) · [Cache-first policy](#cache-first-data-policy-no-redundant-api-calls)

### Prerequisites

Install these **once** on your Mac before cloning or opening the project:

| Requirement | Why | Install |
|-------------|-----|---------|
| **Python 3.12** | Project targets 3.12 | `brew install python@3.12` |
| **libomp** | LightGBM needs OpenMP on macOS | `brew install libomp` |
| **Odds API key** | Live prop lines from sportsbooks | Free key at [the-odds-api.com](https://the-odds-api.com/) |
| **Git** (optional) | Clone from GitHub | `brew install git` |

You do **not** need a paid Odds API plan for the daily board. Paid history is only needed for [historical backtesting](#historical-odds--backtesting-phase-1).

### First-time setup (from zero)

**1. Open a terminal and go to the project folder.**

```bash
cd /Users/edosaona-enagbare/pfinder_v1/mlb-prop-model
```

**2. Create and activate a virtual environment** (~30 seconds).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` in your prompt. Run `source .venv/bin/activate` again whenever you open a new terminal tab.

**3. Install Python dependencies** (~1–3 minutes).

```bash
pip install -r requirements.txt
```

**4. Add your API key** (~1 minute).

```bash
cp .env.example .env
```

Edit `.env` and set `ODDS_API_KEY=your_key_here`. Never commit `.env` — it is gitignored.

**5. (Optional) Download training-season data** — only if `models/v1/` or `models/v2/` are empty and you need to train from scratch. Skip if pre-trained `.pkl` files already exist under `models/`. See [Initial data (one time)](#initial-data-one-time) for full commands (~10–30 minutes for Statcast + features + train).

At this point the environment is ready. The next section prepares **all four version-compare columns**.

### Prepare all versions for Version Compare

Version compare reads **four prediction CSV files** on disk. Each column maps to a project generation. You do not need to run all steps every day — only when files are missing or you want a refresh.

#### Overview: files each version needs

| Column | Model folder | Predictions file | How to create |
|--------|--------------|------------------|---------------|
| **V1** | `models/v1/*.pkl` | `data/predictions/predictions.csv` | `predict.py --version v1` |
| **V2** | `models/v2/*.pkl` | `data/predictions/predictions_v2.csv` | `./run_daily.sh` or `predict.py --version v2` |
| **V3** | `models/v2/*.pkl` (same weights as V2) | `data/predictions/predictions_v3.csv` | **Manual copy** from frozen v3 snapshot |
| **Main** | `models/v2/*.pkl` | `data/predictions/predictions_v2.csv` (same file as V2) | `./run_daily.sh` |

**Feature window for compare:** [`ui/version_compare.py`](ui/version_compare.py) ranks props scored from **2026-03-25** through **2026-08-16** (opening day through a fixed end date). Use those dates in `predict.py` when generating V1 CSVs so columns align.

**Timing:** A full first-time path (Statcast + features + daily pipeline + V1 predict + V3 copy) can take **20–45 minutes**. A normal `./run_daily.sh` rerun with warm caches is usually **2–8 minutes**.

---

#### Cache-first data & `DISABLE_LIVE_FETCH` (read before Step A)

This project is designed **cache-first**: live APIs are called only during explicit **fetch** steps. Everything else reads files from disk. That keeps Odds API credits and Statcast downloads from being wasted on backtests, Streamlit reruns, or version compare.

**Three layers (simple mental model):**

| Layer | What runs | Hits network? |
|-------|-----------|---------------|
| **1. Fetch** | `fetch_data.py`, `fetch_probables.py`, `fetch_historical_odds.py`, and `./run_daily.sh` (steps 1–4) | **Yes** — downloads once, saves to `data/raw/` and `data/processed/` |
| **2. Disk** | Parquets and CSVs under `data/raw/`, `data/processed/`, `data/predictions/`, `data/backtest/` | N/A — single source of truth |
| **3. Compute** | `predict.py`, Streamlit, backtests, version compare, Batter Score enrichment | **No** — reads disk + model `.pkl` files only |

**During this quick-start workflow:**

- **Step A (`./run_daily.sh`)** = **online mode** — may fetch Statcast (if features stale), props, game lines, and probables, then runs read-only `predict.py`.
- **Steps B–D (`predict.py --version v1`, V3 file copy, Streamlit, version compare)** = **offline-friendly** — they only read existing parquets/CSVs. They do **not** call the Odds API or Statcast if data is already on disk.
- **Streamlit reruns** (clicking filters, opening compare) = **no API** — reads `predictions_*.csv` and cached parquet enrichment.

**`DISABLE_LIVE_FETCH=1`** is a safety switch that **blocks all network calls** if something tries to fetch accidentally (backtests, offline predict, etc.). Implemented in [`utils.py`](utils.py) via `require_live_fetch()`.

| When to leave it **unset** (default) | When to set **`DISABLE_LIVE_FETCH=1`** |
|--------------------------------------|----------------------------------------|
| First-time setup, `./run_daily.sh`, manual `fetch_data.py` | Backtests, Batter Score validation, re-running predict against cached props |
| Refreshing Statcast through `ensure_features.py --fix` | Streamlit / version compare when CSVs already exist |
| Historical odds backfill | Saving API credits: `./run_daily.sh --skip-props --skip-probables` reuses yesterday’s parquets |

```bash
# Offline examples (zero API spend — files must already exist on disk)
DISABLE_LIVE_FETCH=1 python predict.py --start 2026-03-25 --end 2026-08-16 --version v2
python scripts/backtest_batter_score.py --start 2025-04-01 --end 2025-06-30   # auto-sets the flag
streamlit run app.py   # no flag needed; never calls APIs on rerun

# Online refresh (uses API credits)
unset DISABLE_LIVE_FETCH
./run_daily.sh
```

If a job fails with `Live fetch blocked … DISABLE_LIVE_FETCH=1 is set`, either unset the variable for a fetch step or ensure the required parquet/CSV already exists from a prior `./run_daily.sh`.

**Full reference:** [Cache-first data policy](#cache-first-data-policy-no-redundant-api-calls) (blocked modules, Streamlit `@st.cache_data`, daily skip flags).

---

#### Step A — V2 + Main (daily pipeline)

This is the **main path**. It refreshes today’s lines and writes `predictions_v2.csv`, which powers both the **V2** and **Main** compare columns.

```bash
cd /Users/edosaona-enagbare/pfinder_v1/mlb-prop-model
./run_daily.sh
```

**What it does (in order):**

1. `scripts/ensure_features.py --fix` — validates or rebuilds V2 feature parquets for the current season through **yesterday** (~1–15 min if rebuild needed; seconds if parquets are fresh).
2. `fetch_data.py --props` — downloads today’s player prop lines → `data/processed/current_props.parquet` + an intraday snapshot under `data/raw/odds/snapshots/`.
3. `fetch_data.py --game-lines` — game totals/spreads → `data/processed/current_game_lines.parquet`.
4. `fetch_probables.py` — probable starting pitchers → `data/processed/daily_probables.parquet` (feeds Batter Score on the board).
5. `predict.py --version v2` — scores every prop → `data/predictions/predictions_v2.csv` and `predictions_v2_best.csv` (~30 s–2 min).

**Recommended daily rhythm (evening + morning)** — books usually post **tomorrow’s slate** around **~8pm PT / 11pm ET**, after that day’s games finish. The pipeline is built for a **two-pass** schedule:

| When | Command | Why |
|------|---------|-----|
| **Evening** (~8pm PT, props live) | `./run_daily.sh --streamlit` | Fresh props, game lines, and probables for **tomorrow’s slate**; scores the board. Do **not** use `--skip-props` or `--skip-probables` — you want new lines and SP names. |
| **Next morning** (Statcast posted) | `./run_daily.sh --skip-props --streamlit` | Refreshes rolling stats through **calendar yesterday** without overwriting last night’s pre-game lines. |
| **Pre-game** (~1–2 hr before first pitch) | `./run_official_lineups.sh` | Pulls Rotowire **Today's Lineup** for slate teams → [Hitter's Life](#hitters-life-board) lineup filter uses official 1–9 order (see [Official lineups](#official-rotowire-lineups-pre-game)). Reload Streamlit after. |

**What lines up correctly on the evening run:**

- **Props & game lines** — Odds API returns upcoming events (tomorrow once posted).
- **SP props** (`pitcher_strikeouts`, walks, hits allowed, outs) — scored when books post them.
- **Probables** — fetched for every **Eastern slate date** in `current_props.parquet` (feeds Batter Score opposing-SP lookup).
- **Fantasy lines** — PrizePicks (`prizepicks_fantasy_lines.parquet`) and Underdog (`underdog_fantasy_lines.parquet`) refresh on every successful `--props` pass (feeds [batter score boards](#batter-score)).
- **Over % / Edge / EV** — computed at predict time from each player’s latest feature row (main LightGBM classifiers).
- **Stuff K (v2)** — on **Pitcher Strikeouts** rows only, if `models/v2/pitcher_strikeouts_stuff.pkl` exists; shows expected K and Poisson Over % from Statcast SwStr/chase/velocity (does **not** change Edge/EV). First-time setup: [`./run_pitcher_strikeout_stuff.sh`](#run_pitcher_strikeout_stuffsh--stuff-k-v2-pipeline).

**One timing caveat:** `run_daily.sh` uses `YESTERDAY=$(date -v-1d)` for features. On an **evening** run, that is still **calendar yesterday**, so today’s just-finished box scores are **not** in rolling stats yet — the board is correct for tomorrow’s lines, but L5/L10 and model form are **one slate behind** until the morning pass. Check the player page caption **Game logs through YYYY-MM-DD** to confirm feature freshness.

**Terminal checks after the evening run:** look for `OK — SP prop coverage looks complete` and probables counts (`primary day has N home / M away SP named`). If Batter Score shows **Partial · SP TBD** everywhere, re-run `python fetch_data.py --probables` once MLB has named starters.

See [Shell scripts — quick walkthrough](#shell-scripts--quick-walkthrough) for flag details on each script.

**Expected files after success:**

```text
data/processed/batter_features_v2_2026-03-25_YYYY-MM-DD.parquet
data/processed/pitcher_features_v2_2026-03-25_YYYY-MM-DD.parquet
data/processed/current_props.parquet
data/processed/prizepicks_fantasy_lines.parquet
data/processed/underdog_fantasy_lines.parquet
data/processed/current_game_lines.parquet
data/processed/daily_probables.parquet
data/predictions/predictions_v2.csv
models/v2/*.pkl                    # at least one market model
models/v2/pitcher_strikeouts_stuff.pkl   # optional — Stuff K (v2) column; create via ./run_pitcher_strikeout_stuff.sh
```

**If `./run_daily.sh` fails:**

| Symptom | Fix |
|---------|-----|
| `models/v2/` empty | Run `./run_daily.sh --train` once (needs 2025 training data — see [Initial data](#initial-data-one-time)) |
| Feature columns missing | Let `ensure_features.py --fix` finish; see [Troubleshooting](#troubleshooting) |
| `OUT_OF_USAGE_CREDITS` | `./run_daily.sh --skip-props` uses cached props if `current_props.parquet` exists |
| `LightGBM` / libomp error | `brew install libomp` |
| No MLB games today | Props fetch may return zero rows — normal on off-days |

---

#### Step B — V1 predictions

V1 uses **older, simpler models** in `models/v1/`.

```bash
source .venv/bin/activate
python predict.py --start 2026-03-25 --end 2026-08-16 --version v1
```

**Writes:** `data/predictions/predictions.csv`  
**Requires:** `models/v1/*.pkl` and V1 feature parquets (`batter_features_v1_*`, `pitcher_features_v1_*`). Build with [Initial data](#initial-data-one-time) if missing.  
**Time:** ~1–3 minutes when models and features exist.

**If V1 models are missing:** run the V1 block in [Initial data](#initial-data-one-time), or copy `models/v1/` from [`mlb-prop-model-v1/`](../mlb-prop-model-v1/) if you have that snapshot.

---

#### Step C — V3 predictions (frozen snapshot)

**V3 is not auto-generated** in this workspace. It represents the **frozen git tag `v3`** state. Copy a predictions file from the sibling snapshot folder:

```bash
cp ../mlb-prop-model-v3/data/predictions/predictions_v2.csv \
   data/predictions/predictions_v3.csv
```

**Writes:** `data/predictions/predictions_v3.csv`  
**Requires:** The [`mlb-prop-model-v3/`](../mlb-prop-model-v3/) folder exists beside this repo (created when the v3 snapshot was copied).  
**Time:** Instant (file copy).

If you do not have `mlb-prop-model-v3/`, clone or copy it from the **`v3`** tag on [GitHub MON3Y](https://github.com/EZ94PHEN0M52/MON3Y/tags), or leave V3 empty — the compare table shows **—** for that column.

---

#### Step D — Verify all four slots

```bash
ls -la data/predictions/predictions.csv \
       data/predictions/predictions_v2.csv \
       data/predictions/predictions_v3.csv
ls models/v1/*.pkl 2>/dev/null | head -3
ls models/v2/*.pkl 2>/dev/null | head -3
```

You want at least **one** `.pkl` per version folder and **non-empty** CSVs for each generation you care about. V2 and Main intentionally share `predictions_v2.csv`, so those two columns will match until you change that wiring in [`utils.py`](utils.py).

**Shortcut on the compare page:** click **Generate missing predictions** — this runs `predict.py` for **V1** and/or **V2** when CSVs are missing and models exist. It **does not** create V3 (copy only) and **does not** duplicate Main beyond the shared V2 file.

---

### Open Streamlit and Version Compare

**Launch the app** (after `./run_daily.sh` or at least one successful `predict.py` run):

```bash
streamlit run app.py
# or: ./run_daily.sh --streamlit   # runs pipeline first, then opens the app
```

Open **http://localhost:8501** in your browser.

**Go to Version Compare** any of these ways:

- **URL:** `http://localhost:8501/?view=compare`
- **Sidebar:** **Version compare**
- **Main board:** link **Version compare** under the Top Over / Top Under previews

Stop the server with **Ctrl+C** in the terminal.

### What you'll see on the Version Compare page

1. **Title and caption** — explains that the table shows the **top 30** unique `(player, market)` props, one best book per version, for feature window **2026-03-25 → 2026-08-16**.

2. **Version sources expander** — lists each generation (V1, V2, V3, Main), its CSV filename, whether predictions and models were found (**loaded** vs **missing**), and the model directory (`models/v1/` or `models/v2/`).

3. **Generate missing predictions** — primary button to backfill V1/V2 CSVs from disk models.

4. **Comparison table** — columns: **Player**, **Market**, **Line**, then for each loaded version **Over %** and **Under %** (e.g. **V1 Over**, **V1 Under**, … **Main Over**, **Main Under**). Missing versions show **—**.

5. **Ranking** — rows sorted by **Main |edge|** when Main is loaded; otherwise max **|edge|** or max **Over %** across loaded versions.

6. **Footer disclaimer** — model estimates only, not betting advice.

The compare view uses [`@st.cache_data`](ui/version_compare.py) keyed on prediction CSV **modification times** — it reloads when you regenerate predictions, not on every widget click.

For implementation details (merge keys, dedupe, slot config), see [Version compare (reference)](#version-compare-v1--v2--v3--main) below.

---

## Shell scripts — quick walkthrough

Five bash entry points wrap the Python pipeline. Run them from the project root (`./script.sh`, not bare `script.sh` on zsh). Each auto-activates `.venv`.

**Typical season schedule:**

```text
Evening (props post)        →  ./run_daily.sh --streamlit
Next morning (fresh form)     →  ./run_daily.sh --skip-props --streamlit
Pre-game (official lineups)   →  ./run_official_lineups.sh   # then reload Streamlit
First-time / refresh Stuff K  →  ./run_pitcher_strikeout_stuff.sh   # then daily only
Weekly / after backfill     →  ./run_evaluation.sh
Optional (outs learning)    →  ./run_pitcher_outs_learning.sh
```

Full flag tables live in [Command reference](#command-reference). This section is the **order-of-operations cheat sheet**.

---

### 1. `./run_daily.sh` — run first, every game day

**Purpose:** Refresh features (if needed), fetch today’s slate from sportsbooks, pull probables, score every prop, optionally open the board.

**When:** **Evening** when tomorrow’s props are live (~8pm PT); **morning** to refresh Statcast/form (`--skip-props`).

```bash
./run_daily.sh [--train] [--skip-props] [--skip-game-lines] [--skip-probables] [--streamlit] [--port N]
```

| Step | What runs | Skipped by |
|------|-----------|------------|
| 1 | `ensure_features.py --fix` — season features through **yesterday** | Never |
| 2 | `fetch_data.py --props` → `current_props.parquet`, `prizepicks_fantasy_lines.parquet`, `underdog_fantasy_lines.parquet` | `--skip-props` |
| 3 | `fetch_data.py --game-lines` → `current_game_lines.parquet` | `--skip-game-lines` |
| 4 | `fetch_data.py --probables` → `daily_probables.parquet` | `--skip-probables` |
| 5 | `train.py` (2025 window) | Only `--train` |
| 6 | `predict.py --version v2` → `predictions_v2.csv` | Never |
| 7 | `streamlit run app.py` | Only with `--streamlit` |

| Flag | What it does | When to use |
|------|--------------|-------------|
| *(none)* | Full online pipeline | Evening run when props just posted |
| `--streamlit` | Opens board after predict | Normal daily UX |
| `--skip-props` | Reuse cached props/lines from last fetch | **Morning** refresh; after games start (keeps pre-game lines) |
| `--skip-probables` | Reuse cached SP list | Repeat run same slate; save MLB API calls |
| `--skip-game-lines` | Reuse cached totals/spreads | Re-predict only |
| `--train` | Retrain all V2 classifiers on 2025 window | First setup, schema bump, infrequent refresh — **not** daily |
| `--port N` | Streamlit port (default 8501) | Port conflict |

**Common combos:**

```bash
./run_daily.sh --streamlit                              # evening: full fetch + board
./run_daily.sh --skip-props --streamlit                   # morning: fresh form, same lines
./run_daily.sh --skip-props --skip-probables --streamlit  # minimal API; offline re-score
```

**Writes:** `data/processed/current_*.parquet`, fantasy-line parquets from `--props`, `data/predictions/predictions_v2.csv`, optionally refreshes feature parquets.

---

### 2. `./run_evaluation.sh` — run separately (not daily)

**Purpose:** Offline **Phase 6** evaluation on a **historical** window — backtest ROI, fit probability calibrators, fit distributional (Poisson) heads for K/walks/outs.

**When:** After you have historical props + feature parquets for the window; typically **weekly or after backfill**, not before every board session.

```bash
./run_evaluation.sh [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--version v2] [--min-edge N] [--min-ev N]
```

| Step | What runs |
|------|-----------|
| 1 | `scripts/backtest.py` — simulated picks vs historical lines |
| 2 | `scripts/fit_calibrators.py` — isotonic/Platt from backtest CSV |
| 3 | `scripts/fit_distributional.py` — Poisson regressors → `models/v2/dist/*.pkl` |

| Flag | Default | Meaning |
|------|---------|---------|
| `--start` / `--end` | `2025-04-01` → `2025-06-30` | Historical evaluation window |
| `--version` | `v2` | Model directory |
| `--min-edge` / `--min-ev` | Script defaults | Backtest filter thresholds |

**Does not:** fetch live props, retrain main LightGBM classifiers, or regenerate today’s board.

**After evaluation:** run `./run_daily.sh` (or `--streamlit`) so live predict picks up new calibrators and **Pred #** / **Dist Over %** models.

---

### 3. `./run_pitcher_outs_learning.sh` — optional Track 1 loop

**Purpose:** Self-learning loop for **pitcher_strikeouts**, **pitcher_walks**, and **pitcher_outs** — log predictions → join post-game outcomes → retrain classifiers → re-predict. Use `--market MARKET` for one market only.

**When:** **Weekly** or when enough logged outcomes exist; not required for a normal board day.

```bash
./run_pitcher_outs_learning.sh [--fit-distributional] [--streamlit] [--market MARKET] [skip flags...]
```

| Step | What runs | Skipped by |
|------|-----------|------------|
| 1 | `./run_daily.sh` (logging during predict — all three count markets) | `--skip-daily` |
| 2 | `scripts/log_outcomes.py` — join actuals to log (per market) | `--skip-join` |
| 3 | `scripts/retrain_market.py` — classifier retrain (per market) | `--skip-retrain` |
| 3b | `scripts/fit_distributional.py` (per market) | Only `--fit-distributional` |
| 4 | `predict.py` — refresh board with new models | `--skip-repredict` |

| Flag | Meaning |
|------|---------|
| `--market` | One market only (default: strikeouts + walks + outs) |
| `--fit-distributional` | Also train Poisson heads → **Pred #** / **Dist Over %** for each market |
| `--skip-daily` | Outcomes + retrain only (no fresh predict log) |
| `--skip-join` / `--skip-retrain` / `--skip-repredict` | Run subset of steps |
| `--join-start` / `--join-end` | Outcome join window (default season start → yesterday) |
| `--train-start` / `--train-end` | Retrain feature window (default season start → yesterday) |
| `--fetch-props` | Refresh props during step 1 (default skips props fetch) |
| `--streamlit` | Open board when done |

**Note:** Step 2 needs **played games** in the log — first run after an evening predict may show **0 outcomes joined** until the next day. See [Pitcher outs learning loop](#pitcher-outs-learning-loop-track-1).

---

### 4. `./run_pitcher_strikeout_stuff.sh` — Stuff K (v2) pipeline

**Purpose:** Build Statcast **stuff** columns in pitcher feature parquets, train the **separate** SwStr/chase/velocity strikeout model, and re-predict so the board **Stuff K (v2)** column populates on strikeout props.

**When:** **First time** after clone, after [`pitcher_stuff.py`](pitcher_stuff.py) / stuff-metric fixes, or when the stuff model is missing. **Not** required every game day — `./run_daily.sh` re-scores existing stuff predictions automatically.

```bash
./run_pitcher_strikeout_stuff.sh [--skip-features] [--skip-fit] [--skip-predict] [--streamlit] [--start DATE] [--end DATE]
```

| Step | What runs | Skipped by |
|------|-----------|------------|
| 1 | `scripts/ensure_features.py --fix` — fetch Statcast if missing/stale, rebuild pitcher stuff columns | `--skip-features` |
| 2 | `scripts/fit_pitcher_strikeout_stuff.py` → `models/v2/pitcher_strikeouts_stuff.pkl` | `--skip-fit` |
| 3 | `predict.py` — writes `stuff_predicted_count` / `stuff_over_probability` to CSV | `--skip-predict` |

| Flag | Meaning |
|------|---------|
| `--skip-features` | Parquet already rebuilt (e.g. morning `./run_daily.sh` just ran `ensure_features --fix`) |
| `--skip-fit` | Model already trained; only re-predict |
| `--streamlit` | Open board after predict |

**Does not:** change main **Model %** / **Edge** / **EV** (still `pitcher_strikeouts.pkl`). Does **not** fetch props — run `./run_daily.sh` first if `current_props.parquet` is stale.

Full detail: [Stuff strikeout model (v2)](#stuff-strikeout-model-v2) · [Command reference → run_pitcher_strikeout_stuff.sh](#run_pitcher_strikeout_stuffsh--stuff-k-v2-pipeline)

---

### 5. `./run_official_lineups.sh` — pre-game (Hitter's Life lineups)

**Purpose:** Fetch Rotowire **Today's Lineup** for today's slate teams and merge **OFFICIAL** rows into `data/processed/rotowire_lineups.parquet`. The [Hitter's Life](#hitters-life-board) lineup filter prefers these over default vs RHP/LHP when cached.

**When:** **~1–2 hours before first pitch** (or poll with `--watch` until Rotowire posts). Requires a prior `./run_daily.sh` so `current_props.parquet` lists slate teams. **Does not** re-run predict or change prop edge/EV.

```bash
./run_official_lineups.sh [--dry-run] [--teams ABBR,...] [--watch SECONDS] [--max-runs N] [python flags…]
```

| Step | What runs |
|------|-----------|
| 1 | Resolve slate teams from `current_props.parquet` (or `--teams`) |
| 2 | `scripts/update_official_lineups.py` — fetch each team's Rotowire batting-orders page, parse **Today's Lineup** |
| 3 | Validate (8–10 batters, no dupes, slate overlap) — skip if not posted yet |
| 4 | Backup parquet → atomic write to `rotowire_lineups.parquet` |

| Flag | Meaning |
|------|---------|
| *(none)* | Update all slate teams; 2 fetch retries per team |
| `--dry-run` | Fetch + validate only; no cache write |
| `--teams NYY,BOS,…` | Limit to specific Rotowire codes (skip props lookup) |
| `--watch 300` | Poll every 300s until Ctrl+C (optional `--max-runs N`) |
| `--min-players` / `--max-players` | Lineup size bounds (default 8–10) |
| `--skip-slate-check` | Disable cross-check vs prop slate |
| `--no-backup` | Skip timestamped `.bak.parquet` before write |

**After success:** reload Streamlit (or refresh the browser). On Hitter's Life → pick a **Game** → **Lineup filter** caption shows **Today's Lineup** when official rows exist.

**Safety (built-in):** skips empty/unposted lineups; never deletes default RHP/LHP rows; unchanged lineups skip write; `DISABLE_LIVE_FETCH=1` blocked in shell script.

Full detail: [Official Rotowire lineups (pre-game)](#official-rotowire-lineups-pre-game) · [Command reference → run_official_lineups.sh](#run_official_lineupssh--pre-game-lineups)

---

## Version compare (V1 / V2 / V3 / Main)

Side-by-side **Over %** and **Under %** for the same player and market across project generations. Implementation: [`ui/version_compare.py`](ui/version_compare.py); slot definitions in [`utils.py`](utils.py) (`VERSION_COMPARE_SLOTS`).

### Version columns

| Column | Models | Predictions CSV | Notes |
|--------|--------|-----------------|-------|
| **V1** | `models/v1/` | `data/predictions/predictions.csv` | Rolling-form baseline; git tag **`v1`** |
| **V2** | `models/v2/` | `data/predictions/predictions_v2.csv` | Opponent / handedness / park; git tag **`v2`** |
| **V3** | `models/v2/` | `data/predictions/predictions_v3.csv` | Frozen **`v3`** tag — manual copy from [`mlb-prop-model-v3/`](../mlb-prop-model-v3/) |
| **Main** | `models/v2/` | `predictions_v2.csv` (same as V2) | Active **`main`** branch / daily board |

V2 and Main share one CSV today, so those columns match unless you point Main at a different file later. Missing versions show **—** in the table.

### Prep commands (quick reference)

```bash
# V2 + Main
./run_daily.sh

# V1
python predict.py --start 2026-03-25 --end 2026-08-16 --version v1

# V3 — copy from frozen snapshot
cp ../mlb-prop-model-v3/data/predictions/predictions_v2.csv \
   data/predictions/predictions_v3.csv
```

**Ranking:** Main `|edge|` → max `|edge|` across versions → max Over %. Cache busts when prediction CSV mtimes change.

---

## Version snapshots

| | V1 (frozen) | V2 (frozen) | V3 (frozen) | Active (`main`) |
|--|-------------|-------------|-------------|-----------------|
| **Location** | `mlb-prop-model-v1/` | `mlb-prop-model-v2/` | `mlb-prop-model-v3/` | `mlb-prop-model/` |
| **Git tag** | `v1` | `v2` | `v3` | — (branch **`main`** on [MON3Y](https://github.com/EZ94PHEN0M52/MON3Y)) |
| **Player features** | Rolling L3/L5/L10/L20/season | + opponent, handedness, park | + game lines, stolen bases | Same as V3 |
| **Odds pipeline** | Live props only | Live props only | Phases 1–6 (history, multi-book, movement, calibration) | Same as V3 |
| **UI** | Basic board | Basic board | Batter Score, board filters, Pick Builder, L5/L10 % | Same as V3 + fantasy-line batter score boards, **Hitter's Life** board, **official Rotowire lineups** script, **Batter Score v2**, **Batter Score Pick Builder**, ranking-table Game & time, conditional cell highlights, H2H stat history, Rotowire lineup filter |
| **Models** | `models/*.pkl` | `models/v1/`, `models/v2/` | `models/v2/` + calibrators + dist | Same as V3 |
| **Predictions** | `predictions.csv` | `predictions_v2.csv` | `predictions_v2.csv` + `_best.csv` | Same as V3 |
| **Default CLI** | N/A | `--version v2` | `--version v2` | `--version v2` |

All snapshots can run side-by-side on the same machine. Frozen folders are never modified; active development continues here on **`main`**.

**Next architecture:** PickFinder IP integration (Predictable v3.1) — see [Predictable plan](#predictable--pickfinder-ip-integration-v31-plan) and [docs/ROADMAP.md](docs/ROADMAP.md). Dual-head pitcher models and Batter Score validation are **done** on `main` (see [Changelog](#changelog)).

---

## Standalone frozen copies (optional)

If you want to run a **frozen generation** without touching this active folder, use the sibling snapshot directories. Each has its own `data/`, `models/`, and `.env`. Virtual environments are **not** copied — recreate `.venv` on first use.

### Standalone v3 setup (recommended frozen copy)

[`mlb-prop-model-v3/`](../mlb-prop-model-v3/) is the **frozen V3 snapshot** (git tag `v3`) — a self-contained folder copy sibling to active development in [`mlb-prop-model/`](../mlb-prop-model/). Use it to run the full Phases 1–6 pipeline, Batter Score, board filters, and Pick Builder without modifying the active workspace.

The snapshot includes `data/`, `models/`, and `.env`. Virtual environments are **not** copied; recreate `.venv` locally on first use:

```bash
cd /Users/edosaona-enagbare/pfinder_v1/mlb-prop-model-v3
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./run_daily.sh --streamlit
```

**What's in v3 vs v2:** Historical odds + backtesting, multi-book devig/consensus/best-price, real-line training, intraday line movement, game-line features, stolen bases prop, calibration + distributional models + CLV, Batter Score Phases A–D, board market filters + Pick Builder, player stat history with L5/L10 %.

## Standalone v2 setup (legacy baseline)

[`mlb-prop-model-v2/`](../mlb-prop-model-v2/) is the **pre–Phases 1–6 frozen snapshot** (git tag `v2`). Use only when you need the simpler pre-expansion baseline.

```bash
cd /Users/edosaona-enagbare/pfinder_v1/mlb-prop-model-v2
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./run_daily.sh --streamlit
```

---

## Initial data (one time)

Statcast is shared under `data/raw/`. If you already downloaded in V1, skip re-fetching.

**Training data (2025, used for both V1 and V2 models):**

```bash
python fetch_data.py --statcast --start 2025-04-01 --end 2025-06-30

# V1 features + models (optional; pre-trained models may exist in models/v1/)
python build_features.py --start 2025-04-01 --end 2025-06-30 --version v1
python train.py --start 2025-04-01 --end 2025-06-30 --version v1

# V2 features + models
python build_features.py --start 2025-04-01 --end 2025-06-30 --version v2
python train.py --start 2025-04-01 --end 2025-06-30 --version v2
```

**Inference data (current season, adjust end to yesterday):**

```bash
python fetch_data.py --statcast --start 2026-03-25 --end 2026-08-16

python build_features.py --start 2026-03-25 --end 2026-08-16 --version v1
python build_features.py --start 2026-03-25 --end 2026-08-16 --version v2
```

---

## Cache-first data policy (no redundant API calls)

All inference, UI, and backtest paths read **local parquets and CSVs** first. Live downloads are confined to explicit fetch scripts and the daily pipeline — never triggered implicitly by `predict.py`, backtests, or Streamlit reruns.

### Single source of truth

| Directory | Contents |
|-----------|----------|
| **`data/raw/`** | Statcast pitch-level (`statcast_{start}_{end}.parquet`), historical odds partitions (`odds/historical/date=…/`), intraday prop snapshots (`odds/snapshots/`) |
| **`data/processed/`** | Batter/pitcher feature parquets, `current_props.parquet`, `current_game_lines.parquet`, `daily_probables.parquet`, `prizepicks_fantasy_lines.parquet`, `underdog_fantasy_lines.parquet`, `rotowire_lineups.parquet` (default RHP/LHP on first Hitter's Life use; **OFFICIAL** rows from [`./run_official_lineups.sh`](#official-rotowire-lineups-pre-game)) |
| **`data/predictions/`** | `predictions_v2.csv`, `predictions_v2_best.csv` (written by `predict.py`; read by Streamlit) |
| **`data/backtest/`** | Backtest CSVs and `batter_score_validation.json` |

Downstream code must not call pybaseball, the Odds API, or the MLB Stats API directly — only read from these paths (or raise if a required file is missing).

### What hits the network vs local disk

**Live fetch scripts** (the only modules that download):

| Script | API | Writes |
|--------|-----|--------|
| `fetch_data.py --statcast` | pybaseball Statcast | `data/raw/statcast_{start}_{end}.parquet` |
| `fetch_data.py --props` | Odds API (via `odds_api.py`) + Underdog public API (fantasy lines) | `data/processed/current_props.parquet`, `prizepicks_fantasy_lines.parquet`, `underdog_fantasy_lines.parquet` + snapshot |
| `fetch_data.py --underdog-fantasy` | Underdog public API (via [`fetch_underdog_fantasy.py`](fetch_underdog_fantasy.py)) | `data/processed/underdog_fantasy_lines.parquet` |
| `fetch_data.py --game-lines` | Odds API | `data/processed/current_game_lines.parquet` |
| `fetch_data.py --probables` | MLB Stats API (via `fetch_probables.py`) | `data/processed/daily_probables.parquet` |
| `fetch_probables.py` | MLB Stats API | same probables parquet |
| `fetch_historical_odds.py` | Odds API | `data/raw/odds/historical/date=…/` |
| [`fetch_rotowire_lineups.py`](fetch_rotowire_lineups.py) | Rotowire batting-orders HTML | `rotowire_lineups.parquet` (defaults on demand; **OFFICIAL** via [`./run_official_lineups.sh`](#official-rotowire-lineups-pre-game)) |
| `odds_api.py` | Odds API | used by fetch scripts only — not imported by inference |

**Read-only paths** (parquet/CSV + models only; zero network):

| Module | Reads |
|--------|-------|
| `build_features.py` | `data/raw/statcast_*.parquet` only (no live Statcast download) |
| `predict.py` | Feature parquets, `current_props.parquet`, `current_game_lines.parquet`, model pickles |
| `scripts/backtest.py` | Feature parquets, `data/raw/odds/historical/`, models |
| `scripts/backtest_batter_score.py` | `data/processed/batter_features_*.parquet` only |
| `batter_score_data.py` | Feature parquets (game logs), `daily_probables.parquet` (SP lookup), `data/raw/statcast_*.parquet` via `@lru_cache` for Phase D arsenal — never live Statcast |
| `app.py` (Streamlit) | Predictions CSV + enrichment from the parquets above |

`build_features.py` may call `pybaseball.playerid_reverse_lookup` for ID resolution; it does **not** call `statcast()` — Statcast must already exist under `data/raw/`.

### `DISABLE_LIVE_FETCH=1` guard

Set this env var to block accidental network calls in shared helpers. Implemented in [`utils.py`](utils.py) via `require_live_fetch()` / `live_fetch_disabled()`.

**Blocked when set:**

| Module | Operation blocked |
|--------|-------------------|
| `fetch_data.py` | `--statcast`, `--props`, `--game-lines` |
| `fetch_probables.py` | MLB Stats API probables download |
| `fetch_historical_odds.py` | Historical props and game lines |
| `odds_api.py` | Any Odds API request |
| `scripts/ensure_features.py --fix` | Statcast **refresh** when raw parquet is stale or missing games through `--end` |

**Auto-set:** `scripts/backtest_batter_score.py` calls `os.environ.setdefault("DISABLE_LIVE_FETCH", "1")` at import — safe to run without remembering the flag.

**When to set:** Offline backtests, CI, re-running `predict.py` / Streamlit against cached data, or any job that must not spend API quota.

**When to unset:** Initial data download, daily pipeline (`./run_daily.sh`), manual `fetch_data.py` / `fetch_historical_odds.py` runs, or `ensure_features.py --fix` when Statcast raw needs a refresh.

```bash
# Example: prop backtest with zero API calls (historical odds + features must exist on disk)
DISABLE_LIVE_FETCH=1 python scripts/backtest.py --start 2025-04-01 --end 2025-06-30 --version v2
```

### Zero-API run patterns

These workflows assume prerequisites are already on disk (`build_features.py`, fetch scripts, or a prior `./run_daily.sh`):

```bash
# Batter Score validation — auto-sets DISABLE_LIVE_FETCH; Phase A only (game_context=None)
python scripts/backtest_batter_score.py --start 2025-04-01 --end 2025-06-30

# Prop-model backtest — historical odds + feature parquets + models
DISABLE_LIVE_FETCH=1 python scripts/backtest.py --start 2025-04-01 --end 2025-06-30 --version v2

# Predictions from cached props/features (no live fetch unless ensure_features rebuilds Statcast)
DISABLE_LIVE_FETCH=1 python predict.py --start 2026-03-25 --end 2026-08-16 --version v2

# Streamlit — reads predictions CSV + local parquets only; no API on rerun
streamlit run app.py
# Version compare: see top of README (?view=compare)
```

**Streamlit caching:** [`app.py`](app.py) wraps board enrichment in `@st.cache_data(show_spinner=False)`, keyed on the predictions CSV **mtime** (`predictions_mtime`). Widget reruns reuse the cached dataframe until `predict.py` rewrites the CSV. Batter Score / L5-L10 enrichment inside the cache uses `@lru_cache` on feature and Statcast **parquet** loads in [`batter_score_data.py`](batter_score_data.py) — not live API calls.

### Daily pipeline: one fetch per run

[`run_daily.sh`](run_daily.sh) is designed for **at most one live fetch per resource per daily pass**:

1. **`ensure_features.py --fix`** once at startup (may fetch Statcast only if raw is stale — skipped when `DISABLE_LIVE_FETCH=1` and rebuild is needed).
2. **`fetch_data.py --props`** — unless `--skip-props` → uses cached `current_props.parquet`.
3. **`fetch_data.py --game-lines`** — unless `--skip-game-lines` → cached `current_game_lines.parquet`.
4. **`fetch_data.py --probables`** — unless `--skip-probables` → cached `daily_probables.parquet`.
5. **`predict.py`** — read-only (no fetch).

Use skip flags for intraday re-predicts or when odds/probables were fetched earlier the same day. Full flag table: [Daily workflow (V2)](#daily-workflow-v2).

Historical odds backfill (`fetch_historical_odds.py`) and evaluation ([`./run_evaluation.sh`](#evaluation-pipeline-phase-6)) are **outside** `run_daily.sh` — run them separately when needed.

### Dual-head pitcher K / walks / outs (brief)

Post-v3 upgrade for count markets **`pitcher_strikeouts`**, **`pitcher_walks`**, and **`pitcher_outs`**: a lightweight **dual-head** — existing LightGBM **classifier** (beat the line?) plus a Poisson **regressor** (expected count μ). Classifier probability remains the source of truth for **edge** and **EV**; regressor outputs (`predicted_count`, `dist_over_probability`) appear on the board for context. Inference: [`predict.py`](predict.py) (`DUAL_HEAD_MARKETS` in [`distributional.py`](distributional.py)). Training: [`scripts/fit_distributional.py`](scripts/fit_distributional.py) → `models/v2/dist/{market}.pkl` (via [`./run_evaluation.sh`](#evaluation-pipeline-phase-6) or [`./run_pitcher_outs_learning.sh --fit-distributional`](#pitcher-outs-learning-loop-track-1)). A 50/50 classifier+regressor probability blend is deferred — see [docs/ROADMAP.md](docs/ROADMAP.md).

**Track 1 self-learning** for pitcher K / walks / outs (log → join outcomes → retrain) is documented in [Pitcher count-market learning loop](#pitcher-outs-learning-loop-track-1).

**Stuff K (v2)** is a **third**, independent strikeout path — Statcast SwStr/chase/velocity, not dual-head and not the main classifier. See [Stuff strikeout model (v2)](#stuff-strikeout-model-v2).

---

## Historical odds & backtesting (Phase 1)

Phase 1 adds historical sportsbook prop storage and backtesting against Statcast outcomes. This establishes a **baseline** for evaluating model edges on real book prices. See the full [Roadmap: Phases 1–6](#roadmap-phases-16) for Phases 4–6 (line movement, market expansion, model refinement); [Phases 2–3](#phase-2-multi-bookmaker-intelligence) (multi-book intelligence and real-line training) are implemented.

Historical fetch is **not** part of `run_daily.sh` — run it separately when you want more history. Each MLB event costs roughly **~120 API credits** (1 credit for the event list + 10 × markets × regions for the odds snapshot), so backfilling long windows is intentionally opt-in rather than daily.

**Fetch historical props** (paid Odds API plan required; player props history available from **2023-05-03**):

```bash
# Fetch historical props (not in run_daily.sh — run when you want more history)
python fetch_historical_odds.py --start 2025-04-01 --end 2025-06-30

# Dry-run: list dates/events without spending credits
python fetch_historical_odds.py --start 2025-04-01 --end 2025-06-30 --dry-run

# Re-fetch dates already on disk
python fetch_historical_odds.py --start 2025-04-01 --end 2025-04-01 --force
```

Output is partitioned by snapshot date:

```text
data/raw/odds/historical/date=YYYY-MM-DD/props.parquet
```

Each row matches live `current_props.parquet` schema plus `snapshot_date` and `fetched_at`. Dates already fetched are skipped unless `--force`. Quota errors stop the run without overwriting existing files.

**Backtest** (requires feature parquets, historical odds parquets, and models for the same date range). **Offline-safe:** set `DISABLE_LIVE_FETCH=1` — reads only `data/processed/` feature parquets and `data/raw/odds/historical/` (see [Cache-first data policy](#cache-first-data-policy-no-redundant-api-calls)).

```bash
# Backtest
python scripts/backtest.py --start 2025-04-01 --end 2025-06-30 --version v2

python scripts/backtest.py --start 2025-04-01 --end 2025-06-30 --version v2 --min-edge 0.03 --min-ev 0.05
python scripts/backtest.py --start 2025-04-01 --end 2025-04-03 --version v2 --market batter_hits
```

Build features first if needed:

```bash
python build_features.py --start 2025-04-01 --end 2025-06-30 --version v2
```

Detailed results: `data/backtest/backtest_{start}_{end}.csv`. Summary prints win rate, flat-bet ROI (filtered by `--min-edge` / `--min-ev`), average edge, and Brier score per market.

**API notes:** Historical player props use the event-level endpoint `/v4/historical/sports/baseball_mlb/events` (1 credit) plus `/v4/historical/sports/baseball_mlb/events/{id}/odds` (10 × markets × regions per event). Pre-game snapshots use each event's `commence_time` as the query timestamp.

---

## Phase 2: Multi-bookmaker intelligence

Phase 2 improves how the pipeline compares model probabilities to the market when multiple sportsbooks post the same prop. Part of the [Roadmap: Phases 1–6](#roadmap-phases-16) (✅ Done).

**Devigging** — When a book posts both Over and Under prices, `devig_two_way()` in [`utils.py`](utils.py) removes two-way vig and produces fair probabilities that sum to 1. **Edge** and **EV** use the devigged probability; **Market %** in the UI still shows raw implied probability for reference.

**Consensus line** — [`odds_aggregation.py`](odds_aggregation.py) groups props by `(player, market, event_id)`, computes the median line across books, and a weighted devigged Over probability (optional sharp-book weights via `SHARP_BOOK_WEIGHTS` in `utils.py`: Pinnacle, Circa, etc.).

**Best price** — For each `(player, market, line, side)`, the book with the highest EV is flagged `is_best_price=True`. Outputs:

- `data/predictions/predictions_v2.csv` — all books (enhanced with devigged edge)
- `data/predictions/predictions_v2_best.csv` — one row per player/market at best EV (deduped across books/lines/sides)

The Streamlit [main board](#streamlit-ui) always shows **one row per (player, market)** — the highest-EV book only — via [`dedupe_best_prop()`](odds_aggregation.py). There is no **All books** toggle on the board; see the [player page](#player-pages-uplayerpy) for every book/line on a market. `predictions_v2_best.csv` uses the same dedupe keys.

Backtests (`scripts/backtest.py`) apply the same devigged edge logic for consistency with live predictions.

---

## Roadmap: Phases 1–6

Post-**V2 snapshot** expansion plan for this repo (Odds API history, multi-book intelligence, and model upgrades). The **`v2` git tag** (`mlb-prop-model-v2/`) marks the baseline **before** Phases 1–6; the **`v3` git tag** (`mlb-prop-model-v3/`) marks completion of Phases 1–6 plus Batter Score and UI upgrades. Active development continues here on **`main`** in `mlb-prop-model/`. **Next:** dual-head pitcher models — see [docs/ROADMAP.md](docs/ROADMAP.md).

**Why this order:** Store and score real book prices first (Phase 1), then interpret them correctly across books (Phase 2), then teach models on those lines (Phase 3), then capture how lines move (Phase 4), expand markets only where Statcast supports outcomes (Phase 5), and finally tighten probability quality and bet ranking (Phase 6). Each phase builds on the previous without skipping evaluation.

| Phase | Status | Summary |
|-------|--------|---------|
| **1 — Historical odds + backtesting** | ✅ Done | Historical prop storage, backtest vs Statcast outcomes |
| **2 — Multi-bookmaker intelligence** | ✅ Done | Devig, consensus line, best price, board dedupe (one row per player/market) |
| **3 — Train on real book lines** | ✅ Done | Historical consensus lines in training; synthetic fallback |
| **4 — Line movement & intraday snapshots** | ✅ Done | Append-only odds history, steam/movement features |
| **5 — Expand markets strategically** | ✅ Done | Game lines as features; stolen bases player prop |
| **6 — Model refinement** | ✅ Done | Calibration, distributional models, CLV, vig-aware EV ranking |

### Phase 1 — Historical odds + backtesting ✅ Done

**Goal:** Establish a **baseline** for model edges on real sportsbook prices before changing training or adding movement features.

**Implemented:**
- `fetch_historical_odds.py` — backfill pre-game player props into `data/raw/odds/historical/date=YYYY-MM-DD/props.parquet`
- `scripts/backtest.py` — score historical props with trained models, join Statcast outcomes from feature parquets, write `data/backtest/backtest_{start}_{end}.csv`
- **Baseline metrics:** flat-bet ROI, win rate, average edge, Brier score per market (calibration proxy)

**Commands & cost:** See [Historical odds & backtesting (Phase 1)](#historical-odds--backtesting-phase-1). Historical fetch is opt-in (not in `run_daily.sh`). Each MLB event costs roughly **~120 API credits** (1 for the event list + 10 × markets × regions per odds snapshot); long backfills add up quickly on a paid Odds API plan.

### Phase 2 — Multi-bookmaker intelligence ✅ Done

**Goal:** Compare model probabilities to the market fairly when multiple books post the same prop.

**Implemented:**
- **Devigging** — `devig_two_way()` in [`utils.py`](utils.py); edge and EV use devigged probability
- **Consensus line** — median line + weighted devigged Over probability in [`odds_aggregation.py`](odds_aggregation.py)
- **Best price** — `dedupe_best_prop()` keeps highest-EV row per `(player, market)`; `is_best_price` still marks best book per `(player, market, line, side)` inside `predictions_v2.csv`; `predictions_v2_best.csv` is fully deduped
- **UI** — [Main board](#streamlit-ui) and [Top Over / Top Under](#top-over--top-under-pages-uitop_listspy) always dedupe to one best-EV row per `(player, market)`; [player pages](#player-pages-uplayerpy) show all books

**Details:** See [Phase 2: Multi-bookmaker intelligence](#phase-2-multi-bookmaker-intelligence). Dedupe logic: [`dedupe_best_prop()`](odds_aggregation.py). Backtests use the same devigged edge logic as live predictions.

### Phase 3 — Train on real book lines ✅ Done

**Goal:** Stop training only on synthetic threshold grids; learn from prices the market actually posted.

**Implemented:**
- **`training_odds.py`** — `load_historical_props`, `build_consensus_lines` (median line + weighted devigged Over probability per player/market/date), `match_props_to_features` (join to Statcast feature rows via fuzzy player match)
- **`train.py --line-source`** — `real` (consensus lines only), `synthetic` (fixed threshold grids), or `auto` (default: real when historical props exist and yield ≥100 rows per market, else synthetic)
- **Target:** `actual_stat > line` on posted consensus lines (same grading as inference/backtest)
- **Derived model inputs:** `market_implied_over_prob`, `line_vs_season_avg` (computed at train and infer time, not stored in feature parquets)
- **`run_daily.sh --train`** passes `--line-source auto`

**Commands:**

```bash
# Train on real lines when historical props exist (default via run_daily.sh --train)
python train.py --start 2025-04-01 --end 2025-06-30 --version v2 --line-source auto

# Force synthetic threshold grids (legacy behavior)
python train.py --start 2025-04-01 --end 2025-06-30 --version v2 --line-source synthetic

# Real lines only (fails if no historical props)
python train.py --start 2025-04-01 --end 2025-06-30 --version v2 --line-source real
```

Fetch historical props first if needed: `python fetch_historical_odds.py --start 2025-04-01 --end 2025-06-30`

### Phase 4 — Line movement & intraday snapshots ✅ Done

**Goal:** Capture how lines and odds change before first pitch, not just a single pre-game snapshot.

**Implemented:**
- **`odds_snapshots.py`** — `save_live_snapshot()` appends each live fetch to `data/raw/odds/snapshots/props_{YYYYMMDD_HHMMSS}.parquet` (separate from historical backfill partitions and from `current_props.parquet`, which is still overwritten each fetch)
- **`fetch_data.py --props`** — saves an append-only snapshot after writing `current_props.parquet`; ensures `fetched_at` on all rows
- **`odds_movement.py`** — `compute_movement_features()` compares current props to the day's opening snapshot (earliest fetch before `commence_time` per player/market/book/side)
- **Movement columns:** `opening_line`, `opening_odds`, `line_delta`, `odds_delta`, `steam_flag` (line moved ≥0.5 toward the opening favorite side)
- **`predict.py`** — joins movement features before enrichment; columns included in `predictions_v2.csv` and `predictions_v2_best.csv`
- **UI** — optional **Line Δ** and **Steam** (🔥) columns on the board and player pages; glossary entries in `ui/glossary.py`
- **`run_daily.sh`** — props fetch saves snapshots automatically; optional intraday cron documented in script header

**Commands:**

```bash
# Daily pipeline (saves one snapshot per props fetch)
./run_daily.sh

# Manual props fetch (also appends snapshot)
python fetch_data.py --props

# Optional intraday cron on game days (every 2–4 hours), e.g. crontab:
# 0 8,12,16,20 * * * cd /path/to/mlb-prop-model && .venv/bin/python fetch_data.py --props >> logs/props_fetch.log 2>&1

# Unit tests (mock snapshots; no models or API)
python scripts/test_odds_movement.py
```

**Snapshot path example:**

```text
data/raw/odds/snapshots/props_20260819_143052.parquet
```

Movement features require at least two snapshots on a game day (opening + later fetch). With only one fetch, opening equals current and deltas are zero. Historical backfill partitions (`data/raw/odds/historical/date=YYYY-MM-DD/`) are unchanged — snapshots are live intraday only.

**Note:** `scripts/backtest.py` does not compute movement features (intraday snapshots are not available for historical backfill windows).

### Phase 5 — Expand markets strategically ✅ Done

**Goal:** Add signal where Statcast and the Odds API overlap; avoid scope creep.

**Implemented:**
- **Game totals and run lines as model features** — `fetch_data.py --game-lines` stores live lines in `data/processed/current_game_lines.parquet`; historical backfill via `fetch_historical_odds.py --game-lines` → `data/raw/odds/historical/date=YYYY-MM-DD/game_lines.parquet`
- **Feature columns** (V2 only, joined in `features_v2.py` / `game_lines.py`): `game_total_line`, `game_run_line`, `game_implied_total_over_prob`
- **Stolen bases player prop** — `batter_stolen_bases` in Odds API fetch, Statcast rolling features, training, and inference
- **Schema bump** — `PARQUET_FEATURE_SCHEMA_VERSION="3"` in `train.py` (rebuild features after upgrade)
- **Deferred:** cross-sport, same-game parlays, F5 innings, standalone game-line betting models

**Commands:**

```bash
# Live game lines (also run by ./run_daily.sh)
python fetch_data.py --game-lines

# Historical game lines for training feature merge
python fetch_historical_odds.py --start 2025-04-01 --end 2025-06-30 --game-lines

# Rebuild features + retrain after Phase 5 upgrade
./run_daily.sh --train
```

See [Phase 5 — Expand markets strategically](#phase-5--expand-markets-strategically--done) below for feature details.

### Phase 6 — Model refinement ✅ Done

**Goal:** Improve probability quality and bet ranking after real-line training and movement data exist.

**Implemented:**
- **`calibration.py`** — isotonic regression or Platt scaling per market on held-out backtest rows; calibrators saved to `models/v2/calibrators/{market}.pkl`
- **`scripts/fit_calibrators.py`** — fit calibrators from a backtest window (or `--from-csv`); only markets with ≥100 labeled outcomes
- **`predict.py` / `prop_scoring.py`** — `raw_model_probability` and `calibrated_probability` columns; edge and EV use calibrated probs when calibrators exist (graceful fallback to raw)
- **`distributional.py`** — Poisson rate models for `batter_hits` and `pitcher_strikeouts`; P(stat > line) at any posted line; saved to `models/v2/dist/{market}.pkl`
- **`scripts/fit_distributional.py`** — train distributional rate models from feature parquets
- **`clv.py` + `scripts/backtest.py`** — closing line value when multiple historical snapshots exist (`clv`, `model_clv` columns; avg CLV in summary)
- **UI** — [Main board](#streamlit-ui) deduped to best EV per `(player, market)`, sorted by **EV**; optional **Calibrated %** column when present

**Commands:**

```bash
# Full Phase 6 evaluation (backtest → calibrators → distributional)
./run_evaluation.sh

# Or run steps individually:
python scripts/backtest.py --start 2025-04-01 --end 2025-06-30 --version v2
python scripts/fit_calibrators.py --start 2025-04-01 --end 2025-06-30 --from-csv data/backtest/backtest_2025-04-01_2025-06-30.csv
python scripts/fit_distributional.py --start 2025-04-01 --end 2025-06-30 --version v2

# Unit tests
python scripts/test_calibration.py
```

See [Evaluation pipeline (Phase 6)](#evaluation-pipeline-phase-6) for when to run and flag details.

See [Phase 6: Model refinement](#phase-6-model-refinement) below for details.

---

## Batter Score

**Status:** **Phase A ✅** (season baseline + recent form), **Phase B ✅** (SP L5 ERA + H2H in pitcher form), **Phase C ✅** (probable SP pipeline via MLB Stats API), **Phase D ✅** (pitch-type matchup grade from Statcast), **Validation backtest ✅** (Spearman gate passed 2026-08-22 — player pages show **✓ Batter Score validated**). **2026-08-19 fixes:** [team abbr mapping](#5-team-abbreviation-mapping-critical-fix), [NaN `sp_id` / TBD IDs](#6-nan-sp_id--tbd-starter-ids-2026-08-19). Spec: [`batter_score.py`](batter_score.py) · data/UI: [`batter_score_data.py`](batter_score_data.py) · board UI: [`ui/batter_score_board.py`](ui/batter_score_board.py) · Phase D: [`pitch_matchup.py`](pitch_matchup.py).

### 1. Overview

**Batter Score** is a **0–100 composite rating** for a batter's upcoming game. It blends four interpretable components:

| Component | Nominal weight | Core input |
|-----------|----------------|------------|
| **Season baseline** | 30% | Per-game **H + TB + BB** raw points, scaled to 100 (benchmark max = 6.0) |
| **Recent form** | 25% | **0.7×L5 + 0.3×L10** blend of the same H+TB+BB stat |
| **Matchup grade** | 30% | Usage-weighted **wOBA (35%) + AVG (65%)** vs each pitch bucket in the opposing SP's arsenal (Phase D) |
| **Pitcher form** | 15% | Opposing starter **ERA over last 5 starts** (letter-graded) + optional **H2H** blend (≥10 PA) |

**Relationship to LightGBM:** Batter Score is **orthogonal to the prop models** — an interpretable **UI and ranking layer** ([main board](#main-board-apppy--uiboardpy) column + [player page](#player-pages-uplayerpy) breakdown), **not** a replacement for per-market Over/Under probability, edge, or EV.

**Where it runs:** Batter Score is **computed at Streamlit load time** via `enrich_with_batter_score()` in [`batter_score_data.py`](batter_score_data.py). It is **not** stored in `predictions_v2.csv`. After fetching probables, **restart Streamlit** (or re-run [`./run_daily.sh --streamlit`](#daily-workflow-v2)) so the board picks up fresh SP data. See [Troubleshooting](#troubleshooting) if enrichment crashes or scores look stale.

**UI surfaces:**

- **[Main board](#main-board-apppy--uiboardpy)** — sortable **Batter Score** column with labels **Full** / **Partial** / **Partial · SP TBD** / **Form only** (glossary tooltip); player names show **(L)/(R)** bat/throw hand when known
- **[Main board → Top 10 batter score](#main-board-apppy--uiboardpy)** — highest Batter Score per player (respects Market type filter). Columns: Player, **Game & time**, Opposing SP **(L)/(R)**, Vs pitcher, **PP fantasy**, **UD fantasy**, **L5 / L10 %** (vs PP line), **Batter score**, **Batter score v2**. **Conditional highlights:** orange **UD fantasy** when Underdog line &lt; PrizePicks; sky blue when PP = UD; light green **Vs pitcher** when H2H AVG &gt; .300; yellow/green **L5 / L10 %** when L5 ≥ 80% with L10 below/above 80%; red row outline when UD lower + L5/L10 green
- **[Main board → Batter score by game](#main-board-apppy--uiboardpy)** (Hitter's Life) — **all** slate batters with Top 10 PP/UD/L5-L10/Vs pitcher columns and highlights, plus **Batting average** (Szn / L5 / L10) and **TB per game (L5)** with the same color rules as the [Batting average board](#hitters-life-board); **Game** selectbox above the table filters to one matchup (no **Game & time** column); **[Batter Score Pick Builder](#pick-builder-uipick_builderpy)** add controls ([`ui/batter_score_board.py`](ui/batter_score_board.py))
- **[Main board → Hot batters — batter score](#main-board-apppy--uiboardpy)** — top **20** Batter Scores among hitters in the top **15** L5 batting averages **and** a Hitter's Life batting-AVG highlight (green / orange / yellow); includes **Batting average**, **TB per game (L5)**, and PP/UD/L5-L10/Vs pitcher columns with independent color rules; tie-break favors blue TB soarer then AVG color ([`ui/main_bottom_boards.py`](ui/main_bottom_boards.py))
- **[Hitter's Life](#hitters-life-board)** (`?view=hitters_life`) — batting-context board: Vs SP (name + H2H), **Arsenal wOBA**, season/L5/L10 AVG, pitch-type wOBA selector, **SP arsenal** (Savant pitch names), TB game log; Rotowire lineup filter (**Today's Lineup** when [`./run_official_lineups.sh`](#official-rotowire-lineups-pre-game) has run, else default vs SP hand)
- **[Player page](#player-pages-uplayerpy)** — component breakdown (season baseline, recent form, matchup, pitcher form), SP ERA L5 + H2H detail, H+TB+BB last-10 Altair chart; opposing SP name shows throw hand **(L)/(R)** when known
- **[Stat history](#player-pages-uplayerpy)** — market dropdown (all batter/pitcher prop markets), **All / H2H** scope toggle (H2H = games vs today's slate opponent), **L5 / L10** window toggle, rolling averages, per-game Altair bar chart ([`ui/player_stats.py`](ui/player_stats.py))

### 2. Phases A–D (all ✅)

| Phase | Scope | Label when active | Notes |
|-------|--------|-------------------|-------|
| **A** | Season baseline + recent form | **Form only** (when SP TBD) or **Partial · SP TBD** | [`compute_batter_score_partial()`](batter_score.py); `PHASE_A_GATES` |
| **B** | + opposing SP **ERA L5** + optional **H2H** (≥10 PA) | **Partial** (matchup still gated) | [`compute_batter_score_phase_b()`](batter_score.py); `PHASE_B_GATES` |
| **C** | SP identification pipeline | Enables Phase B/D when SP known | [`fetch_probables.py`](fetch_probables.py) → `daily_probables.parquet`; [Daily workflow](#daily-workflow-v2) step 4 |
| **D** | Usage-weighted pitch-type matchup | **Full** (all four components) | [`pitch_matchup.py`](pitch_matchup.py) + [`compute_batter_score_phase_d()`](batter_score.py); `PHASE_D_GATES` |
| **D v2** | Same composite; matchup uses **Savant pitch types** (4-Seam, Sinker, Sweeper, …) not five buckets | **Full** when detailed arsenal ready | [`build_opponent_pitcher_arsenal_detailed()`](pitch_matchup.py) + [`score_batter_v2()`](batter_score_data.py); shown as **Batter score v2** column on batter score boards |

Scoring path in [`batter_score_data.py`](batter_score_data.py) `score_batter()`: Phase D when SP + Statcast arsenal ready → Phase B when SP + ERA L5 ready → Phase A otherwise. **`score_batter_v2()`** swaps in the detailed Savant arsenal for the matchup component only; season, form, and pitcher form unchanged. ERA L5 can resolve via **SP name** when `sp_id` is missing (see [NaN sp_id fix](#6-nan-sp_id--tbd-starter-ids-2026-08-19)); H2H and arsenal require a valid numeric ID.

### 3. Component gating and weight renormalization

Gated components are **excluded entirely** — never imputed with league averages or fake arsenals. Active weights are **renormalized to sum to 100%** via `renormalize_weights()` in [`batter_score.py`](batter_score.py).

**Example effective weights** (nominal 30 / 25 / 30 / 15):

| Scenario | Active components | Renormalized weights |
|----------|-------------------|----------------------|
| SP TBD, no proxy | Season + form | **54.5%** season · **45.5%** form → label **Form only** or **Partial · SP TBD** |
| SP known, no arsenal yet | Season + form + pitcher | **42.9%** season · **35.7%** form · **21.4%** pitcher → label **Partial** |
| SP + arsenal ready (Phase D) | All four | **30%** · **25%** · **30%** · **15%** (nominal) → label **Full** |

**Partial vs Full:** A **Full** score uses all four inputs. **Partial** scores omit one or more components and renormalize — they are useful for ranking within the same label but **not directly comparable** to Full scores on the same sort (see [Troubleshooting](#troubleshooting)).

**Never impute:** No invented pitch arsenals, usage %, or SP ERA for unknown starters. Optional team-level `opp_team_earned_runs_season` proxy exists (`USE_TEAM_PITCHING_PROXY=False` by default) — label **Partial · SP TBD (team proxy)** if enabled.

### 4. Starting pitcher pipeline (Phase C)

**Primary source:** [MLB Stats API](https://statsapi.mlb.com/api/v1/schedule?sportId=1&hydrate=probablePitcher) via [`fetch_probables.py`](fetch_probables.py).

**Output:** `data/processed/daily_probables.parquet` — `game_date`, `home_team`, `away_team`, `home_sp_name`, `away_sp_name`, `home_sp_id`, `away_sp_id`, `fetched_at`, `source`.

**TBD starters:** MLB often lists a probable **name** before assigning a pitcher ID — `home_sp_id` / `away_sp_id` may be **NaN** while the name column is populated. [`coerce_mlb_id()`](#6-nan-sp_id--tbd-starter-ids-2026-08-19) treats NaN as “no ID”; ERA L5 still resolves via name fallback; H2H and Phase D arsenal stay gated until a real ID exists.

**Commands:** Step 4 of [Daily workflow](#daily-workflow-v2) (`fetch_data.py --probables`); standalone options in [Command reference → fetch_probables.py](#fetch_probablespy). Use `--skip-probables` on `run_daily.sh` to reuse cached probables.

On MLB API failure the fetch logs a warning and writes an empty/partial parquet without crashing.

### 5. Team abbreviation mapping (critical fix)

Feature parquets store **`team` as abbreviations** (e.g. `SF`, `CLE`). Probables and Odds API props use **full franchise names** (e.g. `San Francisco Giants`, `Cleveland Guardians`). SP lookup failed silently when abbrs did not match — every row showed **Partial · SP TBD** even with probables populated.

**Fix:** [`utils.py`](utils.py) `TEAM_ABBR_TO_ODDS` maps Statcast/feature abbrs to Odds API names; `canonical_odds_team_key()` normalizes any input for joins. [`fetch_probables.py`](fetch_probables.py) `lookup_opposing_sp()` uses `canonical_odds_team_key()` on home, away, and batter team.

**If all Batter Score rows show Partial · SP TBD:** (1) run probables fetch (see [Daily workflow](#daily-workflow-v2) step 4), (2) verify `daily_probables.parquet` has named SPs, (3) confirm abbr mapping covers your teams in `TEAM_ABBR_TO_ODDS`, (4) restart Streamlit. See [Troubleshooting → All SP TBD](#troubleshooting).

**Other name quirks:** `MLB_TO_ODDS_TEAM` maps `Oakland Athletics` → `Athletics`; `fetch_probables.normalize_team_for_odds()` applies this on ingest.

### 6. NaN sp_id / TBD starter IDs (2026-08-19)

**Symptom:** Streamlit crashed during Batter Score enrichment with `ValueError: cannot convert float NaN to integer` — often when probables listed an SP **name** but left `sp_id` as NaN (TBD assignment).

**Cause:** Pandas/parquet stores missing MLB pitcher IDs as float NaN. Code paths that called `int(sp_id)` or joined Statcast on ID without a guard would raise.

**Fix:** [`utils.py`](utils.py) **`coerce_mlb_id()`** — returns `int | None` for valid IDs; `None` for missing, NaN, or non-numeric values. Applied in [`fetch_probables.py`](fetch_probables.py) `lookup_opposing_sp()`, [`batter_score_data.py`](batter_score_data.py) `build_batter_inputs()` / H2H, and [`pitch_matchup.py`](pitch_matchup.py) arsenal lookups.

**Behavior after fix:**

| Input | ERA L5 (Phase B) | H2H | Arsenal (Phase D) |
|-------|------------------|-----|-------------------|
| SP name + valid `sp_id` | ✅ | ✅ (≥10 PA) | ✅ when usage sums ~1.0 |
| SP name + NaN `sp_id` | ✅ via **name fallback** in `_pitcher_rows_by_sp()` | ❌ gated | ❌ gated → label **Partial** |
| No SP name | ❌ | ❌ | ❌ → **Form only** / **Partial · SP TBD** |

**Hint:** After pulling this fix, **restart Streamlit** (Ctrl+C, then [`./run_daily.sh --streamlit`](#daily-workflow-v2)) — enrichment runs at UI load, not during `predict.py`.

### 7. Phase D — pitch-type matchup

[`pitch_matchup.py`](pitch_matchup.py):

- Maps Statcast `pitch_type` codes → **Fastball / Slider / Curveball / Changeup / Other** buckets (**Batter Score v1** matchup grade)
- **Batter Score v2** and **Hitter's Life → SP arsenal** use individual **Baseball Savant** pitch names from Statcast `pitch_name` (4-Seam Fastball, Sinker, Sweeper, etc.) via `aggregate_pitcher_arsenal_usage_detailed()` / `build_opponent_pitcher_arsenal_detailed()`
- **Batter:** wOBA + AVG per bucket (v1) or per Savant pitch type (v2) from Statcast raw (balls in play)
- **SP:** arsenal usage % by bucket (v1) or Savant pitch type (v2) over **last 5 starts**
- Builds `PitchTypeMatchup` rows for `matchup_grade_index()` in [`batter_score.py`](batter_score.py)

Phase D activates when `arsenal_ready()` returns true (usage sums to ~1.0), SP ERA L5 is available, and **`sp_id` is a valid integer** (not NaN/TBD).

### 8. Validation backtest (offline)

Orthogonal validation track — does **not** affect board edge, EV, or prop-model ranking. Confirms that point-in-time Batter Score (Phase A) ranks batters vs same-game outcomes.

**Command:**

```bash
python scripts/backtest_batter_score.py --start 2025-04-01 --end 2025-06-30
python scripts/backtest_batter_score.py --start 2025-04-01 --end 2025-06-30 --min-sample 100 --min-spearman 0.15 --write-detail
```

**Prerequisite:** `build_features.py` (or `ensure_features.py --fix`) for a batter feature parquet covering the date range.

**Data policy:** Reads `data/processed/batter_features_*.parquet` only — no Statcast download, no Odds API, no MLB Stats API. Sets `DISABLE_LIVE_FETCH=1` automatically. Scores each batter-game with **`game_context=None`** so validation stays **Phase A only** (season baseline + recent form from pre-game history; no probables lookup, no SP ERA/H2H, no raw Statcast arsenal per row).

**Target outcome:** Same-game **H + TB + BB** raw points (the Batter Score input stat).

**Validation gates** (written to `data/backtest/batter_score_validation.json`; drives player-page **✓ Batter Score validated**):

| Gate | Default |
|------|---------|
| `sample_size` | ≥ 100 scored batter-games |
| `spearman_correlation` | ≥ 0.15 (primary metric — robust to non-linear 0–100 index) |

Also reports Pearson correlation and MAE on implied raw points. Optional `--write-detail` saves per-game rows to `data/backtest/batter_score_validation_detail.parquet`.

**Status (2026-08-22):** Gates **passed** on 33,148 batter-games (Spearman **0.161**) — `batter_score_validation.json` has `"validated": true`. Player pages show **✓ Batter Score validated**. See [Changelog](#2026-08-22--batter-score-validation-passed).

See [Cache-first data policy](#cache-first-data-policy-no-redundant-api-calls) for the full offline run matrix.

### 9. Risks, open items, and what's next

| Risk / item | Mitigation / status |
|-------------|---------------------|
| Small sample wOBA/AVG vs pitch type | Defaults to batter overall rates when bucket sample is thin |
| SP scratches / late changes | Re-fetch probables on game days; restart Streamlit; stale-SP UI badge still TODO |
| TBD `sp_id` with known name | ERA L5 + **Partial** label; H2H/arsenal wait for ID — see [NaN sp_id fix](#6-nan-sp_id--tbd-starter-ids-2026-08-19) |
| Doubleheaders | Same `(game_date, home, away)` join keys — `commence_time` disambiguation TODO |
| H2H noise | Gated at **MIN_PA_H2H = 10**; ERA-only below threshold |
| Not a validated edge | Validation backtest **passed** (2026-08-22) — player page shows **✓ Batter Score validated**; board edge/ranking still unchanged |
| Partial vs Full sorting | UI shows label; avoid comparing unlike labels on one sort |

**Open polish:** doubleheader `commence_time` join, stale probables badge, optional team ERA proxy validation.

**Deferred:** Phase 6 extras (negative binomial, calibrators for all 13 markets).

---

## Predictable — PickFinder IP integration (v3.1 plan)

The active workspace is evolving into **Predictable** — the same MLB prop research pipeline with planned enrichment from **PickFinder implied probability (IP)** on the main board. This section documents the **v3.1** integration plan. **Phase 1 discovery** can start from the current `main` baseline (Batter Score game filter and fantasy-line boards are already on `main`).

> **⚠️ Pre-flight — snapshot before v3.1 work**
>
> Before starting PickFinder v3.1 integration, create a new **snapshot subversion** from the current baseline (**v3.0 → v3.1 Predictable**). Tag or copy the repo as **`mlb-prop-model-v3.1`** (or similar) so v3.0 remains a clean rollback point while PickFinder fetch/join/UI work proceeds on `main`.

### Four-phase plan

| Phase | Goal | Deliverable |
|-------|------|-------------|
| **1 — Discover API** | Reverse-engineer PickFinder's authenticated prop endpoints via browser DevTools | Documented request URLs, headers, auth cookies/tokens, and response JSON shape |
| **2 — Fetch script** | Nightly (or on-demand) pull of PickFinder props | [`fetch_pickfinder.py`](fetch_pickfinder.py) → `data/processed/pickfinder_props.parquet` with columns **`player`**, **`market`**, **`line`**, **`pf_over_pct`**, **`pf_under_pct`** |
| **3 — Board join** | Enrich predictions at Streamlit load (same pattern as Batter Score) | Left join on **`(player, market, line)`** during board enrichment in [`app.py`](app.py) / [`predict.py`](predict.py) pipeline |
| **4 — UI subscripts** | Show PickFinder IP beside model Over % / Under % | Model value remains **primary**; PickFinder Over % / Under % render as **subscript** under the main cell |

### Double subscript when lines differ

When the board's posted **line ≠ PickFinder line**, retain both for analysis by showing the PickFinder line in the cell identifier, e.g. **`16.5 (PF: 17.5)`** — board line first, PickFinder line in parentheses. Subscript percentages always refer to the PickFinder line on that row; the model % stays tied to the board line.

### PickFinder ↔ internal market map

Map PickFinder display labels to this repo's `market` keys before join:

| PickFinder label | Internal `market` |
|------------------|-------------------|
| Hits | `batter_hits` |
| Home Runs | `batter_home_runs` |
| Total Bases | `batter_total_bases` |
| RBIs | `batter_rbis` |
| Runs | `batter_runs_scored` |
| Batter Walks | `batter_walks` |
| Hits + Runs + RBIs | `batter_hits_runs_rbis` |
| Stolen Bases | `batter_stolen_bases` |
| Strikeouts | `pitcher_strikeouts` |
| Pitcher Walks | `pitcher_walks` |
| Hits Allowed | `pitcher_hits_allowed` |
| Pitching Outs | `pitcher_outs` |
| Earned Runs | `pitcher_earned_runs` |

Confirm exact PickFinder strings in Phase 1 — aliases may differ slightly from the table above.

### Phase 1 — DevTools discovery (start here)

PickFinder requires an **active subscription**. Use a logged-in browser session to capture live API traffic:

1. Open [PickFinder](https://pickfinder.app/) (or your subscription URL) in **Chrome** or **Edge**.
2. Open **DevTools** → **Network** tab → enable **Preserve log**.
3. Filter by **Fetch/XHR** (or type `api` in the filter box).
4. Navigate to **MLB player props** and open a market you care about (e.g. Pitching Outs, Strikeouts).
5. Click a player row or refresh props so new requests appear.
6. Inspect each XHR request:
   - **Request URL** and query params
   - **Request headers** — especially `Authorization`, `Cookie`, custom `x-*` tokens
   - **Response** — JSON fields for player name, market, line, over/under percentages
7. **Right-click** a representative request → **Copy** → **Copy as cURL** — save to a local notes file (never commit secrets).
8. Repeat for 2–3 markets to confirm one endpoint vs per-market routes.
9. Document: base URL, auth mechanism (cookie vs bearer), rate limits, and whether lines are American odds or implied % only.

**Output of Phase 1:** a short internal doc (or README subsection update) with endpoint templates and sample response keys — enough to implement `fetch_pickfinder.py` without guessing.

---

## Phase 4: Line movement & intraday snapshots

Phase 4 captures how player prop lines move through the day on game slates. Part of the [Roadmap: Phases 1–6](#roadmap-phases-16) (✅ Done).

**Append-only snapshots** — Each `fetch_data.py --props` run writes `data/raw/odds/snapshots/props_{YYYYMMDD_HHMMSS}.parquet` in addition to overwriting `data/processed/current_props.parquet`. Snapshots are separate from Phase 1 historical partitions (`data/raw/odds/historical/date=YYYY-MM-DD/`).

**Movement features** — [`odds_movement.py`](odds_movement.py) compares the current fetch to the day's opening snapshot (earliest `fetched_at` before `commence_time` per player/market/book/side):

| Column | Description |
|--------|-------------|
| `opening_line` | Line at opening snapshot |
| `opening_odds` | American odds at opening snapshot |
| `line_delta` | Current line − opening line |
| `odds_delta` | Current odds − opening odds |
| `steam_flag` | Line moved ≥0.5 toward the opening favorite side |

**Scheduled fetches** — `./run_daily.sh` saves one snapshot per run. For intraday history, schedule additional props fetches on game days (cron example in `run_daily.sh` header).

**UI** — Streamlit board and player pages show optional **Line Δ** and **Steam** (🔥) columns when predictions include movement data.

See [Phase 4 — Line movement & intraday snapshots](#phase-4--line-movement--intraday-snapshots--done) in the roadmap for commands and snapshot paths.

---

## Phase 5: Game line context & stolen bases

Phase 5 adds game-level market context as **features** for existing player prop models and introduces the **stolen bases** batter prop. Part of the [Roadmap: Phases 1–6](#roadmap-phases-16) (✅ Done).

**Game line features** — [`game_lines.py`](game_lines.py) builds consensus totals and run lines from Odds API `totals` and `spreads` markets:

| Column | Description |
|--------|-------------|
| `game_total_line` | Median over/under total runs for the event |
| `game_run_line` | Median spread for the player's team |
| `game_implied_total_over_prob` | Weighted devigged Over probability on the game total |

Historical lines merge into feature parquets during `build_features.py --version v2`. At inference, `predict.py` attaches live lines from `current_game_lines.parquet` using each prop's event (home/away/commence time).

**Stolen bases** — Statcast `stolen_base*` events aggregated by runner; rolling features (`stolen_bases_l3` … `stolen_bases_season`) feed `batter_stolen_bases` model training and scoring.

**Schema version** — `PARQUET_FEATURE_SCHEMA_VERSION="3"`. Run `./run_daily.sh` (or `ensure_features.py --fix`) to rebuild parquets; `./run_daily.sh --train` to retrain all 13 player prop models.

**Tests** — `scripts/test_phase5.py` (consensus game lines, feature merge, stolen-base column presence).

---

## Phase 6: Model refinement

Phase 6 improves probability calibration, adds distributional count models, tracks closing line value in backtests, and ranks bets by vig-aware EV in the UI. Part of the [Roadmap: Phases 1–6](#roadmap-phases-16) (✅ Done).

**Calibration** — [`calibration.py`](calibration.py) fits isotonic regression (default) or Platt scaling per market on held-out backtest rows. Calibrators live at `models/v2/calibrators/{market}.pkl`. At inference, `predict.py` emits `raw_model_probability` and `calibrated_probability`; edge and EV use calibrated values when calibrators exist.

**Distributional models** — [`distributional.py`](distributional.py) trains Poisson rate LightGBM regressors for **hits** and **pitcher strikeouts**. At inference, P(stat > line) is derived from the predicted rate for any posted line. Models saved to `models/v2/dist/{market}.pkl`. Negative binomial and additional markets are deferred.

**CLV** — [`clv.py`](clv.py) compares bet-time devigged prices to closing devigged prices when multiple historical snapshots exist for the same prop. Backtest CSVs include `clv` (market beat-close) and `model_clv` (model edge vs close).

**UI ranking** — [Main board](#streamlit-ui) deduped to best EV per `(player, market)`, sorted by **EV** (calibrated model prob × decimal odds − 1). Optional **Calibrated %** column when predictions include calibration columns.

**Tests** — `scripts/test_calibration.py` (synthetic isotonic/Platt, Poisson over probabilities).

---

## Evaluation pipeline (Phase 6)

**Separate from [`run_daily.sh`](#daily-workflow-v2).** Use `./run_evaluation.sh` (note the **`./` prefix** — bare `run_evaluation.sh` is not on zsh PATH) to refresh backtest metrics, probability calibrators, and distributional rate models on a historical window. The script auto-activates `.venv` when present. It does **not** fetch live props, retrain the main LightGBM classifiers, or run predictions.

**When to run:**

- After [historical props](#historical-odds--backtesting-phase-1) are fetched for the evaluation window
- After `./run_daily.sh --train` (or `train.py`) so scored models exist
- Periodically (e.g. monthly) to refresh calibrators and distributional models as more labeled outcomes accumulate

**Single command** (default window matches `run_daily.sh` training dates: `2025-04-01` → `2025-06-30`):

```bash
./run_evaluation.sh
./run_evaluation.sh --start 2025-04-01 --end 2025-06-30 --version v2
./run_evaluation.sh --min-edge 0.03          # passed to backtest ROI filters
./run_evaluation.sh --help
```

**Timing:** A full backtest on the default 3-month window takes **~15 minutes**. For quick iteration, pass a shorter range (e.g. `--start 2025-04-01 --end 2025-04-03`).

**Pipeline order:**

1. **`scripts/backtest.py`** — score historical props, join Statcast outcomes, attach CLV → `data/backtest/backtest_{start}_{end}.csv`
2. **`scripts/fit_calibrators.py --from-csv …`** — fit isotonic/Platt calibrators per market from that CSV → `models/{version}/calibrators/`
3. **`scripts/fit_distributional.py`** — train Poisson rate models from feature parquets (independent of backtest) → `models/{version}/dist/`

Requires V2 feature parquets for the window (`ensure_features.py --fix` or `./run_daily.sh --train`).

**After evaluation:** run `./run_daily.sh` (or `./run_daily.sh --streamlit`) to regenerate live predictions with the new calibrators and distributional models. See [Phase 6: Model refinement](#phase-6-model-refinement) for calibration/dist details.

---

## Daily workflow (V2)

**Preferred path:** run the full pipeline from the project root with `./run_daily.sh`. No manual `source .venv/bin/activate` is required — the script sources `.venv/bin/activate` internally. It validates or rebuilds V2 feature files, fetches props, game lines, probables, and generates predictions. Fetch control and skip flags are documented in [Cache-first data policy → Daily pipeline](#daily-pipeline-one-fetch-per-run).

```bash
cd /Users/edosaona-enagbare/pfinder_v1/mlb-prop-model

./run_daily.sh              # ensure features → props → game lines → probables → predict
./run_daily.sh --streamlit  # same, then open http://localhost:8501
./run_daily.sh --train      # also verify training features and retrain models
./run_daily.sh --skip-props       # skip Odds API fetch; use cached current_props.parquet
./run_daily.sh --skip-probables   # skip MLB probables fetch; use cached daily_probables.parquet
./run_daily.sh --skip-game-lines  # skip game totals/spreads fetch
./run_daily.sh --train --streamlit
./run_daily.sh --help       # print usage
```

**What `run_daily.sh` does:**

1. **`scripts/ensure_features.py --fix`** — validates V2 batter/pitcher feature parquets for the current season (`SEASON_START=2026-03-25` → yesterday). Compares each file to columns required by `train.feature_columns_for_version()`, parquet schema fingerprint (`PARQUET_FEATURE_SCHEMA_VERSION`), and `build_features.py` / `features_v2.py` / `game_lines.py` mtimes. If anything is missing or stale, `--fix` removes old parquets, fetches Statcast when needed, and runs `build_features.py`, then re-verifies (pipeline aborts if still broken). Edits to `train.py` or `training_odds.py` (derived line features, line-source logic) do **not** trigger rebuilds. After [Phase 5](#phase-5--expand-markets-strategically--done), expect a **one-time schema rebuild** when `PARQUET_FEATURE_SCHEMA_VERSION` bumps to `"3"`.
2. **`fetch_data.py --props`** — today's sportsbook player prop lines; PrizePicks standard markets from Odds API `us_dfs`; **PrizePicks fantasy** lines → `prizepicks_fantasy_lines.parquet`; **Underdog fantasy** lines → `underdog_fantasy_lines.parquet` (Underdog public API via [`fetch_underdog_fantasy.py`](fetch_underdog_fantasy.py) — not available from Odds API). Also appends an intraday snapshot to `data/raw/odds/snapshots/` ([Phase 4](#phase-4--line-movement--intraday-snapshots--done))
3. **`fetch_data.py --game-lines`** — today's game totals and run lines → `data/processed/current_game_lines.parquet`
4. **`fetch_data.py --probables`** — today's probable starting pitchers → `data/processed/daily_probables.parquet` ([Batter Score Phase C](#batter-score))
5. **`train.py --version v2 --line-source auto`** — only with `--train`; uses `TRAIN_START=2025-04-01`, `TRAIN_END=2025-06-30`. Trains on **real book consensus lines** when historical props exist (`data/raw/odds/historical/`), otherwise falls back to synthetic thresholds. With `--train`, step 1 also re-validates and rebuilds training-window features for that range.
6. **`predict.py --version v2`** — generates `predictions_v2.csv` + `predictions_v2_best.csv` (merges live game line features at scoring time)
7. **`streamlit run app.py`** — only with `--streamlit`; [Batter Score](#batter-score) is enriched at UI load — restart after probables fetch to refresh SP-dependent scores

Dates are computed inside the script: `YESTERDAY=$(date -v-1d +%Y-%m-%d)` (macOS). Update `SEASON_START` and `TRAIN_START`/`TRAIN_END` in `run_daily.sh` when the season or training window changes.

**Skip flags summary:**

| Flag | Skips | Uses cached |
|------|-------|-------------|
| `--skip-props` | Odds API prop fetch + snapshot | `current_props.parquet` |
| `--skip-game-lines` | Game totals/spreads fetch | `current_game_lines.parquet` |
| `--skip-probables` | MLB Stats API probables fetch | `daily_probables.parquet` |

See [Command reference → run_daily.sh](#run_dailysh--pipeline-flags) for full flag details and common combinations. Evaluation ([`./run_evaluation.sh`](#evaluation-pipeline-phase-6)) is separate — not part of the daily pipeline.

---

### Manual workflow (step-by-step)

Use these steps when you want explicit control, or to understand what `run_daily.sh` / `ensure_features.py --fix` rebuild under the hood. All commands assume you are in the project folder with the virtual environment active:

```bash
cd /Users/edosaona-enagbare/pfinder_v1/mlb-prop-model
source .venv/bin/activate
```

Replace the `--end` date below with **yesterday** (never today — you need completed games for rolling stats). On macOS:

```bash
YESTERDAY=$(date -v-1d +%Y-%m-%d)
SEASON_START=2026-03-25   # Opening day; update each season
```

---

### Step 1 — Fetch today's sportsbook lines

```bash
python fetch_data.py --props
```

**What it does:** Calls The Odds API for every MLB game on today's slate and downloads current player prop lines for all [supported markets](#supported-prop-markets) (hits, HR, total bases, RBI, runs, batter walks, hits+runs+RBIs, strikeouts, pitcher walks, hits allowed, pitcher outs, earned runs) from US books (DraftKings, FanDuel, BetMGM, etc.). Also merges **PrizePicks** standard props from Odds API `us_dfs`, writes **PrizePicks fantasy score** lines to `prizepicks_fantasy_lines.parquet`, and fetches **Underdog fantasy score** lines from Underdog's public API into `underdog_fantasy_lines.parquet` (feeds [batter score boards](#batter-score)).

**When to use:** Run **first**, every time you want fresh predictions. Props move through the day; run again if you want updated lines closer to first pitch.

**Why it matters:** This is the **market side** of the model. Without this file, `predict.py` has nothing to compare your model probabilities against.

**Output:** `data/processed/current_props.parquet`, `data/processed/prizepicks_fantasy_lines.parquet`, `data/processed/underdog_fantasy_lines.parquet`, and an append-only snapshot at `data/raw/odds/snapshots/props_{YYYYMMDD_HHMMSS}.parquet`

**Notes:**
- Uses `ODDS_API_KEY` from `.env`. Each event costs one API call; expect ~10–20 calls on a full slate.
- If zero events are returned, there may be no MLB games scheduled today.
- You do **not** need to re-run this unless you want updated odds.

---

### Step 2 — Refresh current-season Statcast

```bash
python fetch_data.py --statcast --start 2026-03-25 --end 2026-08-16
```

**What it does:** Downloads pitch-by-pitch Statcast data from pybaseball for the date range and saves it as a parquet file. If the file already exists for that exact range, it skips the download and loads from disk.

**When to use:** Run **every morning** during the season so yesterday's games are included. The `--start` should be **opening day of the current season**. The `--end` should be **yesterday** (last completed game day).

**Why it matters:** Your prediction features (L3, L5, L10, season averages, opponent stats in V2) are computed from this data. Stale Statcast = stale player form = wrong probabilities.

**Output:** `data/raw/statcast_{start}_{end}.parquet`

**Notes:**
- First download of a long range can take several minutes. Subsequent runs with an extended `--end` re-download the whole range unless you use monthly chunks.
- During offseason, skip this step until games resume.
- Training data (2025) is separate — you rarely re-fetch that.

**Example with dynamic yesterday (macOS):**

```bash
python fetch_data.py --statcast --start 2026-03-25 --end $(date -v-1d +%Y-%m-%d)
```

---

### Step 3 — Build V2 feature tables

```bash
python build_features.py --start 2026-03-25 --end 2026-08-16 --version v2
```

**What it does:** Reads the Statcast parquet from Step 2 and builds:
- **Batter game logs** (hits, HR, TB, RBI, runs, walks, H+R+RBI) with rolling averages (L3/L5/L10/L20/season)
- **Pitcher game logs** (K, BB, hits allowed, outs recorded, earned runs) with the same rolling windows
- **V2 extras:** opponent team pitching/batting strength, batter stand, vs LHP/RHP splits, home park proxy

All rolling features use `shift(1)` so today's game is never included in today's features (no leakage).

**When to use:** Run **after Step 2**, every morning. Must use the **same `--start` and `--end` dates** as the Statcast file you just fetched.

**Why it matters:** `predict.py` looks up each player's **most recent row** in these files to represent "current form going into today's game."

**Output:**
- `data/processed/batter_features_v2_{start}_{end}.parquet`
- `data/processed/pitcher_features_v2_{start}_{end}.parquet`

**Notes:**
- `--version v2` is the default but include it explicitly to avoid confusion with V1 files.
- This overwrites the feature files for that date range each run (expected).
- Does **not** retrain models — only refreshes inputs to prediction.

---

### Step 4 — Generate predictions and edges

```bash
python predict.py --start 2026-03-25 --end 2026-08-16 --version v2
```

**What it does:**
1. Loads today's props (`current_props.parquet`)
2. Loads V2 feature files (same dates as Step 3)
3. For each sportsbook line, fuzzy-matches the player name to Statcast
4. Runs the trained V2 LightGBM model for that market with the sportsbook's line
5. Computes **model probability**, **Over %**, **Under %**, **market implied probability** (raw), **devigged market probability**, **edge** (devigged), **consensus line/edge**, **best book/EV**, **EV**, and **line movement** (`opening_line`, `line_delta`, `odds_delta`, `steam_flag` when intraday snapshots exist)
6. Saves ranked results (`predictions_v2.csv` all books + `predictions_v2_best.csv` best price)

**When to use:** Run **after Steps 1 and 3**. This is the step that produces your betting board.

**Why it matters:** This is where model meets market. Edge = model probability minus devigged market probability (when Over/Under pairs exist). Positive edge means the model thinks the bet is better than the fair price suggests (not a guarantee of profit).

**Output:** `data/predictions/predictions_v2.csv` and `data/predictions/predictions_v2_best.csv`

**Notes:**
- `--start` / `--end` must match the feature files from Step 3.
- Requires trained models in `models/v2/`. If missing, run `train.py --version v2` (weekly/monthly, not daily).
- Unmatched player names are skipped silently (Odds API name vs Statcast name mismatch).

---

### Step 5 — Launch the Streamlit app

```bash
streamlit run app.py
# or: ./run_daily.sh --streamlit
```

**What it does:** Starts a local web UI at **http://localhost:8501** that reads the predictions CSV and provides the [main board](#streamlit-ui), [player detail pages](#player-pages-uplayerpy), [Top Over/Under lists](#top-over--top-under-pages-uitop_listspy), and live [Batter Score](#batter-score) enrichment.

**When to use:** Run **after Step 4** (or `./run_daily.sh --streamlit`) whenever you want to browse picks visually. Restart after re-running `predict.py` to see new numbers; **restart after probables fetch** to refresh Batter Score SP-dependent components.

**Why it matters:** Easier than reading raw CSV. Use the sidebar to switch between **V2** and **V1** predictions if both CSVs exist.

**Output:** Browser UI (no new files)

**Notes:**
- Stop with **Ctrl+C** in the terminal.
- If you see "No predictions found", run Step 4 first (or check sidebar version matches the CSV you generated).
- Shortcut without activate: `.venv/bin/streamlit run app.py`

See [Streamlit UI](#streamlit-ui) for board dedupe, filters, player pages, L5/L10 %, and charts.

---

### Adding a new feature

When you extend the model with new rolling stats or V2-only columns, keep the build and training column lists in sync, then let the daily pipeline rebuild feature parquets automatically.

**Workflow:**

1. **`build_features.py`** — Add the base stat to `batter_stats` or `pitcher_stats` in `build_all_features()` (rolling L3/L5/L10/L20/season columns are generated from these lists). For V2-only inputs (opponent stats, handedness, park proxy), add columns in [`features_v2.py`](features_v2.py) instead.
2. **`train.py`** (and `features_v2.py` for V2 extras) — Add matching column names to `BATTER_FEATURES` / `PITCHER_FEATURES`, or to `BATTER_FEATURES_V2_EXTRA` / `PITCHER_FEATURES_V2_EXTRA` for V2-only fields. `train.py` consumes V2 extras via `feature_columns_for_version("v2")`.
3. **`train.py`** — Bump `PARQUET_FEATURE_SCHEMA_VERSION` so existing parquets are treated as stale (only when parquet column lists change; derived model inputs do not require a bump).
4. **Rebuild** — Run `./run_daily.sh`. Its first step calls `scripts/ensure_features.py --fix` automatically. If columns drifted, you should see output like:

   ```text
   New model features detected (walks_l3, walks_l5, ...), rebuilding...
   ```

   `--fix` removes stale parquets, fetches Statcast when the raw file is missing, runs `build_features.py`, and re-verifies (the pipeline aborts if validation still fails).

**After adding features, retrain models** so LightGBM sees the new inputs:

```bash
./run_daily.sh --train
```

To rebuild features without the full daily pipeline:

```bash
python scripts/ensure_features.py --start 2026-03-25 --end $(date -v-1d +%Y-%m-%d) --version v2 --fix
```

See [Feature validation (`ensure_features.py`)](#feature-validation-scriptsensure_featurespy) below for a short summary, or the full write-up in [Command reference → ensure_features.py](#ensure_featurespy--flags-and-fix).

---

### Feature validation (`scripts/ensure_features.py`)

Quick summary — **`ensure_features.py --fix`** is the pipeline’s “make sure player feature files are ready for predict” step. It runs automatically at the start of [`./run_daily.sh`](#run_dailysh--pipeline-flags).

```bash
python scripts/ensure_features.py --start YYYY-MM-DD --end YYYY-MM-DD [--version v1|v2]
python scripts/ensure_features.py --start YYYY-MM-DD --end YYYY-MM-DD --version v2 --fix
```

| Flag | Required | Meaning |
|------|----------|---------|
| `--start` | Yes | First date of the feature window (usually season opening day) |
| `--end` | Yes | Last date to cover — use **yesterday**, not today |
| `--version` | No | `v2` (default in daily pipeline) or `v1` |
| `--fix` | No | If checks fail, delete stale parquets, refresh Statcast when needed, rebuild features, re-verify |

Without `--fix`: prints problems and exits **1** ([`run_daily.sh`](#run_dailysh--pipeline-flags) aborts). With `--fix`: repairs when possible; aborts only if validation still fails after rebuild.

**Full documentation:** [Command reference → ensure_features.py](#ensure_featurespy--flags-and-fix) (checks, rebuild triggers, examples, what it does *not* do).

**Example (current season):**

```bash
YESTERDAY=$(date -v-1d +%Y-%m-%d)
python scripts/ensure_features.py --start 2026-03-25 --end $YESTERDAY --version v2 --fix
```

**Manual equivalent of `./run_daily.sh` (copy-paste):**

```bash
cd /Users/edosaona-enagbare/pfinder_v1/mlb-prop-model
source .venv/bin/activate

SEASON_START=2026-03-25
YESTERDAY=$(date -v-1d +%Y-%m-%d)

python scripts/ensure_features.py --start $SEASON_START --end $YESTERDAY --version v2 --fix
python fetch_data.py --props
python fetch_data.py --game-lines
python fetch_data.py --probables
python predict.py --start $SEASON_START --end $YESTERDAY --version v2
streamlit run app.py
```

### What you do NOT run daily

| Command | When |
|---------|------|
| `train.py --version v2` | After adding features, new training data, or weekly/monthly refresh — use `./run_daily.sh --train` |
| `fetch_data.py --statcast` for 2025 | Only for initial model training setup |
| `fetch_historical_odds.py` | Separate backfill; not daily — see [Historical odds & backtesting](#historical-odds--backtesting) |
| `./run_evaluation.sh` | After historical props + training; refreshes calibrators and distributional models — see [Evaluation pipeline (Phase 6)](#evaluation-pipeline-phase-6) |
| `scripts/backtest.py` | After historical props + features exist for the window (or use `./run_evaluation.sh`) |
| `build_features.py --version v1` | Only if you want to refresh V1 comparison predictions |
| Manual Statcast/feature steps | Usually unnecessary; `run_daily.sh` runs `ensure_features.py --fix` first |

---

Use the sidebar in Streamlit to switch between **V2** and **V1** predictions.

**Run V1 from this repo** (without using the frozen folder):

```bash
python predict.py --start 2026-03-25 --end 2026-08-16 --version v1
```

Requires V1 feature files and models in `models/v1/`.

---

## Supported prop markets

All markets below are fetched from The Odds API (`odds_api.py` `PROP_MARKETS`), trained in `train.py`, and scored in `predict.py` / `prop_scoring.py` (`MODEL_MAP`).

**UI note:** `batter_home_runs` and `batter_stolen_bases` are still trained and written to prediction CSVs, but are **hidden from Streamlit** via `EXCLUDED_UI_MARKETS` in [`ui/market_filters.py`](ui/market_filters.py) (board, player pages, top lists, version compare, Pick Builder).

| Odds API market | Display name | Model file | Role | UI |
|-----------------|--------------|------------|------|-----|
| `batter_hits` | Hits | `batter_hits.pkl` | Batter | shown |
| `batter_home_runs` | Home Runs | `batter_home_runs.pkl` | Batter | hidden |
| `batter_total_bases` | Total Bases | `batter_total_bases.pkl` | Batter | shown |
| `batter_rbis` | RBIs | `batter_rbi.pkl` | Batter | shown |
| `batter_runs_scored` | Runs | `batter_runs.pkl` | Batter | shown |
| `batter_walks` | Walks | `batter_walks.pkl` | Batter | shown |
| `batter_hits_runs_rbis` | Hits + Runs + RBIs | `batter_hits_runs_rbis.pkl` | Batter | shown |
| `batter_stolen_bases` | Stolen Bases | `batter_stolen_bases.pkl` | Batter | hidden |
| `pitcher_strikeouts` | Strikeouts | `pitcher_strikeouts.pkl` | Pitcher | shown |
| `pitcher_walks` | Walks | `pitcher_walks.pkl` | Pitcher | shown |
| `pitcher_hits_allowed` | Hits Allowed | `pitcher_hits_allowed.pkl` | Pitcher | shown |
| `pitcher_outs` | Pitcher Outs | `pitcher_outs.pkl` | Pitcher | shown |
| `pitcher_earned_runs` | Earned Runs | `pitcher_earned_runs.pkl` | Pitcher | shown |

**Training lines** (`train.py`):

- **Default (`--line-source auto`):** consensus historical book lines from `data/raw/odds/historical/date=YYYY-MM-DD/props.parquet` when available; synthetic threshold grids otherwise
- **Synthetic fallback thresholds** (used when no historical props, or `--line-source synthetic`):

- **Batters:** hits (0.5–2.5), home runs (0.5–1.5), total bases (0.5–4.5), RBI (0.5–2.5), runs (0.5–1.5), walks (0.5–1.5), hits+runs+RBIs (0.5–3.5), stolen bases (0.5–1.5)
- **Pitchers:** strikeouts (2.5–8.5), walks (0.5–2.5), hits allowed (0.5–4.5), outs (14.5–20.5), earned runs (1.5–4.5)

---

## Streamlit UI

Launch with `streamlit run app.py` or [`./run_daily.sh --streamlit`](#daily-workflow-v2). Routing uses query params: `?player=Name` for player pages, `?view=top_over` / `?view=top_under` for full ranked lists, `?view=hitters_life` for the [Hitter's Life board](#hitters-life-board), `?view=compare` for version compare.

### Pick Builder (`ui/pick_builder.py`)

Session favorites slip (Pickfinder-style, **no export**). Picks live in `st.session_state` for the current browser session only.

- **Sidebar — Prop Pick Builder:** Always visible below the model version selector — pick count badge; each pick card shows **Player**, **Market**, **Side**, **Line**, **Over % / Under %** (picked side bold), **Edge %**, **Game** (teams + commence time ET), and **Batter Score** on batter markets (with partial label when applicable); **Remove** per pick, **Clear all**
- **Sidebar — Batter Score Pick Builder:** Separate slip for batters from **Batter score by game**; duplicate key `(player, game)`; pick cards use the same PP/UD/L5-L10/Vs pitcher **color highlights** as the board ([`ui/batter_score_highlights.py`](ui/batter_score_highlights.py))
- **Duplicate prevention:** Same `(player, market, side, line, book)` cannot be added twice to the prop slip
- **Main board:** **Add to Pick Builder** expander below the filtered table — multiselect visible rows, **Add selected**, or **Add top EV**
- **Batter score by game:** **Add selected** / **Add top batter score** controls ([`ui/batter_score_board.py`](ui/batter_score_board.py))
- **Player page:** Per-market row selector + **Add** / **Add best EV** below each market table

### Hitter's Life board

Dedicated batting-context page at **`?view=hitters_life`** (link **Hitter's Life** on the main board next to Top Over / Top Under). Code: [`ui/hitters_life_page.py`](ui/hitters_life_page.py), [`ui/hitters_life_board.py`](ui/hitters_life_board.py), [`hitters_life_data.py`](hitters_life_data.py).

- **Columns:** Player, **Game & time**, **Vs pitcher** (SP name + H2H hits/AB or SP ERA L5), **Arsenal wOBA** (usage-weighted vs SP mix), **Batting average** (Szn / L5 / L10), **wOBA vs {pitch type}** (selectbox: Fastball, Slider, …), **SP arsenal** (Savant pitch names, usage order), **TB per game** (last 5 games, space-separated)
- **Highlights:** light green when season AVG &gt; .300 or H2H AVG &gt; .300; light green TB log when every game is non-zero ([`ui/hitters_life_highlights.py`](ui/hitters_life_highlights.py))
- **Game filter:** same pattern as batter score by game
- **Lineup filter** (when one game selected): prefers Rotowire **Today's Lineup** when [`./run_official_lineups.sh`](#official-rotowire-lineups-pre-game) has cached **OFFICIAL** rows; otherwise **default vs opposing SP hand** ([`fetch_rotowire_lineups.py`](fetch_rotowire_lineups.py) → `data/processed/rotowire_lineups.parquet`); orders batters 1–9 per team
- Respects **Market type** filter only (not Edge / EV)

### Main board (`app.py` → `ui/board.py`)

The board always shows **one row per (player, market)** — the book with the highest **EV** — via [`dedupe_best_prop()`](odds_aggregation.py) in [`apply_top_level_filters()`](ui/board.py). There is **no All books toggle** on the board. `predictions_v2.csv` still contains every book; dedupe happens at render time (and in `predictions_v2_best.csv` at write time).

**Hint:** To compare prices across books for one player/market, open the **[player page](#player-pages-uplayerpy)** — it lists all books/lines per market with consensus line, devigged %, and best book/EV columns.

- **Sidebar:** V1 / V2 model version selector; **[Prop Pick Builder](#pick-builder-uipick_builderpy)** and **[Batter Score Pick Builder](#pick-builder-uipick_builderpy)** favorites slips
- **Market type** (always visible): full-width multiselect at the top — limits prop categories (Hits, **Batter Walks** / **Pitcher Walks** as distinct labels via [`ui/market_filters.py`](ui/market_filters.py)); synced with Top Over / Under previews. Home runs and stolen bases are excluded from the UI (`EXCLUDED_UI_MARKETS`).
- **Filters & columns** popover: minimum Edge / EV sliders and **Show columns** visibility
- **Active filter chips:** when any filter is on, a summary row shows **Active filters:** with each constraint as a chip and a **Clear all filters** button
- **Filter by column** expander: labeled filters in a 3-column grid for every table column except Market — player text search, game/book multiselect, side, line/odds ranges, min Over % / Under % / Model % / Market % / Devigged % / L5–L10 % / Edge / Consensus Edge / EV / Best EV, etc. Active filters show **subscript indices** (e.g. Line₂) matching the chip order
- **Column header sort buttons:** click a header to sort (up to **3 columns** — first click descending, second click ascending, third click removes; subscript ₁₂₃ shows sort priority). **Clear sort** resets to EV descending. Headers also show filter subscripts when a column filter is active
- **AND logic:** Market type, min Edge, min EV, and every column filter combine with **AND** — a row must pass all active filters
- **Summary metrics:** Prop count, best edge, best EV, unique players (reflect Market / Edge / EV filters)
- **Top Over / Top Under previews:** Top 10 by model Over % / Under % (same Market / Edge / EV filters as the board); columns include **Player** (link, **(L)/(R)** hand when known), **Game & time**, market, book, line, side, Over/Under %, **L5 / L10 %**, Edge; links to full lists, **[Hitter's Life](#hitters-life-board)**, and **[Version compare](#version-compare-v1--v2--v3--main)**
- **Top 10 batter score** — highest Batter Score among batters on the slate (best row per player; respects Market type filter; independent of Edge / EV filters). See [Batter Score → UI surfaces](#batter-score) for PP/UD fantasy columns, **Batter score v2**, and conditional cell highlights
- **Batter score by game** (Hitter's Life) — all slate batters with Top 10 fantasy/score columns plus **Batting average** and **TB per game (L5)** (Hitter's Life color rules); **Game** selectbox filters to one matchup; **Batter Score Pick Builder** add controls
- **Hot batters — batter score** — top **20** batter scores among elite L5 AVG hitters with batting-board highlights; **Top props by market** table sits above it ([`ui/main_bottom_boards.py`](ui/main_bottom_boards.py))
- **Columns:** Player (link, **(L)/(R)** when known), game, market, book, side, line, odds, Over %, Under %, Model %, Market %, Devigged %, L5 / L10 %, **[Batter Score](#batter-score)** (Full / Partial · SP TBD / Partial / Form only), **Pred #** and **Dist Over %** on pitcher K/walks/outs (dual-head), **Stuff K (v2)** on pitcher strikeouts only (Statcast stuff model — separate from Model % / Edge), Edge %, Consensus Edge %, Best Book, Best EV %, EV %, Line Δ, Steam
- **L5 / L10 %:** Share of the player's last 5 / 10 completed games where the stat strictly exceeded the posted line (from feature parquets via [`ui/player_stats.py`](ui/player_stats.py))

Board enrichment is **`@st.cache_data`** in [`app.py`](app.py), keyed on predictions CSV **mtime** — see [Cache-first → Streamlit caching](#zero-api-run-patterns).

See also: [Phase 2 dedupe](#phase-2-multi-bookmaker-intelligence) · [Batter Score](#batter-score) · [Version compare](#version-compare-v1--v2--v3--main)

### Player pages (`ui/player.py`)

Open by clicking a player name on the board (`?player=...`).

- Game, first-pitch time, **[Batter Score](#batter-score)** breakdown (season baseline, recent form, matchup, pitcher form; SP ERA L5 / H2H when known; opposing SP **(L)/(R)** throw hand), **✓ Batter Score validated** badge when [validation backtest](#8-validation-backtest-offline) gates pass, best edge / EV / prop count / market count
- **Stat history** — market dropdown (all batter or pitcher markets on the player's slate), **All / H2H** scope toggle (H2H limits game logs to matchups vs today's opponent on the slate), **L5 / L10** segmented toggle, rolling averages, Altair bar chart for last N games ([`ui/player_stats.py`](ui/player_stats.py)); caption **Game logs through YYYY-MM-DD** from max `game_date` in feature parquets
- Batter Score section: H+TB+BB last-10 Altair chart (composite input stat)
- **Per-market sections (all books):** every book/line for that player and market, plus consensus line caption and last-10-games chart for that market — this is where multi-book comparison lives (not on the main board)
- **Pick Builder:** row selector + **Add** / **Add best EV** under each market table
- Over %, Under %, Model %, Market %, Devigged %, Edge %, Consensus Edge %, Best Book, Best EV %, EV %, Line Δ, Steam per row

### Top Over / Top Under pages (`ui/top_lists.py`)

Same dedupe rule as the [main board](#main-board-apppy--uiboardpy): one best-EV row per `(player, market)` via [`dedupe_best_prop()`](odds_aggregation.py). See [Phase 2 dedupe](#phase-2-multi-bookmaker-intelligence).

- **Top Over %** (`?view=top_over`): full table ranked by Over %; columns match board previews — **Player**, **Game & time**, market, book, line, side, Over %, Under %, **L5 / L10 %**, Edge
- **Top Under %** (`?view=top_under`): same, ranked by Under %
- Market / Edge / EV filters via [`apply_top_level_filters()`](ui/board.py) (uses [`ui/market_filters.py`](ui/market_filters.py) for distinct walk labels)

Board previews share the main board's Market / Edge / EV session state. Full list pages use independent filter state.

### Version compare (`ui/version_compare.py`)

Side-by-side **Over %** / **Under %** for V1, V2, V3, and Main — see [Quick start → Version Compare](#open-streamlit-and-version-compare) and [Version compare (reference)](#version-compare-v1--v2--v3--main).

---

## Command reference

Detailed flags, when to use them, and how each script fits the daily pipeline. See also [Shell scripts — quick walkthrough](#shell-scripts--quick-walkthrough) for order-of-operations and [Daily workflow (V2)](#daily-workflow-v2) for the ordered step list.

### run_daily.sh — pipeline flags

Main entry point for the V2 daily loop. Activates `.venv` automatically — no manual `source` needed.

```bash
./run_daily.sh [--train] [--skip-props] [--skip-game-lines] [--skip-probables] [--streamlit] [--port N]
./run_daily.sh --help
```

**What it runs (in order):**

| Step | Script | Skipped by |
|------|--------|------------|
| 1 | `scripts/ensure_features.py --fix` | Never — always runs |
| 2 | `fetch_data.py --props` | `--skip-props` |
| 3 | `fetch_data.py --game-lines` | `--skip-game-lines` |
| 4 | `fetch_data.py --probables` | `--skip-probables` |
| 5 | `train.py` (optional) | Only with `--train` |
| 6 | `predict.py` | Never |
| 7 | `streamlit run app.py` | Only with `--streamlit` |

Date context: `SEASON_START=2026-03-25`, `YESTERDAY=$(date -v-1d +%Y-%m-%d)` (macOS). Feature validation and predict both use **season start → yesterday** — not today’s date.

#### Flags

| Flag | What it does | When to use |
|------|--------------|-------------|
| *(none)* | Full pipeline: ensure features → props → game lines → probables → predict | Normal morning run before or after lineups |
| `--train` | Also runs `train.py` on a fixed 2025 window before predict | First setup, after feature/schema changes, or periodic retrain — **not** needed daily |
| `--skip-props` | Skips Odds API prop fetch; uses `data/processed/current_props.parquet` | Odds API quota exhausted; games already started and you want pre-game lines; re-predict without burning credits |
| `--skip-game-lines` | Skips totals/spreads fetch; uses `data/processed/current_game_lines.parquet` | Re-run predict with cached game context |
| `--skip-probables` | Skips MLB Stats API probables; uses `data/processed/daily_probables.parquet` | SP list unchanged; save API calls on repeat runs |
| `--streamlit` | Launches Streamlit after predict | Daily board workflow |
| `--port N` | Streamlit port (default `8501`) | Multiple apps or port conflict — use with `--streamlit` |

Skip flags are **independent** — combine as needed (e.g. all three skips + `--streamlit` for a fully offline re-predict from cache).

#### Common combinations

| Command | Use case |
|---------|----------|
| `./run_daily.sh --streamlit` | **Evening** — props just posted (~8pm PT); full fetch + board for tomorrow |
| `./run_daily.sh --skip-props --streamlit` | **Morning** — refresh Statcast/form; keep last night’s pre-game lines |
| `./run_daily.sh` | Standard full refresh (any time) |
| `./run_daily.sh --skip-props --skip-probables --streamlit` | Minimal API usage — only feature check may hit Statcast if stale |
| `./run_daily.sh --train --streamlit` | Full retrain + predict + UI (infrequent) |
| `./run_daily.sh --skip-props --skip-game-lines --skip-probables` | Offline predict only (requires valid cache + features) |

**After games start:** A fresh `--props` fetch may pull live-game lines and overwrite pre-game cache. Use `--skip-props` to keep yesterday’s pre-game snapshot. Step 1 (`ensure_features --fix`) may still download Statcast if feature parquets stop before yesterday — see [ensure_features.py](#ensure_featurespy--flags-and-fix) below.

**Quota / credits:** On `OUT_OF_USAGE_CREDITS`, `fetch_data.py --props` preserves existing `current_props.parquet` when non-empty. Run with `--skip-props` to continue the pipeline.

### run_evaluation.sh

See [Evaluation pipeline (Phase 6)](#evaluation-pipeline-phase-6) for timing, prerequisites, and post-steps.

```bash
./run_evaluation.sh [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--version v1|v2] [--min-edge N] [--min-ev N]
./run_evaluation.sh --help
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--start` / `--end` | Script defaults | Backtest / calibration window |
| `--version` | `v2` | Model version directory |
| `--min-edge` / `--min-ev` | Script defaults | Filter thresholds for evaluation output |

Must be invoked as `./run_evaluation.sh` (not bare `run_evaluation.sh` on zsh). Auto-activates `.venv`. **Not** part of the daily pipeline — run separately when fitting distributional models or running backtests.

### fetch_data.py — live data flags

Single script for all live fetches. Each flag is independent; pass only what you need.

```bash
python fetch_data.py --props
python fetch_data.py --game-lines
python fetch_data.py --probables
python fetch_data.py --underdog-fantasy
python fetch_data.py --statcast --start YYYY-MM-DD --end YYYY-MM-DD [--force]
python fetch_data.py --props --game-lines --probables   # combine flags
```

| Flag | Requires | Output | API / source |
|------|----------|--------|--------------|
| `--props` | — | `data/processed/current_props.parquet` (+ optional snapshot under `data/raw/odds/snapshots/`) | Odds API — player props across books; merges **PrizePicks** standard markets from `us_dfs` when available; also writes `prizepicks_fantasy_lines.parquet` and fetches `underdog_fantasy_lines.parquet` (Underdog API) |
| `--underdog-fantasy` | — | `data/processed/underdog_fantasy_lines.parquet` | Underdog public pick'em API ([`fetch_underdog_fantasy.py`](fetch_underdog_fantasy.py)) — standalone refresh without a full `--props` run |
| `--game-lines` | — | `data/processed/current_game_lines.parquet` | Odds API — game totals and run lines for today’s slate |
| `--probables` | — | `data/processed/daily_probables.parquet` | MLB Stats API — probable starting pitchers |
| `--statcast` | `--start`, `--end` | `data/raw/statcast_{start}_{end}.parquet` | pybaseball → Baseball Savant pitch-level data |
| `--force` | With `--statcast` | Re-downloads Statcast even when cache exists or when cached max date is before `--end` | Use after late-posted box scores or when `ensure_features --fix` reports stale data |

**Notes:**

- `--props` on zero rows: preserves non-empty cache and exits non-zero (quota / no games).
- Underdog fantasy fetch runs at the end of every successful `--props` pass; failures print a **WARNING** and leave the previous `underdog_fantasy_lines.parquet` in place.
- Probables use **US Eastern** schedule dates (aligned with slate `game_date` from commence times) — avoids UTC date mismatch that caused mass **SP TBD** on the batter score board.
- Statcast is **not** run on every daily pass — only when [`ensure_features.py --fix`](#ensure_featurespy--flags-and-fix) detects missing or incomplete raw data.

### fetch_probables.py

```bash
python fetch_probables.py [--date YYYY-MM-DD]
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--date` | Today (Eastern) | Schedule date for probables lookup |

Standalone probable-SP fetch (also invoked by `fetch_data.py --probables`). Output feeds [Batter Score Phase C](#4-starting-pitcher-pipeline-phase-c).

When loading the board, `ensure_probables_for_props_slate()` can auto-fetch missing slate dates in the background. Terminal may print SP coverage warnings if probables are partial — the board still loads with **Partial · SP TBD** where SP data is missing (no blocking UI banner).

Use `--skip-probables` on [`run_daily.sh`](#run_dailysh--pipeline-flags) to reuse cached probables. If everyone shows **SP TBD**, see [team abbr mapping](#5-team-abbreviation-mapping-critical-fix) and [Troubleshooting](#troubleshooting).

### fetch_historical_odds.py

```bash
python fetch_historical_odds.py --start YYYY-MM-DD --end YYYY-MM-DD [--markets m1,m2] [--force] [--dry-run]
python fetch_historical_odds.py --start YYYY-MM-DD --end YYYY-MM-DD --game-lines [--force] [--dry-run]
```

| Flag | Meaning |
|------|---------|
| `--start` / `--end` | Date range (required) |
| `--markets` | Comma-separated prop market keys (props mode only) |
| `--game-lines` | Fetch game lines instead of player props |
| `--force` | Re-download even if parquet exists |
| `--dry-run` | Print plan without writing |

Writes `data/raw/odds/historical/date=YYYY-MM-DD/props.parquet` or `.../game_lines.parquet`. See [Historical odds & backtesting](#historical-odds--backtesting).

### scripts/backtest.py

```bash
python scripts/backtest.py --start YYYY-MM-DD --end YYYY-MM-DD [--version v2] [--min-edge 0.03] [--min-ev 0.05] [--market batter_hits]
```

| Flag | Meaning |
|------|---------|
| `--start` / `--end` | Backtest window |
| `--version` | `v1` or `v2` models |
| `--min-edge` / `--min-ev` | Bet filters |
| `--market` | Single market key (optional) |

Scores historical props with trained models, joins Statcast outcomes from feature parquets, writes `data/backtest/backtest_{start}_{end}.csv`. Offline-safe with `DISABLE_LIVE_FETCH=1` when historical odds and features exist on disk.

### scripts/backtest_batter_score.py

```bash
python scripts/backtest_batter_score.py --start YYYY-MM-DD --end YYYY-MM-DD [--min-sample 100] [--min-spearman 0.15]
```

| Flag | Meaning |
|------|---------|
| `--start` / `--end` | Validation window |
| `--min-sample` | Minimum paired rows |
| `--min-spearman` | Pass threshold for rank correlation |

Validates Batter Score vs same-game H+TB+BB outcomes; writes `data/backtest/batter_score_validation.json` (drives the player-page **✓ Batter Score validated** flag). **Offline-only** — see [§8 Validation backtest](#8-validation-backtest-offline).

### build_features.py

```bash
python build_features.py --start YYYY-MM-DD --end YYYY-MM-DD [--version v1|v2]
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--start` / `--end` | Required | Statcast window → rolling features per player per game |
| `--version` | `v2` | Feature column set (`features_v2.py` for v2) |

Normally invoked by `ensure_features.py --fix`, not run manually. Rebuilds `data/processed/batter_features_v2_{start}_{end}.parquet` and `pitcher_features_v2_{start}_{end}.parquet` (L3/L5/L10/L20/season rolls, opponent strength, etc.).

**Pitcher parquet extras (Stuff K v2):** when [`build_features.py`](build_features.py) runs, it also merges Statcast **stuff** metrics via [`pitcher_stuff.py`](pitcher_stuff.py) — SwStr%, CSW%, chase%, whiff%, velocity (L3/L5/L10/L20/season) and `batters_faced`. These columns are **not** part of main `PITCHER_FEATURES` in [`train.py`](train.py); they feed only [`pitcher_strikeout_stuff.py`](pitcher_strikeout_stuff.py). After upgrading stuff logic, run [`./run_pitcher_strikeout_stuff.sh`](#run_pitcher_strikeout_stuffsh--stuff-k-v2-pipeline) or at minimum `build_features.py` + `fit_pitcher_strikeout_stuff.py`.

### fit_pitcher_strikeout_stuff.py — train Stuff K (v2) model

```bash
python scripts/fit_pitcher_strikeout_stuff.py --start YYYY-MM-DD --end YYYY-MM-DD [--version v2]
```

Trains Ridge regression **K% ~ SwStr% + chase% + velocity** (L5 rolls) on pitcher-game rows with 50+ pitches; saves `models/v2/pitcher_strikeouts_stuff.pkl`. Usually invoked by [`./run_pitcher_strikeout_stuff.sh`](#run_pitcher_strikeout_stuffsh--stuff-k-v2-pipeline), not run alone unless debugging.

### ensure_features.py — flags and `--fix`

**`ensure_features.py --fix` is the pipeline’s “make sure player feature files are ready for predict” step.** It checks whether batter/pitcher feature parquets are complete and up to date; if not, it downloads Statcast and rebuilds them. [`run_daily.sh`](#run_dailysh--pipeline-flags) always runs this with `--fix` at step 1.

```bash
python scripts/ensure_features.py --start YYYY-MM-DD --end YYYY-MM-DD [--version v1|v2]
python scripts/ensure_features.py --start YYYY-MM-DD --end YYYY-MM-DD --version v2 --fix
```

| Flag | Required | Meaning |
|------|----------|---------|
| `--start` | Yes | Window start (season opening day in daily pipeline) |
| `--end` | Yes | Window end — **yesterday**, not today |
| `--version` | No | `v2` (default in pipeline) or `v1` |
| `--fix` | No | Repair: delete stale parquets, refresh Statcast if needed, rebuild, re-verify |

#### What it checks (without `--fix`)

For the date window `run_daily.sh` passes (**season start → yesterday**), it validates two files:

- `data/processed/batter_features_v2_{start}_{end}.parquet`
- `data/processed/pitcher_features_v2_{start}_{end}.parquet`

It **fails** (exit code 1) if any of these are true:

| Trigger | Meaning |
|---------|---------|
| **Missing file** | First run, or parquets were deleted |
| **Missing columns** | `train.py` expects feature columns that aren’t in the parquet |
| **Stale schema** | Feature schema version bumped (`PARQUET_FEATURE_SCHEMA_VERSION` in `train.py` — column lists changed) |
| **Stale source code** | `build_features.py`, `features_v2.py` (v2), or `game_lines.py` edited after the parquet was built |
| **Stale data** | Parquet’s latest `game_date` is behind `--end` (yesterday’s games not in features yet) |

If issues are found **without** `--fix`, it prints the problems and exits — [`run_daily.sh`](#run_dailysh--pipeline-flags) aborts.

Rebuilds are **not** triggered by edits to `train.py` training logic, `training_odds.py`, or derived-only model inputs (`market_implied_over_prob`, `line_vs_season_avg`). When adding parquet columns, follow [Adding a new feature](#adding-a-new-feature) (sync lists, bump schema version, then `--fix`).

#### What `--fix` actually does

When something is wrong:

1. **Deletes** stale batter/pitcher feature parquets
2. **Optionally re-downloads Statcast** via `fetch_data.py --statcast` (pybaseball → Baseball Savant) if raw data is missing or doesn’t cover through `--end`
3. **Rebuilds features** via `build_features.py` → new rolling stats (L5/L10, opponent strength, etc.) per player per game
4. **Re-checks** that the new parquets pass validation

If everything is already OK, it prints **`Feature check OK`** and does nothing — no Statcast download, no rebuild. Most days this is a quick check with zero API calls.

#### What those parquets are used for

Everything downstream depends on them:

| Consumer | Uses features for |
|----------|-------------------|
| **`predict.py`** | Match props to players, build model inputs |
| **Batter score board** | Game logs, SP ERA L5, H2H vs pitcher, PP/UD fantasy lines |
| **L5 / L10 % columns** | Recent game history vs lines (board + batter score L5/L10 vs PP fantasy line) |
| **Player pages** | Stat history, game logs |

No valid features → players get skipped at predict time or show no batter score (e.g. recent call-ups with zero Statcast history).

#### Concrete example: Sunday morning after Saturday games

Scenario: `./run_daily.sh --skip-props --streamlit` on Sunday morning after Saturday night games.

1. `run_daily.sh` sets `YESTERDAY` to Saturday
2. Feature parquets were last built Friday → latest game in parquet is Friday
3. `ensure_features` sees: *“feature data stops before requested end date”*
4. **`--fix` runs:**
   - Fetches Statcast through Saturday (`statcast_2026-03-25_2026-08-21.parquet` updated)
   - Rebuilds batter/pitcher features including Saturday box scores
5. `predict.py` can score today’s slate with fresh rolling form

**Without `--fix`:** pipeline would fail at step 1, or you’d predict using stale L5/L10 and outdated opponent stats.

#### Other useful cases

| Situation | Why `--fix` helps |
|-----------|-------------------|
| **Recent call-up** (e.g. new Yankee) | After a few MLB games, Statcast has data; `--fix` pulls it in so name matching + batter score can work (still needs ≥10 games for full batter score) |
| **Feature code change** | Edited `features_v2.py` to add a rolling column; stale parquets rebuild on next daily run |
| **Fresh clone / new machine** | No parquets exist; `--fix` does initial Statcast download + feature build (can take a while) |
| **Schema bump** | Project updates `PARQUET_FEATURE_SCHEMA_VERSION`; old parquets are invalid and get rebuilt |
| **Early-morning run** | Baseball Savant may not have posted yesterday yet; re-run later or `fetch_data.py --statcast --force` then `--fix` |
| **Evening re-run after early fetch** | If the morning pass built parquets while Statcast still stopped at **two days ago**, an evening `./run_daily.sh` used to print **Feature check OK** and skip re-fetch (fixed 2026-08-30 — see [changelog](#2026-08-30--batter-score-by-game-columns-hot-batters-board-tb-log-colors)). Verify max `game_date` in the parquet filename's end date matches **Game logs through** on the player page |

#### What it does **not** do

- Does **not** fetch odds/props (Odds API)
- Does **not** fetch probables (MLB Stats API)
- Does **not** retrain models (unless you pass `--train` on `run_daily.sh`)
- Does **not** run heavy work on every `--skip-props` day if parquets are already current

**Summary:** `--fix` keeps the **Statcast → feature parquets** layer healthy so predict and batter score have fresh player history.

### train.py

```bash
python train.py --start YYYY-MM-DD --end YYYY-MM-DD [--version v1|v2] [--line-source auto|real|synthetic]
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--start` / `--end` | Required | Training window (2025 dates for initial models) |
| `--version` | `v2` | Output directory `models/v1/` or `models/v2/` |
| `--line-source` | `auto` | `auto` = real historical lines when available, else synthetic grids; `real` / `synthetic` force one mode |

Trains one LightGBM model per market (see [Supported prop markets](#supported-prop-markets)). Adds derived inputs `market_implied_over_prob` and `line_vs_season_avg` at train/infer time.

### predict.py

```bash
python predict.py [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--version v1|v2]
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--start` | `2026-03-25` | Feature window start (must match built parquets) |
| `--end` | Script default | Feature window end |
| `--version` | `v2` | Model + feature version |

Requires `data/processed/current_props.parquet` from `--props`. Writes `predictions_v2.csv` (and `predictions_v2_best.csv`). Output columns include `over_probability`, `under_probability`, `model_probability`, `edge`, `ev`, `consensus_line`, `best_book`, `best_odds`, `steam_flag`, etc. Optional `dist_over_probability` and `predicted_rate` for hits/strikeouts when distributional models exist. Optional `stuff_predicted_count` / `stuff_over_probability` on **pitcher_strikeouts** rows when `models/v2/pitcher_strikeouts_stuff.pkl` exists (board **Stuff K (v2)**). Batter score is **not** in the CSV — computed at Streamlit load in `app.py`.

### app.py

```bash
streamlit run app.py [--server.port 8501]
```

Reads `predictions.csv` (V1) or `predictions_v2.csv` (V2) from sidebar selection. Enrichment is cache-first — see [Cache-first data policy → Streamlit caching](#zero-api-run-patterns). See [Streamlit UI](#streamlit-ui).

**Batter score boards (V2):** **Top 10 batter score** shows **PP fantasy**, **UD fantasy**, **Game & time**, **L5 / L10 %** vs the PP line, and conditional cell highlights. **Batter score by game** (Hitter's Life) adds **Batting average** and **TB per game (L5)** with Hitter's Life color rules and omits **Game & time** (use the Game filter instead). **Hot batters — batter score** on the main board lists up to **20** qualified hitters — see [Batter Score → UI surfaces](#batter-score). These columns are **not** on the main prop table.

### run_official_lineups.sh — pre-game lineups

Fetches Rotowire **Today's Lineup** for slate teams and merges **OFFICIAL** rows into `rotowire_lineups.parquet`. See [Official Rotowire lineups (pre-game)](#official-rotowire-lineups-pre-game).

```bash
./run_official_lineups.sh [--dry-run] [--teams ABBR,...] [--watch SECONDS] [--max-runs N] [--help]
# Python flags pass through, e.g.:
./run_official_lineups.sh --dry-run --min-players 8 --skip-slate-check
python scripts/update_official_lineups.py --help
```

| Flag | Meaning |
|------|---------|
| *(none)* | All teams from `current_props.parquet`; 2 fetch retries per team |
| `--dry-run` | Validate only; no parquet write |
| `--teams` | Comma-separated Rotowire codes (bypass props lookup) |
| `--watch SECONDS` | Poll loop (min 30s interval); optional `--max-runs N` |
| `--min-players` / `--max-players` | Lineup size bounds (default 8–10) |
| `--min-slate-overlap` | Min batters matching prop slate (default 3; `0` or `--skip-slate-check` disables) |
| `--no-backup` | Skip timestamped backup before write |

**Requires:** `.venv`, live network (`DISABLE_LIVE_FETCH` must be unset), and usually `current_props.parquet` from `./run_daily.sh`. **After:** reload Streamlit for Hitter's Life lineup filter.

### run_pitcher_strikeout_stuff.sh — Stuff K (v2) pipeline

One-shot wrapper: **features → stuff model fit → predict**. Independent of main `pitcher_strikeouts.pkl` and Track 1 learning.

```bash
./run_pitcher_strikeout_stuff.sh [--skip-features] [--skip-fit] [--skip-predict] [--streamlit] [--start DATE] [--end DATE]
./run_pitcher_strikeout_stuff.sh --help
```

| Step | Script |
|------|--------|
| 1 | `scripts/ensure_features.py --fix` — fetch Statcast if needed, then `build_features.py` |
| 2 | `scripts/fit_pitcher_strikeout_stuff.py` → `models/v2/pitcher_strikeouts_stuff.pkl` |
| 3 | `predict.py` — `stuff_*` fields on strikeout rows; board **Stuff K (v2)** column |

| Flag | Meaning |
|------|---------|
| `--skip-features` | Parquet already has stuff columns (e.g. after `./run_daily.sh`) |
| `--skip-fit` | Model file already exists; re-predict only |
| `--skip-predict` | Features + fit only |
| `--start` / `--end` | Override default season start → yesterday |
| `--streamlit` | Open board when done |

**Typical use:** first-time setup or after [`pitcher_stuff.py`](pitcher_stuff.py) fixes. Daily board refresh stays on `./run_daily.sh` — it scores Stuff K v2 when the `.pkl` exists.

See [Stuff strikeout model (v2)](#stuff-strikeout-model-v2).

### run_pitcher_outs_learning.sh — Track 1 self-learning loop

Orchestrates the full **pitcher_strikeouts / pitcher_walks / pitcher_outs** collect → join → retrain → re-predict cycle. See [Pitcher count-market learning loop (Track 1)](#pitcher-outs-learning-loop-track-1) for context.

```bash
./run_pitcher_outs_learning.sh [--fit-distributional] [--streamlit] [--market MARKET] [--skip-daily|--skip-join|--skip-retrain|--skip-repredict]
./run_pitcher_outs_learning.sh --help
```

| Flag | Meaning |
|------|---------|
| *(default)* | All four steps for **all three** count markets |
| `--market` | Run join/retrain/dist for one market only (e.g. `pitcher_walks`) |
| `--skip-props` (via daily) | Default — uses cached `current_props.parquet` |
| `--fetch-props` | Refresh props before predict |
| `--fit-distributional` | Also train Poisson regressor → **Pred #** / **Dist Over %** on board for outs |
| `--skip-daily` / `--skip-join` / `--skip-retrain` / `--skip-repredict` | Run subset of steps |
| `--join-start` / `--join-end` | Outcome join window (default season start → yesterday) |
| `--train-start` / `--train-end` | Classifier retrain window (default season start → yesterday) |
| `--streamlit` | Launch board after pipeline |

---

## V2 feature additions

Implemented in [`features_v2.py`](features_v2.py) (`build_all_features_v2`):

**Batters**
- Opponent team pitching: K, BB, hits allowed, ER (season rolling, lagged)
- Batter stand (L/R via `batter_stand_L`)
- Hits vs LHP / RHP season rates (`hits_vs_lhp_season`, `hits_vs_rhp_season`)
- Home park offense proxy (`park_home_hits_season`, `park_home_tb_season`)

**Pitchers**
- Opponent team batting: hits, TB, HR, RBI, runs (season rolling, lagged)
- Opponent team batter K rate (`opp_team_batter_k_rate_season`)
- Pitcher throwing hand (L/R via `pitcher_throws_L`)

**Pitcher stat columns (V1 base + V2):** strikeouts, walks, hits allowed, **outs**, **earned runs** — with L3/L5/L10/L20/season rolling windows.

**Stuff columns (extra parquet only, Stuff K v2):** SwStr%, CSW%, chase%, whiff%, avg/max velocity — same rolling windows; plus per-game `pitches`, `batters_faced`. Built by [`pitcher_stuff.py`](pitcher_stuff.py); consumed by [`pitcher_strikeout_stuff.py`](pitcher_strikeout_stuff.py), **not** by main `PITCHER_FEATURES` LightGBM training.

Column lists: `BATTER_FEATURES_V2_EXTRA` and `PITCHER_FEATURES_V2_EXTRA` in `features_v2.py`; consumed by `train.py` via `feature_columns_for_version("v2")`.

---

## Project layout

```text
mlb-prop-model/
├── run_daily.sh           # Daily pipeline: ensure_features → props → game lines → probables → predict [→ streamlit]
├── run_official_lineups.sh     # Pre-game: Rotowire Today's Lineup → rotowire_lineups.parquet (OFFICIAL rows)
├── run_pitcher_outs_learning.sh  # Track 1: log → join outcomes → retrain outs → re-predict
├── run_pitcher_strikeout_stuff.sh  # Stuff K v2: features → stuff model → predict
├── run_evaluation.sh      # Phase 6: backtest → calibrators → distributional (not daily)
├── docs/
│   └── ROADMAP.md         # Predictable v3.1 + future tracks
├── fetch_data.py          # --props | --game-lines | --probables | --underdog-fantasy | --statcast
├── fetch_underdog_fantasy.py  # Underdog batter fantasy-point lines → underdog_fantasy_lines.parquet
├── fetch_rotowire_lineups.py  # Rotowire lineups: default (vs RHP/LHP) + OFFICIAL (Today's Lineup)
├── hitters_life_data.py       # Hitter's Life board data (AVG, wOBA, arsenal, TB log)
├── fetch_probables.py     # MLB Stats API probable starting pitchers
├── batter_score.py        # Batter Score composite (Phases A–D implemented)
├── pitch_matchup.py       # Phase D pitch-type arsenal + batter vs pitch stats
├── batter_score_data.py   # Feature-parquet loading + board enrichment
├── fetch_historical_odds.py  # Historical props backfill (--start --end)
├── odds_api.py            # Shared Odds API helpers (live + historical)
├── odds_snapshots.py      # Append-only intraday props snapshots (Phase 4)
├── odds_movement.py       # Line/odds movement and steam features (Phase 4)
├── odds_aggregation.py    # Devig, consensus line, best-price aggregation
├── training_odds.py       # Historical props → training rows (Phase 3)
├── calibration.py         # Isotonic/Platt calibration (Phase 6)
├── distributional.py      # Poisson rate models; DUAL_HEAD_MARKETS (K, walks, outs)
├── learning_log.py        # Track 1 append-only prediction/outcome logs
├── clv.py                 # Closing line value for backtests (Phase 6)
├── prop_scoring.py        # Shared model scoring (predict + backtest)
├── game_lines.py          # Game totals/spreads consensus (Phase 5)
├── build_features.py      # --start --end [--version v1|v2]; merges Statcast stuff into pitcher parquet
├── pitcher_stuff.py       # SwStr / chase / velocity from pitch-level Statcast
├── pitcher_strikeout_stuff.py  # Separate Stuff K (v2) fit + score (Ridge K% → Poisson)
├── features_v2.py         # V2-only feature logic (outs, ER, opponent, handedness, park)
├── train.py               # Per-market LightGBM training
├── predict.py             # Props + features → predictions CSV
├── app.py                 # Streamlit entry (routes to ui/)
├── utils.py               # Paths, odds math, coerce_mlb_id, TEAM_ABBR_TO_ODDS
├── scripts/
│   ├── ensure_features.py # --start --end [--version v1|v2] [--fix]
│   ├── log_outcomes.py    # Join predictions_log → actual outs (Track 1)
│   ├── retrain_market.py  # Single-market retrain (pitcher_outs)
│   ├── test_learning_log.py
│   ├── backtest.py        # Historical props vs model edges + outcomes + CLV
│   ├── fit_calibrators.py # Fit per-market probability calibrators (Phase 6)
│   ├── fit_distributional.py  # Train Poisson rate models (Phase 6)
│   ├── fit_pitcher_strikeout_stuff.py  # Train Stuff K (v2) Ridge model
│   ├── test_pitcher_strikeout_stuff.py # Stuff metrics + model unit tests
│   ├── test_batter_score.py   # Batter Score unit tests + board styling
│   ├── test_underdog_fantasy.py  # Underdog fantasy API parse tests
│   ├── test_hitters_life.py      # Hitter's Life + Savant arsenal + official lineup tests
│   ├── test_main_bottom_boards.py  # Hot batters + market top props board tests
│   ├── update_official_lineups.py  # CLI for Rotowire Today's Lineup fetch/merge
│   ├── test_player_stats.py   # L5/L10, PP/UD fantasy lookup tests
│   ├── test_calibration.py    # Unit tests for Phase 6 calibration/dist
│   ├── test_odds_movement.py  # Unit tests for Phase 4 snapshots/movement
│   └── test_phase5.py         # Game lines + stolen bases tests
├── ui/
│   ├── board.py           # Main prop board, filters, top Over/Under previews
│   ├── main_bottom_boards.py  # Hot batters + top props by market (bottom of main board)
│   ├── batter_score_board.py  # Top 10 + by-game batter score tables (v1 + v2 columns)
│   ├── batter_score_highlights.py  # Shared PP/UD/L5/L10/Vs pitcher styling (board + pick cards)
│   ├── hitters_life_page.py   # Hitter's Life page shell (?view=hitters_life)
│   ├── hitters_life_board.py  # Hitter's Life table, game + lineup filters
│   ├── hitters_life_highlights.py  # AVG / H2H / TB log cell colors
│   ├── market_filters.py  # Shared market multiselect (distinct walk labels)
│   ├── pick_builder.py    # Session favorites slip (Pick Builder)
│   ├── player.py          # Player detail pages + stat history (All/H2H)
│   ├── top_lists.py       # Full Top Over % / Top Under % pages
│   ├── player_stats.py    # L5/L10 %, PP/UD fantasy lines, game-log loading
│   ├── batter_score.py    # Batter Score player-page UI
│   ├── formatting.py      # Display helpers, game/time, hand labels, styling
│   ├── version_compare.py # V1/V2/V3/Main side-by-side board
│   └── glossary.py        # Tooltips and market labels
├── models/
│   ├── v1/                # V1 models (*.pkl per market)
│   └── v2/                # V2 models (*.pkl per market, calibrators/, dist/)
├── data/
│   ├── raw/               # statcast_{start}_{end}.parquet, odds/historical/, odds/snapshots/
│   ├── processed/         # Features, current_props.parquet, daily_probables.parquet
│   ├── backtest/          # backtest_{start}_{end}.csv
│   ├── learning/          # predictions_log.parquet, outcomes_log.parquet (Track 1)
│   └── predictions/       # predictions.csv, predictions_v2.csv, predictions_v2_best.csv
├── ../mlb-prop-model-v1/  # Frozen V1 copy (do not edit)
├── ../mlb-prop-model-v2/  # Frozen V2 snapshot (tag v2)
└── ../mlb-prop-model-v3/  # Frozen V3 snapshot (tag v3)
```

---

## Git

Local git repo with annotated tags for frozen snapshots. Remote: **[EZ94PHEN0M52/MON3Y](https://github.com/EZ94PHEN0M52/MON3Y)** on GitHub (`origin`).

| Tag | Snapshot folder | Commit (approx.) | Contents |
|-----|-----------------|------------------|----------|
| **`v1`** | `mlb-prop-model-v1/` | `c4c9f8e` | Pre–V2 rolling-form baseline |
| **`v2`** | `mlb-prop-model-v2/` | `fec6236` | Pre–Phases 1–6 baseline |
| **`v3`** | `mlb-prop-model-v3/` | `3de111a` | Phases 1–6 + Batter Score A–D + board filters + Pick Builder |

Active development continues on **`main`** in `mlb-prop-model/`. Post-v3 work (dual-head K/walks/outs, Track 1 outs learning, validated Batter Score, board/fantasy-line UX) lands on **`main`** and is documented in [Changelog](#changelog) and [docs/ROADMAP.md](docs/ROADMAP.md).

```bash
git remote -v
git status
git log --oneline
git tag -l                  # v1, v2, v3
git show v3 --no-patch      # v3 tag metadata
git push origin main        # push active branch (when ready)
git push origin --tags      # push all version tags
```

Never commit `.env`, `data/`, or `models/` (see `.gitignore`).

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `Unsupported version` | Use `--version v1` or `--version v2` |
| V2 models missing | Run `train.py --version v2` or [`./run_daily.sh --train`](#daily-workflow-v2) |
| V2 features missing | Run `build_features.py --version v2` or `scripts/ensure_features.py ... --fix` |
| Feature columns missing / stale (e.g. `walks_l*`) | See [Adding a new feature](#adding-a-new-feature): `./run_daily.sh` runs `ensure_features.py --fix` first; or manually: `python scripts/ensure_features.py --start ... --end ... --version v2 --fix` |
| Feature schema rebuild every run after Phase 5 | **Expected once** when `PARQUET_FEATURE_SCHEMA_VERSION` bumped to `"3"` (game line columns + stolen bases). Let `ensure_features.py --fix` rebuild; subsequent daily runs should be fast |
| `ensure_features` exits 1 without `--fix` | Expected when parquets are missing or columns drifted; re-run with `--fix` |
| `ensure_features` still fails after `--fix` | Follow [Adding a new feature](#adding-a-new-feature): confirm `BATTER_FEATURES` / `PITCHER_FEATURES` match `batter_stats` / `pitcher_stats` in `build_features.py`, V2 extras match `features_v2.py`, and `PARQUET_FEATURE_SCHEMA_VERSION` was bumped |
| V1 predictions empty in app | Run `predict.py --version v1` |
| "No predictions found" in Streamlit | Run `predict.py` (or `./run_daily.sh`) for the sidebar version |
| **UD fantasy** shows **—** on batter score board | Re-run `python fetch_data.py --props` (or `python fetch_data.py --underdog-fantasy`). Underdog lines come from [`fetch_underdog_fantasy.py`](fetch_underdog_fantasy.py), not Odds API. Check terminal for **WARNING: Underdog fantasy fetch failed**. Player name must fuzzy-match Underdog's `full_name` |
| L5/L10 % or player charts show "—" | Rebuild V2 features; player name must match feature parquet `player_name` |
| Player stats / L5–L10 stuck one day behind (e.g. last game 8/28 on 8/30) | `./run_daily.sh` uses **yesterday** as `--end` (`date -v-1d` on macOS) — **today's** box scores never appear until tomorrow's run. If **Game logs through** is more than one day behind **yesterday**, Statcast or features are stale: the raw file may be named `statcast_*_{end}.parquet` while max `game_date` inside is still earlier (common after an early-morning fetch before Savant posts). **`ensure_features.py --fix`** compares feature max to the **MLB schedule date** required for `--end` and re-fetches when behind (fixed 2026-08-30: previously it only compared features to stale Statcast and could report **Feature check OK** while skipping refresh). Manual: `python fetch_data.py --statcast --start 2026-03-25 --end $(date -v-1d +%Y-%m-%d) --force` then re-run `./run_daily.sh`. After rebuild, [`ui/player_stats.py`](ui/player_stats.py) **auto-reloads** when feature parquet mtime changes. Player page **Game logs through YYYY-MM-DD** should match max `game_date` in `data/processed/*_features_v2_*.parquet` |
| Duplicate **Walk** in market filters / pitcher walks missing | Fixed in [`ui/market_filters.py`](ui/market_filters.py): multiselect options are market **keys** with distinct labels **Batter Walks** / **Pitcher Walks** from [`ui/glossary.py`](ui/glossary.py) `MARKET_LABELS`. Table display uses `market_label()` via [`ui/formatting.py`](ui/formatting.py). Pitcher walks may have few rows but filter correctly when selected |
| Hitter's Life lineup filter shows **default** not **Today's Lineup** | Lineups not posted yet on Rotowire, or script not run. Run `./run_official_lineups.sh` (or `--watch 300`); reload Streamlit. Check terminal for `skipped: empty lineup`. See [Official lineups](#official-rotowire-lineups-pre-game) |
| Unknown option in `run_daily.sh` | Use `--train`, `--skip-props`, `--skip-game-lines`, `--skip-probables`, `--streamlit`, or `--help` |
| Streamlit crash: `ValueError: cannot convert float NaN to integer` during Batter Score | Probable has SP **name** but **NaN `sp_id`** (TBD). Fixed by [`coerce_mlb_id()`](#6-nan-sp_id--tbd-starter-ids-2026-08-19) in [`utils.py`](utils.py). Pull latest code, then **restart Streamlit** — score is enriched at UI load, not in `predictions_v2.csv` |
| Batter Score shows **Form only** for everyone | Run probables fetch ([Daily workflow](#daily-workflow-v2) step 4, or `fetch_data.py --probables`); check `data/processed/daily_probables.parquet`; **restart Streamlit** after fetch — see [Batter Score → Where it runs](#batter-score) |
| **All SP TBD** / every row **Partial · SP TBD** even when probables show names | **Common fix (2026-08-22):** UTC vs Eastern date mismatch — re-run probables without `--skip-probables`. Also verify (1) `daily_probables.parquet` exists, (2) [`TEAM_ABBR_TO_ODDS`](#5-team-abbreviation-mapping-critical-fix) team mapping, (3) restart Streamlit. See [Changelog → Probables Eastern date](#2026-08-22--probables-eastern-date-sp-coverage--player-matching) |
| **Luis Garcia Jr.** / suffix players missing from board | Fixed: Jr./Sr. stripped in [`utils.py`](utils.py). Re-run `predict.py`; player must exist in feature parquets |
| **Brett Bateman** / recent call-ups missing | No Statcast history — run `ensure_features.py --fix` after games accumulate; batter score needs ≥10 games |
| **Pitcher outs learning** — empty predictions log | Run `predict.py` first; logging is automatic for `pitcher_strikeouts`, `pitcher_walks`, and `pitcher_outs`. Check `data/learning/predictions_log.parquet` |
| **Pred # / Dist Over %** blank for Pitcher K / Walks / Outs | `./run_pitcher_outs_learning.sh --fit-distributional` then re-predict. Edge still from classifier |
| **Stuff K (v2)** shows **—** on all strikeout rows | Run `./run_pitcher_strikeout_stuff.sh` once to create `models/v2/pitcher_strikeouts_stuff.pkl`. `./run_daily.sh` only scores stuff if that file exists. If model exists but column still empty, re-run `predict.py` |
| **Stuff K (v2)** all **—** after feature rebuild | Pitcher parquet missing stuff columns — run `./run_pitcher_strikeout_stuff.sh --skip-fit` or full script. Check parquet has `swstr_pct_l5`, `chase_pct_l5`, etc. |
| **`fit_pitcher_strikeout_stuff.py` — no training rows** | Usually fixed chase columns (see [Changelog → Stuff K v2](#2026-08-29--stuff-strikeout-model-v2-statcast-swstrchasevelocity)). Re-run `./run_pitcher_strikeout_stuff.sh` (or `ensure_features.py --fix` + fit) |
| **`FileNotFoundError: statcast_*`** running stuff script | Fixed: `./run_pitcher_strikeout_stuff.sh` now uses `ensure_features.py --fix` (fetches Statcast like `./run_daily.sh`). If fetch fails (no games posted yet), pass `--end YYYY-MM-DD` for the latest date you have raw data for |
| **Pitcher count-market learning** — zero outcome joins | Log must cover completed dates; feature parquets need box scores. Adjust `--join-start` / `--join-end` on `log_outcomes.py` |
| Batter Score **Partial** vs **Full** — scores look incomparable | **By design:** **Full** = all four components (season, form, matchup, pitcher). **Partial** / **Partial · SP TBD** / **Form only** omit gated components and **renormalize** weights — rank within the same label, not across labels. See [Component gating](#3-component-gating-and-weight-renormalization) |
| Batter Score unchanged after probables update | Computed at **Streamlit load**, not written to CSV. Restart after `./run_daily.sh` or probables fetch ([Where it runs](#batter-score)) |
| LightGBM libomp error | `brew install libomp` |
| API key error | Set `ODDS_API_KEY` in `.env` |
| `OUT_OF_USAGE_CREDITS` / quota exhausted | Odds API monthly credits used up. If `data/processed/current_props.parquet` exists, fetch keeps the cache and exits non-zero. Run `./run_daily.sh --skip-props` to skip the fetch and use cached props. |
| Zero MLB events from `--props` | No games scheduled today, or API key/quota issue |
| Pipeline or Streamlit accidentally suspended (**Ctrl+Z**) | Use **Ctrl+C** to stop cleanly. If suspended, run `jobs` then `kill %1` (or relevant job number) before re-running `./run_daily.sh`. Do not suspend mid `ensure_features.py --fix` — partial parquets may corrupt |
| Version compare shows all **—** | Run [Quick start → Prepare all versions](#prepare-all-versions-for-version-compare); at minimum `./run_daily.sh` for V2/Main. Copy V3 CSV from `mlb-prop-model-v3/`. Run V1 `predict.py` when `models/v1/` exists |
| Version compare **Generate missing predictions** does nothing | CSV may already exist, or `models/v1/` / `models/v2/` is empty — check **Version sources** expander on compare page |
| Range filter shows "All values: X" instead of slider | Column has only one unique numeric value — expected; filter is a no-op until lines vary |

See also: [Operational note — suspended jobs](#operational-note--suspended-jobs-and-long-running-pipeline-steps) in the changelog.

---

## Official Rotowire lineups (pre-game)

Rotowire publishes two lineup types on each team's [batting orders](https://www.rotowire.com/baseball/batting-orders.php) page:

| Rotowire block | When available | Used by |
|----------------|----------------|---------|
| **Default vs. RHP / vs. LHP** | Always (projected platoon order) | Auto-cached the first time you open [Hitter's Life → Lineup filter](#hitters-life-board) |
| **Today's Lineup** | Close to first pitch (confirmed order) | Cached by **`./run_official_lineups.sh`** — board **prefers** this when present |

**Scope:** Display-only — reorders/filters batters on the **Hitter's Life** board. Does **not** change `predictions_v2.csv`, edge, EV, or Batter Score math.

### When to run

| Timing | Command |
|--------|---------|
| After evening/morning `./run_daily.sh` | Ensures `current_props.parquet` lists slate teams |
| **~1–2 hours before first pitch** | `./run_official_lineups.sh` |
| Lineups not posted yet | `./run_official_lineups.sh --watch 300` (poll every 5 min) or re-run manually |

Then **reload Streamlit** (or refresh the browser). The lineup popover shows **Today's Lineup** vs **default vs {SP hand}** per team.

### Quick commands

```bash
./run_official_lineups.sh                    # all slate teams from current_props.parquet
./run_official_lineups.sh --dry-run            # fetch + validate, no write
./run_official_lineups.sh --teams NYY,BOS,LAD  # specific Rotowire codes only
./run_official_lineups.sh --watch 300 --max-runs 12   # poll up to 12 times
python scripts/update_official_lineups.py --help      # full Python CLI flags
```

### What gets written

**File:** `data/processed/rotowire_lineups.parquet`

| Column | Meaning |
|--------|---------|
| `team_abbr` | Rotowire team code (e.g. `NYY`, `LAD`) |
| `vs_hand` | `RHP` / `LHP` (defaults) or **`OFFICIAL`** (Today's Lineup) |
| `slot` | Batting order 1–9 |
| `player_name` | Rotowire display name |
| `fetched_at` | UTC ISO timestamp |

Official updates **replace OFFICIAL rows only** for teams that pass validation. Default RHP/LHP rows are never deleted by this script.

### Safety and redundancy checks

Before writing, each team's **Today's Lineup** must pass:

1. **Non-empty** — empty `<ol>` or “lineup has yet to be announced” → **skipped** (cache unchanged)
2. **Size** — 8–10 batters (configurable `--min-players` / `--max-players`)
3. **No blank or duplicate names**
4. **Slate overlap** — at least 3 lineup batters must match today's prop slate for that team (feature-based team lookup); disable with `--skip-slate-check`
5. **Unchanged** — identical to cached OFFICIAL → skip write
6. **Fetch retries** — 2 attempts per team, 5s apart (shell default)
7. **Backup** — timestamped `rotowire_lineups.{stamp}.bak.parquet` before each write (`--no-backup` to skip)
8. **Atomic save** — write `.tmp.parquet` then replace
9. **`DISABLE_LIVE_FETCH=1`** — blocked by shell script (live Rotowire fetch required)

### Code map

| Module | Role |
|--------|------|
| [`run_official_lineups.sh`](run_official_lineups.sh) | Shell wrapper, pre-flight checks, optional `--watch` |
| [`scripts/update_official_lineups.py`](scripts/update_official_lineups.py) | CLI: slate discovery, validation, summary |
| [`fetch_rotowire_lineups.py`](fetch_rotowire_lineups.py) | Parse **Today's Lineup**, `update_official_lineups()`, `lineup_for_team()` |
| [`ui/hitters_life_board.py`](ui/hitters_life_board.py) | Lineup filter prefers official → default fallback |

**Tests:** [`scripts/test_hitters_life.py`](scripts/test_hitters_life.py) (`test_validate_official_lineup`, Today's Lineup HTML parse).

---

## Stuff strikeout model (v2)

Separate strikeout research column on the main board — **not** the LightGBM `pitcher_strikeouts.pkl` path and **not** the dual-head Poisson regressor (**Pred #** / **Dist Over %**).

### What it predicts

| Step | Detail |
|------|--------|
| **Inputs** | Rolling L5 Statcast process metrics from local pitch data: **SwStr%**, **chase (O-Swing%)**, **avg velocity** |
| **Model** | Ridge regression on **K%** (strikeouts ÷ batters faced) per pitcher-game |
| **Output** | Expected strikeouts = predicted K% × recent batters faced (L5); **Poisson P(Over line)** |
| **Board** | **Stuff K (v2)** column — e.g. `5.7 K · 67% Over` on **Pitcher Strikeouts** rows only |

Main **Model %**, **Edge**, and **EV** are unchanged (still the v1 classifier on box-score rolling stats).

### Files

| Path | Role |
|------|------|
| [`pitcher_stuff.py`](pitcher_stuff.py) | Derive stuff metrics from `data/raw/statcast_*.parquet` pitch rows |
| [`build_features.py`](build_features.py) | Merge stuff + `batters_faced` into pitcher feature parquets |
| [`pitcher_strikeout_stuff.py`](pitcher_strikeout_stuff.py) | Fit, load, score; writes `stuff_predicted_count`, `stuff_over_probability` |
| `models/v2/pitcher_strikeouts_stuff.pkl` | Trained Ridge package |
| [`run_pitcher_strikeout_stuff.sh`](run_pitcher_strikeout_stuff.sh) | One-shot pipeline wrapper |

### When to run

| Situation | Command |
|-----------|---------|
| **First time** / new clone | `./run_pitcher_strikeout_stuff.sh --streamlit` |
| **Daily board** (model already exists) | `./run_daily.sh` only — `predict.py` auto-scores Stuff K v2 |
| Parquet stale but model OK | `./run_pitcher_strikeout_stuff.sh --skip-fit` |
| Re-fit after more games | `./run_pitcher_strikeout_stuff.sh --skip-features` *(if features fresh)* or full script |

### Relationship to other K columns

| Column | Source | Drives Edge/EV? |
|--------|--------|-----------------|
| **Model %** / **Over %** | `pitcher_strikeouts.pkl` (LightGBM) | **Yes** |
| **Pred #** / **Dist Over %** | `models/v2/dist/pitcher_strikeouts.pkl` (Poisson on box-score features) | No (display only in Phase 1) |
| **Stuff K (v2)** | `pitcher_strikeouts_stuff.pkl` (Statcast stuff) | **No** (display / research) |

At **pitcher-season** aggregation, SwStr% vs K% correlation is much stronger (~R² 0.4+) than single-game K% (~R² 0.08) — game-level strikeout counts are noisy; the column is still useful for comparing process vs market on tonight’s line.

### Bugs fixed during implementation (2026-08-29)

| Issue | Symptom | Fix |
|-------|---------|-----|
| **Zone boolean bug** | `chase_pct_*` all NaN; `pitches_outside_zone` negative | `~` on integer zone flags from `np.where` — fixed by casting `in_zone` to **bool** in [`pitcher_stuff.py`](pitcher_stuff.py) |
| **Unsafe division** | `build_features.py` crash on zero outside-zone pitches | Use `np.divide(..., where=)` with numpy arrays for chase/whiff/velocity rates |
| **Missing BF rolls** | Stuff expected-K NaN at predict | Added `batters_faced` to pitcher game logs + rolling windows in `build_features.py` |

**Tests:** [`scripts/test_pitcher_strikeout_stuff.py`](scripts/test_pitcher_strikeout_stuff.py)

---

## Pitcher count-market learning loop (Track 1)

Self-learning pipeline for **`pitcher_strikeouts`**, **`pitcher_walks`**, and **`pitcher_outs`** — collect what the model predicted, join post-game actuals, retrain classifiers (and optionally Poisson regressors), then refresh the board CSV. Spec: [docs/ROADMAP.md](docs/ROADMAP.md#track-1--pitcher-count-market-learning-loop-v1-shipped).

### Why this exists

The daily board uses **frozen** LightGBM classifiers for the dual-head count markets. Track 1 adds a **feedback path** so you can:

1. **Log** every K / walks / outs prop the model scores (line, probability, edge, book)
2. **Measure** how those predictions performed vs actual stats from feature parquets
3. **Retrain** one or all three classifiers when you choose — without retraining all 13 markets
4. **Optionally** fit Poisson **regressors** so **Pred #** and **Dist Over %** populate on the board

This is **not** fully automatic nightly retrain yet — you run the loop when you want to refresh the count models (e.g. weekly).

### Board / UI impact

| Stage | Main board | Batter score board | Other markets |
|-------|------------|-------------------|---------------|
| **Logging only** (default after `predict.py`) | No visible change | No change | No change |
| **After classifier retrain + re-predict** | **Pitcher K / Walks / Outs** rows: Over %, Edge, EV may change | No change | No change |
| **After `--fit-distributional`** | Those rows: **Pred #** and **Dist Over %** fill in (existing columns) | No change | No change |

**Unchanged by design:** Edge and EV still come from the **classifier**, not the Poisson head. No new columns are added to the board layout.

### Data files (gitignored)

| File | Written by | Contents |
|------|------------|----------|
| `data/learning/predictions_log.parquet` | `predict.py` (automatic) | Every Track 1 count-market row scored, with model prob / edge / line / book / `game_date` |
| `data/learning/outcomes_log.parquet` | `scripts/log_outcomes.py` | Joined actual stat, `over_hit`, `prediction_error` per logged prop |

### Recommended: one executable script

**`./run_pitcher_outs_learning.sh`** wraps all four steps for all three markets (filename kept for compatibility). Use `--market pitcher_outs` for one market only.

```bash
# Full loop (default: skip props fetch, use cached lines)
./run_pitcher_outs_learning.sh

# One market only
./run_pitcher_outs_learning.sh --market pitcher_walks

# Also train Poisson regressors → Pred # / Dist Over % on board
./run_pitcher_outs_learning.sh --fit-distributional

# Open board when done
./run_pitcher_outs_learning.sh --fit-distributional --streamlit
```

### Step-by-step (manual equivalent)

#### Step 1 — Daily predict + auto-log

```bash
./run_daily.sh --skip-props
```

**What happens:**

| Sub-step | Script | Purpose |
|----------|--------|---------|
| Features | `ensure_features.py --fix` | Refreshes Statcast/features through **yesterday** so rolling outs stats are current |
| Props | *(skipped with `--skip-props`)* | Uses cached `current_props.parquet` — saves Odds API credits |
| Game lines / probables | `fetch_data.py` | Still runs unless you also pass `--skip-game-lines` / `--skip-probables` |
| Predict | `predict.py` | Scores all markets → `predictions_v2.csv` |
| **Log** | `learning_log.py` | Appends **pitcher_strikeouts**, **pitcher_walks**, and **pitcher_outs** rows to `data/learning/predictions_log.parquet` |

**Why `--skip-props`:** After games start, a fresh props fetch can overwrite pre-game lines with live numbers. For learning, you usually want the same cached slate you already predicted against. Use `./run_daily.sh` without `--skip-props` when you intentionally want fresh lines.

Logging is **non-blocking** — if the log write fails, predict still completes and the board is unaffected.

#### Step 2 — Join actual outs (after games finish)

```bash
python scripts/log_outcomes.py \
  --start 2026-04-01 \
  --end $(date -v-1d +%Y-%m-%d)
```

**What happens:** Reads `predictions_log.parquet`, matches each `(game_date, player)` to **`outs`** in pitcher feature parquets (fuzzy name match), computes whether the prop hit (`actual > line`), and appends to `outcomes_log.parquet`.

**Run when:** At least one **completed** game day exists in the log (typically morning after night games). Re-running is safe — duplicate keys are updated, not duplicated.

**Dry run:** add `--dry-run` to print match count without writing.

#### Step 3 — Retrain outs classifier (and optional Poisson head)

```bash
python scripts/retrain_market.py \
  --market pitcher_outs \
  --start 2026-03-25 \
  --end 2026-08-21

# Optional dual-head regressor:
python scripts/retrain_market.py \
  --market pitcher_outs \
  --start 2026-03-25 \
  --end 2026-08-21 \
  --fit-distributional
```

**What happens:** Rebuilds training rows from pitcher feature parquets (same path as `train.py`, per market), fits new `pitcher_strikeouts.pkl`, `pitcher_walks.pkl`, and/or `pitcher_outs.pkl`. With `--fit-distributional`, also writes matching `models/v2/dist/{market}.pkl` files.

**When to run:** After you have enough logged outcomes to justify a refresh, or on a schedule (weekly). The shell script defaults to **current season start → yesterday** (same feature window as `run_daily.sh` predict). Override with `--train-start` / `--train-end` if needed.

**Board impact starts here** — but only after step 4.

#### Step 4 — Re-predict

```bash
python predict.py
```

**What happens:** Loads the new count-market model(s) (and dist models if present), rewrites `predictions_v2.csv`. Restart Streamlit if it is already open (or use `--streamlit` on the learning script).

Filter the main board to **Pitcher Outs** to compare Over % / Edge / Pred # before and after.

### Script flags reference

See [Command reference → run_pitcher_outs_learning.sh](#run_pitcher_outs_learningsh--track-1-self-learning-loop).

---

## Updating this README

Keep this file in sync when adding new CLI flags, paths, or workflow steps. Update the V1 README if frozen-folder paths change.

---

## Changelog

### 2026-08-29 — Stuff strikeout model (v2) — Statcast SwStr/chase/velocity

**Context:** Users wanted a **separate** strikeout prediction path from process metrics (SwStr%, chase%, velocity), not mixed into the main LightGBM `pitcher_strikeouts.pkl` classifier — analogous to Batter Score v1 vs v2.

**Added:**
- [`pitcher_stuff.py`](pitcher_stuff.py) — pitch-level Statcast flags + per-game stuff aggregates + pitch-weighted rolling rates
- [`pitcher_strikeout_stuff.py`](pitcher_strikeout_stuff.py) — Ridge K% model → expected K → Poisson Over %; `models/v2/pitcher_strikeouts_stuff.pkl`
- [`run_pitcher_strikeout_stuff.sh`](run_pitcher_strikeout_stuff.sh) — one-shot: `build_features.py` → `fit_pitcher_strikeout_stuff.py` → `predict.py`
- [`scripts/fit_pitcher_strikeout_stuff.py`](scripts/fit_pitcher_strikeout_stuff.py), [`scripts/test_pitcher_strikeout_stuff.py`](scripts/test_pitcher_strikeout_stuff.py)
- Main board **Stuff K (v2)** column ([`ui/board.py`](ui/board.py)); CSV fields `stuff_predicted_count`, `stuff_over_probability` ([`predict.py`](predict.py))

**Changed:**
- [`build_features.py`](build_features.py) — merges stuff metrics + `batters_faced` into pitcher parquets (extra columns; **not** added to main `PITCHER_FEATURES` in [`train.py`](train.py))

**Fixes (stuff pipeline):**
- **Zone boolean bug** — `outside_zone = ~in_zone` on 0/1 integers produced negative pitch counts and all-NaN `chase_pct_*`; fixed with explicit bool `in_zone`
- **Zero-denominator crashes** — safe `np.divide` for chase/whiff/velocity when a pitcher-game has no outside-zone pitches

**Board impact:** **Stuff K (v2)** on strikeout rows only. **Model %**, **Edge**, and **EV** unchanged. First-time: `./run_pitcher_strikeout_stuff.sh`; daily `./run_daily.sh` re-scores when `.pkl` exists.

**Follow-up:** `./run_pitcher_strikeout_stuff.sh` step 1 now uses `ensure_features.py --fix` (auto Statcast fetch) instead of bare `build_features.py` — fixes `FileNotFoundError: statcast_*` when yesterday’s raw shard was not on disk yet.

---

### 2026-08-30 — Batter score by game columns, hot batters board, TB log colors, Statcast refresh fix

**Context:** Hitter's Life **Batter score by game** should surface batting context without duplicating **Game & time** when a Game filter is already present. Main board needed a larger hot-batters shortlist. Evening `./run_daily.sh` runs could leave L5/TB stats stale when an early-morning Statcast fetch stopped before calendar yesterday.

**Added / changed:**
- [`ui/batter_score_board.py`](ui/batter_score_board.py) — **Batter score by game** drops **Game & time**; adds **Batting average** and **TB per game (L5)** with the same highlights as the [Batting average board](#hitters-life-board) (TB color legend under the caption)
- [`ui/main_bottom_boards.py`](ui/main_bottom_boards.py) — **Hot batters — batter score** limit raised from **10 → 20**; shares batting-AVG and TB-log styling via [`ui/hitters_life_highlights.py`](ui/hitters_life_highlights.py)
- [`ui/hitters_life_highlights.py`](ui/hitters_life_highlights.py) — TB log **orange hot** rule extended (exactly one zero + ≥3 games with 2+ TB); **yellow warm** path for one zero + two 2+ TB games both in the most recent pair
- Tests: [`scripts/test_batter_score.py`](scripts/test_batter_score.py), [`scripts/test_main_bottom_boards.py`](scripts/test_main_bottom_boards.py), [`scripts/test_hitters_life.py`](scripts/test_hitters_life.py)

**Fix (Statcast / feature freshness):**
- [`utils.py`](utils.py) — **`feature_parquet_needs_refresh()`** now compares feature max `game_date` to **`required_max_game_date()`** (last MLB game on or before `--end`) **before** checking whether features merely match stale Statcast. **Symptom:** after a morning run, an evening `./run_daily.sh` printed **Feature check OK** even though `statcast_*_{yesterday}.parquet` and feature parquets still stopped at **two days ago** — TB log, L5/L10, and **Game logs through** looked one+ slate behind. **After fix:** `ensure_features.py --fix` re-fetches Statcast and rebuilds when features are behind the schedule date.

**Board impact:** Display-only on Hitter's Life by-game table and main-board hot batters section. Top 10 batter score and main prop edge/EV unchanged. Rolling stats refresh when features rebuild.

---

### 2026-08-24 — Official Rotowire lineups (pre-game)

**Context:** Close to first pitch, Rotowire posts confirmed orders under **Today's Lineup**. Hitter's Life should use those instead of default vs RHP/LHP when available.

**Added:**
- [`run_official_lineups.sh`](run_official_lineups.sh) — pre-game wrapper with `--dry-run`, `--watch`, pre-flight checks
- [`scripts/update_official_lineups.py`](scripts/update_official_lineups.py) — fetch, validate, merge **OFFICIAL** rows into `rotowire_lineups.parquet`
- [`fetch_rotowire_lineups.py`](fetch_rotowire_lineups.py) — parse Today's Lineup, `lineup_for_team()` (official → default fallback), backup + atomic save
- [Hitter's Life](#hitters-life-board) lineup popover shows **Today's Lineup** vs default source per team

**Board impact:** Hitter's Life lineup filter/order only. Prop edge/EV unchanged.

---

### 2026-08-24 — Hitter's Life board, Batter Score v2, Batter Score Pick Builder, Savant arsenals

**Context:** Extend batting research beyond the main prop table — detailed pitch-type matchup, a dedicated AVG/wOBA board, and a separate batter favorites slip.

**Added / changed:**
- **[Hitter's Life](#hitters-life-board)** — `?view=hitters_life`; columns for Vs SP, Arsenal wOBA, Szn/L5/L10 AVG, pitch-type wOBA, **SP arsenal** (Savant names), TB game log; conditional highlights; Rotowire lineup filter ([`fetch_rotowire_lineups.py`](fetch_rotowire_lineups.py), [`hitters_life_data.py`](hitters_life_data.py), [`ui/hitters_life_*.py`](ui/hitters_life_page.py))
- **Batter Score v2** — same composite as v1 but matchup grade uses **Savant pitch types** (Sinker, Sweeper, …) via [`build_opponent_pitcher_arsenal_detailed()`](pitch_matchup.py) / [`score_batter_v2()`](batter_score_data.py); new column on Top 10 and By game batter score boards
- **Batter Score Pick Builder** — sidebar slip separate from prop Pick Builder; add from **Batter score by game**; colored PP/UD/L5-L10/Vs pitcher cards ([`ui/batter_score_highlights.py`](ui/batter_score_highlights.py))
- [`pitch_matchup.py`](pitch_matchup.py) — `PITCH_CODE_TO_SAVANT_NAME`, `aggregate_pitcher_arsenal_usage_detailed()`, `aggregate_batter_pitch_stats_detailed()`, `build_opponent_pitcher_arsenal_detailed()`
- Track 1 learning extended to **pitcher_strikeouts** and **pitcher_walks** (in addition to outs) in [`learning_log.py`](learning_log.py), [`run_pitcher_outs_learning.sh`](run_pitcher_outs_learning.sh)
- Tests: [`scripts/test_hitters_life.py`](scripts/test_hitters_life.py), expanded [`scripts/test_batter_score.py`](scripts/test_batter_score.py)

**Board impact:** Main prop edge/EV unchanged. Batter Score v1 column unchanged. New display-only boards/slips.

---

### 2026-08-23 — Underdog fantasy, ranking-table UX, batter score highlights, H2H stat history

**Context:** Post-v3 board polish — DFS fantasy lines for research, clearer slate context on ranking tables, visual signals on batter score boards, and opponent-scoped stat history on player pages.

**Added / changed:**
- [`fetch_underdog_fantasy.py`](fetch_underdog_fantasy.py) — Underdog public API → `data/processed/underdog_fantasy_lines.parquet` (Odds API `us_dfs` does not return MLB batter fantasy scores for Underdog)
- [`fetch_data.py`](fetch_data.py) — runs Underdog fantasy fetch after every successful `--props`; standalone `--underdog-fantasy` flag
- [`ui/batter_score_board.py`](ui/batter_score_board.py) — **PP fantasy** / **UD fantasy** columns; **Game & time**; **L5 / L10 %** vs PP line; conditional highlights (orange UD when UD &lt; PP; yellow/green L5/L10 at 80% threshold; red row outline on combo)
- [`ui/board.py`](ui/board.py) / [`ui/top_lists.py`](ui/top_lists.py) — **Game & time** and **L5 / L10 %** on Top Over / Top Under previews and full list pages
- [`ui/formatting.py`](ui/formatting.py) — shared `format_game_time()`; **(L)/(R)** hand suffix on player/SP names across main board and batter score boards
- [`ui/player.py`](ui/player.py) / [`ui/player_stats.py`](ui/player_stats.py) — stat history **All / H2H** scope toggle vs today's slate opponent
- Tests: [`scripts/test_underdog_fantasy.py`](scripts/test_underdog_fantasy.py), styling coverage in [`scripts/test_batter_score.py`](scripts/test_batter_score.py)

**Board impact:** Main prop edge/EV unchanged. Batter score and ranking tables are display-only additions/highlights.

---

### 2026-08-22 — Pitcher outs learning loop (Track 1) + dual-head outs

**Context:** Post-v3 work needed a **single-market self-learning path** for Pitcher Outs without retraining all props or changing board layout. Users also wanted Poisson **Pred #** / **Dist Over %** on the board for outs (same as K/walks).

**Added:**
- [`learning_log.py`](learning_log.py) — append-only `data/learning/predictions_log.parquet` and `outcomes_log.parquet`
- [`predict.py`](predict.py) — auto-logs `pitcher_outs` rows after enrich (non-blocking)
- [`scripts/log_outcomes.py`](scripts/log_outcomes.py) — join logged preds to actual `outs` from feature parquets
- [`scripts/retrain_market.py`](scripts/retrain_market.py) — retrain `pitcher_outs.pkl` only; optional `--fit-distributional`
- [`run_pitcher_outs_learning.sh`](run_pitcher_outs_learning.sh) — one-shot wrapper for log → join → retrain → re-predict
- [`distributional.py`](distributional.py) — `pitcher_outs` in `DISTRIBUTIONAL_MARKETS` and `DUAL_HEAD_MARKETS`
- [`scripts/test_learning_log.py`](scripts/test_learning_log.py), updated [`scripts/test_distributional.py`](scripts/test_distributional.py)
- README [Pitcher outs learning loop (Track 1)](#pitcher-outs-learning-loop-track-1); [docs/ROADMAP.md](docs/ROADMAP.md) Track 1 section

**Board impact:** Logging none. Retrain + re-predict changes **Pitcher Outs** Over % / Edge / EV only. `--fit-distributional` fills existing **Pred #** and **Dist Over %** columns for outs rows; edge still classifier.

---

### 2026-08-22 — Probables Eastern date, SP coverage, player matching, PrizePicks

**Context:** Mass **Partial · SP TBD** on the batter score board; Jr./Sr. suffix players not matching feature parquets; PrizePicks props and fantasy lines requested.

**Fixes / changes:**
- [`fetch_probables.py`](fetch_probables.py), [`utils.py`](utils.py), [`game_lines.py`](game_lines.py) — **US Eastern** schedule dates aligned with slate `game_date` (UTC mismatch caused all SP lookups to fail)
- [`utils.py`](utils.py) `strip_name_suffix` — **Jr./Sr.** normalization for prop ↔ feature matching (e.g. Luis Garcia Jr.)
- [`odds_api.py`](odds_api.py) / [`fetch_data.py`](fetch_data.py) — merge PrizePicks standard props; **PP fantasy** column on batter score board (Underdog fantasy added 2026-08-23 — see [changelog entry](#2026-08-23--underdog-fantasy-ranking-table-ux-batter-score-highlights-h2h-stat-history))
- Terminal-only SP coverage warnings (no blocking Streamlit banner)
- README [Command reference](#command-reference) expanded (`ensure_features --fix`, pipeline flags)

---

### 2026-08-22 — Batter Score validation passed

**Context:** Track 1 ROADMAP item #4 — confirm Batter Score ranks batters vs same-game H+TB+BB outcomes.

**Result:** `scripts/backtest_batter_score.py` on 2026-04-01 → 2026-08-21: **33,148** rows, Spearman **0.161** (gate ≥ 0.15) → `data/backtest/batter_score_validation.json` `"validated": true`. Player pages show **✓ Batter Score validated**. Board edge/ranking unchanged.

---

### 2026-08-19 — Post-v3 board filter UX (AND combine, chips, header sort)

**Context:** After the v3 snapshot, the main board needed clearer filter composition, visible active-state feedback, and faster multi-column sorting without a separate sort dropdown.

**Added / changed:**
- [`ui/board.py`](ui/board.py) — **Active filters** chip row with **Clear all filters**; Market + min Edge + min EV + column filters combine with **AND** logic; caption shows filtered row count
- **Column header buttons** — click to sort up to **3 columns** (desc → asc → remove); subscript priority markers (₁₂₃) and ↑/↓ on headers; **Clear sort** toolbar
- **Filter subscripts** on headers and in **Filter by column** panel labels — index matches chip order when multiple column filters are active
- **Range filter slider fix** — when min equals max (single unique value), show caption instead of an invalid Streamlit slider
- [`ui/glossary.py`](ui/glossary.py) — `header_click_help`, `header_click_sort` tooltips

---

### 2026-08-19 — Version compare board (V1 / V2 / V3 / Main)

**Context:** Users needed one view to compare model **Over %** / **Under %** across project generations without switching model versions on the main board.

**Added:**
- [`ui/version_compare.py`](ui/version_compare.py) — side-by-side compare table (top 30 deduped props); **Version sources** expander; **Generate missing predictions** for V1/V2
- [`utils.py`](utils.py) — `VERSION_COMPARE_SLOTS`, `compare_predictions_path()`, `version_has_models()`
- [`app.py`](app.py) — routing via `?view=compare` and sidebar link; board link under Top Over / Under previews
- Prediction files: `predictions.csv` (V1), `predictions_v2.csv` (V2/Main), `predictions_v3.csv` (manual copy from frozen v3)

Documented in [Quick start → Version Compare](#open-streamlit-and-version-compare).

---

### 2026-08-19 — Cache-first data policy and `DISABLE_LIVE_FETCH`

**Context:** Backtests, Streamlit reruns, and `predict.py` were at risk of accidental Odds API / Statcast calls, burning quota and slowing the UI.

**Added:**
- README [Cache-first data policy](#cache-first-data-policy-no-redundant-api-calls) — single source of truth under `data/raw/`, `data/processed/`, `data/predictions/`, `data/backtest/`
- [`utils.py`](utils.py) — `live_fetch_disabled()`, `require_live_fetch()`; blocks fetch scripts and Statcast refresh when `DISABLE_LIVE_FETCH=1`
- Zero-API run patterns for backtest, predict, and Streamlit; daily pipeline skip flags documented
- [`app.py`](app.py) — `@st.cache_data` on board load keyed by predictions CSV **mtime**; [`ui/version_compare.py`](ui/version_compare.py) caches compare table by CSV mtimes

---

### 2026-08-19 — Dual-head pitcher K / walks / outs (classifier + regressor)

**Context:** Count props (strikeouts, walks, outs) benefit from an expected-count view alongside the existing Over/Under classifier. Post-v3 architecture per [docs/ROADMAP.md](docs/ROADMAP.md).

**Added (2026-08-19 — K/walks; outs added 2026-08-22):**
- [`distributional.py`](distributional.py) — `DUAL_HEAD_MARKETS` (`pitcher_strikeouts`, `pitcher_walks`, `pitcher_outs`); Poisson regressor outputs `predicted_count`, `dist_over_probability`
- [`scripts/fit_distributional.py`](scripts/fit_distributional.py) → `models/v2/dist/{market}.pkl`; [`./run_evaluation.sh`](#evaluation-pipeline-phase-6)
- [`predict.py`](predict.py) — dual-head inference for K/walks only; **classifier remains source of truth for edge/EV**
- Board columns **Pred #** and **Dist Over %**; [`scripts/test_distributional.py`](scripts/test_distributional.py)

**Deferred:** 50/50 classifier+regressor probability blend for edge (Phase 2+ in ROADMAP).

---

### 2026-08-19 — Batter Score validation backtest + validated flag

**Context:** Batter Score was shown on player pages but not validated against outcomes; board edge/ranking should stay unchanged until gates pass.

**Added:**
- [`scripts/backtest_batter_score.py`](scripts/backtest_batter_score.py) — point-in-time scoring vs same-game **H + TB + BB**; Spearman ρ ≥ 0.15 and n ≥ 100 gates
- [`batter_score_data.py`](batter_score_data.py) — `load_batter_score_validation()`, `is_batter_score_validated()`; auto-sets `DISABLE_LIVE_FETCH=1`
- [`ui/batter_score.py`](ui/batter_score.py) — player page **✓ Batter Score validated** when `data/backtest/batter_score_validation.json` has `"validated": true`
- Board edge/ranking **unchanged** until explicit future wiring

---

### 2026-08-19 — GitHub MON3Y remote and `v1` tag

**Context:** Full v3 snapshot needed a remote backup; all three generation tags should be fetchable from GitHub.

**Changes:**
- Remote **`origin`** → [https://github.com/EZ94PHEN0M52/MON3Y](https://github.com/EZ94PHEN0M52/MON3Y); **`main`** and tags pushed
- Annotated tags **`v1`** (`c4c9f8e`), **`v2`** (`fec6236`), **`v3`** (`3de111a`) on GitHub
- Commit **`3de111a`** — `v3: full MLB prop model snapshot` (code + docs; `.env` / `data/` / `models/` remain local/gitignored)

---

### 2026-08-19 — v3 git tag and frozen snapshot (`mlb-prop-model-v3/`)

**Context:** Phases 1–6 (historical odds, multi-book intelligence, real-line training, line movement, game-line features, stolen bases, calibration, distributional models, CLV), Batter Score Phases A–D, board market filters, Pick Builder, and player stat history were complete. A reproducible baseline was needed before starting the dual-head pitcher K/walks upgrade ([docs/ROADMAP.md](docs/ROADMAP.md)).

**Changes:** Tagged commit **`v3`** and created sibling folder [`mlb-prop-model-v3/`](../mlb-prop-model-v3/) — a self-contained copy with `data/`, `models/`, and `.env` for standalone runs. Post-v3 architecture work (50/50 dual-head classifier + regressor for `pitcher_strikeouts` / `pitcher_walks`, Batter Score validation track) continues on **`main`** in this repo.

**Snapshot note (in copied folder):** The v3 copy README includes a banner identifying it as the frozen v3 snapshot; recreate `.venv` locally — virtual environments are not copied.

---

### 2026-08-19 — Pick Builder (session favorites slip)

**Context:** Users wanted Pickfinder-style favorites to track props while browsing — UI-only, session-persistent, no export.

**Added:**
- [`ui/pick_builder.py`](ui/pick_builder.py) — `pick_key()`, `add_pick()`, `remove_pick()`, `clear_picks()`, sidebar panel, board multiselect + player-page add controls
- [`app.py`](app.py) — sidebar Pick Builder always visible
- [`ui/board.py`](ui/board.py) — **Add to Pick Builder** expander below filtered table
- [`ui/player.py`](ui/player.py) — per-market **Add** / **Add best EV**
- [`ui/glossary.py`](ui/glossary.py) — `pick_builder`, `pick_builder_add` tooltips

Duplicate prevention on `(player, market, side, line, book)`. Picks stored in `st.session_state` only.

---

### 2026-08-19 — Player stats cache bust, market filters, unified board popover

**Context:** After `./run_daily.sh` rebuilt features to 8/18, player stat history and L5/L10 % still showed 8/17 because `@lru_cache` on `_kind_player_game_cache(kind, version)` ignored parquet path/mtime — Streamlit kept stale in-memory game logs. Market multiselects could show duplicate **Walk** labels (`batter_walks` vs `pitcher_walks`). The main board had separate top filter row plus a **Columns & filters** popover.

**Fix:**
- [`ui/player_stats.py`](ui/player_stats.py) — `_kind_player_game_cache` keyed by `(kind, version, path, mtime)`; `get_features_max_game_date()`; player page caption **Game logs through YYYY-MM-DD**
- [`batter_score_data.py`](batter_score_data.py) — probables/statcast `@lru_cache` keyed by path + mtime
- [`ui/market_filters.py`](ui/market_filters.py) — `render_market_multiselect()` with distinct **Batter Walks** / **Pitcher Walks** labels
- [`ui/board.py`](ui/board.py) — single **Filters & columns** popover (market, edge, EV, columns, column filters); `_apply_data_filters()` without widgets; Top Over/Under previews share the same market filter; **Sort by** / **Ascending** outside popover

**Note:** Restart Streamlit is optional after `./run_daily.sh` — feature caches bust automatically when parquet mtime changes.

**Verified:** Feature parquets max `game_date` = **2026-08-18** (`batter_features_v2_*`, `pitcher_features_v2_*`).

---

### 2026-08-19 — Statcast staleness, walk market labels, board columns panel

**Context:** On 8/19, player stat history and L5/L10 % stopped at **8/17** even though `./run_daily.sh` targeted yesterday (**8/18**). The cached Statcast parquet was created early on 8/19 before Baseball Savant had 8/18 games; `fetch_data.py` skipped re-download when the file existed. Feature parquets inherited the same cutoff. Market filters showed two identical **Walk** labels (both batter). The main board had a wide row of per-column Filter popovers.

**Root cause (stats):** `fetch_statcast()` returned early when `statcast_{start}_{end}.parquet` existed, without checking whether `max(game_date)` reached `--end`. `ensure_features.py --fix` also skipped re-fetch in that case.

**Fix:**
- [`utils.py`](utils.py) — `statcast_needs_refresh()`, `parquet_max_game_date()`, `feature_parquet_needs_refresh()` (compares features to MLB schedule date via `required_max_game_date()`, not only stale Statcast max)
- [`fetch_data.py`](fetch_data.py) — re-fetch when cached Statcast stops before `--end`; new `--force` flag
- [`scripts/ensure_features.py`](scripts/ensure_features.py) — stale-data detection triggers Statcast re-fetch + feature rebuild
- [`ui/player_stats.py`](ui/player_stats.py) — feature parquet cache keyed by file mtime (reload after rebuild without process restart)
- [`ui/glossary.py`](ui/glossary.py) — **Batter Walks** / **Pitcher Walks** distinct labels
- [`ui/board.py`](ui/board.py) — single **Columns & filters** popover + **Sort by** / **Ascending** controls (replaces per-column header buttons)

**Daily pipeline note:** `run_daily.sh` sets `YESTERDAY=$(date -v-1d +%Y-%m-%d)` and passes it to `ensure_features.py` and `predict.py`. Completed games must appear in Statcast before features and L5/L10 % update; re-run the pipeline after Savant posts if the morning run was too early.

**Tests:** [`scripts/test_ensure_features.py`](scripts/test_ensure_features.py) — stale feature/statcast data detection

---

### 2026-08-19 — NaN sp_id fix (Batter Score enrichment crash)

**Context:** MLB probables often list a starter **name** before a pitcher ID is assigned — parquet stores missing IDs as float **NaN**. Enrichment called `int(sp_id)` and crashed Streamlit with `ValueError: cannot convert float NaN to integer`.

**Fix:**
- [`utils.py`](utils.py) — **`coerce_mlb_id()`** returns `int | None` for valid IDs; `None` for NaN/missing
- [`fetch_probables.py`](fetch_probables.py) — `lookup_opposing_sp()` coerces `home_sp_id` / `away_sp_id` on read
- [`batter_score_data.py`](batter_score_data.py) — ERA L5 via **`_pitcher_rows_by_sp()` name fallback** when ID missing; **H2H and arsenal only when valid ID**
- [`pitch_matchup.py`](pitch_matchup.py) — arsenal build skips invalid IDs

**Hint:** Restart Streamlit after update — Batter Score is computed at UI load. Documented in [NaN sp_id fix](#6-nan-sp_id--tbd-starter-ids-2026-08-19) and [Troubleshooting](#troubleshooting).

**Tests:** [`scripts/test_batter_score.py`](scripts/test_batter_score.py) — `test_coerce_mlb_id_handles_nan`

---

### 2026-08-19 — Board dedupe, top-list market filters, Batter Score Phase D

**Context:** Main board and top lists now show one best-EV row per player/market. Batter Score Phase D wires pitch-type matchup from Statcast when the opposing SP is known.

**Board / top lists:**
- [`odds_aggregation.py`](odds_aggregation.py) — `dedupe_best_prop()` on `(player, market)` by highest EV
- [`ui/board.py`](ui/board.py) — removed **All books** toggle; always deduped; ranking previews get **Market type** multiselect
- [`ui/top_lists.py`](ui/top_lists.py) — full Over/Under lists deduped + market type filter
- [`predict.py`](predict.py) — `predictions_v2_best.csv` uses `dedupe_best_prop()`

**Batter Score Phase D + SP abbr fix:**
- [`pitch_matchup.py`](pitch_matchup.py) — Statcast pitch buckets, batter wOBA/AVG, SP arsenal usage (L5 starts)
- [`batter_score.py`](batter_score.py) — `PHASE_D_GATES`, `compute_batter_score_phase_d()`, **Full** label when all components active
- [`batter_score_data.py`](batter_score_data.py) — arsenal lookup via probables + Statcast; Phase D when SP + arsenal ready
- [`utils.py`](utils.py) — **`TEAM_ABBR_TO_ODDS`** + **`canonical_odds_team_key()`** — maps feature-parquet abbrs (`SF`, `CLE`) to Odds API full names; [`fetch_probables.py`](fetch_probables.py) `lookup_opposing_sp()` uses this (fixes all rows stuck on **Partial · SP TBD**)

**Tests:** [`scripts/test_batter_score.py`](scripts/test_batter_score.py) — dedupe, pitch buckets, Phase D composite, SP lookup abbr mapping, `coerce_mlb_id` NaN handling (see [NaN sp_id changelog](#2026-08-19--nan-sp_id-fix-batter-score-enrichment-crash))

---

### 2026-08-19 — Batter Score Phase A + B, SP probables, player stat history

**Context:** Implemented Phases A–C of the Batter Score roadmap: form-only baseline with gating/renormalization, SP L5 ERA + H2H pitcher form, MLB Stats API probables pipeline, and all-market player stat history. Batter Score stays orthogonal to LightGBM prop models.

**Phase A (season baseline + recent form):**
- [`batter_score.py`](batter_score.py) — `compute_batter_score_partial()`, `ComponentGates`, `PHASE_A_GATES`, weight renormalization, **Form only** partial label
- [`batter_score_data.py`](batter_score_data.py) — game logs from batter feature parquets, `enrich_with_batter_score()` for board
- Streamlit board column (sortable); player-page component breakdown + H+TB+BB last-10 Altair chart

**Phase B (SP L5 ERA + H2H, component gating):**
- [`batter_score.py`](batter_score.py) — `PHASE_B_GATES`, `compute_batter_score_phase_b()`, H2H blend in `pitcher_form_index()` (`MIN_PA_H2H=10`), **Partial · SP TBD** / **Partial** labels; optional team proxy (`USE_TEAM_PITCHING_PROXY=False`)
- [`batter_score_data.py`](batter_score_data.py) — SP lookup via `daily_probables.parquet`, ERA L5 from pitcher game logs, H2H from Statcast

**Phase C (SP sourcing):**
- [`fetch_probables.py`](fetch_probables.py) — MLB Stats API → `data/processed/daily_probables.parquet`; `fetch_data.py --probables`; `./run_daily.sh` step 4 (`--skip-probables`)

**Player stat history:**
- [`ui/player.py`](ui/player.py) — market dropdown (all batter/pitcher markets), L5/L10 toggle, averages, Altair chart
- [`ui/player_stats.py`](ui/player_stats.py) — `infer_player_kind()`, `get_stat_history()`, `batter_stolen_bases` market map

**Tests:** [`scripts/test_batter_score.py`](scripts/test_batter_score.py)

**Deferred:** Batter Score vs prop backtest validation (Phase D since implemented — see [entry above](#2026-08-19--board-dedupe-top-list-market-filters-batter-score-phase-d)).

---

### 2026-08-19 — Batter Score feasibility & plan documented

**Context:** Composite **Batter Score** (0–100) spec captured in-repo before implementation.

**Changes:** Added [Batter Score](#batter-score) section — overview (30/25/30/15 weights), phased rollout (A–D), unknown-SP gating/renormalization strategy, risks, and TODO checklist. Phases A–C since implemented; see entry above.

---

### 2026-08-19 — Backtest scoring crash (`ValueError: truth value of Series is ambiguous`)

**Context:** Scoring historical props in `derived_line_features` failed when duplicate `(game_date, player_name)` keys returned a Series instead of a scalar (e.g. `season_avg`).

**Fix:** Hardened `prop_scoring.py` with `_safe_notna()` and `_scalar_value()`; added `_lookup_feature_row()` in `scripts/backtest.py` for deduplicated feature lookups. Full-window backtest (`2025-04-01` → `2025-06-30`) now completes.

---

### 2026-08-19 — Evaluation pipeline script (`run_evaluation.sh`)

**Changes:** Added `./run_evaluation.sh` — unified Phase 6 pipeline in order: backtest → `fit_calibrators.py --from-csv` → `fit_distributional.py`. Separate from `run_daily.sh`; auto-activates `.venv`; must be invoked as `./run_evaluation.sh` (not bare `run_evaluation.sh` on zsh). Defaults to the training window (`2025-04-01` → `2025-06-30`). After evaluation, run `./run_daily.sh --streamlit` to apply calibrators to live predictions. Documented in [Evaluation pipeline (Phase 6)](#evaluation-pipeline-phase-6).

---

### 2026-08-19 — Phase 6: model refinement

**Context:** After real-line training (Phase 3) and movement features (Phase 4), probabilities needed calibration and bets needed vig-aware EV ranking. Count props benefit from full distributional models rather than fixed-threshold classifiers alone.

**Changes:**
- Added `calibration.py` — isotonic/Platt calibrators per market; `scripts/fit_calibrators.py` fits from backtest windows
- `prop_scoring.py` / `predict.py` — `raw_model_probability`, `calibrated_probability`; edge/EV use calibrated probs with graceful fallback
- Added `distributional.py` — Poisson rate models for `batter_hits` and `pitcher_strikeouts`; `scripts/fit_distributional.py`
- Added `clv.py` — closing line value in `scripts/backtest.py` when multiple historical snapshots exist
- UI board defaults to sort by **EV**; optional **Calibrated %** column; glossary updates
- Added `scripts/test_calibration.py`

**Deferred:** Negative binomial distributional models for over-dispersed counts; calibrators for all 13 markets require sufficient backtest history per market.

---

### 2026-08-19 — Phase 5: game line features & stolen bases

**Context:** Player prop models lacked game-level context (expected total runs, run line) and did not cover stolen bases despite Statcast and Odds API support.

**Changes:**
- Added `game_lines.py` — consensus totals/spreads, merge into V2 feature parquets, live enrichment at inference
- Extended `odds_api.py` with `GAME_MARKETS` (`totals`, `spreads`); `fetch_data.py --game-lines` → `current_game_lines.parquet`
- `fetch_historical_odds.py --game-lines` → historical `game_lines.parquet` partitions
- V2 feature columns: `game_total_line`, `game_run_line`, `game_implied_total_over_prob`
- New market `batter_stolen_bases` — Statcast steal events, rolling features, model training/scoring
- Bumped `PARQUET_FEATURE_SCHEMA_VERSION` to `"3"`; `./run_daily.sh` fetches game lines by default
- Added `scripts/test_phase5.py`; glossary entries for game line features and stolen bases

---

### 2026-08-19 — Phase 4: line movement & intraday snapshots

**Context:** Phases 1–3 stored and scored single pre-game snapshots (historical backfill) or one live fetch (`current_props.parquet`). Bettors needed to see how lines moved through the day before first pitch.

**Changes:**
- Added `odds_snapshots.py` — append-only live snapshots at `data/raw/odds/snapshots/props_{YYYYMMDD_HHMMSS}.parquet`; `save_live_snapshot()` called from `fetch_data.py --props`
- Added `odds_movement.py` — `compute_movement_features()` derives `opening_line`, `opening_odds`, `line_delta`, `odds_delta`, and `steam_flag` from earliest intraday snapshot before `commence_time`
- `predict.py` joins movement features; outputs included in `predictions_v2.csv` / `predictions_v2_best.csv`
- Streamlit board and player pages show optional **Line Δ** and **Steam** columns; glossary entries added
- `run_daily.sh` documents optional intraday cron for additional props fetches on game days
- Added `scripts/test_odds_movement.py` — unit tests with mock snapshot parquets (no models/API)

---

### 2026-08-19 — Phase 3 parquet rebuild false-alarm fix

**Context:** After Phase 3 landed, `ensure_features.py --fix` began rebuilding V2 feature parquets on every `./run_daily.sh` run even though inference-season files (2026) were already valid. A one-time rebuild for the 2025 training window was expected when new parquet columns were added; repeated rebuilds for the current season were not.

**Changes:** Introduced `PARQUET_FEATURE_SCHEMA_VERSION="2"` in `train.py`, separate from model-only schema tracking. Derived inputs `market_implied_over_prob` and `line_vs_season_avg` are computed at train/infer time and no longer affect parquet staleness.

**Fix:** Removed `train.py` and `training_odds.py` from `ensure_features.py` rebuild triggers (`SOURCE_FILES` now only includes `build_features.py` and, for V2, `features_v2.py`). Mtime staleness and column checks apply only to files that change parquet contents. Inference 2026 parquets remain valid without rebuild; the 2025 training-window V2 build is still legitimately required once when training features gain new columns.

---

### 2026-08-19 — v2 git tag and frozen snapshot (`mlb-prop-model-v2/`)

**Context:** Active development on `main` was expanding into Phases 1–6 (Odds API history, multi-book intelligence, real-line training). A reproducible baseline was needed before that roadmap work continued.

**Changes:** Tagged commit `v2` and created sibling folder [`mlb-prop-model-v2/`](../mlb-prop-model-v2/) — a self-contained copy with `data/`, `models/`, and `.env` for standalone runs without touching the active workspace. Post-v2 expansion continues in this repo on `main`.

---

### 2026-08-19 — Phase 3: train on real book lines

**Context:** Models trained only on synthetic threshold grids (e.g. hits at 0.5, 1.5, 2.5) did not reflect lines sportsbooks actually posted. Phase 1 historical props made it possible to grade training rows on consensus market lines.

**Changes:**
- Added `training_odds.py` — loads historical props, builds consensus lines (median line + weighted devigged Over probability), fuzzy-matches players to Statcast feature rows
- Added `train.py --line-source` — `real`, `synthetic`, or `auto` (default: real when historical props yield ≥100 rows per market, else synthetic)
- Target is `actual_stat > line` on posted consensus lines (same grading as inference/backtest)
- Derived model inputs `market_implied_over_prob` and `line_vs_season_avg` attached at train and infer time
- `./run_daily.sh --train` passes `--line-source auto`

---

### 2026-08-18 — Phase 2: multi-bookmaker intelligence

**Context:** Multiple US books post the same prop at different lines and with different vig. Comparing model probability to a single raw implied probability overstated or understated edge depending on which book was chosen.

**Changes:**
- **Devigging** — `devig_two_way()` in `utils.py` removes two-way vig when Over/Under pairs exist; edge and EV use devigged probability
- **Consensus line** — `odds_aggregation.py` computes median line and weighted devigged Over probability per `(player, market, event_id)` (optional sharp-book weights via `SHARP_BOOK_WEIGHTS`)
- **Best price** — highest-EV book per `(player, market, line, side)` flagged `is_best_price=True`
- Outputs `predictions_v2.csv` (all books) and `predictions_v2_best.csv` (best price only)
- Streamlit [main board](#streamlit-ui) deduped to best EV per `(player, market)`; [player pages](#player-pages-uplayerpy) show per-book breakdown
- `scripts/backtest.py` uses the same devigged edge logic for consistency with live predictions

---

### 2026-08-18 — Phase 1: historical odds fetch and backtesting

**Context:** Before changing training or adding movement features, the pipeline needed a baseline for evaluating model edges against real sportsbook prices and Statcast outcomes.

**Changes:**
- Added `fetch_historical_odds.py` — backfills pre-game player props into `data/raw/odds/historical/date=YYYY-MM-DD/props.parquet` (paid Odds API plan; history from 2023-05-03)
- Added `odds_api.py` — shared helpers for live and historical Odds API calls
- Added `prop_scoring.py` — model scoring shared by `predict.py` and backtest
- Added `scripts/backtest.py` — scores historical props with trained models, joins Statcast outcomes from feature parquets, writes `data/backtest/backtest_{start}_{end}.csv` with win rate, flat-bet ROI, average edge, and Brier score per market
- Historical fetch is opt-in (not in `run_daily.sh`); quota errors stop the run without overwriting existing files

---

### 2026-08-18 — Auto-rebuild when new market columns missing (walks KeyError)

**Context:** Adding batter walks and other new markets extended `train.py` feature lists (`walks_l3`, `walks_l5`, …) before existing feature parquets contained those columns. `predict.py` and training failed with `KeyError` on missing columns.

**Changes:** `scripts/ensure_features.py --fix` detects missing columns from `train.feature_columns_for_version()`, removes stale parquets, fetches Statcast when needed, and rebuilds via `build_features.py`. Integrated as step 1 of `./run_daily.sh`.

**Fix:** Documented workflow in [Adding a new feature](#adding-a-new-feature): sync `build_features.py` / `features_v2.py` with `train.py` column lists, bump `PARQUET_FEATURE_SCHEMA_VERSION` when parquet columns change, then run `./run_daily.sh` or `ensure_features.py --fix`.

---

### 2026-08-18 — Odds API quota exhaustion and empty props fetch

**Context:** When monthly Odds API credits ran out (`OUT_OF_USAGE_CREDITS`) or all event fetches failed, `fetch_data.py --props` could return zero rows and overwrite a valid `current_props.parquet` cache, breaking the daily pipeline.

**Changes:** On zero rows collected, fetch now preserves existing cache when non-empty and exits non-zero with a clear message. Added `./run_daily.sh --skip-props` to skip the props step and use cached lines.

**Fix:** Graceful failure path in `fetch_data.py`; `odds_api.py` detects quota errors. Troubleshooting table documents `--skip-props` for quota-exhausted days.

---

### 2026-08-18 — Streamlit LinkColumn player name display

**Context:** Player links on the main board used raw URL strings in `LinkColumn`, showing encoded query params instead of the player's name.

**Changes:** `ui/formatting.py` `player_path()` appends a URL fragment (`#Player Name`) and `LinkColumn` uses `display_text=r"#(.*)$"` to render the name. Fragment is ignored by Streamlit routing; `?player=` query param still drives navigation.

**Fix:** Display text comes from the fragment because `LinkColumn` cannot read another dataframe column for labels.

---

### 2026-08-17 — Streamlit UI: player pages and board enhancements

**Context:** Raw CSV output was hard to scan; bettors needed sortable filters, recent-form context, and drill-down per player without leaving the app.

**Changes:**
- **Main board** (`ui/board.py`) — market multiselect, min Edge/EV sliders, sortable columns, per-column filter popovers, Top Over/Under previews (top 10, one prop per player), L5/L10 % columns (share of last 5/10 games exceeding the posted line via `ui/player_stats.py`)
- **Player detail pages** (`ui/player.py`) — per-market book breakdown, consensus line caption, last-10-games Altair bar chart, best edge/EV summary
- **Top lists** (`ui/top_lists.py`) — full ranked Top Over % / Top Under % pages (`?view=top_over`, `?view=top_under`)
- Routing via query params: `?player=Name`, `?view=top_over` / `?view=top_under`

---

### 2026-08-17 — New prop markets: pitcher outs/ER, batter runs, H+R+RBI, walks

**Context:** The Odds API exposes more player props than the original hits/HR/TB/RBI/K set. Expanding markets required matching Statcast stat columns, training thresholds, and UI labels.

**Changes:** Added markets in `odds_api.py` `PROP_MARKETS`, `train.py`, `predict.py`, and `prop_scoring.py`:

| Market | Role |
|--------|------|
| `batter_runs_scored` | Batter |
| `batter_walks` | Batter |
| `batter_hits_runs_rbis` | Batter |
| `pitcher_walks` | Pitcher |
| `pitcher_outs` | Pitcher |
| `pitcher_earned_runs` | Pitcher |

Synthetic fallback thresholds documented in [Supported prop markets](#supported-prop-markets). V2 `features_v2.py` adds outs and earned runs to pitcher rolling stats.

---

### 2026-08-17 — Daily pipeline (`run_daily.sh`) and feature validation

**Context:** Manual multi-step workflows (Statcast → features → props → predict) were error-prone and easy to skip when dates or column lists drifted.

**Changes:**
- Added `run_daily.sh` — activates `.venv`, runs `ensure_features.py --fix`, fetches props, optionally retrains (`--train`), predicts, optionally launches Streamlit (`--streamlit`, `--skip-props`)
- Added `scripts/ensure_features.py` — validates batter/pitcher parquets against `train.feature_columns_for_version()`, schema fingerprint, and source-file mtimes; `--fix` rebuilds when stale
- Introduced `PARQUET_FEATURE_SCHEMA_VERSION` workflow: bump when parquet column lists change; derived model inputs do not require a bump

---

### 2026-08-17 — V2 core: opponent, handedness, and park features

**Context:** V1 models used rolling player form only. Matchup context (opponent strength, platoon splits, home park) was the main V2 modeling upgrade over the frozen V1 copy.

**Changes:**
- Added `features_v2.py` (`build_all_features_v2`) — opponent team pitching/batting season stats (lagged), batter stand and vs LHP/RHP hit rates, home park offense proxy, pitcher throwing hand
- Versioned artifacts: `batter_features_v2_{dates}.parquet`, `models/v2/*.pkl`, `predictions_v2.csv`
- Default CLI `--version v2`; V1 remains runnable side-by-side from the same repo
- Dual README workflow: active development here; V1 frozen in `mlb-prop-model-v1/`

---

### Operational note — suspended jobs and long-running pipeline steps

**Context:** `./run_daily.sh --train` and `ensure_features.py --fix` can run for several minutes (Statcast download, feature rebuild, LightGBM training). Accidentally suspending the shell (e.g. **Ctrl+Z**) leaves background jobs holding locks or partial files.

**Changes:** Use **Ctrl+C** to stop Streamlit or a pipeline step cleanly. If a job was suspended, run `jobs` and `kill %1` (or the relevant job number) before re-running `./run_daily.sh`. Do not suspend mid-rebuild — let `ensure_features.py --fix` finish or abort with Ctrl+C so parquets are not left in a half-written state.
