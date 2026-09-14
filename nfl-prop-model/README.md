# NFL Prop Model

Sibling research project to [`mlb-prop-model/`](../) (same git repo, separate folder). Weekly NFL player-prop board: model Over/Under vs sportsbook lines in Streamlit.

This is **not** a betting service. It is a statistical research dashboard.

---

## Design plan

### Intent

Replicate the MLB **V1 → V2** pipeline pattern for NFL:

1. Download weekly player stats + injury/status + live prop lines  
2. Build rolling features (no leakage)  
3. Train / load LightGBM classifiers: **P(actual > posted line)**  
4. Show a Streamlit board with Over %, Under %, edge, EV, and injury flags  

**Stats source is not Statcast.** Statcast is MLB-only. NFL uses **nflverse** (`nflreadpy`) for weekly stats, schedules, snaps, and injury reports. Sportsbook lines still come from **The Odds API** (`americanfootball_nfl`), same API key family as MLB.

### Isolation from MLB

| Rule | Detail |
|------|--------|
| Own folder | Everything lives under `nfl-prop-model/` |
| Own artifacts | Separate `data/`, `models/`, predictions CSVs |
| No shared runtime | Do not import MLB modules; copy patterns only |
| Same repo | Push on the same remote; keep READMEs separate |

### Markets (V1)

| Odds API market | Outcome stat |
|-----------------|--------------|
| `player_pass_yds` | Passing yards |
| `player_pass_tds` | Passing TDs |
| `player_rush_yds` | Rushing yards |
| `player_receptions` | Receptions |
| `player_reception_yds` | Receiving yards |
| `player_rush_reception_yds` | Rush + receiving yards |

**Out of V1:** anytime TD, defense/kicking, longest-play props, alternate lines, multi-book consensus, historical odds backfill, calibration / CLV (MLB Phases 1–6).

### Modeling versions

| Version | Features | Training lines | Artifacts |
|---------|----------|----------------|-----------|
| **V1** | Rolling L3 / L5 / season form **+ injury/status features + board gates** | Synthetic threshold grids | `models/v1/`, `predictions_v1.csv` |
| **V2** | V1 + opponent defense, home/away, snap % / target share / rush share | Synthetic first; real lines later | `models/v2/`, `predictions_v2.csv` |

Classifier-first (same as early MLB): edge/EV from **P(over)**. Distributional / dual-head deferred.

### Injury noise (first-class, not deferred)

NFL availability moves all week. Design assumes injuries are refreshed **before each board run**, with highest trust closer to kickoff (e.g. Sunday morning).

**V1 (ships with MVP):**

1. **Ingest** weekly injury / practice status (nflverse injury reports; depth where available)  
2. **Board filters** — hide or flag **Out / Doubtful / IR**; warn on **Questionable**  
3. **Features** — status encoding (active / Q / D / O) and simple “teammate ahead on depth” when known  
4. **Predict gate** — Out players excluded from Top lists by default; optional “show inactive” toggle  

**V2:** same injury gates; adds snap % / target share / rush share rolling usage and opponent yards-allowed context. Depth chart RB1/WR1 labels remain optional future polish.

### Pipeline

```text
fetch nflverse stats + injuries + schedule
  → build features (--version v1|v2)
  → fetch Odds API props → current_props.parquet
  → predict
  → streamlit board
```

- Orchestrator: `./run_weekly.sh` (keyed by **NFL week / kickoffs**, not MLB “yesterday”)  
- **Cache-first:** `predict.py` and Streamlit only read local parquets/CSVs; network stays in fetch scripts  
- Board: one useful row per player/market, injury badges, market filters  

### Delivery order

| Step | Status | Deliverable |
|------|--------|-------------|
| 1. Scaffold | **Done** | Folder, README, requirements, stubs |
| 2. Fetch layer | **Done** | nflverse weekly stats, schedule, injury ingest, Odds API props |
| 3. V1 features + train + predict | **Done** | Six markets, synthetic lines, injury gates |
| 4. Streamlit board | **Done** | Over/Under/edge/EV + injury UI |
| 5. `run_weekly.sh` | **Done** | End-to-end current week |
| 6. V2 | **Done** | Opponent defense, home/away, usage/snaps |

### Explicit non-goals (this phase)

- MLB Phases 1–6 (historical odds, consensus, movement, calibration)  
- Shared code with `mlb-prop-model/` beyond “same repo + Odds API key”  
- DFS (PrizePicks / Underdog / Sleeper) until the core board works  
- Profit guarantees  

### Risks (accepted)

- **Small samples** (~17 games/season) → noisier models than MLB; keep V1 expectations modest  
- **Odds API credits** — props are per-event; stick to the six markets above  
- **Injury lag** — status can change after the last fetch; re-run fetch before kickoff  

---

## Prerequisites

- Python 3.12  
- libomp (LightGBM on macOS): `brew install libomp`  
- Odds API key: [the-odds-api.com](https://the-odds-api.com/)  

## First-time setup

```bash
cd nfl-prop-model
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env: ODDS_API_KEY=your_key
```

## Fetch commands (step 2)

```bash
source .venv/bin/activate

# Weekly player stats + schedule (cached under data/raw/ unless --force)
python fetch_data.py --stats --schedule

# Injury + roster snapshot → data/processed/current_injuries.parquet
# Always re-downloads injury reports; use --week N to target a week.
python fetch_data.py --injuries
python fetch_data.py --injuries --week 1

# Live Odds API props for the six V1 markets → current_props.parquet
python fetch_data.py --props

# Combined (typical weekly refresh before features/predict)
python fetch_data.py --stats --schedule --injuries --props --force
```

**Injury columns used by the board/predict gates:** `report_status`, `report_status_norm` (`active` / `questionable` / `inactive`), `is_inactive`, `is_questionable`, plus roster `status` / `depth_chart_position` when available.

Set `DISABLE_LIVE_FETCH=1` to block network calls (cache-only mode).

## Features / train / predict (step 3)

```bash
source .venv/bin/activate

# Rolling L3/L5/season features (+ injury flags when raw injuries exist)
python build_features.py --seasons 2024 2025 2026 --version v1

# Train one LightGBM classifier per V1 market → models/v1/*.pkl
python train.py --version v1

# Score current_props.parquet → data/predictions/predictions_v1.csv
# Joins current_injuries.parquet; sets exclude_from_top for inactive players
python predict.py --version v1
```

**Typical weekly refresh:**

```bash
python fetch_data.py --stats --schedule --injuries --props --force
python build_features.py --seasons 2024 2025 2026 --version v1
python predict.py --version v1
```

Retrain when feature schema or training window changes (`python train.py --version v1`).

## Streamlit board (step 4)

```bash
source .venv/bin/activate
streamlit run app.py
# opens http://localhost:8501
```

Board defaults:
- Hide inactive (Out / Doubtful / IR)
- Filters for market, side, min edge / EV / model %, player search
- Injury column: `INACTIVE (…)`, `Q`, or `—`
- Download filtered CSV

## Weekly orchestrator (step 5)

```bash
./run_weekly.sh --help

# Full refresh + board (default --version v2)
./run_weekly.sh --streamlit

# Form + injuries only (keep cached props)
./run_weekly.sh --skip-props --streamlit

# First-time / schema change: also retrain
./run_weekly.sh --train --streamlit

# Stay on V1
./run_weekly.sh --version v1 --streamlit
```

Steps: stats/schedule (+ snaps/team-stats for V2) → injuries → props → `build_features` → optional `train` → `predict` → optional Streamlit.

## V2 extras (step 6)

```bash
# One-time / force refresh of V2 raw inputs
python fetch_data.py --snaps --team-stats --season 2024 --force
python fetch_data.py --snaps --team-stats --season 2025 --force
python fetch_data.py --snaps --team-stats --season 2026 --force

python build_features.py --seasons 2024 2025 2026 --version v2
python train.py --version v2
python predict.py --version v2
```

V2 adds rolling **target_share / rush_share / offense_snap_pct** and opponent **pass/rush/rec yards allowed** (prior-game means). At predict time, slate opponent + `is_home` are overlaid from the prop event.

## Project layout (target)

```text
nfl-prop-model/
├── README.md                 # This design plan + ops notes
├── requirements.txt
├── .env.example
├── utils.py                  # Paths, env, NFL week helpers
├── odds_api.py               # americanfootball_nfl prop markets
├── fetch_data.py             # nflverse stats / injuries / props
├── build_features.py         # Rolling features (--version v1|v2)
├── features_v2.py            # V2-only context + usage
├── train.py
├── predict.py
├── app.py                    # Streamlit entry
├── run_weekly.sh
├── ui/                       # Board, filters, glossary
├── data/
│   ├── raw/                  # nflverse + odds caches
│   ├── processed/            # features, current_props, injuries
│   └── predictions/
└── models/
    ├── v1/
    └── v2/
```

## Related

- MLB active workspace: [`../README.md`](../README.md)  
- MLB early architecture: V1 rolling form → V2 opponent/handedness/park → V3 odds phases  
