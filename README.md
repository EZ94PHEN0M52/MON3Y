# MLB Prop Model V2 (Active Development)

**Active workspace** for the MLB prop pipeline. V1 is frozen in [`../mlb-prop-model-v1`](../mlb-prop-model-v1).

---

## V1 vs V2

| | V1 (frozen copy) | V2 (this repo) |
|--|------------------|----------------|
| **Location** | `mlb-prop-model-v1/` | `mlb-prop-model/` |
| **Player features** | Rolling L3/L5/L10/L20/season | Same + opponent team stats |
| **New in V2** | — | Handedness, vs LHP/RHP splits, park proxy |
| **Models** | `models/*.pkl` | `models/v1/*.pkl`, `models/v2/*.pkl` |
| **Feature files** | `batter_features_{dates}.parquet` | `batter_features_v2_{dates}.parquet` |
| **Predictions** | `predictions.csv` | `predictions_v2.csv` |
| **Default CLI** | N/A | `--version v2` |

Both versions can run on the same machine. V1 folder is never modified.

---

## Prerequisites

- **Python 3.12:** `brew install python@3.12`
- **libomp** (LightGBM on macOS): `brew install libomp`
- **Odds API key:** [the-odds-api.com](https://the-odds-api.com/)

---

## First-time setup

```bash
cd /Users/edosaona-enagbare/pfinder_v1/mlb-prop-model

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env: ODDS_API_KEY=your_key
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

## Daily workflow (V2)

```bash
cd /Users/edosaona-enagbare/pfinder_v1/mlb-prop-model
source .venv/bin/activate

python fetch_data.py --props

python fetch_data.py --statcast --start 2026-03-25 --end 2026-08-16

python build_features.py --start 2026-03-25 --end 2026-08-16 --version v2

python predict.py --start 2026-03-25 --end 2026-08-16 --version v2

streamlit run app.py
```

Use the sidebar in Streamlit to switch between **V2** and **V1** predictions.

**Run V1 from this repo** (without using the frozen folder):

```bash
python predict.py --start 2026-03-25 --end 2026-08-16 --version v1
```

---

## Command reference

### fetch_data.py

```bash
python fetch_data.py --props
python fetch_data.py --statcast --start YYYY-MM-DD --end YYYY-MM-DD
```

### build_features.py

```bash
python build_features.py --start YYYY-MM-DD --end YYYY-MM-DD [--version v1|v2]
```

Default: `--version v2`

### train.py

```bash
python train.py --start YYYY-MM-DD --end YYYY-MM-DD [--version v1|v2]
```

Trains on **synthetic** threshold lines (not historical book prices). Saves to `models/v1/` or `models/v2/`.

### predict.py

```bash
python predict.py [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--version v1|v2]
```

Requires `data/processed/current_props.parquet` from `--props`.

### app.py

```bash
streamlit run app.py
```

Reads `predictions.csv` (V1) or `predictions_v2.csv` (V2) based on sidebar selection.

---

## V2 feature additions

Implemented in [`features_v2.py`](features_v2.py):

**Batters**
- Opponent team pitching: K, BB, hits allowed (season rolling, lagged)
- Batter stand (L/R)
- Hits vs LHP / RHP season rates
- Home park offense proxy

**Pitchers**
- Opponent team batting: hits, TB, HR, RBI (season rolling, lagged)
- Opponent team batter K rate
- Pitcher throwing hand (L/R)

---

## Project layout

```text
mlb-prop-model/
├── features_v2.py       # V2-only feature logic
├── build_features.py    # --version v1|v2
├── train.py
├── predict.py
├── app.py
├── models/
│   ├── v1/              # V1 models
│   └── v2/              # V2 models
├── data/
│   ├── raw/             # Statcast parquets
│   ├── processed/       # Features + current_props.parquet
│   └── predictions/     # predictions.csv, predictions_v2.csv
└── ../mlb-prop-model-v1/  # Frozen V1 copy (do not edit)
```

---

## Roadmap

| Version | Status | Focus |
|---------|--------|-------|
| V1 | Frozen in `-v1` folder | Rolling features, proof of pipeline |
| **V2** | **In progress** | Opponent, handedness, park |
| V3 | Planned | Consensus line, best price, line movement |
| V4 | Planned | Historical sportsbook lines for training |
| V5 | Planned | Distributional models |

---

## Git

Local git repo. V1 baseline is tagged; V2 work continues on `main`.

```bash
git status
git log --oneline
```

Never commit `.env`, `data/`, or `models/` (see `.gitignore`).

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `Unsupported version` | Use `--version v1` or `--version v2` |
| V2 models missing | Run `train.py --version v2` |
| V2 features missing | Run `build_features.py --version v2` |
| V1 predictions empty in app | Run `predict.py --version v1` |
| LightGBM libomp error | `brew install libomp` |
| API key error | Set `ODDS_API_KEY` in `.env` |

---

## Updating this README

Keep this file in sync when adding new CLI flags, paths, or workflow steps. Update the V1 README if frozen-folder paths change.
