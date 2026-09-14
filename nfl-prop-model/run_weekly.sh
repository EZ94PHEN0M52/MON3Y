#!/usr/bin/env bash
#
# NFL Prop Model — weekly pipeline (V1 / V2)
#
# Usage:
#   ./run_weekly.sh                      # default: version v2
#   ./run_weekly.sh --version v1
#   ./run_weekly.sh --skip-props         # refresh form/injuries; reuse cached props
#   ./run_weekly.sh --train --streamlit  # retrain + board
#   ./run_weekly.sh --force
#
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -d .venv ]]; then
  echo "Error: .venv missing. Create it with:" >&2
  echo "  python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt" >&2
  exit 1
fi

# shellcheck source=/dev/null
source .venv/bin/activate

RUN_TRAIN=false
RUN_STREAMLIT=false
SKIP_STATS=false
SKIP_INJURIES=false
SKIP_PROPS=false
FORCE=false
STREAMLIT_PORT=8501
SEASONS="2024 2025 2026"
WEEK=""
SEASON=""
VERSION="v2"

usage() {
  cat <<'EOF'
run_weekly.sh — NFL Prop Model weekly pipeline

Usage:
  ./run_weekly.sh [flags]

Default steps:
  1. fetch stats + schedule (+ snaps/team-stats for v2)
  2. fetch injuries → current_injuries.parquet
  3. fetch Odds API props → current_props.parquet
  4. build_features.py --version {v1|v2}
  5. predict.py --version {v1|v2}
  6. optional: train.py / streamlit

Flags:
  --version v1|v2   Model version (default: v2).
  --streamlit       Open Streamlit board after predict.
  --port N          Streamlit port (default 8501).
  --train           Retrain classifiers (first-time or schema change).
  --skip-stats      Skip nflverse stats/schedule/snaps/team-stats fetch.
  --skip-injuries   Skip injury refresh.
  --skip-props      Skip Odds API props (reuse cache).
  --force           Re-download raw caches even when files exist.
  --seasons A B C   Seasons for feature build (default: 2024 2025 2026).
  --season YYYY     Season for fetch scripts (default: nflverse current).
  --week N          Injury snapshot week (default: current week).
  -h, --help        Show this help.

Typical schedule:
  ./run_weekly.sh --streamlit
  ./run_weekly.sh --skip-props --streamlit
  ./run_weekly.sh --version v1 --train --streamlit
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train) RUN_TRAIN=true; shift ;;
    --streamlit) RUN_STREAMLIT=true; shift ;;
    --skip-stats) SKIP_STATS=true; shift ;;
    --skip-injuries) SKIP_INJURIES=true; shift ;;
    --skip-props) SKIP_PROPS=true; shift ;;
    --force) FORCE=true; shift ;;
    --version)
      if [[ -z "${2:-}" || "$2" == --* ]]; then
        echo "Error: --version requires v1 or v2" >&2
        exit 1
      fi
      VERSION="$2"
      shift 2
      ;;
    --port)
      if [[ -z "${2:-}" || "$2" == --* ]]; then
        echo "Error: --port requires a port number" >&2
        exit 1
      fi
      STREAMLIT_PORT="$2"
      shift 2
      ;;
    --seasons)
      SEASONS=""
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        SEASONS+="${SEASONS:+ }$1"
        shift
      done
      if [[ -z "$SEASONS" ]]; then
        echo "Error: --seasons requires at least one year" >&2
        exit 1
      fi
      ;;
    --season)
      if [[ -z "${2:-}" || "$2" == --* ]]; then
        echo "Error: --season requires a year" >&2
        exit 1
      fi
      SEASON="$2"
      shift 2
      ;;
    --week)
      if [[ -z "${2:-}" || "$2" == --* ]]; then
        echo "Error: --week requires a week number" >&2
        exit 1
      fi
      WEEK="$2"
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown flag: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "$VERSION" != "v1" && "$VERSION" != "v2" ]]; then
  echo "Error: --version must be v1 or v2 (got $VERSION)" >&2
  exit 1
fi

FORCE_ARGS=()
if [[ "$FORCE" == true ]]; then
  FORCE_ARGS+=(--force)
fi

SEASON_ARGS=()
if [[ -n "$SEASON" ]]; then
  SEASON_ARGS+=(--season "$SEASON")
fi

WEEK_ARGS=()
if [[ -n "$WEEK" ]]; then
  WEEK_ARGS+=(--week "$WEEK")
fi

echo
echo "============================================================"
echo "NFL PROP MODEL — WEEKLY PIPELINE ($VERSION)"
echo "============================================================"
echo "seasons (features): $SEASONS"
echo "train=$RUN_TRAIN streamlit=$RUN_STREAMLIT skip_stats=$SKIP_STATS skip_injuries=$SKIP_INJURIES skip_props=$SKIP_PROPS force=$FORCE"
echo

ensure_season_file() {
  local flag="$1"
  local yr="$2"
  local path="$3"
  if [[ ! -f "$path" || "$FORCE" == true ]]; then
    echo "  → ensuring $flag for $yr"
    python fetch_data.py "$flag" --season "$yr" "${FORCE_ARGS[@]}"
  fi
}

# ----------------------------------------------------------
# 1. Stats + schedule (+ V2 extras)
# ----------------------------------------------------------
if [[ "$SKIP_STATS" == true ]]; then
  echo "[1/5] Skipping stats/schedule fetch"
else
  echo "[1/5] Fetching nflverse stats + schedule…"
  python fetch_data.py --stats --schedule "${SEASON_ARGS[@]}" "${FORCE_ARGS[@]}"
  if [[ "$VERSION" == "v2" ]]; then
    python fetch_data.py --snaps --team-stats "${SEASON_ARGS[@]}" "${FORCE_ARGS[@]}"
  fi
  for yr in $SEASONS; do
    if [[ -n "$SEASON" && "$yr" == "$SEASON" ]]; then
      continue
    fi
    ensure_season_file --stats "$yr" "data/raw/player_stats_${yr}.parquet"
    ensure_season_file --schedule "$yr" "data/raw/schedules_${yr}.parquet"
    if [[ "$VERSION" == "v2" ]]; then
      ensure_season_file --snaps "$yr" "data/raw/snap_counts_${yr}.parquet"
      ensure_season_file --team-stats "$yr" "data/raw/team_stats_${yr}.parquet"
    fi
  done
fi

# ----------------------------------------------------------
# 2. Injuries
# ----------------------------------------------------------
if [[ "$SKIP_INJURIES" == true ]]; then
  echo "[2/5] Skipping injury refresh"
else
  echo "[2/5] Refreshing injuries + roster snapshot…"
  python fetch_data.py --injuries "${SEASON_ARGS[@]}" "${WEEK_ARGS[@]}" "${FORCE_ARGS[@]}"
fi

# ----------------------------------------------------------
# 3. Props
# ----------------------------------------------------------
if [[ "$SKIP_PROPS" == true ]]; then
  echo "[3/5] Skipping props fetch (using cached current_props.parquet)"
  if [[ ! -f data/processed/current_props.parquet ]]; then
    echo "Error: no cached current_props.parquet — run without --skip-props" >&2
    exit 1
  fi
else
  echo "[3/5] Fetching Odds API props…"
  python fetch_data.py --props
fi

# ----------------------------------------------------------
# 4–5. Features / train / predict
# ----------------------------------------------------------
echo "[4/5] Building features ($VERSION)…"
# shellcheck disable=SC2086
python build_features.py --seasons $SEASONS --version "$VERSION"

if [[ "$RUN_TRAIN" == true ]]; then
  echo "[4b/5] Training $VERSION models…"
  python train.py --version "$VERSION"
fi

echo "[5/5] Predicting ($VERSION)…"
python predict.py --version "$VERSION"

PRED_FILE="data/predictions/predictions_${VERSION}.csv"
if [[ -f "$PRED_FILE" ]]; then
  ROWS=$(python -c "import pandas as pd; print(len(pd.read_csv('$PRED_FILE')))")
  echo
  echo "Done. Predictions: $PRED_FILE ($ROWS rows)"
else
  echo "Error: predict finished but $PRED_FILE is missing" >&2
  exit 1
fi

if [[ "$RUN_STREAMLIT" == true ]]; then
  echo
  echo "Starting Streamlit on port $STREAMLIT_PORT…"
  exec streamlit run app.py --server.port "$STREAMLIT_PORT"
fi
