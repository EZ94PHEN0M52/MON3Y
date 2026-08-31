#!/usr/bin/env bash
#
# MLB Prop Model V2 — daily pipeline
#
# Usage:
#   ./run_daily.sh              # ensure features → props → game lines → probables → predict
#   ./run_daily.sh --train      # also retrain models (not needed daily)
#   ./run_daily.sh --skip-props # skip Odds API prop fetch (use cached props)
#   ./run_daily.sh --skip-game-lines # skip game totals/spreads fetch
#   ./run_daily.sh --skip-probables # skip MLB probable SP fetch
#   ./run_daily.sh --include-today  # feature window through today (evening, after games + Statcast)
#   ./run_daily.sh --streamlit --include-today
#   ./run_daily.sh --streamlit --port 8502
#
# Intraday snapshots (Phase 4): fetch_data.py --props appends to
# data/raw/odds/snapshots/props_YYYYMMDD_HHMMSS.parquet on each run.
# Optional cron on game days (every 2–4 hours), e.g.:
#   0 8,12,16,20 * * * cd /path/to/mlb-prop-model && .venv/bin/python fetch_data.py --props >> logs/props_fetch.log 2>&1
#
# Distributional / dual-head regressor models (Phase 1): fit via
# ./run_evaluation.sh (not run on every daily pass). After fitting,
# run_daily.sh picks up models/v2/dist/*.pkl at predict time.
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# shellcheck source=/dev/null
source .venv/bin/activate

SEASON_START=2026-03-25

TRAIN_START=2025-04-01
TRAIN_END=2025-06-30

RUN_TRAIN=false
RUN_STREAMLIT=false
SKIP_PROPS=false
SKIP_GAME_LINES=false
SKIP_PROBABLES=false
INCLUDE_TODAY=false
STREAMLIT_PORT=8501

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train)
      RUN_TRAIN=true
      shift
      ;;
    --streamlit)
      RUN_STREAMLIT=true
      shift
      ;;
    --skip-props)
      SKIP_PROPS=true
      shift
      ;;
    --skip-game-lines)
      SKIP_GAME_LINES=true
      shift
      ;;
    --skip-probables)
      SKIP_PROBABLES=true
      shift
      ;;
    --include-today)
      INCLUDE_TODAY=true
      shift
      ;;
    --port)
      if [[ -z "${2:-}" || "$2" == --* ]]; then
        echo "Error: --port requires a port number" >&2
        exit 1
      fi
      STREAMLIT_PORT="$2"
      shift 2
      ;;
    -h|--help)
      sed -n '2,13p' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Usage: $0 [--train] [--skip-props] [--skip-game-lines] [--skip-probables] [--include-today] [--streamlit] [--port PORT]" >&2
      exit 1
      ;;
  esac
done

if $INCLUDE_TODAY; then
  FEATURE_END=$(date +%Y-%m-%d)
  FEATURE_END_LABEL="today"
else
  FEATURE_END=$(date -v-1d +%Y-%m-%d)
  FEATURE_END_LABEL="yesterday"
fi

echo "=== MLB Prop Model V2 pipeline ==="
echo "Season start: $SEASON_START | Feature end ($FEATURE_END_LABEL): $FEATURE_END"
echo ""

echo ">>> Checking inference features ($SEASON_START → $FEATURE_END, v2)..."
if ! python scripts/ensure_features.py \
  --start "$SEASON_START" \
  --end "$FEATURE_END" \
  --version v2 \
  --fix; then
  echo "Feature ensure failed for inference window; aborting pipeline." >&2
  exit 1
fi

echo ">>> Building PP fantasy score archive (L5/L10 source)..."
python scripts/build_pp_fantasy_scores.py --version v2

if $RUN_TRAIN; then
  echo ">>> Checking training features ($TRAIN_START → $TRAIN_END, v2)..."
  if ! python scripts/ensure_features.py \
    --start "$TRAIN_START" \
    --end "$TRAIN_END" \
    --version v2 \
    --fix; then
    echo "Feature ensure failed for training window; aborting pipeline." >&2
    exit 1
  fi
fi

if $SKIP_PROPS; then
  echo ">>> Skipping props fetch (--skip-props); using cached current_props.parquet"
else
  echo ">>> Fetching today's sportsbook props (also saves intraday snapshot)..."
  python fetch_data.py --props
fi

if $SKIP_GAME_LINES; then
  echo ">>> Skipping game lines fetch (--skip-game-lines); using cached current_game_lines.parquet"
else
  echo ">>> Fetching today's game totals and run lines..."
  python fetch_data.py --game-lines
fi

if $SKIP_PROBABLES; then
  echo ">>> Skipping probables fetch (--skip-probables); using cached daily_probables.parquet"
else
  echo ">>> Fetching today's probable starting pitchers..."
  python fetch_data.py --probables
fi

if $RUN_TRAIN; then
  echo ">>> Training V2 models ($TRAIN_START → $TRAIN_END, real lines when available)..."
  python train.py --start "$TRAIN_START" --end "$TRAIN_END" --version v2 --line-source auto
fi

echo ">>> Generating predictions..."
python predict.py --start "$SEASON_START" --end "$FEATURE_END" --version v2

echo ""
echo "=== Pipeline complete ==="

if $RUN_STREAMLIT; then
  echo ">>> Launching Streamlit at http://localhost:${STREAMLIT_PORT} ..."
  streamlit run app.py --server.port "$STREAMLIT_PORT"
fi
