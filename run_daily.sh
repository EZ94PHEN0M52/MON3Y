#!/usr/bin/env bash
#
# MLB Prop Model V2 — daily pipeline
#
# Usage:
#   ./run_daily.sh              # ensure features → props → predict
#   ./run_daily.sh --train      # also retrain models (not needed daily)
#   ./run_daily.sh --skip-props # skip Odds API fetch (use cached props)
#   ./run_daily.sh --streamlit  # launch Streamlit app after pipeline
#   ./run_daily.sh --train --streamlit
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# shellcheck source=/dev/null
source .venv/bin/activate

SEASON_START=2026-03-25
YESTERDAY=$(date -v-1d +%Y-%m-%d)

TRAIN_START=2025-04-01
TRAIN_END=2025-06-30

RUN_TRAIN=false
RUN_STREAMLIT=false
SKIP_PROPS=false

for arg in "$@"; do
  case "$arg" in
    --train)
      RUN_TRAIN=true
      ;;
    --streamlit)
      RUN_STREAMLIT=true
      ;;
    --skip-props)
      SKIP_PROPS=true
      ;;
    -h|--help)
      sed -n '2,10p' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *)
      echo "Unknown option: $arg" >&2
      echo "Usage: $0 [--train] [--skip-props] [--streamlit]" >&2
      exit 1
      ;;
  esac
done

echo "=== MLB Prop Model V2 pipeline ==="
echo "Season start: $SEASON_START | End (yesterday): $YESTERDAY"
echo ""

echo ">>> Checking inference features ($SEASON_START → $YESTERDAY, v2)..."
if ! python scripts/ensure_features.py \
  --start "$SEASON_START" \
  --end "$YESTERDAY" \
  --version v2 \
  --fix; then
  echo "Feature ensure failed for inference window; aborting pipeline." >&2
  exit 1
fi

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
  echo ">>> Fetching today's sportsbook props..."
  python fetch_data.py --props
fi

if $RUN_TRAIN; then
  echo ">>> Training V2 models ($TRAIN_START → $TRAIN_END)..."
  python train.py --start "$TRAIN_START" --end "$TRAIN_END" --version v2
fi

echo ">>> Generating predictions..."
python predict.py --start "$SEASON_START" --end "$YESTERDAY" --version v2

echo ""
echo "=== Pipeline complete ==="

if $RUN_STREAMLIT; then
  echo ">>> Launching Streamlit at http://localhost:8501 ..."
  streamlit run app.py
fi
