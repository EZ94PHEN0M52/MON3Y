#!/usr/bin/env bash
#
# Track 1 — pitcher_outs self-learning loop
#
# Runs the full collect → join → retrain → re-predict cycle for Pitcher Outs.
# Logging is automatic during predict; this script wires the offline steps.
#
# Usage:
#   ./run_pitcher_outs_learning.sh
#   ./run_pitcher_outs_learning.sh --skip-daily          # outcomes + retrain only
#   ./run_pitcher_outs_learning.sh --skip-retrain        # daily + join + re-predict
#   ./run_pitcher_outs_learning.sh --fit-distributional  # also train Poisson dist head
#   ./run_pitcher_outs_learning.sh --streamlit           # open board after pipeline
#   ./run_pitcher_outs_learning.sh --help
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# shellcheck source=/dev/null
source .venv/bin/activate

SEASON_START=2026-03-25
YESTERDAY=$(date -v-1d +%Y-%m-%d)
# Retrain on current-season features (same window as predict / run_daily season pass).
TRAIN_START=$SEASON_START
TRAIN_END=$YESTERDAY
VERSION=v2

RUN_DAILY=true
RUN_JOIN=true
RUN_RETRAIN=true
RUN_REPREDICT=true
FIT_DIST=false
RUN_STREAMLIT=false
SKIP_PROPS=true
SKIP_GAME_LINES=false
SKIP_PROBABLES=false
STREAMLIT_PORT=8501
JOIN_START=""
JOIN_END=""
TRAIN_START_OVERRIDE=""
TRAIN_END_OVERRIDE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-daily)
      RUN_DAILY=false
      shift
      ;;
    --skip-join)
      RUN_JOIN=false
      shift
      ;;
    --skip-retrain)
      RUN_RETRAIN=false
      shift
      ;;
    --skip-repredict)
      RUN_REPREDICT=false
      shift
      ;;
    --fit-distributional)
      FIT_DIST=true
      shift
      ;;
    --streamlit)
      RUN_STREAMLIT=true
      shift
      ;;
    --fetch-props)
      SKIP_PROPS=false
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
    --join-start)
      JOIN_START="$2"
      shift 2
      ;;
    --join-end)
      JOIN_END="$2"
      shift 2
      ;;
    --train-start)
      TRAIN_START_OVERRIDE="$2"
      shift 2
      ;;
    --train-end)
      TRAIN_END_OVERRIDE="$2"
      shift 2
      ;;
    --port)
      STREAMLIT_PORT="$2"
      shift 2
      ;;
    -h|--help)
      sed -n '2,14p' "$0" | sed 's/^# \?//'
      echo ""
      echo "Steps (default: all enabled):"
      echo "  1. ./run_daily.sh [--skip-props ...]  → ensure features + predict (logs pitcher_outs)"
      echo "  2. scripts/log_outcomes.py          → join actual outs to predictions log"
      echo "  3. scripts/retrain_market.py          → retrain pitcher_outs classifier"
      echo "  4. predict.py                         → refresh predictions CSV for the board"
      echo ""
      echo "Optional: --fit-distributional runs fit_distributional.py --market pitcher_outs after step 3."
      echo "Board: Pred # and Dist Over % populate for outs rows when dist model exists."
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Run $0 --help" >&2
      exit 1
      ;;
  esac
done

JOIN_START="${JOIN_START:-$SEASON_START}"
JOIN_END="${JOIN_END:-$YESTERDAY}"
TRAIN_START="${TRAIN_START_OVERRIDE:-$TRAIN_START}"
TRAIN_END="${TRAIN_END_OVERRIDE:-$TRAIN_END}"

echo ""
echo "========================================================================"
echo "PITCHER OUTS LEARNING LOOP"
echo "========================================================================"
echo "Season features through: $YESTERDAY"
echo "Outcome join window:     $JOIN_START → $JOIN_END"
echo "Retrain window:          $TRAIN_START → $TRAIN_END ($VERSION)"
echo ""

if [[ "$RUN_DAILY" == true ]]; then
  echo ">>> Step 1/4 — daily pipeline (predict + auto-log pitcher_outs)"
  DAILY_ARGS=()
  [[ "$SKIP_PROPS" == true ]] && DAILY_ARGS+=(--skip-props)
  [[ "$SKIP_GAME_LINES" == true ]] && DAILY_ARGS+=(--skip-game-lines)
  [[ "$SKIP_PROBABLES" == true ]] && DAILY_ARGS+=(--skip-probables)
  [[ "$RUN_STREAMLIT" == true ]] && DAILY_ARGS+=(--streamlit --port "$STREAMLIT_PORT")
  ./run_daily.sh "${DAILY_ARGS[@]}"
  echo ""
fi

if [[ "$RUN_JOIN" == true ]]; then
  echo ">>> Step 2/4 — join post-game outs to predictions log"
  python scripts/log_outcomes.py \
    --market pitcher_outs \
    --start "$JOIN_START" \
    --end "$JOIN_END" \
    --version "$VERSION"
  echo ""
fi

if [[ "$RUN_RETRAIN" == true ]]; then
  echo ">>> Step 3/4 — retrain pitcher_outs classifier"
  python scripts/retrain_market.py \
    --market pitcher_outs \
    --start "$TRAIN_START" \
    --end "$TRAIN_END" \
    --version "$VERSION"
  if [[ "$FIT_DIST" == true ]]; then
    echo ""
    echo ">>> Step 3b — fit Poisson regressor (dual-head Pred # / Dist Over %)"
    python scripts/fit_distributional.py \
      --start "$TRAIN_START" \
      --end "$TRAIN_END" \
      --version "$VERSION" \
      --market pitcher_outs
  fi
  echo ""
fi

if [[ "$RUN_REPREDICT" == true ]]; then
  echo ">>> Step 4/4 — re-predict (board picks up new outs model"
  if [[ "$FIT_DIST" == true ]]; then
    echo "    and distributional Pred # / Dist Over % when model exists)"
  else
    echo "    probabilities; run with --fit-distributional for Pred # columns)"
  fi
  python predict.py \
    --start "$SEASON_START" \
    --end "$YESTERDAY" \
    --version "$VERSION"
  echo ""
fi

if [[ "$RUN_STREAMLIT" == true && "$RUN_DAILY" == false ]]; then
  echo ">>> Launching Streamlit (predictions already refreshed)"
  streamlit run app.py --server.port "$STREAMLIT_PORT"
fi

echo "Done. Filter the main board to Pitcher Outs to inspect changes."
echo "Learning logs: data/learning/predictions_log.parquet, outcomes_log.parquet"
