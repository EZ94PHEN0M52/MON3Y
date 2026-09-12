#!/usr/bin/env bash
#
# Stuff K (v2) — build features → train stuff model → predict
#
# Separate from the main pitcher_strikeouts LightGBM path. Populates the
# board "Stuff K (v2)" column on strikeout props.
#
# Usage:
#   ./run_pitcher_strikeout_stuff.sh
#   ./run_pitcher_strikeout_stuff.sh --skip-features   # features already built
#   ./run_pitcher_strikeout_stuff.sh --skip-fit        # model already trained
#   ./run_pitcher_strikeout_stuff.sh --streamlit       # open board after predict
#   ./run_pitcher_strikeout_stuff.sh --help
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# shellcheck source=/dev/null
source .venv/bin/activate

SEASON_START=2026-03-25
YESTERDAY=$(date -v-1d +%Y-%m-%d)
VERSION=v2

START="$SEASON_START"
END="$YESTERDAY"

RUN_FEATURES=true
RUN_FIT=true
RUN_PREDICT=true
RUN_STREAMLIT=false
STREAMLIT_PORT=8501

usage() {
  cat <<'EOF'
run_pitcher_strikeout_stuff.sh — Stuff K (v2) pipeline

Usage:
  ./run_pitcher_strikeout_stuff.sh [flags]

Populates the board "Stuff K (v2)" column on pitcher strikeout props.
Separate from the main pitcher_strikeouts LightGBM classifier.

Flags:
  --start YYYY-MM-DD   Feature window start (default: season start).
  --end YYYY-MM-DD     Feature window end (default: yesterday).
  --skip-features      Skip ensure_features (Statcast + stuff columns).
  --skip-fit           Skip fit_pitcher_strikeout_stuff.py (model already trained).
  --skip-predict       Skip predict.py refresh.
  --streamlit          Open board after predict.
  --port N             Streamlit port (default: 8501).

Steps (default: all enabled):
  1. scripts/ensure_features.py --fix
  2. scripts/fit_pitcher_strikeout_stuff.py → models/v2/pitcher_strikeouts_stuff.pkl
  3. predict.py → Stuff K (v2) on strikeout rows

Run once, then ./run_daily.sh picks up the stuff model automatically.

Run ./help.sh for all pipeline commands.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --start)
      START="$2"
      shift 2
      ;;
    --end)
      END="$2"
      shift 2
      ;;
    --version)
      VERSION="$2"
      shift 2
      ;;
    --skip-features)
      RUN_FEATURES=false
      shift
      ;;
    --skip-fit)
      RUN_FIT=false
      shift
      ;;
    --skip-predict)
      RUN_PREDICT=false
      shift
      ;;
    --streamlit)
      RUN_STREAMLIT=true
      shift
      ;;
    --port)
      STREAMLIT_PORT="$2"
      shift 2
      ;;
    -h|-help|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Run $0 --help" >&2
      exit 1
      ;;
  esac
done

echo ""
echo "========================================================================"
echo "STUFF STRIKEOUT MODEL (v2)"
echo "========================================================================"
echo "Feature window:  $START → $END ($VERSION)"
echo ""

if [[ "$RUN_FEATURES" == true ]]; then
  echo ">>> Step 1/3 — ensure features (Statcast fetch + SwStr / chase / velocity)"
  if ! python scripts/ensure_features.py \
    --start "$START" \
    --end "$END" \
    --version "$VERSION" \
    --fix; then
    echo "Feature ensure failed; aborting stuff pipeline." >&2
    exit 1
  fi
  echo ""
fi

if [[ "$RUN_FIT" == true ]]; then
  echo ">>> Step 2/3 — train stuff → K model"
  python scripts/fit_pitcher_strikeout_stuff.py \
    --start "$START" \
    --end "$END" \
    --version "$VERSION"
  echo ""
fi

if [[ "$RUN_PREDICT" == true ]]; then
  echo ">>> Step 3/3 — predict (Stuff K v2 column on strikeout rows)"
  python predict.py \
    --start "$START" \
    --end "$END" \
    --version "$VERSION"
  echo ""
fi

if [[ "$RUN_STREAMLIT" == true ]]; then
  echo ">>> Launching Streamlit"
  streamlit run app.py --server.port "$STREAMLIT_PORT"
fi

echo "Done. Filter the main board to Strikeouts — look for the Stuff K (v2) column."
echo "Model: models/v2/pitcher_strikeouts_stuff.pkl"
