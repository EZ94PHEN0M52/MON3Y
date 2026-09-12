#!/usr/bin/env bash
#
# Pipeline help index — list entry points or show flags for one script.
#
# Usage:
#   ./help.sh
#   ./help.sh fetch_data
#   ./help.sh run_daily
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

show_index() {
  cat <<'EOF'
MLB Prop Model — pipeline help index

Run ./help.sh <name> or the command below with --help / -help

Shell scripts (run from project root):
  ./help.sh run_daily              Daily V2 pipeline (props, predict, optional Streamlit)
  ./help.sh run_evaluation         Phase 6 backtest + calibrators + distributional fit
  ./help.sh run_pitcher_outs_learning   Track 1 K / walks / outs learning loop
  ./help.sh run_pitcher_strikeout_stuff Stuff K (v2) feature + train + predict
  ./help.sh run_official_lineups   Rotowire official lineups for Hitter's Life

Python entry points:
  ./help.sh fetch_data             Statcast, props, game lines, probables, Sleeper
  ./help.sh predict                Score props → predictions CSV

Examples:
  ./run_daily.sh --help
  python fetch_data.py --help
  python fetch_data.py -help
  ./help.sh fetch_data
EOF
}

run_help() {
  local name="$1"
  case "$name" in
    run_daily|daily)
      ./run_daily.sh --help
      ;;
    run_evaluation|evaluation|eval)
      ./run_evaluation.sh --help
      ;;
    run_pitcher_outs_learning|outs_learning|learning)
      ./run_pitcher_outs_learning.sh --help
      ;;
    run_pitcher_strikeout_stuff|stuff|stuff_k)
      ./run_pitcher_strikeout_stuff.sh --help
      ;;
    run_official_lineups|lineups|rotowire)
      ./run_official_lineups.sh --help
      ;;
    fetch_data|fetch)
      if [ -f .venv/bin/activate ]; then
        # shellcheck source=/dev/null
        source .venv/bin/activate
      fi
      python fetch_data.py --help
      ;;
    predict)
      if [ -f .venv/bin/activate ]; then
        # shellcheck source=/dev/null
        source .venv/bin/activate
      fi
      python predict.py --help
      ;;
    *)
      echo "Unknown help target: $name" >&2
      echo "Run ./help.sh with no arguments for the index." >&2
      exit 1
      ;;
  esac
}

if [[ $# -eq 0 ]]; then
  show_index
  exit 0
fi

if [[ "$1" == "-h" || "$1" == "--help" || "$1" == "-help" || "$1" == "help" ]]; then
  show_index
  exit 0
fi

run_help "$1"
