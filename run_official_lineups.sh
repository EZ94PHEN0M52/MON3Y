#!/usr/bin/env bash
#
# Fetch Rotowire **Today's Lineup** for today's slate and update board cache.
#
# Run 1–2 hours before first pitch (and again after lineups post). Boards use
# official lineups when cached; otherwise they fall back to default vs RHP/LHP.
#
# Usage:
#   ./run_official_lineups.sh
#   ./run_official_lineups.sh --dry-run
#   ./run_official_lineups.sh --teams NYY,BOS,LAD
#   ./run_official_lineups.sh --watch 300          # poll every 5 minutes
#   ./run_official_lineups.sh --watch 300 --max-runs 12
#   ./run_official_lineups.sh --help
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

WATCH_INTERVAL=""
MAX_RUNS=0
PYTHON_ARGS=()

usage() {
  sed -n '2,14p' "$0" | sed 's/^# \?//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --watch)
      if [[ -z "${2:-}" || "$2" == --* ]]; then
        echo "Error: --watch requires interval seconds" >&2
        exit 1
      fi
      WATCH_INTERVAL="$2"
      shift 2
      ;;
    --max-runs)
      if [[ -z "${2:-}" || "$2" == --* ]]; then
        echo "Error: --max-runs requires a number" >&2
        exit 1
      fi
      MAX_RUNS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      PYTHON_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ! -d ".venv" ]]; then
  echo "Error: .venv not found. Create the virtualenv before running." >&2
  exit 1
fi

# shellcheck source=/dev/null
source .venv/bin/activate

if [[ "${DISABLE_LIVE_FETCH:-}" == "1" ]]; then
  echo "Error: DISABLE_LIVE_FETCH=1 blocks Rotowire downloads. Unset it first." >&2
  exit 1
fi

if [[ ! -f "data/processed/current_props.parquet" ]]; then
  has_teams=false
  for arg in "${PYTHON_ARGS[@]}"; do
    if [[ "$arg" == --teams* ]]; then
      has_teams=true
      break
    fi
  done
  if [[ "$has_teams" == false ]]; then
    echo "Warning: data/processed/current_props.parquet missing." >&2
    echo "Run ./run_daily.sh first, or pass --teams ABBR1,ABBR2,…" >&2
    exit 1
  fi
fi

run_once() {
  echo "============================================================"
  echo "Rotowire official lineups — $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "============================================================"
  PYTHONPATH="$SCRIPT_DIR" python scripts/update_official_lineups.py \
    --retries 2 \
    --retry-wait 5 \
    "${PYTHON_ARGS[@]}"
}

if [[ -z "$WATCH_INTERVAL" ]]; then
  run_once
  exit $?
fi

if ! [[ "$WATCH_INTERVAL" =~ ^[0-9]+$ ]] || [[ "$WATCH_INTERVAL" -lt 30 ]]; then
  echo "Error: --watch interval must be an integer ≥ 30 seconds" >&2
  exit 1
fi

run_count=0
while true; do
  run_once
  exit_code=$?
  run_count=$((run_count + 1))

  if [[ "$MAX_RUNS" -gt 0 && "$run_count" -ge "$MAX_RUNS" ]]; then
    echo "Reached --max-runs $MAX_RUNS; exiting."
    exit "$exit_code"
  fi

  echo "Sleeping ${WATCH_INTERVAL}s before next check…"
  sleep "$WATCH_INTERVAL"
done
