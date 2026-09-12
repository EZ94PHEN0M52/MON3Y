#!/usr/bin/env bash
#
# MLB Prop Model V2 — Phase 6 evaluation pipeline
#
# Usage:
#   ./run_evaluation.sh                                    # default training window
#   ./run_evaluation.sh --start 2025-04-01 --end 2025-06-30
#   ./run_evaluation.sh --version v2 --min-edge 0.03
#   ./run_evaluation.sh --min-ev 0.05
#   ./run_evaluation.sh --help
#
# Runs backtest → fit_calibrators (--from-csv) → fit_distributional in order.
# Separate from run_daily.sh — use after historical props are fetched and models trained.
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f .venv/bin/activate ]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
fi

TRAIN_START=2025-04-01
TRAIN_END=2025-06-30

START="$TRAIN_START"
END="$TRAIN_END"
VERSION="v2"
MIN_EDGE=""
MIN_EV=""

usage() {
  cat <<'EOF'
run_evaluation.sh — Phase 6 evaluation pipeline (offline)

Usage:
  ./run_evaluation.sh [flags]

Runs in order: backtest → fit_calibrators → fit_distributional.
Separate from run_daily.sh — use when historical props and models exist.

Flags:
  --start YYYY-MM-DD   Backtest / calibration window start (default: 2025-04-01).
  --end YYYY-MM-DD     Window end (default: 2025-06-30).
  --version v1|v2      Model version (default: v2).
  --min-edge N         Minimum edge filter passed to backtest.py.
  --min-ev N           Minimum EV filter passed to backtest.py.

Does not fetch live props or regenerate today's board. After evaluation, run
./run_daily.sh so live predict picks up new calibrators and dist models.

Run ./help.sh for all pipeline commands.
EOF
}

while [ $# -gt 0 ]; do
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
    --min-edge)
      MIN_EDGE="$2"
      shift 2
      ;;
    --min-ev)
      MIN_EV="$2"
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

BACKTEST_CSV="data/backtest/backtest_${START}_${END}.csv"

BACKTEST_EXTRA=()
if [ -n "$MIN_EDGE" ]; then
  BACKTEST_EXTRA+=(--min-edge "$MIN_EDGE")
fi
if [ -n "$MIN_EV" ]; then
  BACKTEST_EXTRA+=(--min-ev "$MIN_EV")
fi

echo "=== MLB Prop Model V2 — evaluation pipeline (Phase 6) ==="
echo "Window: $START → $END | Version: $VERSION"
echo ""

echo ">>> Step 1/3: Backtest historical props (outcomes + CLV)..."
python scripts/backtest.py \
  --start "$START" \
  --end "$END" \
  --version "$VERSION" \
  "${BACKTEST_EXTRA[@]}"

echo ""
echo ">>> Step 2/3: Fit probability calibrators from backtest CSV..."
python scripts/fit_calibrators.py \
  --start "$START" \
  --end "$END" \
  --version "$VERSION" \
  --from-csv "$BACKTEST_CSV"

echo ""
echo ">>> Step 3/3: Train distributional Poisson rate models..."
python scripts/fit_distributional.py \
  --start "$START" \
  --end "$END" \
  --version "$VERSION"

echo ""
echo "=== Evaluation pipeline complete ==="
echo "Backtest CSV: $BACKTEST_CSV"
echo "Calibrators:  models/${VERSION}/calibrators/"
echo "Dist models:  models/${VERSION}/dist/"
