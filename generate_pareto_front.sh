#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./generate_pareto_front.sh [metrics_dir] [output_dir] [target_path]
# - metrics_dir: root directory containing per-method metric JSONs
#                (default: $RESULTS_ROOT/metrics_results)
# - output_dir : where Pareto summaries will be written
#                (default: $RESULTS_ROOT/pareto_fronts)
# - target_path: optional dataset/model path to limit processing.
#                When omitted, every dataset/model under metrics_dir is processed.

RESULTS_ROOT=${RESULTS_ROOT:-results/hc_combo_20260110_024805}
METRICS_DIR=${1:-${RESULTS_ROOT}/metrics_results}
OUTPUT_DIR=${2:-${RESULTS_ROOT}/pareto_fronts_new}
TARGET_PATH=${3:-}

PYTHON_BIN=${PYTHON_BIN:-python3}

CMD=(
  "$PYTHON_BIN" generate_pareto_fronts.py
  --mode metrics
  --metrics-dir "$METRICS_DIR"
  --output-dir "$OUTPUT_DIR"
  --methods autoxai_lime autoxai_shap
)

if [[ -n "$TARGET_PATH" ]]; then
  CMD+=("$TARGET_PATH")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
