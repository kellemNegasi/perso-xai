#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./generate_pareto_front.sh [metrics_dir] [output_dir] [target_path]
# - metrics_dir: root directory containing per-method metric JSONs
#                (default: results/full_run_dec8/metrics_results)
# - output_dir : where Pareto summaries will be written
#                (default: results/full_run_dec8/pareto_fronts)
# - target_path: optional dataset/model path to limit processing.
#                When omitted, every dataset/model under metrics_dir is processed.

METRICS_DIR=${1:-results/hc_combo_20251228_050331/metrics_results}
OUTPUT_DIR=${2:-results/hc_combo_20251228_050331/pareto_fronts}
TARGET_PATH=${3:-}

PYTHON_BIN=${PYTHON_BIN:-python3}

CMD=(
  "$PYTHON_BIN" generate_pareto_fronts.py
  --mode metrics
  --metrics-dir "$METRICS_DIR"
  --output-dir "$OUTPUT_DIR"
)

if [[ -n "$TARGET_PATH" ]]; then
  CMD+=("$TARGET_PATH")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
