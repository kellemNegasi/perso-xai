#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./generate_pareto_front.sh [metrics_dir] [output_dir] [target_path]
# - metrics_dir: root directory containing per-method metric JSONs
#                (default: experiment_results/metrics_results)
# - output_dir : where Pareto summaries will be written
#                (default: experiment_results/pareto_fronts)
# - target_path: optional dataset/model path to limit processing.
#                When omitted, every dataset/model under metrics_dir is processed.

METRICS_DIR=${1:-experiment_results/metrics_results}
OUTPUT_DIR=${2:-experiment_results/pareto_fronts}
TARGET_PATH=${3:-}

CMD=(
  python generate_pareto_fronts.py
  --mode metrics
  --metrics-dir "$METRICS_DIR"
  --output-dir "$OUTPUT_DIR"
)

if [[ -n "$TARGET_PATH" ]]; then
  CMD+=("$TARGET_PATH")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
