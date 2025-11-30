#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <experiment_suite> [model_name]" >&2
  exit 1
fi

EXPERIMENT_SUITE="$1"
MODEL_OVERRIDE="${2:-}"

RESULTS_DIR=experiment_results
DETAIL_DIR="$RESULTS_DIR/detailed_explanations"
METRICS_DIR="$RESULTS_DIR/metrics_results"

mkdir -p "$RESULTS_DIR" "$DETAIL_DIR" "$METRICS_DIR"

cmd=(
  python -m src.cli.main "$EXPERIMENT_SUITE"
  --reuse-trained-models
  --use-tuned-params
  --write-detailed-explanations
  --detailed-output-dir "$DETAIL_DIR"
  --write-metric-results
  --metrics-output-dir "$METRICS_DIR"
  --output-dir "$RESULTS_DIR"
  --model-store-dir saved_models
  --log-level INFO
)

if [[ -n "$MODEL_OVERRIDE" ]]; then
  cmd+=(--model "$MODEL_OVERRIDE")
fi

start_time=$(date +%s)

"${cmd[@]}"

end_time=$(date +%s)
elapsed=$((end_time - start_time))
printf "Metrics run for '%s'%s finished in %02d:%02d:%02d\n" \
  "$EXPERIMENT_SUITE" \
  "${MODEL_OVERRIDE:+ (model: $MODEL_OVERRIDE)}" \
  $((elapsed / 3600)) $(((elapsed % 3600) / 60)) $((elapsed % 60))
