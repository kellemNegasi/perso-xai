#!/usr/bin/env bash
set -euo pipefail

# Ordered list of experiment suites to execute
EXPERIMENT_SUITES=(
  # openml_adult_suite # very large datasets, skip for now
  openml_bank_suite
  openml_german_suite
  open_compas_suite
)

MODEL_OVERRIDE="${1:-}"

BASE_RESULTS_DIR="results"
RUN_TIMESTAMP=$(date "+%d_%m_%d_%H_%M")
RESULTS_DIR="$BASE_RESULTS_DIR/$RUN_TIMESTAMP"
DETAIL_DIR="$RESULTS_DIR/detailed_explanations"
METRICS_DIR="$RESULTS_DIR/metrics_results"

mkdir -p "$DETAIL_DIR" "$METRICS_DIR"

for EXPERIMENT_SUITE in "${EXPERIMENT_SUITES[@]}"; do
  cmd=(
    python -m src.cli.main "$EXPERIMENT_SUITE"
    --reuse-trained-models
    --tune-models
    --use-tuned-params
    --write-detailed-explanations
    --detailed-output-dir "$DETAIL_DIR"
    --write-metric-results
    --skip-existing-experiments
    --skip-existing-methods
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
done
