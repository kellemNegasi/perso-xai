#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <experiment_suite> [model_name]" >&2
  exit 1
fi

EXPERIMENT_SUITE="$1"
MODEL_OVERRIDE="${2:-}"

cmd=(
  python -m src.cli.main "$EXPERIMENT_SUITE"
  --reuse-trained-models
  --use-tuned-params
  --reuse-detailed-explanations
  --write-detailed-explanations
  --detailed-output-dir saved_models/detailed_explanations
  --write-metric-results
  --metrics-output-dir saved_models/metrics_results
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
