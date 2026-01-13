#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <experiment_suite> <model_name> [run_id]" >&2
  exit 1
fi

EXPERIMENT_SUITE="$1"
MODEL_NAME="$2"
RUN_ID="${3:-${RUN_ID:-}}"

BASE_RESULTS_DIR="${BASE_RESULTS_DIR:-results}"
if [[ -z "${RUN_ID}" ]]; then
  RUN_ID="$(date "+%Y_%m_%d_%H_%M_%S")"
fi

RESULTS_DIR="${BASE_RESULTS_DIR}/${RUN_ID}"
DETAIL_DIR="${RESULTS_DIR}/detailed_explanations"
METRICS_DIR="${RESULTS_DIR}/metrics_results"

mkdir -p "${DETAIL_DIR}" "${METRICS_DIR}"

MODEL_STORE_DIR="${MODEL_STORE_DIR:-saved_models}"
TUNING_OUTPUT_DIR="${TUNING_OUTPUT_DIR:-${MODEL_STORE_DIR}/tuning_results}"

	cmd=(
	  python -m src.cli.main "${EXPERIMENT_SUITE}"
	  --reuse-trained-models
	  --tune-models
	  --use-tuned-params
	  --write-detailed-explanations
	  --detailed-output-dir "${DETAIL_DIR}"
	  --write-metric-results
	  --skip-existing-experiments
	  --skip-existing-methods
	  --metrics-output-dir "${METRICS_DIR}"
	  --output-dir "${RESULTS_DIR}"
	  --experiment-results-subdir experiment_results
	  --model-store-dir "${MODEL_STORE_DIR}"
	  --tuning-output-dir "${TUNING_OUTPUT_DIR}"
	  --log-level INFO
	  --model "${MODEL_NAME}"
	)

if [[ -n "${MAX_INSTANCES:-}" ]]; then
  cmd+=(--max-instances "${MAX_INSTANCES}")
fi

start_time=$(date +%s)
"${cmd[@]}"
end_time=$(date +%s)

elapsed=$((end_time - start_time))
printf "Metrics run for '%s' (model: %s, run_id: %s) finished in %02d:%02d:%02d\n" \
  "${EXPERIMENT_SUITE}" \
  "${MODEL_NAME}" \
  "${RUN_ID}" \
  $((elapsed / 3600)) $(((elapsed % 3600) / 60)) $((elapsed % 60))
