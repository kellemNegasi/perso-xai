#!/usr/bin/env bash
set -euo pipefail

# Local runner equivalent to `granular_submit.sh` (no Slurm).
# Runs one process per (suite, model) in parallel (local background jobs).
#
# Usage:
#   ./run_local_autoxai_all.sh [run_id]
#
# Optional env overrides:
#   SUITES="autoxai_openml_adult_suite autoxai_openml_bank_suite ..."
#   MODELS="decision_tree random_forest ..."
#   JOBS=4                      # max parallel jobs (default: nproc)
#   MAX_INSTANCES=100
#   BASE_RESULTS_DIR=results
#   MODEL_STORE_DIR=...         # optional override (defaults to saved_models)
#   TUNING_OUTPUT_DIR=...       # optional override (defaults to <MODEL_STORE_DIR>/tuning_results)

RUN_ID="${1:-hc_combo_$(date +%Y%m%d_%H%M%S)}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  if [[ -f ".venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
  fi
fi

if ! command -v python >/dev/null 2>&1; then
  echo "ERROR: python not found. Activate your venv first (e.g. 'source .venv/bin/activate')." >&2
  exit 1
fi

if [[ -n "${SUITES:-}" ]]; then
  read -r -a SUITES_ARR <<<"${SUITES}"
else
  SUITES_ARR=(
    # autoxai_openml_adult_suite
    autoxai_openml_bank_suite
    autoxai_openml_german_suite
    autoxai_open_compas_suite
  )
fi

if [[ -n "${MODELS:-}" ]]; then
  read -r -a MODELS_ARR <<<"${MODELS}"
else
  MODELS_ARR=(
    decision_tree
    random_forest
    mlp_classifier
    gradient_boosting
    logistic_regression
    svm_rbf
  )
fi
JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"

BASE_RESULTS_DIR="${BASE_RESULTS_DIR:-results}"
LOG_DIR="${BASE_RESULTS_DIR}/${RUN_ID}/logs"
mkdir -p "${LOG_DIR}"

export BASE_RESULTS_DIR

echo "[local-run] RUN_ID=${RUN_ID}"
echo "[local-run] suites: ${SUITES_ARR[*]}"
echo "[local-run] models: ${MODELS_ARR[*]}"
echo "[local-run] JOBS=${JOBS}"

running=0
failures=0
declare -a pids=()
declare -a labels=()

start_job() {
  local suite="$1"
  local model="$2"
  local label="local__${suite}__${model}"
  local log_path="${LOG_DIR}/${label}.log"
  echo "[local-run] start ${label} (log: ${log_path})"
  (
    bash scripts/run_metrics_task.sh "${suite}" "${model}" "${RUN_ID}"
  ) >"${log_path}" 2>&1 &
  pids+=("$!")
  labels+=("${label}")
  running=$((running + 1))
}

wait_one() {
  # wait -n is available in bash 4.3+
  if wait -n; then
    :
  else
    failures=$((failures + 1))
  fi
  running=$((running - 1))
}

for SUITE in "${SUITES_ARR[@]}"; do
  for MODEL in "${MODELS_ARR[@]}"; do
    while [[ "${running}" -ge "${JOBS}" ]]; do
      wait_one
    done
    start_job "${SUITE}" "${MODEL}"
  done
done

while [[ "${running}" -gt 0 ]]; do
  wait_one
done

if [[ "${failures}" -ne 0 ]]; then
  echo "ERROR: ${failures} job(s) failed. Check logs under ${LOG_DIR}" >&2
  exit 1
fi

echo "[local-run] all jobs completed successfully."
