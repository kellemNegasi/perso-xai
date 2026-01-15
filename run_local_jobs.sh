#!/usr/bin/env bash
set -euo pipefail

# Local runner equivalent to `granular_submit.sh` (no Slurm).
# Runs (suite, model) tasks with bounded parallelism.
#
# Usage:
#   ./run_local_autoxai_all.sh [run_id]
#
# Optional env overrides:
#   SUITES="suite1 suite2 ..."
#   MODELS="model1 model2 ..."
#   JOBS=1                      # 1 = sequential (default), 2/3/... = parallelism
#   MAX_INSTANCES=100
#   BASE_RESULTS_DIR=results
#   MODEL_STORE_DIR=...         # optional override (defaults to saved_models)
#   TUNING_OUTPUT_DIR=...       # optional override (defaults to <MODEL_STORE_DIR>/tuning_results)

RUN_ID="${1:-hc_combo_$(date +%Y%m%d_%H%M%S)}"
# RUN_ID="hc_combo_20260110_024805_testing"  # fixed for testing
# RUN_ID="hc_combo_20260114_132532"  # fixed for testing
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
    # random_forest
    mlp_classifier
    gradient_boosting
    logistic_regression
    # svm_rbf
  )
fi

# IMPORTANT: sequential by default to reduce memory pressure
JOBS="${JOBS:-1}"

BASE_RESULTS_DIR="${BASE_RESULTS_DIR:-results}"
LOG_DIR="${BASE_RESULTS_DIR}/${RUN_ID}/logs"
mkdir -p "${LOG_DIR}"

export BASE_RESULTS_DIR

echo "[local-run] RUN_ID=${RUN_ID}"
echo "[local-run] suites: ${SUITES_ARR[*]}"
echo "[local-run] models: ${MODELS_ARR[*]}"
echo "[local-run] JOBS=${JOBS}  (1 = sequential)"

running=0
failures=0

# Map PID -> label for better error reporting
declare -a pids=()
declare -A pid_to_label=()

start_job() {
  local suite="$1"
  local model="$2"
  local label="local__${suite}__${model}"
  local log_path="${LOG_DIR}/${label}.log"

  echo "[local-run] start ${label} (log: ${log_path})"
  (
    bash scripts/run_metrics_task.sh "${suite}" "${model}" "${RUN_ID}"
  ) >"${log_path}" 2>&1 &

  local pid="$!"
  pids+=("${pid}")
  pid_to_label["${pid}"]="${label}"
  running=$((running + 1))
}

# Remove a PID from pids array (portable)
remove_pid_from_list() {
  local target="$1"
  local new=()
  local p
  for p in "${pids[@]}"; do
    [[ "$p" == "$target" ]] || new+=("$p")
  done
  pids=("${new[@]}")
}

wait_one() {
  local finished_pid=""
  local status=0

  # If bash supports: wait -n -p var  (bash >= 5.1)
  if wait -n -p finished_pid 2>/dev/null; then
    status=0
  else
    status=$?
    # Fallback: poll until one PID is no longer running, then wait on it.
    # This is slower but works on older bash versions.
    while :; do
      local pid
      for pid in "${pids[@]}"; do
        if ! kill -0 "${pid}" 2>/dev/null; then
          finished_pid="${pid}"
          wait "${pid}" || status=$?
          break 2
        fi
      done
      sleep 0.2
    done
  fi

  # If wait -n -p succeeded, we still need the exit status:
  # - In bash >= 5.1, exit status is in $?
  if [[ -n "${finished_pid}" ]]; then
    # If previous branch used wait -n -p, $? is already the status.
    # If fallback path ran, status is already set.
    if [[ "${status}" -eq 0 ]]; then
      : # ok
    fi

    local label="${pid_to_label[${finished_pid}]:-unknown}"
    if ! wait "${finished_pid}" 2>/dev/null; then
      # If already waited, this will fail; in that case rely on status variable.
      if [[ "${status}" -ne 0 ]]; then
        echo "[local-run] FAILED ${label} (pid=${finished_pid})" >&2
        failures=$((failures + 1))
      fi
    else
      # If we got here, it means we hadn't waited yet and it succeeded.
      :
    fi

    # In the wait -n -p case, we haven't actually reaped it above,
    # but the wait -n already reaped *some* job. So don't double-wait.
    # For correctness, just use the status from wait -n:
    if [[ "${status}" -ne 0 ]]; then
      echo "[local-run] FAILED ${label} (pid=${finished_pid})" >&2
      failures=$((failures + 1))
    fi

    remove_pid_from_list "${finished_pid}"
  else
    # Extremely unlikely, but handle gracefully
    if ! wait -n; then
      failures=$((failures + 1))
    fi
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
