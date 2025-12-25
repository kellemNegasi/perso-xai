#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Run AutoXAI holdout evaluation for every dataset/model that has an HC-XAI preference-learning split.

This runs AutoXAI on the same held-out instances used by HC-XAI preference-learning and writes
reports that include `top_k_evaluation` (when pair labels exist).

Usage:
  src/baseline/run_autoxai_holdout_eval_all.sh [options]

Options:
  --results-root PATH        HC-XAI run directory (default: results/full_run_dec8)
  --output-dir PATH          Where to write JSON reports (default: <results-root>/baslines/holdout_eval)
  --persona NAME             layperson | regulator (default: layperson)
  --methods "m1 m2 ..."      Methods to include (default: "lime shap integrated_gradients causal_shap")
  --split-set NAME           train | test | all (default: test)
  --top-k "k1 k2 ..."        Top-k values (default: "3 5")
  --hpo MODE                 grid | random | gp (default: grid)
  --epochs N                 HPO epochs for random/gp (default: 20)
  --hpo-seed N               HPO seed (default: 0)
  --scaling NAME             Std | MinMax (default: Std)
  --scaling-scope NAME       trial | global | instance (default: trial)
  --jobs N                   Max parallel jobs (default: 1)
  --overwrite                Recompute even if output exists
  --no-redirect              Do not redirect stdout/stderr to per-run log files
  --dry-run                  Print commands without running

Notes:
  - This script enumerates: <results-root>/preference_learning/<persona>/*/processed/splits.json
  - It expects pair labels at: <results-root>/candidate_pair_rankings_<persona>/<dataset>__<model>_pareto_pair_labels.parquet
  - It expects method metrics at: <results-root>/metrics_results/<dataset>/<model>/<method>_metrics.json
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RESULTS_ROOT="${ROOT}/results/full_run_dec8"
OUTPUT_DIR=""
PERSONA="layperson"
METHODS_STR="lime shap integrated_gradients causal_shap"
SPLIT_SET="test"
TOPK_STR="3 5"
HPO_MODE="grid"
EPOCHS=20
HPO_SEED=0
SCALING="Std"
SCALING_SCOPE="trial"
JOBS=1
OVERWRITE=0
REDIRECT=1
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --results-root) RESULTS_ROOT="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --persona) PERSONA="$2"; shift 2 ;;
    --methods) METHODS_STR="$2"; shift 2 ;;
    --split-set) SPLIT_SET="$2"; shift 2 ;;
    --top-k) TOPK_STR="$2"; shift 2 ;;
    --hpo) HPO_MODE="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --hpo-seed) HPO_SEED="$2"; shift 2 ;;
    --scaling) SCALING="$2"; shift 2 ;;
    --scaling-scope) SCALING_SCOPE="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    --no-redirect) REDIRECT=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${PERSONA}" != "layperson" && "${PERSONA}" != "regulator" ]]; then
  echo "ERROR: --persona must be layperson or regulator (got: ${PERSONA})." >&2
  exit 2
fi

if [[ "${SPLIT_SET}" != "train" && "${SPLIT_SET}" != "test" && "${SPLIT_SET}" != "all" ]]; then
  echo "ERROR: --split-set must be train|test|all (got: ${SPLIT_SET})." >&2
  exit 2
fi

if ! [[ "${JOBS}" =~ ^[0-9]+$ ]] || [[ "${JOBS}" -lt 1 ]]; then
  echo "ERROR: --jobs must be a positive integer (got: ${JOBS})." >&2
  exit 2
fi

if [[ ${DRY_RUN} -eq 1 ]]; then
  exec 1>&2
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${RESULTS_ROOT}/baslines/holdout_eval"
fi

PY="${ROOT}/.venv/bin/python"
if [[ ! -x "${PY}" ]]; then
  PY="$(command -v python3 || true)"
fi
if [[ -z "${PY}" ]]; then
  echo "ERROR: could not find python3 or ${ROOT}/.venv/bin/python" >&2
  exit 1
fi

SPLITS_ROOT="${RESULTS_ROOT}/preference_learning/${PERSONA}"
if [[ ! -d "${SPLITS_ROOT}" ]]; then
  echo "ERROR: preference_learning directory not found: ${SPLITS_ROOT}" >&2
  exit 1
fi

read -r -a METHODS <<<"${METHODS_STR}"
read -r -a TOPK <<<"${TOPK_STR}"

LOG_DIR="${OUTPUT_DIR}/logs"
if [[ ${DRY_RUN} -eq 0 ]]; then
  mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
fi

wait_for_slot() {
  while true; do
    local running
    running="$(jobs -pr | wc -l | tr -d ' ')"
    if [[ "${running}" -lt "${JOBS}" ]]; then
      break
    fi
    sleep 0.2
  done
}

run_one() {
  local dataset="$1"
  local model="$2"
  local splits_json="$3"
  local pair_labels="$4"

  local out="${OUTPUT_DIR}/autoxai_holdout_eval__${dataset}__${model}__${PERSONA}__${SPLIT_SET}.json"
  local log="${LOG_DIR}/autoxai_holdout_eval__${dataset}__${model}__${PERSONA}__${SPLIT_SET}.log"

  if [[ -f "${out}" && ${OVERWRITE} -eq 0 ]]; then
    echo "[skip] exists: ${out}"
    return 0
  fi

  local cmd=(
    "${PY}" -m src.baseline.autoxai
    --results-root "${RESULTS_ROOT}"
    --dataset "${dataset}"
    --model "${model}"
    --methods "${METHODS[@]}"
    --persona "${PERSONA}"
    --hpo "${HPO_MODE}"
    --epochs "${EPOCHS}"
    --hpo-seed "${HPO_SEED}"
    --scaling "${SCALING}"
    --scaling-scope "${SCALING_SCOPE}"
    --pair-labels "${pair_labels}"
    --hc-xai-split-json "${splits_json}"
    --split-set "${SPLIT_SET}"
    --top-k "${TOPK[@]}"
    --output "${out}"
    --require-write
  )

  if [[ ${DRY_RUN} -eq 1 ]]; then
    printf '[dry-run] '
    printf '%q ' "${cmd[@]}"
    echo
    return 0
  fi

  echo "[run] dataset=${dataset} model=${model} persona=${PERSONA} split=${SPLIT_SET} -> ${out}"
  if [[ ${REDIRECT} -eq 1 ]]; then
    if ! "${cmd[@]}" >"${log}" 2>&1; then
      echo "[fail] dataset=${dataset} model=${model} (see ${log})" >&2
      tail -n 80 "${log}" >&2 || true
      return 1
    fi
  else
    "${cmd[@]}"
  fi
}

failures=0
skipped_missing=0
pids=()

while IFS= read -r -d '' splits_json; do
  # splits_json: .../<dataset>__<model>_pareto/processed/splits.json
  stem="$(basename "$(dirname "$(dirname "${splits_json}")")")"
  dataset="${stem%%__*}"
  rest="${stem#*__}"
  model="${rest%_pareto}"

  if [[ -z "${dataset}" || -z "${model}" ]]; then
    echo "[skip] could not parse dataset/model from splits path: ${splits_json}" >&2
    skipped_missing=$((skipped_missing + 1))
    continue
  fi

  pair_labels="${RESULTS_ROOT}/candidate_pair_rankings_${PERSONA}/${dataset}__${model}_pareto_pair_labels.parquet"
  if [[ ! -f "${pair_labels}" ]]; then
    echo "[skip] missing pair labels: ${pair_labels}"
    skipped_missing=$((skipped_missing + 1))
    continue
  fi

  metrics_dir="${RESULTS_ROOT}/metrics_results/${dataset}/${model}"
  missing=0
  for method in "${METHODS[@]}"; do
    if [[ ! -f "${metrics_dir}/${method}_metrics.json" ]]; then
      missing=1
      break
    fi
  done
  if [[ ${missing} -eq 1 ]]; then
    echo "[skip] missing metrics for dataset=${dataset} model=${model} (need: ${METHODS_STR})"
    skipped_missing=$((skipped_missing + 1))
    continue
  fi

  if [[ ${DRY_RUN} -eq 1 || ${JOBS} -le 1 ]]; then
    if ! run_one "${dataset}" "${model}" "${splits_json}" "${pair_labels}"; then
      failures=$((failures + 1))
    fi
    continue
  fi

  wait_for_slot
  (
    if ! run_one "${dataset}" "${model}" "${splits_json}" "${pair_labels}"; then
      exit 1
    fi
  ) &
  pids+=("$!")

done < <(find "${SPLITS_ROOT}" -type f -path '*/processed/splits.json' -print0 | sort -z)

if [[ ${DRY_RUN} -eq 0 && ${JOBS} -gt 1 ]]; then
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failures=$((failures + 1))
    fi
  done
fi

if [[ ${failures} -ne 0 ]]; then
  echo "Done with ${failures} failures (${skipped_missing} skipped). Logs: ${LOG_DIR}" >&2
  exit 1
fi

echo "Done (${skipped_missing} skipped). Outputs: ${OUTPUT_DIR}"
