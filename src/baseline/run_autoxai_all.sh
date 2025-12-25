#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Run AutoXAI baseline over all dataset/model combos present in an HC-XAI results root.

Usage:
  src/baseline/run_autoxai_all.sh [options]

Options:
  --results-root PATH        HC-XAI run directory (default: results/full_run_dec8)
  --output-dir PATH          Where to write JSON reports (default: <results-root>/baslines)
  --persona NAME             autoxai | layperson | regulator (default: autoxai)
  --methods "m1 m2 ..."       Methods to include (default: "lime shap")
  --hpo MODE                 grid | random | gp (default: grid)
  --epochs N                 HPO epochs for random/gp (default: 20)
  --hpo-seed N               HPO seed (default: 0)
  --scaling NAME             Std | MinMax (default: Std)
  --scaling-scope NAME       trial | global | instance (default: trial)
  --overwrite                Recompute even if output exists
  --no-redirect              Do not redirect stdout/stderr to log files
  --dry-run                  Print commands without running

Notes:
  - For persona=layperson/regulator, the script will auto-add --pair-labels when the
    corresponding parquet exists under <results-root>/candidate_pair_rankings_<persona>/.
  - hpo=gp requires scikit-optimize (skopt) to be installed.
  - In --dry-run mode this script prints to stderr; use `2>&1 | head` to preview.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RESULTS_ROOT="${ROOT}/results/full_run_dec8"
OUTPUT_DIR=""
PERSONA="autoxai"
METHODS_STR="lime shap"
HPO_MODE="grid"
EPOCHS=20
HPO_SEED=0
SCALING="Std"
SCALING_SCOPE="trial"
OVERWRITE=0
REDIRECT=1
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --results-root)
      RESULTS_ROOT="$2"; shift 2 ;;
    --output-dir)
      OUTPUT_DIR="$2"; shift 2 ;;
    --persona)
      PERSONA="$2"; shift 2 ;;
    --methods)
      METHODS_STR="$2"; shift 2 ;;
    --hpo)
      HPO_MODE="$2"; shift 2 ;;
    --epochs)
      EPOCHS="$2"; shift 2 ;;
    --hpo-seed)
      HPO_SEED="$2"; shift 2 ;;
    --scaling)
      SCALING="$2"; shift 2 ;;
    --scaling-scope)
      SCALING_SCOPE="$2"; shift 2 ;;
    --overwrite)
      OVERWRITE=1; shift ;;
    --no-redirect)
      REDIRECT=0; shift ;;
    --dry-run)
      DRY_RUN=1; shift ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

# In --dry-run mode, write messages to stderr so piping stdout (e.g. to `head`) does not error.
if [[ ${DRY_RUN} -eq 1 ]]; then
  exec 1>&2
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${RESULTS_ROOT}/baslines"
fi

PY="${ROOT}/.venv/bin/python"
if [[ ! -x "${PY}" ]]; then
  PY="$(command -v python3 || true)"
fi
if [[ -z "${PY}" ]]; then
  echo "ERROR: could not find python3 or ${ROOT}/.venv/bin/python" >&2
  exit 1
fi

METRICS_DIR="${RESULTS_ROOT}/metrics_results"
if [[ ! -d "${METRICS_DIR}" ]]; then
  echo "ERROR: metrics_results not found: ${METRICS_DIR}" >&2
  exit 1
fi

LOG_DIR="${OUTPUT_DIR}/logs"
if [[ ${DRY_RUN} -eq 0 ]]; then
  mkdir -p "${OUTPUT_DIR}"
  mkdir -p "${LOG_DIR}"
fi

read -r -a METHODS <<<"${METHODS_STR}"

run_one() {
  local dataset="$1"
  local model="$2"

  local out="${OUTPUT_DIR}/autoxai_baseline__${dataset}__${model}__${PERSONA}.json"
  local log="${LOG_DIR}/autoxai_baseline__${dataset}__${model}__${PERSONA}.log"

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
    --output "${out}"
    --require-write
  )

  if [[ "${PERSONA}" == "layperson" || "${PERSONA}" == "regulator" ]]; then
    local pair_labels="${RESULTS_ROOT}/candidate_pair_rankings_${PERSONA}/${dataset}__${model}_pareto_pair_labels.parquet"
    if [[ -f "${pair_labels}" ]]; then
      cmd+=(--pair-labels "${pair_labels}")
    fi
  fi

  if [[ ${DRY_RUN} -eq 1 ]]; then
    printf '[dry-run] '
    printf '%q ' "${cmd[@]}"
    echo
    return 0
  fi

  echo "[run] dataset=${dataset} model=${model} -> ${out}"
  if [[ ${REDIRECT} -eq 1 ]]; then
    if ! "${cmd[@]}" >"${log}" 2>&1; then
      echo "[fail] dataset=${dataset} model=${model} (see ${log})" >&2
      tail -n 60 "${log}" >&2 || true
      return 1
    fi
  else
    "${cmd[@]}"
  fi
}

failures=0
while IFS= read -r -d '' model_dir; do
  dataset="$(basename "$(dirname "${model_dir}")")"
  model="$(basename "${model_dir}")"

  missing=0
  for method in "${METHODS[@]}"; do
    if [[ ! -f "${model_dir}/${method}_metrics.json" ]]; then
      missing=1
      break
    fi
  done
  if [[ ${missing} -eq 1 ]]; then
    echo "[skip] missing metrics for dataset=${dataset} model=${model} (need: ${METHODS_STR})"
    continue
  fi

  if ! run_one "${dataset}" "${model}"; then
    failures=$((failures + 1))
  fi

done < <(find "${METRICS_DIR}" -mindepth 2 -maxdepth 2 -type d -print0 | sort -z)

if [[ ${failures} -ne 0 ]]; then
  echo "Done with ${failures} failures. Logs: ${LOG_DIR}" >&2
  exit 1
fi

echo "Done. Outputs: ${OUTPUT_DIR}"
