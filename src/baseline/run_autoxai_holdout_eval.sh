#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Run AutoXAI baseline on the exact held-out HC-XAI test split and emit HC-XAI-style top-k metrics.

Usage:
  src/baseline/run_autoxai_holdout_eval.sh --results-root PATH --dataset NAME --model NAME [options]

Options:
  --results-root PATH        HC-XAI run directory (default: results/full_run_dec8)
  --dataset NAME             Dataset key (e.g. open_compas)
  --model NAME               Model key (e.g. mlp_classifier)
  --persona NAME             layperson | regulator (default: layperson)
  --methods "m1 m2 ..."      Methods to include (default: "lime shap integrated_gradients causal_shap")
  --output-dir PATH          Where to write JSON report (default: <results-root>/baslines)
  --split-set NAME           train | test | all (default: test)
  --top-k "k1 k2 ..."        Top-k values (default: "3 5")
  --hpo MODE                 grid | random | gp (default: grid)
  --epochs N                 HPO epochs for random/gp (default: 20)
  --hpo-seed N               HPO seed (default: 0)
  --scaling NAME             Std | MinMax (default: Std)
  --scaling-scope NAME       trial | global | instance (default: trial)
  --no-redirect              Do not redirect stdout/stderr to a log file
  --dry-run                  Print the python command without running

Notes:
  - This script auto-wires:
      * --pair-labels: <results-root>/candidate_pair_rankings_<persona>/<dataset>__<model>_pareto_pair_labels.parquet
      * --hc-xai-split-json: <results-root>/preference_learning/<persona>/<dataset>__<model>_pareto/processed/splits.json
  - persona must be layperson/regulator because those are the personas used by HC-XAI preference-learning.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RESULTS_ROOT="${ROOT}/results/full_run_dec8"
DATASET=""
MODEL=""
PERSONA="layperson"
METHODS_STR="lime shap integrated_gradients causal_shap"
OUTPUT_DIR=""
SPLIT_SET="test"
TOPK_STR="3 5"
HPO_MODE="grid"
EPOCHS=20
HPO_SEED=0
SCALING="Std"
SCALING_SCOPE="trial"
REDIRECT=1
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --results-root) RESULTS_ROOT="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --persona) PERSONA="$2"; shift 2 ;;
    --methods) METHODS_STR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --split-set) SPLIT_SET="$2"; shift 2 ;;
    --top-k) TOPK_STR="$2"; shift 2 ;;
    --hpo) HPO_MODE="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --hpo-seed) HPO_SEED="$2"; shift 2 ;;
    --scaling) SCALING="$2"; shift 2 ;;
    --scaling-scope) SCALING_SCOPE="$2"; shift 2 ;;
    --no-redirect) REDIRECT=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${DATASET}" || -z "${MODEL}" ]]; then
  echo "ERROR: --dataset and --model are required." >&2
  usage >&2
  exit 2
fi

if [[ "${PERSONA}" != "layperson" && "${PERSONA}" != "regulator" ]]; then
  echo "ERROR: --persona must be layperson or regulator (got: ${PERSONA})." >&2
  exit 2
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

PAIR_LABELS="${RESULTS_ROOT}/candidate_pair_rankings_${PERSONA}/${DATASET}__${MODEL}_pareto_pair_labels.parquet"
SPLITS_JSON="${RESULTS_ROOT}/preference_learning/${PERSONA}/${DATASET}__${MODEL}_pareto/processed/splits.json"

if [[ ! -f "${PAIR_LABELS}" ]]; then
  echo "ERROR: pair labels not found: ${PAIR_LABELS}" >&2
  exit 1
fi
if [[ ! -f "${SPLITS_JSON}" ]]; then
  echo "ERROR: HC-XAI split file not found: ${SPLITS_JSON}" >&2
  exit 1
fi

read -r -a METHODS <<<"${METHODS_STR}"
read -r -a TOPK <<<"${TOPK_STR}"

OUT="${OUTPUT_DIR}/autoxai_holdout_eval__${DATASET}__${MODEL}__${PERSONA}.json"
LOG="${OUTPUT_DIR}/logs/autoxai_holdout_eval__${DATASET}__${MODEL}__${PERSONA}.log"

cmd=(
  "${PY}" -m src.baseline.autoxai
  --results-root "${RESULTS_ROOT}"
  --dataset "${DATASET}"
  --model "${MODEL}"
  --methods "${METHODS[@]}"
  --persona "${PERSONA}"
  --hpo "${HPO_MODE}"
  --epochs "${EPOCHS}"
  --hpo-seed "${HPO_SEED}"
  --scaling "${SCALING}"
  --scaling-scope "${SCALING_SCOPE}"
  --pair-labels "${PAIR_LABELS}"
  --hc-xai-split-json "${SPLITS_JSON}"
  --split-set "${SPLIT_SET}"
  --top-k "${TOPK[@]}"
  --output "${OUT}"
  --require-write
)

if [[ ${DRY_RUN} -eq 1 ]]; then
  printf '[dry-run] '
  printf '%q ' "${cmd[@]}"
  echo
  exit 0
fi

mkdir -p "${OUTPUT_DIR}" "${OUTPUT_DIR}/logs"
echo "[run] dataset=${DATASET} model=${MODEL} persona=${PERSONA} split=${SPLIT_SET} -> ${OUT}"
if [[ ${REDIRECT} -eq 1 ]]; then
  if ! "${cmd[@]}" >"${LOG}" 2>&1; then
    echo "[fail] see ${LOG}" >&2
    tail -n 80 "${LOG}" >&2 || true
    exit 1
  fi
  echo "[ok] log: ${LOG}"
else
  "${cmd[@]}"
fi

echo "[ok] output: ${OUT}"
