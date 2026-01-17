#!/bin/bash
#
# SLURM submission script to run each (dataset suite, model) in a separate array task.
#
# Example:
#   RUN_ID="slurm_${USER}_$(date +%Y%m%d_%H%M%S)" sbatch --array=0-23%6 hc_xai_long_array_job.sh
#
# Notes:
#   - Adjust SUITES/MODELS below to match what you want to run.
#   - Use the %N limiter on --array to control max concurrent tasks.
#
# Job metadata -----------------------------------------------------------------
#SBATCH -J hc_xai_array
#SBATCH --output=slurm-%x.%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=3-00:00:00
#SBATCH --partition=main
#SBATCH --mem=64G

# Environment setup ------------------------------------------------------------
module load python/3.12.3
source .venv/bin/activate

# Workload matrix --------------------------------------------------------------
SUITES=(
  openml_bank_suite
  openml_german_suite
  open_compas_suite
)

MODELS=(
  decision_tree
  random_forest
  gradient_boosting
  mlp_classifier
  logistic_regression
  svm_rbf
)

NUM_MODELS=${#MODELS[@]}
NUM_SUITES=${#SUITES[@]}
TOTAL=$((NUM_MODELS * NUM_SUITES))

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID is not set (submit with sbatch --array=0-$((TOTAL - 1)))." >&2
  exit 2
fi
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= TOTAL )); then
  echo "ERROR: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} out of range [0, $((TOTAL - 1))]." >&2
  exit 2
fi

SUITE_IDX=$((SLURM_ARRAY_TASK_ID / NUM_MODELS))
MODEL_IDX=$((SLURM_ARRAY_TASK_ID % NUM_MODELS))

EXPERIMENT_SUITE="${SUITES[$SUITE_IDX]}"
MODEL_NAME="${MODELS[$MODEL_IDX]}"

RUN_ID="${RUN_ID:-slurm_${SLURM_ARRAY_JOB_ID}}"

echo "[$(date --iso-8601=seconds)] Task ${SLURM_ARRAY_TASK_ID}/${TOTAL}: suite=${EXPERIMENT_SUITE} model=${MODEL_NAME} run_id=${RUN_ID} host=${HOSTNAME}"

bash scripts/run_metrics_task.sh "${EXPERIMENT_SUITE}" "${MODEL_NAME}" "${RUN_ID}"

echo "[$(date --iso-8601=seconds)] Task completed: suite=${EXPERIMENT_SUITE} model=${MODEL_NAME}"

