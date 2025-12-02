#!/bin/bash
#
# SLURM submission script for running HC-XAI experiments on the long queue.
# Customize the environment variables below (PROJECT_ROOT, VENV_PATH, etc.)
# before calling `sbatch scripts/hc_xai_long_job.sh`.
#
# Job metadata -----------------------------------------------------------------
#SBATCH -J hc_xai_long
#SBATCH --output=slurm-%x.%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=3-00:00:00
#SBATCH --partition=main
#SBATCH --mem=64G
## Uncomment to receive e-mail updates:
## #SBATCH --mail-type=BEGIN,END,FAIL
## #SBATCH --mail-user=you@example.com

# Environment setup ------------------------------------------------------------
# Assume this script is submitted from the project root where .venv lives.
module load python/3.12.3

# if [[ ! -d ".venv" ]]; then
#   echo "Expected virtual environment '.venv' in $(pwd) but it does not exist." >&2
#   exit 1
# fi

source .venv/bin/activate

# Workload ---------------------------------------------------------------------
echo "[$(date --iso-8601=seconds)] Running scripts/run_metrics.sh on ${HOSTNAME}"
bash scripts/run_metrics.sh
echo "[$(date --iso-8601=seconds)] Job completed"
