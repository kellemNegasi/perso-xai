RUN_ID="hc_combo_$(date +%Y%m%d_%H%M%S)"

# AutoXAI BO-on-randint explainer HPO suites (SHAP + LIME only).
SUITES=(
  # autoxai_openml_adult_suite
  autoxai_openml_bank_suite
  autoxai_openml_german_suite
  autoxai_open_compas_suite
)

MODELS=(
  decision_tree
  random_forest
  gradient_boosting
  mlp_classifier
  logistic_regression
  # svm_rbf
)

for SUITE in "${SUITES[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    sbatch -J "hc_${SUITE}__${MODEL}" \
      --output="slurm-hc_${SUITE}__${MODEL}.%j.out" \
      --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=64G --time=3-00:00:00 --partition=amd \
      --export=ALL,RUN_ID="${RUN_ID}",SUITE="${SUITE}",MODEL="${MODEL}" \
      --wrap 'module load python/3.12.3; source .venv/bin/activate; bash scripts/run_metrics_task.sh "$SUITE" "$MODEL" "$RUN_ID"'
  done
done
