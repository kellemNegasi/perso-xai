RUN_ID="hc_split_$(date +%Y%m%d_%H%M%S)"

for SUITE in openml_bank_suite openml_german_suite open_compas_suite; do
  sbatch -J "hc_${SUITE}" \
    --output="slurm-hc_${SUITE}.%j.out" \
    --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=64G --time=3-00:00:00 --partition=main \
    --export=ALL,SUITE="${SUITE}",RUN_ID="${RUN_ID}" \
    --wrap 'module load python/3.12.3; source .venv/bin/activate; python -m src.cli.main "$SUITE" --reuse-trained-models --tune-models --use-tuned-params --write-detailed-explanations --detailed-output-dir "results/$RUN_ID/detailed_explanations" --write-metric-results --metrics-output-dir "results/$RUN_ID/metrics_results" --output-dir "results/$RUN_ID" --skip-existing-experiments --skip-existing-methods --model-store-dir saved_models --log-level INFO'
done
