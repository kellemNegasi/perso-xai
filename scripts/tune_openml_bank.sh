#!/usr/bin/env bash
set -euo pipefail

MODELS=(decision_tree random_forest gradient_boosting mlp_classifier)

for model in "${MODELS[@]}"; do
  python -m src.cli.main openml_bank_suite \
    --model "$model" \
    --tune-models --use-tuned-params --reuse-trained-models --stop-after-training \
    --tuning-output-dir saved_models/tuning_results \
    --model-store-dir saved_models \
    --log-level INFO
done
