#!/usr/bin/env bash
set -euo pipefail

# Trains preference-learning models over encoded Pareto-front features.
# Defaults assume the `hc_combo_20251228_050331` results root; override via env vars.

RESULTS_ROOT=${RESULTS_ROOT:-results/hc_combo_20251228_050331}
ENCODED_DIR=${ENCODED_DIR:-$RESULTS_ROOT/encoded_pareto_fronts/features_full_lm_stats}
OUTPUT_DIR=${OUTPUT_DIR:-$RESULTS_ROOT/preference_learning_simulation_advanced/basic_features/tuned-svc-best_tau_and_users}
PYTHON_BIN=${PYTHON_BIN:-python3}

"$PYTHON_BIN" -m src.preference_learning.run_all \
  --encoded-dir "$ENCODED_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --personas layperson regulator clinician \
  --exclude-feature-groups statistical landmarking \
  --tune-svc \
  --num-users 40 \
  --tau 0.01 \
