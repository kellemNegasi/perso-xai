#!/usr/bin/env bash
set -euo pipefail

# Trains preference-learning models over encoded Pareto-front features.
# Defaults assume the `hc_combo_20251228_050331` results root; override via env vars.

RESULTS_ROOT=${RESULTS_ROOT:-results/hc_combo_20251228_050331}
ENCODED_DIR=${ENCODED_DIR:-$RESULTS_ROOT/encoded_pareto_fronts/features_full_lm_stats}
OUTPUT_DIR=${OUTPUT_DIR:-$RESULTS_ROOT/preference_learning_simulation_advanced/basic_features/svc_c_sweep}
PYTHON_BIN=${PYTHON_BIN:-python3}

"$PYTHON_BIN" -m src.preference_learning.run_all \
  --encoded-dir "$ENCODED_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --personas layperson regulator clinician \
  --exclude-feature-groups statistical landmarking \
  --num-users 40 \
  --concentration-c-values 1 5 10 20 50 100 200 \
  --concentration-c-results-dir "$OUTPUT_DIR" \
  --tau 0.01 \
