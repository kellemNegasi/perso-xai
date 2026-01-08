#!/usr/bin/env bash
set -euo pipefail

# Trains preference-learning models over encoded Pareto-front features.
# Defaults assume the `hc_combo_20251228_050331` results root; override via env vars.

RESULTS_ROOT=${RESULTS_ROOT:-results/hc_combo_20251228_050331}
ENCODED_DIR=${ENCODED_DIR:-$RESULTS_ROOT/encoded_pareto_fronts/features_full_lm_stats}
PYTHON_BIN=${PYTHON_BIN:-python3}

AUTOXAI_3METRICS_DIR=${AUTOXAI_3METRICS_DIR:-$RESULTS_ROOT/preference_learning_simulation_aligned/aut_xai_comp_3-metrics}
AUTOXAI_ALLMETRICS_DIR=${AUTOXAI_ALLMETRICS_DIR:-$RESULTS_ROOT/preference_learning_simulation_aligned/aut_xai_comp_all-metrics}
PREFERENCE_ONLY_DIR=${PREFERENCE_ONLY_DIR:-$RESULTS_ROOT/preference_learning_simulation_aligned/preference_only_all_personas}

"$PYTHON_BIN" -m src.preference_learning.run_all \
  --encoded-dir "$ENCODED_DIR" \
  --output-dir "$AUTOXAI_3METRICS_DIR" \
  --experiment-mode autoxai-comparison \
  --autoxai-metric-mode auto-xai \
  --exclude-feature-groups statistical landmarking \
  --num-users 40 \
  --tau 0.01 \
  &

"$PYTHON_BIN" -m src.preference_learning.run_all \
  --encoded-dir "$ENCODED_DIR" \
  --output-dir "$AUTOXAI_ALLMETRICS_DIR" \
  --experiment-mode autoxai-comparison \
  --autoxai-metric-mode all-metrics \
  --exclude-feature-groups statistical landmarking \
  --num-users 40 \
  --tau 0.01 \
  &

"$PYTHON_BIN" -m src.preference_learning.run_all \
  --encoded-dir "$ENCODED_DIR" \
  --output-dir "$PREFERENCE_ONLY_DIR" \
  --experiment-mode preference-only \
  --exclude-feature-groups statistical landmarking \
  --num-users 40 \
  --tau 0.01 \
  &

wait
