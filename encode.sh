#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./encode.sh [pareto_dir] [metadata_dir] [encoded_dir]
#
# Defaults can also be provided via:
#   RESULTS_ROOT, PARETO_DIR, METADATA_DIR, ENCODED_DIR, PYTHON_BIN
#
# Encoding outlier handling:
#   METRIC_VALUE_TRANSFORM (default: signed_log1p)
#   TRANSFORM_COUNT_METRICS (default: 1; set 0 to disable)

RESULTS_ROOT=${RESULTS_ROOT:-results/hc_combo_20251228_050331}
PARETO_DIR=${1:-${PARETO_DIR:-$RESULTS_ROOT/pareto_fronts}}
METADATA_DIR=${2:-${METADATA_DIR:-$RESULTS_ROOT/metadata}}
ENCODED_DIR=${3:-${ENCODED_DIR:-$RESULTS_ROOT/encoded_pareto_fronts/features_full_lm_stats}}

PYTHON_BIN=${PYTHON_BIN:-python3}
METRIC_VALUE_TRANSFORM=${METRIC_VALUE_TRANSFORM:-signed_log1p}
TRANSFORM_COUNT_METRICS=${TRANSFORM_COUNT_METRICS:-1}

META_CMD=(
  "$PYTHON_BIN" generate_dataset_metadata.py
  --output-dir "$METADATA_DIR"
)
echo "Running: ${META_CMD[*]}"
"${META_CMD[@]}"

ENCODE_CMD=(
  "$PYTHON_BIN" encode_pareto_fronts.py
  --pareto-dir "$PARETO_DIR"
  --metadata-dir "$METADATA_DIR"
  --output-dir "$ENCODED_DIR"
  --hyperparameters src/configs/explainer_hyperparameters.yml
  --metric-value-transform "$METRIC_VALUE_TRANSFORM"
)
if [[ "$TRANSFORM_COUNT_METRICS" == "1" ]]; then
  ENCODE_CMD+=(--transform-count-metrics)
fi
echo "Running: ${ENCODE_CMD[*]}"
"${ENCODE_CMD[@]}"
