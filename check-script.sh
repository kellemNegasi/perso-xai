#!/usr/bin/env bash
set -euo pipefail

EXP_NAME=openml_german_suite          # experiment from configs/experiments.yml
MODEL_NAME=gradient_boosting              # model within that experiment
INSTANCE_ID=15                        # dataset_index you want to inspect
# Space-delimited metric list from metrics.yml
METRICS="monotonicity_correlation infidelity completeness_drop completeness_random_drop completeness_score"
# Metrics that expose a fast_mode flag
FAST_METRICS="correctness completeness_deletion"
CHECK_ROOT=./checks/${METRICS}   # anywhere outside the repo
DETAIL_ROOT=saved_models/detailed_explanations  # existing cache location

for MODE in fast slow; do
  python - <<PY
import json
from copy import deepcopy
from pathlib import Path
from src.orchestrators.metrics_runner import run_experiment
from src.orchestrators import utils as orch_utils

exp_name = "${EXP_NAME}"
model_name = "${MODEL_NAME}"
instance_id = ${INSTANCE_ID}
metrics = "${METRICS}".split()
check_root = Path("${CHECK_ROOT}")
detail_root = Path("${DETAIL_ROOT}")
mode = "${MODE}"

# Toggle fast/slow mode only for metrics that expose fast_mode
overrides = {}
fast_toggle = {name: None for name in "${FAST_METRICS}".split()}
for metric_name in metrics:
    if metric_name not in fast_toggle:
        continue
    spec = orch_utils.METRIC_CFG[metric_name]
    params = spec.setdefault("params", {})
    overrides[metric_name] = deepcopy(params)
    params["fast_mode"] = (mode == "fast")

result = run_experiment(
    exp_name,
    model_override=model_name,
    max_instances=instance_id + 1,        # grab up to the instance we want
    reuse_trained_models=True,
    reuse_detailed_explanations=True,
    detailed_output_dir=detail_root,
    stop_after_explanations=False,
    write_detailed_explanations=False,
    write_metric_results=False,
)

for metric_name, params in overrides.items():
    orch_utils.METRIC_CFG[metric_name]["params"] = params

# Pick only the requested dataset_index
matching = [
    inst for inst in result["instances"]
    if inst.get("dataset_index") == instance_id
]
if not matching:
    raise SystemExit(f"No instance with dataset_index={instance_id} found.")
payload = {
    "experiment": result["experiment"],
    "dataset": result["dataset"],
    "model": result["model"],
    "mode": mode,
    "instance": matching[0],
}

out_dir = check_root / f"{result['dataset']}__{result['model']}"
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / f"{mode}_instance_{instance_id}.json").write_text(
    json.dumps(payload, indent=2), encoding="utf-8"
)
print(f"Wrote {mode} metrics to {out_dir}")
PY
done
