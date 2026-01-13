import json
import math
from pathlib import Path

def is_neginf(x) -> bool:
    try:
        return isinstance(x, (int, float)) and math.isinf(float(x)) and float(x) < 0
    except Exception:
        return False

def scan_for_bad_hpo_trials(root: Path):
    out = []
    for p in root.rglob("*_metrics.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        h = d.get("hpo")
        if not isinstance(h, dict):
            continue
        trials = h.get("trials") or []
        if not isinstance(trials, list):
            continue
        bad = [t for t in trials if isinstance(t, dict) and is_neginf(t.get("aggregated_score"))]
        if bad:
            out.append((p, len(bad), len(trials)))
    return out

root = Path("results/hc_combo_20260110_024805/metrics_results")  # or Path("results/<run_id>/metrics_results")
bad_files = scan_for_bad_hpo_trials(root)
bad_files.sort(key=lambda x: (-x[1], str(x[0])))

print("files_with_-inf_trials:", len(bad_files))
print("total_-inf_trials:", sum(nbad for _, nbad, _ in bad_files))
for p, nbad, ntotal in bad_files:
    print(f"{nbad}/{ntotal}\t{p}")
