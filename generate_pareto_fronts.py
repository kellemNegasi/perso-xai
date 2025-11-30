#!/usr/bin/env python3
"""
Build per-instance Pareto fronts across explanation metrics.

Reads experiment result JSON files (the ones produced by run_metrics.sh),
filters out a set of metrics we explicitly want to ignore, and computes the
Pareto-optimal explanations (maximisation) for every instance in each file.
Outputs one JSON file per input result under the requested output directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

IGNORED_METRICS = {
    "completeness_drop",
    "completeness_random_drop",
    "contrastivity_pairs",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construct per-instance Pareto fronts for explanation metrics.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experiment_results"),
        help="Directory containing experiment result JSON files (default: experiment_results).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiment_results") / "pareto_fronts",
        help="Directory where Pareto summaries will be written (default: experiment_results/pareto_fronts).",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Optional explicit result JSON files to process. "
        "When omitted, every *.json under --results-dir (non-recursive) is processed.",
    )
    return parser.parse_args(argv)


def collect_input_files(paths: Sequence[str], results_dir: Path) -> List[Path]:
    if paths:
        files: List[Path] = []
        for entry in paths:
            candidate = Path(entry)
            if candidate.is_dir():
                files.extend(sorted(candidate.glob("*.json")))
            else:
                files.append(candidate)
        return [path for path in files if path.is_file()]
    return sorted(results_dir.glob("*.json"))


def clean_metrics(raw_metrics: Dict[str, Any] | None) -> Dict[str, float]:
    cleaned: Dict[str, float] = {}
    if not raw_metrics:
        return cleaned
    for key, value in raw_metrics.items():
        if key in IGNORED_METRICS:
            continue
        try:
            cleaned[key] = float(value)
        except (TypeError, ValueError):
            continue
    return cleaned


def dominates(
    left: Dict[str, float],
    right: Dict[str, float],
    metric_keys: Iterable[str],
) -> bool:
    """Return True when `left` dominates `right` (>= in every metric, > in at least one)."""
    better_or_equal = True
    strictly_better = False
    for key in metric_keys:
        a_val = left.get(key)
        b_val = right.get(key)
        if a_val is None and b_val is None:
            continue
        if a_val is None:
            better_or_equal = False
            break
        if b_val is None:
            strictly_better = True
            continue
        if a_val < b_val:
            better_or_equal = False
            break
        if a_val > b_val:
            strictly_better = True
    return better_or_equal and strictly_better


def pareto_front(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not candidates:
        return []
    metric_keys = sorted({key for cand in candidates for key in cand["metrics"].keys()})
    if not metric_keys:
        return []
    front: List[Dict[str, Any]] = []
    for candidate in candidates:
        dominated = False
        survivors: List[Dict[str, Any]] = []
        for incumbent in front:
            if dominates(incumbent["metrics"], candidate["metrics"], metric_keys):
                dominated = True
                break
            if not dominates(candidate["metrics"], incumbent["metrics"], metric_keys):
                survivors.append(incumbent)
        if dominated:
            continue
        survivors.append(candidate)
        front = survivors
    return front


def process_file(result_path: Path) -> Dict[str, Any]:
    data = json.loads(result_path.read_text(encoding="utf-8"))
    dataset = data.get("dataset")
    model = data.get("model")
    experiment = data.get("experiment")
    instances = data.get("instances") or []

    pareto_instances: List[Dict[str, Any]] = []
    for inst in instances:
        explanations = inst.get("explanations") or []
        candidates: List[Dict[str, Any]] = []
        for expl in explanations:
            metrics = clean_metrics(expl.get("metrics"))
            if not metrics:
                continue
            candidates.append(
                {
                    "method": expl.get("method"),
                    "metrics": metrics,
                    "metadata_key": expl.get("metadata_key"),
                }
            )
        pareto = pareto_front(candidates)
        if not pareto:
            continue
        pareto_instances.append(
            {
                "experiment": experiment,
                "dataset": dataset,
                "model": model,
                "instance_index": inst.get("index"),
                "dataset_index": inst.get("dataset_index"),
                "true_label": inst.get("true_label"),
                "predicted_label": inst.get("predicted_label"),
                "pareto_metrics": sorted({key for cand in pareto for key in cand["metrics"].keys()}),
                "pareto_front": pareto,
            }
        )

    return {
        "experiment": experiment,
        "dataset": dataset,
        "model": model,
        "source_file": str(result_path),
        "n_instances": len(pareto_instances),
        "instances": pareto_instances,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result_dir = args.results_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = collect_input_files(args.paths, result_dir)
    if not inputs:
        raise SystemExit("No result files found to process.")

    for path in inputs:
        payload = process_file(path)
        output_path = output_dir / f"{path.stem}_pareto.json"
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"Wrote Pareto summaries to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
