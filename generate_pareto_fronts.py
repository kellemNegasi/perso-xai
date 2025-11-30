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
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

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
        "--mode",
        choices=("aggregate", "detailed", "metrics"),
        default="aggregate",
        help="Source mode: 'aggregate' reads experiment JSON summaries; "
        "'detailed' scans per-method detailed_explanations directories; "
        "'metrics' loads per-method metrics JSON files (default: aggregate).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experiment_results"),
        help="Directory containing experiment result JSON files (default: experiment_results).",
    )
    parser.add_argument(
        "--detailed-dir",
        type=Path,
        default=Path("experiment_results") / "detailed_explanations",
        help="Base directory for per-method detailed explanation JSONs (used in detailed mode).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiment_results") / "pareto_fronts",
        help="Directory where Pareto summaries will be written (default: experiment_results/pareto_fronts).",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path("experiment_results") / "metrics_results",
        help="Base directory for per-method metric JSONs (used in metrics mode).",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help=(
            "Optional explicit files or directories to process. "
            "Aggregate mode: JSON files. Detailed mode: dataset or model directories."
        ),
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


def collect_detailed_targets(paths: Sequence[str], detailed_dir: Path) -> List[Tuple[str, Path]]:
    targets: List[Tuple[str, Path]] = []
    if paths:
        for entry in paths:
            candidate = Path(entry)
            if candidate.is_file():
                continue
            if candidate.is_dir():
                if candidate.parent == detailed_dir:
                    dataset_name = candidate.name
                    for model_dir in sorted(p for p in candidate.iterdir() if p.is_dir()):
                        targets.append((dataset_name, model_dir))
                elif candidate.parent.parent == detailed_dir:
                    dataset_name = candidate.parent.name
                    targets.append((dataset_name, candidate))
                else:
                    # Accept arbitrary dir -> treat basename as dataset if dataset dir not explicit
                    dataset_name = candidate.name
                    for model_dir in sorted(p for p in candidate.iterdir() if p.is_dir()):
                        targets.append((dataset_name, model_dir))
    else:
        for dataset_dir in sorted(p for p in detailed_dir.iterdir() if p.is_dir()):
            for model_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
                targets.append((dataset_dir.name, model_dir))
    return targets


def collect_metric_targets(paths: Sequence[str], metrics_dir: Path) -> List[Tuple[str, Path]]:
    targets: List[Tuple[str, Path]] = []
    if paths:
        for entry in paths:
            candidate = Path(entry)
            if candidate.is_dir():
                if candidate.parent == metrics_dir:
                    dataset_name = candidate.name
                    for model_dir in sorted(p for p in candidate.iterdir() if p.is_dir()):
                        targets.append((dataset_name, model_dir))
                elif candidate.parent.parent == metrics_dir:
                    dataset_name = candidate.parent.name
                    targets.append((dataset_name, candidate))
                else:
                    dataset_name = candidate.name
                    for model_dir in sorted(p for p in candidate.iterdir() if p.is_dir()):
                        targets.append((dataset_name, model_dir))
    else:
        for dataset_dir in sorted(p for p in metrics_dir.iterdir() if p.is_dir()):
            for model_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
                targets.append((dataset_dir.name, model_dir))
    return targets


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


def process_detailed_model(dataset_name: str, model_dir: Path) -> Dict[str, Any]:
    method_files = sorted(model_dir.glob("*_detailed_explanations.json"))
    if not method_files:
        return {
            "dataset": dataset_name,
            "model": model_dir.name,
            "source_dir": str(model_dir),
            "n_instances": 0,
            "instances": [],
        }

    instance_meta: Dict[int, Dict[str, Any]] = {}
    candidates: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for file_path in method_files:
        method_label = file_path.stem.replace("_detailed_explanations", "")
        try:
            records = json.loads(file_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for record in records:
            dataset_idx = record.get("dataset_index")
            if dataset_idx is None:
                dataset_idx = record.get("instance_id")
            if dataset_idx is None:
                continue
            try:
                dataset_idx_int = int(dataset_idx)
            except (TypeError, ValueError):
                continue
            metrics = clean_metrics(record.get("metrics"))
            if not metrics:
                continue
            meta = instance_meta.setdefault(
                dataset_idx_int,
                {
                    "dataset_index": dataset_idx_int,
                    "instance_id": record.get("instance_id"),
                    "true_label": record.get("true_label"),
                    "predicted_label": record.get("prediction"),
                    "predicted_proba": record.get("prediction_proba"),
                    "metadata": record.get("metadata"),
                },
            )
            if meta.get("true_label") is None:
                meta["true_label"] = record.get("true_label")
            if meta.get("predicted_label") is None:
                meta["predicted_label"] = record.get("prediction")
            if meta.get("predicted_proba") is None:
                meta["predicted_proba"] = record.get("prediction_proba")
            candidates[dataset_idx_int].append(
                {
                    "method": method_label,
                    "metrics": metrics,
                    "source_file": str(file_path),
                    "metadata": record.get("metadata"),
                }
            )

    pareto_instances: List[Dict[str, Any]] = []
    for dataset_idx in sorted(candidates.keys()):
        front = pareto_front(candidates[dataset_idx])
        if not front:
            continue
        metrics_keys = sorted({key for cand in front for key in cand["metrics"].keys()})
        meta = instance_meta.get(dataset_idx, {"dataset_index": dataset_idx})
        pareto_instances.append(
            {
                "dataset_index": dataset_idx,
                "instance_id": meta.get("instance_id"),
                "true_label": meta.get("true_label"),
                "predicted_label": meta.get("predicted_label"),
                "predicted_proba": meta.get("predicted_proba"),
                "pareto_metrics": metrics_keys,
                "pareto_front": front,
            }
        )

    return {
        "dataset": dataset_name,
        "model": model_dir.name,
        "source_dir": str(model_dir),
        "n_instances": len(pareto_instances),
        "instances": pareto_instances,
    }


def process_metrics_model(dataset_name: str, model_dir: Path) -> Dict[str, Any]:
    metric_files = sorted(model_dir.glob("*_metrics.json"))
    if not metric_files:
        return {
            "dataset": dataset_name,
            "model": model_dir.name,
            "source_dir": str(model_dir),
            "n_instances": 0,
            "instances": [],
        }

    method_instances: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    instance_meta: Dict[int, Dict[str, Any]] = {}

    for file_path in metric_files:
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        method_label = payload.get("method") or file_path.stem.replace("_metrics", "")
        for record in payload.get("instances", []):
            dataset_idx = record.get("dataset_index")
            if dataset_idx is None:
                dataset_idx = record.get("instance_id")
            if dataset_idx is None:
                continue
            try:
                dataset_idx_int = int(dataset_idx)
            except (TypeError, ValueError):
                continue
            metrics = clean_metrics(record.get("metrics"))
            if not metrics:
                continue
            meta = instance_meta.setdefault(
                dataset_idx_int,
                {
                    "dataset_index": dataset_idx_int,
                    "instance_id": record.get("instance_id"),
                    "true_label": record.get("true_label"),
                    "predicted_label": record.get("prediction"),
                    "predicted_proba": record.get("prediction_proba"),
                },
            )
            if meta.get("true_label") is None:
                meta["true_label"] = record.get("true_label")
            if meta.get("predicted_label") is None:
                meta["predicted_label"] = record.get("prediction")
            if meta.get("predicted_proba") is None:
                meta["predicted_proba"] = record.get("prediction_proba")
            method_instances[dataset_idx_int].append(
                {
                    "method": method_label,
                    "metrics": metrics,
                    "source_file": str(file_path),
                }
            )

    pareto_instances: List[Dict[str, Any]] = []
    for dataset_idx in sorted(method_instances.keys()):
        front = pareto_front(method_instances[dataset_idx])
        if not front:
            continue
        metric_keys = sorted({key for cand in front for key in cand["metrics"].keys()})
        meta = instance_meta.get(dataset_idx, {"dataset_index": dataset_idx})
        pareto_instances.append(
            {
                "dataset_index": dataset_idx,
                "instance_id": meta.get("instance_id"),
                "true_label": meta.get("true_label"),
                "predicted_label": meta.get("predicted_label"),
                "predicted_proba": meta.get("predicted_proba"),
                "pareto_metrics": metric_keys,
                "pareto_front": front,
            }
        )

    return {
        "dataset": dataset_name,
        "model": model_dir.name,
        "source_dir": str(model_dir),
        "n_instances": len(pareto_instances),
        "instances": pareto_instances,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result_dir = args.results_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "aggregate":
        inputs = collect_input_files(args.paths, result_dir)
        if not inputs:
            raise SystemExit("No result files found to process.")
        for path in inputs:
            payload = process_file(path)
            output_path = output_dir / f"{path.stem}_pareto.json"
            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            print(f"Wrote Pareto summaries to {output_path}")
    elif args.mode == "detailed":
        targets = collect_detailed_targets(args.paths, args.detailed_dir)
        if not targets:
            raise SystemExit("No detailed explanation directories found to process.")
        for dataset_name, model_dir in targets:
            payload = process_detailed_model(dataset_name, model_dir)
            output_path = output_dir / f"{dataset_name}__{model_dir.name}_pareto.json"
            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            print(f"Wrote Pareto summaries to {output_path}")
    else:
        targets = collect_metric_targets(args.paths, args.metrics_dir)
        if not targets:
            raise SystemExit("No metrics directories found to process.")
        for dataset_name, model_dir in targets:
            payload = process_metrics_model(dataset_name, model_dir)
            output_path = output_dir / f"{dataset_name}__{model_dir.name}_pareto.json"
            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            print(f"Wrote Pareto summaries to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
