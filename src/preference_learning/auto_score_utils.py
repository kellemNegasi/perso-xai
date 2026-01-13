"""Helpers for AutoXAI baseline scoring using saved metrics artifacts."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


def default_metrics_results_dir(
    encoded_path: Path,
    *,
    dataset_name: str,
    model_name: str,
    default_results_root: Path,
) -> Path | None:
    """
    Resolve the `metrics_results/<dataset>/<model>/` directory for an encoded Pareto file.

    Prefers a run-local directory (alongside the encoded parquet), falling back to the
    legacy `default_results_root`.
    """
    for parent in encoded_path.parents:
        candidate = parent / "metrics_results" / dataset_name / model_name
        if candidate.exists():
            return candidate
    fallback = default_results_root / "metrics_results" / dataset_name / model_name
    if fallback.exists():
        return fallback
    return None


def extract_hpo_candidate_scores(entries: object) -> List[Dict[str, Any]]:
    if not isinstance(entries, list):
        return []
    out: List[Dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        dataset_index = entry.get("dataset_index")
        method_variant = entry.get("method_variant")
        aggregated_score = entry.get("aggregated_score")
        if not isinstance(method_variant, str) or not method_variant:
            continue
        if not isinstance(dataset_index, (int, float)):
            continue
        if not isinstance(aggregated_score, (int, float)):
            continue
        out.append(
            {
                "dataset_index": int(dataset_index),
                "method_variant": method_variant,
                "aggregated_score": float(aggregated_score),
            }
        )
    return out


def extract_hpo_trial_variant_scores(entries: object) -> Dict[str, float]:
    """
    Extract per-variant HPO trial objective scores.

    These correspond to `hpo.trials[].aggregated_score` produced by the sequential
    trial-history objective in `metrics_runner.py`.
    """
    if not isinstance(entries, list):
        return {}
    best: Dict[str, float] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        method_variant = entry.get("method_variant")
        aggregated_score = entry.get("aggregated_score")
        if not isinstance(method_variant, str) or not method_variant:
            continue
        if not isinstance(aggregated_score, (int, float)):
            continue
        score = float(aggregated_score)
        current = best.get(method_variant)
        if current is None or score > current:
            best[method_variant] = score
    return best


def load_autoxai_hpo_candidate_scores(
    encoded_path: Path,
    *,
    dataset_name: str,
    model_name: str,
    default_results_root: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, float], Path | None]:
    metrics_dir = default_metrics_results_dir(
        encoded_path,
        dataset_name=dataset_name,
        model_name=model_name,
        default_results_root=default_results_root,
    )
    if metrics_dir is None:
        return [], [], {}, None
    overall: List[Dict[str, Any]] = []
    per_instance: List[Dict[str, Any]] = []
    trial_scores: Dict[str, float] = {}
    for path in sorted(metrics_dir.glob("*_metrics.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        hpo = payload.get("hpo")
        if not isinstance(hpo, Mapping):
            continue
        overall.extend(extract_hpo_candidate_scores(hpo.get("candidate_scores_overall_trial_scope")))
        per_instance.extend(extract_hpo_candidate_scores(hpo.get("candidate_scores_per_instance_trial_scope")))
        extracted = extract_hpo_trial_variant_scores(hpo.get("trials"))
        for variant, score in extracted.items():
            current = trial_scores.get(variant)
            if current is None or score > current:
                trial_scores[variant] = score
    return overall, per_instance, trial_scores, metrics_dir


def build_autoxai_hpo_per_instance_scores(
    entries: Sequence[Mapping[str, Any]],
    *,
    allowed_instances: set[int],
    allowed_variants: set[str],
) -> Dict[int, Dict[str, float]]:
    totals: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for entry in entries:
        dataset_index = entry.get("dataset_index")
        variant = entry.get("method_variant")
        score = entry.get("aggregated_score")
        if not isinstance(dataset_index, int) or dataset_index not in allowed_instances:
            continue
        if not isinstance(variant, str) or variant not in allowed_variants:
            continue
        if not isinstance(score, (int, float)):
            continue
        totals[int(dataset_index)][variant] += float(score)
        counts[int(dataset_index)][variant] += 1
    out: Dict[int, Dict[str, float]] = {}
    for dataset_index, by_variant in totals.items():
        out[dataset_index] = {}
        for variant, total in by_variant.items():
            denom = counts[dataset_index].get(variant, 0)
            if denom:
                out[dataset_index][variant] = total / denom
    return out

