from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .autoxai_config import method_metrics_path
from .autoxai_utils import load_json, safe_float


def load_method_instances(
    *,
    results_root: Path,
    dataset: str,
    model: str,
    method: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Returns:
      - instances list (raw JSON)
      - map method_variant -> hyperparameters dict (best-effort)
    """
    path = method_metrics_path(results_root, dataset, model, method)
    payload = load_json(path)
    instances = payload.get("instances") or []
    if not isinstance(instances, list):
        raise ValueError(f"Expected 'instances' list in {path}.")
    variant_hparams: Dict[str, Dict[str, Any]] = {}
    for entry in instances:
        if not isinstance(entry, dict):
            continue
        variant = entry.get("method_variant")
        if not isinstance(variant, str):
            continue
        meta = entry.get("explanation_metadata") or {}
        if isinstance(meta, dict):
            hparams = meta.get("hyperparameters")
            if isinstance(hparams, dict) and variant not in variant_hparams:
                variant_hparams[variant] = dict(hparams)
    return instances, variant_hparams


def build_candidate_metrics(
    *,
    instances_by_method: Mapping[str, List[Dict[str, Any]]],
    allowed_variants: Optional[set[str]] = None,
) -> Tuple[Dict[int, Dict[str, Dict[str, float]]], Dict[str, str]]:
    """
    Returns:
      - dataset_index -> method_variant -> metric_key -> value
      - method_variant -> method
    """
    per_instance: Dict[int, Dict[str, Dict[str, float]]] = {}
    variant_to_method: Dict[str, str] = {}
    for method, instances in instances_by_method.items():
        for entry in instances:
            if not isinstance(entry, dict):
                continue
            dataset_index = entry.get("dataset_index")
            if not isinstance(dataset_index, int):
                continue
            variant = entry.get("method_variant")
            if not isinstance(variant, str):
                continue
            if allowed_variants is not None and variant not in allowed_variants:
                continue
            metrics = entry.get("metrics") or {}
            if not isinstance(metrics, dict):
                continue
            per_instance.setdefault(dataset_index, {})
            metric_values: Dict[str, float] = {}
            for key, value in metrics.items():
                parsed = safe_float(value)
                if parsed is not None:
                    metric_values[str(key)] = parsed
            per_instance[dataset_index][variant] = metric_values
            variant_to_method[variant] = method
    return per_instance, variant_to_method

