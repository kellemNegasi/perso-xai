"""Metric-evaluation helpers shared across orchestrator components."""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple, Optional, Union, List

import numpy as np

LOGGER = logging.getLogger(__name__)


def evaluate_metrics_for_method(
    *,
    metric_objs: Dict[str, Any],
    metric_caps: Dict[str, Dict[str, Any]],
    explainer,
    expl_results: Dict[str, Any],
    dataset_mapping: Dict[int, Union[Tuple[int, Dict[str, Any]], List[Tuple[int, Dict[str, Any]]]]],
    model,
    dataset,
    method_label: str,
    log_progress: bool,
) -> Tuple[Dict[str, float], Dict[int, Dict[str, float]]]:
    """Execute metric evaluators for a single explainer method.

    Returns
    -------
    batch_metrics : Dict[str, float]
    instance_metrics : Dict[int, Dict[int, Dict[str, float]]]
        Mapping dataset_index -> explanation_index (local_idx) -> metric values.
    """
    batch_metrics: Dict[str, float] = {}
    instance_metrics: Dict[int, Dict[int, Dict[str, float]]] = {}
    if not metric_objs:
        return batch_metrics, instance_metrics

    def _iter_entries(mapping_value: Union[Tuple[int, Dict[str, Any]], List[Tuple[int, Dict[str, Any]]]]):
        """Yield (local_idx, explanation) tuples regardless of single/list input."""
        if isinstance(mapping_value, list):
            for entry in mapping_value:
                yield entry
        else:
            yield mapping_value

    for metric_name, metric in metric_objs.items():
        caps = metric_caps[metric_name]
        if caps["per_instance"]:
            if log_progress:
                LOGGER.info(
                    "Running %s metric (per-instance) for %s", metric_name, method_label
                )
            for dataset_idx, mapping_value in dataset_mapping.items():
                for entry in _iter_entries(mapping_value):
                    try:
                        local_idx, _ = entry
                    except (TypeError, ValueError):
                        continue
                    payload = dict(expl_results)
                    payload["current_index"] = local_idx
                    out = metric.evaluate(
                        model=model,
                        explanation_results=payload,
                        dataset=dataset,
                        explainer=explainer,
                    )
                    values = coerce_metric_dict(out)
                    if not values:
                        continue
                    dataset_bucket = instance_metrics.setdefault(int(dataset_idx), {})
                    metrics_bucket = dataset_bucket.setdefault(int(local_idx), {})
                    metrics_bucket.update(values)
            continue

        if not caps["requires_full_batch"]:
            continue
        if log_progress:
            LOGGER.info("Running %s metric (batch) for %s", metric_name, method_label)
        out = metric.evaluate(
            model=model,
            explanation_results=expl_results,
            dataset=dataset,
            explainer=explainer,
        )
        batch_metrics.update(coerce_metric_dict(out))

    return batch_metrics, instance_metrics


def extract_metric_parameters(metric: Any) -> Dict[str, Any]:
    """Return the public JSON-able attributes of a metric instance."""
    params: Dict[str, Any] = {}
    attr_dict = getattr(metric, "__dict__", {})
    for key, value in attr_dict.items():
        if key.startswith("_"):
            continue
        if _is_jsonable(value):
            params[key] = value
    return params


def coerce_metric_dict(values: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """Normalize a metric output mapping to floats, dropping invalid entries."""
    if not values:
        return {}
    coerced: Dict[str, float] = {}
    for key, value in values.items():
        if value is None:
            continue
        try:
            coerced[key] = float(value)
        except (TypeError, ValueError):
            continue
    return coerced


def value_at(sequence, index: int):
    """Safely fetch sequence[index], returning None for out-of-range access."""
    if sequence is None or index < 0:
        return None
    try:
        length = len(sequence)
    except TypeError:
        return None
    if index >= length:
        return None
    try:
        return sequence[index]
    except (IndexError, TypeError):
        return None


def safe_scalar(value: Any) -> Any:
    """Convert numpy scalar/array to a native scalar when possible."""
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = value.ravel()[0]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _is_jsonable(value: Any) -> bool:
    """Internal helper to check whether a value can be serialized to JSON."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_jsonable(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) for k in value.keys()) and all(
            _is_jsonable(v) for v in value.values()
        )
    return False


__all__ = [
    "evaluate_metrics_for_method",
    "extract_metric_parameters",
    "coerce_metric_dict",
    "value_at",
    "safe_scalar",
]
