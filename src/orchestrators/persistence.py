"""Helpers for persisting and loading experiment artifacts."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .metrics_utils import coerce_metric_dict, safe_scalar, value_at
from .utils import to_serializable

LOGGER = logging.getLogger(__name__)


def ensure_dataset_metadata(
    dataset_dir: Path,
    dataset_name: str,
    dataset_type: str,
    feature_names: List[str],
) -> None:
    """Ensure a metadata.json file exists for the dataset directory."""
    meta_path = dataset_dir / "metadata.json"
    if meta_path.exists():
        return
    payload = {
        "dataset": dataset_name,
        "dataset_type": dataset_type,
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "generated_at": datetime.utcnow().isoformat(),
    }
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def checkpoint_explanations(
    *,
    method_label: str,
    path: Path,
    dataset_mapping: Dict[int, Tuple[int, Dict[str, Any]]],
    feature_names: List[str],
    y_pred,
    y_true,
    y_proba,
) -> None:
    """Persist per-instance explanation payload for a method to disk."""
    records: List[Dict[str, Any]] = []
    for dataset_idx in sorted(dataset_mapping.keys()):
        _, explanation = dataset_mapping[dataset_idx]
        metadata = dict(explanation.get("metadata") or {})
        metadata.pop("dataset_index", None)
        predicted_label = safe_scalar(value_at(y_pred, dataset_idx))
        true_label = safe_scalar(value_at(y_true, dataset_idx))
        proba_raw = value_at(y_proba, dataset_idx)
        predicted_proba = None if proba_raw is None else np.asarray(proba_raw).tolist()
        correct_prediction = true_label is not None and predicted_label == true_label
        record: Dict[str, Any] = {
            "instance_id": dataset_idx,
            "dataset_index": dataset_idx,
            "true_label": true_label,
            "prediction": predicted_label,
            "prediction_proba": predicted_proba,
            "correct_prediction": correct_prediction,
            "feature_importance": np.asarray(explanation.get("attributions", [])).tolist(),
            "metadata": to_serializable(metadata) if metadata else {},
        }
        gen_time = explanation.get("generation_time")
        if gen_time is not None:
            record["generation_time"] = float(gen_time)
        records.append(record)
    payload = {
        "metadata": {
            "feature_names": feature_names,
            "n_features": len(feature_names),
        },
        "records": records,
    }
    serialized = json.dumps(to_serializable(payload), indent=2)
    path.write_text(serialized, encoding="utf-8")
    LOGGER.debug(
        "Checkpointed %d explanations for %s at %s",
        len(records),
        method_label,
        path,
    )


def write_completion_flag(
    *,
    status_path: Path,
    explainer_key: str,
    method_label: str,
    dataset_name: str,
    model_name: str,
    detail_path: Path,
    metrics_path: Optional[Path],
    dataset_indices: Iterable[int],
) -> None:
    """Record a completion flag for a method including cache file paths."""
    payload = {
        "explainer": explainer_key,
        "method_label": method_label,
        "dataset": dataset_name,
        "model": model_name,
        "detail_path": str(detail_path),
        "metrics_path": str(metrics_path) if metrics_path else None,
        "dataset_indices": [int(idx) for idx in dataset_indices],
        "completed_at": datetime.utcnow().isoformat(),
    }
    status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_completion_flag(path: Path) -> Optional[Dict[str, Any]]:
    """Load the completion flag metadata if present."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Failed to read completion flag from %s: %s", path, exc)
        return None


def load_cached_explanations(file_path: Path, method_label: str) -> Optional[Dict[str, Any]]:
    """Load previously checkpointed explanations for a method."""
    if not file_path.exists():
        return None
    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.warning("Failed to load cached explanations from %s: %s", file_path, exc)
        return None

    feature_names = None
    if isinstance(raw, dict):
        raw_records = raw.get("records") or raw.get("explanations") or []
        meta_block = raw.get("metadata") or {}
        feature_names = meta_block.get("feature_names") or raw.get("feature_names")
    else:
        raw_records = raw

    explanations: List[Dict[str, Any]] = []
    for record in raw_records:
        metadata = dict(record.get("metadata") or {})
        dataset_idx = record.get("dataset_index")
        if dataset_idx is not None:
            metadata.setdefault("dataset_index", dataset_idx)
        explanation_entry: Dict[str, Any] = {
            "method": method_label,
            "attributions": record.get("feature_importance") or [],
            "metadata": metadata,
            "metadata_key": dataset_idx if dataset_idx is not None else record.get("instance_id"),
        }
        gen_time = record.get("generation_time")
        if gen_time is not None:
            explanation_entry["generation_time"] = float(gen_time)
        explanations.append(explanation_entry)

    return {
        "method": method_label,
        "explanations": explanations,
        "n_explanations": len(explanations),
        "info": {
            "source": "cached_file",
            "path": str(file_path),
            "feature_names": feature_names,
        },
    }


def load_cached_metrics(
    file_path: Path, method_label: str
) -> Tuple[Dict[int, Dict[str, float]], Dict[str, float]]:
    """Load cached per-instance/batch metrics for a method."""
    if not file_path.exists():
        return {}, {}
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.warning("Failed to load cached metrics from %s: %s", file_path, exc)
        return {}, {}
    file_method = payload.get("method", method_label)
    if file_method != method_label:
        LOGGER.warning(
            "Cached metrics file %s method=%s does not match expected label %s",
            file_path,
            file_method,
            method_label,
        )
    instances: Dict[int, Dict[str, float]] = {}
    for record in payload.get("instances", []):
        dataset_idx = record.get("dataset_index", record.get("instance_id"))
        if dataset_idx is None:
            continue
        try:
            dataset_idx_int = int(dataset_idx)
        except (TypeError, ValueError):
            continue
        metrics = coerce_metric_dict(record.get("metrics") or {})
        if metrics:
            instances[dataset_idx_int] = metrics
    batch_metrics = coerce_metric_dict(payload.get("batch_metrics") or {})
    return instances, batch_metrics


def write_metric_results(
    *,
    metrics_dir: Path,
    dataset_name: str,
    model_name: str,
    method_label: str,
    instances: List[Dict[str, Any]],
    batch_metrics: Dict[str, float],
    metric_metadata: Dict[str, Dict[str, Any]],
) -> Optional[str]:
    """Write per-method metric outputs to disk and return the file path."""
    if not instances and not batch_metrics:
        return None
    payload = {
        "dataset": dataset_name,
        "model": model_name,
        "method": method_label,
        "generated_at": datetime.utcnow().isoformat(),
        "metric_metadata": metric_metadata,
        "instances": instances,
        "batch_metrics": batch_metrics,
    }
    file_path = metrics_dir / f"{method_label}_metrics.json"
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(to_serializable(payload), handle, indent=2)
    return str(file_path)


__all__ = [
    "ensure_dataset_metadata",
    "checkpoint_explanations",
    "write_completion_flag",
    "load_completion_flag",
    "load_cached_explanations",
    "load_cached_metrics",
    "write_metric_results",
]
