"""Helpers for persisting and loading experiment artifacts."""

from __future__ import annotations

import json
import logging
import os
import uuid
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
    # Avoid races under multi-process job launches (e.g. HPC array jobs) by using a unique tmp name.
    tmp_path = meta_path.with_name(f"{meta_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    try:
        tmp_path.replace(meta_path)
    except FileNotFoundError:
        # Another process may have won the race and moved its tmp into place.
        if meta_path.exists():
            return
        raise
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


def checkpoint_explanations(
    *,
    method_label: str,
    path: Path,
    dataset_mapping: Dict[int, Any],
    feature_names: List[str],
    y_pred,
    y_true,
    y_proba,
) -> None:
    """Persist per-instance explanation payload for a method to disk."""
    records: List[Dict[str, Any]] = []
    for dataset_idx in sorted(dataset_mapping.keys()):
        mapping_value = dataset_mapping[dataset_idx]
        if isinstance(mapping_value, list):
            entries = mapping_value
        else:
            entries = [mapping_value]

        explanations: List[Dict[str, Any]] = []
        # Preserve the first explanation's metadata (minus dataset_index) for record-level compatibility.
        primary_metadata: Dict[str, Any] = {}
        for local_idx, explanation in entries:
            meta = dict(explanation.get("metadata") or {})
            meta.setdefault("dataset_index", dataset_idx)
            if not primary_metadata:
                primary_metadata = dict(meta)
                primary_metadata.pop("dataset_index", None)
            exp_entry: Dict[str, Any] = {
                "feature_importance": np.asarray(
                    explanation.get("attributions", [])
                ).tolist(),
                "metadata": to_serializable(meta) if meta else {},
                "method_variant": explanation.get("method") or method_label,
            }
            gen_time = explanation.get("generation_time")
            if gen_time is not None:
                exp_entry["generation_time"] = float(gen_time)
            explanations.append(exp_entry)

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
            "explanations": explanations,
            "metadata": to_serializable(primary_metadata) if primary_metadata else {},
        }
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
        record_metadata = dict(record.get("metadata") or {})
        dataset_idx = record.get("dataset_index")
        instance_id = record.get("instance_id")

        nested_explanations = record.get("explanations")
        if nested_explanations:
            for expl in nested_explanations:
                metadata = dict(expl.get("metadata") or {})
                if dataset_idx is not None:
                    metadata.setdefault("dataset_index", dataset_idx)
                elif instance_id is not None:
                    metadata.setdefault("dataset_index", instance_id)
                elif "dataset_index" in metadata:
                    dataset_idx = metadata.get("dataset_index")
                explanation_entry: Dict[str, Any] = {
                    "method": method_label,
                    "attributions": expl.get("feature_importance")
                    or expl.get("attributions")
                    or [],
                    "metadata": metadata,
                    "metadata_key": dataset_idx
                    if dataset_idx is not None
                    else instance_id,
                }
                gen_time = expl.get("generation_time") or record.get("generation_time")
                if gen_time is not None:
                    explanation_entry["generation_time"] = float(gen_time)
                explanations.append(explanation_entry)
            continue

        # Backward-compat: legacy flat records with feature_importance.
        metadata = dict(record_metadata)
        if dataset_idx is not None:
            metadata.setdefault("dataset_index", dataset_idx)
        explanation_entry: Dict[str, Any] = {
            "method": method_label,
            "attributions": record.get("feature_importance") or [],
            "metadata": metadata,
            "metadata_key": dataset_idx if dataset_idx is not None else instance_id,
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
) -> Tuple[Dict[int, Dict[int, Dict[str, float]]], Dict[str, float], Dict[str, Dict[str, float]]]:
    """Load cached per-instance/batch metrics for a method.

    Returns
    -------
    instance_metrics : dict
        Mapping dataset_index -> explanation_index -> metrics dict.
    batch_metrics : dict
    batch_metrics_by_variant : dict
        Mapping method_variant -> batch metrics dict.
    """
    if not file_path.exists():
        return {}, {}, {}
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.warning("Failed to load cached metrics from %s: %s", file_path, exc)
        return {}, {}, {}
    file_method = payload.get("method", method_label)
    if file_method != method_label:
        LOGGER.warning(
            "Cached metrics file %s method=%s does not match expected label %s",
            file_path,
            file_method,
            method_label,
        )
    instances: Dict[int, Dict[int, Dict[str, float]]] = {}
    for record in payload.get("instances", []):
        dataset_idx = record.get("dataset_index", record.get("instance_id"))
        if dataset_idx is None:
            continue
        try:
            dataset_idx_int = int(dataset_idx)
        except (TypeError, ValueError):
            continue
        explanation_index = record.get("explanation_index")
        try:
            expl_idx_int = int(explanation_index) if explanation_index is not None else 0
        except (TypeError, ValueError):
            expl_idx_int = 0
        metrics = coerce_metric_dict(record.get("metrics") or {})
        if metrics:
            bucket = instances.setdefault(dataset_idx_int, {})
            bucket[expl_idx_int] = metrics
    batch_metrics = coerce_metric_dict(payload.get("batch_metrics") or {})
    batch_by_variant_raw = payload.get("batch_metrics_by_variant") or {}
    batch_metrics_by_variant: Dict[str, Dict[str, float]] = {}
    for variant_label, metrics_map in batch_by_variant_raw.items():
        batch_metrics_by_variant[variant_label] = coerce_metric_dict(metrics_map) if metrics_map else {}
    return instances, batch_metrics, batch_metrics_by_variant


def write_metric_results(
    *,
    metrics_dir: Path,
    dataset_name: str,
    model_name: str,
    method_label: str,
    instances: List[Dict[str, Any]],
    batch_metrics: Dict[str, float],
    metric_metadata: Dict[str, Dict[str, Any]],
    batch_metrics_by_variant: Optional[Dict[str, Dict[str, float]]] = None,
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
    if batch_metrics_by_variant:
        payload["batch_metrics_by_variant"] = batch_metrics_by_variant
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
