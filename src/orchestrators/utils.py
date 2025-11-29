"""
Shared helpers for experiment orchestration (config loading, factories, serialization).
"""

from __future__ import annotations

import copy
import importlib
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

from src.datasets import TabularDataset
from src.datasets.adapters import LoaderDatasetAdapter
from src.explainers import make_explainer
from src.models import SklearnModel
from src.models.builder import build_estimator_from_spec
from src.orchestrators.registry import (
    DatasetRegistry,
    ExplainerRegistry,
    ModelRegistry,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

def _load_config(name: str) -> Dict[str, Any]:
    path = CONFIG_DIR / name
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


DATASET_REGISTRY = DatasetRegistry()
MODEL_REGISTRY = ModelRegistry()
EXPLAINER_REGISTRY = ExplainerRegistry()
METRIC_CFG = _load_config("metrics.yml")
EXPERIMENT_CFG = _load_config("experiments.yml")
VALIDATION_CFG = _load_config("validation.yml")


def _import_object(module_name: str, attr: str) -> Any:
    module = importlib.import_module(module_name)
    obj = module
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def instantiate_dataset(name: str, *, data_type: Optional[str] = None) -> TabularDataset:
    spec = DATASET_REGISTRY.get(name)
    dataset_type = spec.get("type")
    if data_type and dataset_type != data_type:
        raise ValueError(
            f"Dataset '{name}' has type '{dataset_type}' which does not match requested '{data_type}'."
        )
    adapter_spec = spec.get("adapter")
    if adapter_spec:
        adapter_cls = _import_object(adapter_spec["module"], adapter_spec["class"])
    else:
        adapter_cls = LoaderDatasetAdapter
    adapter = adapter_cls(name=name, spec=spec)
    return adapter.load()


def instantiate_model(
    name: str,
    *,
    data_type: Optional[str] = None,
    params_override: Optional[Dict[str, Any]] = None,
) -> SklearnModel:
    spec = MODEL_REGISTRY.get(name)
    supported_types = spec.get("supported_data_types", ["tabular"])
    if data_type and data_type not in supported_types:
        raise ValueError(
            f"Model '{name}' does not support data type '{data_type}'. Supported types: {supported_types}."
        )
    estimator = build_estimator_from_spec(spec, params_override=params_override)
    return SklearnModel(name=name, estimator=estimator)


def instantiate_explainer(
    name: str,
    model: Any,
    dataset: TabularDataset,
    *,
    data_type: Optional[str] = None,
    logging_cfg: Optional[Dict[str, Any]] = None,
):
    spec = EXPLAINER_REGISTRY.get(name)
    supported_types = spec.get("supported_data_types", ["tabular"])
    if data_type and data_type not in supported_types:
        raise ValueError(
            f"Explainer '{name}' does not support data type '{data_type}'. Supported types: {supported_types}."
        )
    config = {"type": spec["type"]}
    params = copy.deepcopy(spec.get("params", {}) or {})
    config.update(params)
    if logging_cfg:
        experiment_cfg = config.setdefault("experiment", {})
        current_logging = experiment_cfg.get("logging", {}) or {}
        experiment_cfg["logging"] = {**current_logging, **logging_cfg}
    explainer = make_explainer(config=config, model=model, dataset=dataset)
    explainer.fit(dataset.X_train, dataset.y_train)
    return explainer


def instantiate_metric(name: str):
    spec = METRIC_CFG[name]
    cls = _import_object(spec["module"], spec["class"])
    params = spec.get("params", {}) or {}
    return cls(**params)


def make_serializable_explanation(explanation: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(explanation)
    if "attributions" in out:
        out["attributions"] = np.asarray(out["attributions"]).tolist()
    if "instance" in out:
        out["instance"] = np.asarray(out["instance"]).tolist()
    if "prediction_proba" in out and out["prediction_proba"] is not None:
        out["prediction_proba"] = np.asarray(out["prediction_proba"]).tolist()

    metadata = out.get("metadata") or {}
    clean_meta = {}
    for key, value in metadata.items():
        if isinstance(value, np.ndarray):
            clean_meta[key] = value.tolist()
        else:
            clean_meta[key] = value
    out["metadata"] = clean_meta
    return out


def to_serializable(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    return obj


def metric_capabilities(metric: Any) -> Dict[str, bool]:
    return {
        "per_instance": getattr(metric, "per_instance", True),
        "requires_full_batch": getattr(metric, "requires_full_batch", False),
    }
