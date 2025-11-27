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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.datasets import TabularDataset
from src.datasets.adapters import LoaderDatasetAdapter
from src.explainers import make_explainer
from src.models import SklearnModel

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

def _load_config(name: str) -> Dict[str, Any]:
    path = CONFIG_DIR / name
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


DATASET_CFG = _load_config("dataset.yml")
MODEL_CFG = _load_config("models.yml")
EXPLAINER_CFG = _load_config("explainers.yml")
METRIC_CFG = _load_config("metrics.yml")
EXPERIMENT_CFG = _load_config("experiments.yml")


def _import_object(module_name: str, attr: str) -> Any:
    module = importlib.import_module(module_name)
    obj = module
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def instantiate_dataset(name: str) -> TabularDataset:
    spec = DATASET_CFG[name]
    adapter_spec = spec.get("adapter")
    if adapter_spec:
        adapter_cls = _import_object(adapter_spec["module"], adapter_spec["class"])
    else:
        adapter_cls = LoaderDatasetAdapter
    adapter = adapter_cls(name=name, spec=spec)
    return adapter.load()


def instantiate_model(name: str) -> SklearnModel:
    spec = MODEL_CFG[name]
    model_cls = _import_object(spec["module"], spec["class"])
    params = spec.get("params", {}) or {}
    requires_scaler = spec.get("fit", {}).get("requires_scaler", False)

    if requires_scaler:
        estimator = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("estimator", model_cls(**params)),
            ]
        )
    else:
        estimator = model_cls(**params)

    return SklearnModel(name=name, estimator=estimator)


def instantiate_explainer(
    name: str,
    model: Any,
    dataset: TabularDataset,
    *,
    logging_cfg: Optional[Dict[str, Any]] = None,
):
    spec = EXPLAINER_CFG[name]
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
