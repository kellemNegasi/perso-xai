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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.datasets import TabularDataset
from src.explainers import make_explainer

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


def _maybe_get(obj: Any, attr: str) -> Any:
    if hasattr(obj, attr):
        return getattr(obj, attr)
    getter = getattr(obj, "get", None)
    if callable(getter):
        try:
            return getter(attr)
        except Exception:
            return None
    return None


def instantiate_dataset(name: str) -> TabularDataset:
    spec = DATASET_CFG[name]
    loader = _import_object(spec["loader"]["module"], spec["loader"]["factory"])
    params = spec.get("params", {}) or {}

    obj = loader(**params)
    if isinstance(obj, TabularDataset):
        return obj

    data = _maybe_get(obj, "data")
    target = _maybe_get(obj, "target")
    feature_names = _maybe_get(obj, "feature_names")
    frame = _maybe_get(obj, "frame")

    if frame is not None:
        data = frame.drop(columns=[frame.columns[-1]]).values
        target = frame[frame.columns[-1]].values

    split_cfg = params.get("split", spec.get("split", {})) or {}
    test_size = float(split_cfg.get("test_size", 0.25))
    random_state = split_cfg.get("random_state", 42)
    stratify = None
    if split_cfg.get("stratify", True) and target is not None:
        unique = np.unique(target)
        if unique.size > 1:
            stratify = target

    X_train, X_test, y_train, y_test = train_test_split(
        np.asarray(data),
        np.asarray(target),
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    wrapper = spec.get("wrapper")
    if wrapper:
        wrapper_fn = _import_object(wrapper["module"], wrapper["factory"])
        return wrapper_fn(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_names,
        )
    return TabularDataset.from_arrays(
        X_train, y_train, X_test, y_test, feature_names=feature_names
    )


def instantiate_model(name: str):
    spec = MODEL_CFG[name]
    model_cls = _import_object(spec["module"], spec["class"])
    params = spec.get("params", {}) or {}
    requires_scaler = spec.get("fit", {}).get("requires_scaler", False)

    if requires_scaler:
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("estimator", model_cls(**params)),
            ]
        )
    return model_cls(**params)


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
