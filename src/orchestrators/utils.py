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
from src.models import SklearnModel

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

try:
    import pandas as pd  # type: ignore

    _HAS_PANDAS = True
except Exception:  # pragma: no cover - optional dependency
    pd = None  # type: ignore
    _HAS_PANDAS = False


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

    if frame is not None and _HAS_PANDAS and isinstance(frame, pd.DataFrame):
        data, target, feature_names = _extract_frame_components(
            frame,
            target,
            feature_names,
            dataset_spec=spec,
            loader_params=params,
        )
    elif frame is not None:
        # No pandas support: fallback to numpy conversion.
        data = frame

    split_cfg = params.get("split", spec.get("split", {})) or {}
    test_size = float(split_cfg.get("test_size", 0.25))
    random_state = split_cfg.get("random_state", 42)
    stratify = None
    if split_cfg.get("stratify", True) and target is not None:
        unique = np.unique(target)
        if unique.size > 1:
            stratify = target

    if data is None:
        raise ValueError(f"Dataset loader '{name}' did not return feature data")

    X_array, feature_names = _ensure_numeric_features(data, feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X_array,
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


def _extract_frame_components(
    frame,
    target,
    feature_names,
    *,
    dataset_spec: Dict[str, Any],
    loader_params: Dict[str, Any],
):
    """Separate feature columns and target from a pandas DataFrame."""
    if not _HAS_PANDAS:
        raise RuntimeError("pandas is required to process frame outputs from dataset loaders")

    target_column = (
        loader_params.get("target_column")
        or dataset_spec.get("target_column")
        or (getattr(target, "name", None) if target is not None else None)
    )

    if target_column is None and "target" in frame.columns:
        target_column = "target"
    if target_column is None or target_column not in frame.columns:
        target_column = frame.columns[-1]

    feature_frame = frame.drop(columns=[target_column], errors="ignore")
    target_values = target
    if target_values is None:
        target_values = frame[target_column]

    feature_names = list(feature_frame.columns)
    return feature_frame, np.asarray(target_values), feature_names


def _ensure_numeric_features(data, feature_names):
    """Convert feature matrix to a numeric numpy array, encoding categoricals if needed."""
    if _HAS_PANDAS and isinstance(data, pd.DataFrame):
        encoded = _encode_categorical_dataframe(data)
        if feature_names is None or len(encoded.columns) != len(feature_names):
            feature_names = list(encoded.columns)
        return encoded.to_numpy(dtype=float, copy=True), feature_names

    array = np.asarray(data)
    if array.dtype.kind in {"O", "U", "S"}:
        if not _HAS_PANDAS:
            raise TypeError("Feature matrix contains non-numeric values but pandas is unavailable")
        columns = feature_names or [f"feature_{i}" for i in range(array.shape[1])]
        encoded = _encode_categorical_dataframe(pd.DataFrame(array, columns=columns))
        feature_names = list(encoded.columns)
        return encoded.to_numpy(dtype=float, copy=True), feature_names

    return array, feature_names


def _encode_categorical_dataframe(df):
    if not _HAS_PANDAS:
        raise RuntimeError("pandas is required to encode categorical features")
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    if not categorical_cols.empty:
        df = pd.get_dummies(df, columns=list(categorical_cols), dummy_na=False)
    return df


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
