# src/datasets/adapters.py
"""Dataset adapters that normalize loader outputs into TabularDataset objects."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from src.datasets.tabular import TabularDataset

try:  # pragma: no cover - optional dependency
    import pandas as pd

    _HAS_PANDAS = True
except Exception:  # pragma: no cover - optional dependency
    pd = None  # type: ignore
    _HAS_PANDAS = False


class DatasetAdapter(ABC):
    """Base interface for dataset adapters."""

    def __init__(self, name: str, spec: Dict[str, Any]):
        self.name = name
        self.spec = spec

    @abstractmethod
    def load(self) -> TabularDataset:
        """Return a fully prepared :class:`TabularDataset`."""


class LoaderDatasetAdapter(DatasetAdapter):
    """Adapter that mirrors legacy loader/wrapper behavior from dataset.yml entries."""

    def load(self) -> TabularDataset:
        loader_spec = self.spec.get("loader")
        if not loader_spec:
            raise ValueError(f"Dataset '{self.name}' is missing a loader specification")
        loader = _import_object(loader_spec["module"], loader_spec["factory"])
        loader_params = dict(self.spec.get("params", {}) or {})

        obj = loader(**loader_params)
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
                dataset_spec=self.spec,
                loader_params=loader_params,
            )
        elif frame is not None:
            data = frame

        split_cfg = loader_params.get("split", self.spec.get("split", {})) or {}
        test_size = float(split_cfg.get("test_size", 0.25))
        random_state = split_cfg.get("random_state", 42)
        stratify = None

        target_array = None if target is None else np.asarray(target)
        if split_cfg.get("stratify", True) and target_array is not None:
            unique = np.unique(target_array)
            if unique.size > 1:
                stratify = target_array

        if data is None:
            raise ValueError(f"Dataset loader '{self.name}' did not return feature data")

        X_array, feature_names = _ensure_numeric_features(data, feature_names)

        X_train, X_test, y_train, y_test = _train_test_split(
            X_array,
            target_array,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )

        wrapper = self.spec.get("wrapper")
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


def _train_test_split(
    X: np.ndarray,
    y: Optional[np.ndarray],
    *,
    test_size: float,
    random_state: Optional[int],
    stratify: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    from sklearn.model_selection import train_test_split

    return train_test_split(
        np.asarray(X),
        None if y is None else np.asarray(y),
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


def _import_object(module_name: str, attr: str) -> Any:
    import importlib

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


def _extract_frame_components(
    frame: "pd.DataFrame",
    target,
    feature_names,
    *,
    dataset_spec: Dict[str, Any],
    loader_params: Dict[str, Any],
):
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
    target_values = target if target is not None else frame[target_column]

    feature_names = list(feature_frame.columns)
    return feature_frame, np.asarray(target_values), feature_names


def _ensure_numeric_features(
    data: Any, feature_names: Optional[Sequence[str]]
) -> Tuple[np.ndarray, Optional[Sequence[str]]]:
    if _HAS_PANDAS and isinstance(data, pd.DataFrame):
        encoded = _encode_categorical_dataframe(data)
        if feature_names is None or len(encoded.columns) != len(feature_names):
            feature_names = list(encoded.columns)
        return encoded.to_numpy(dtype=float, copy=True), feature_names

    array = np.asarray(data)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.dtype.kind in {"O", "U", "S"}:
        if not _HAS_PANDAS:
            raise TypeError(
                "Feature matrix contains non-numeric values but pandas is unavailable"
            )
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


__all__ = ["DatasetAdapter", "LoaderDatasetAdapter"]
