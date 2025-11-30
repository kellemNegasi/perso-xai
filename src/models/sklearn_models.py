"""Wrappers around scikit-learn estimators with a unified interface."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from .base import BaseModel


class SklearnModel(BaseModel[Any]):
    """Wrap a scikit-learn estimator to expose the :class:`BaseModel` API."""

    def __init__(self, name: str, estimator: Any):
        super().__init__(name=name, estimator=estimator)
        self.supports_proba = hasattr(estimator, "predict_proba")
        self._label_encoder: Optional[LabelEncoder] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "SklearnModel":
        y_encoded = y
        if y is not None:
            y_arr = np.asarray(y)
            if y_arr.ndim > 1 and y_arr.shape[1] == 1:
                y_arr = y_arr.ravel()
            if _needs_label_encoding(y_arr):
                encoder = LabelEncoder()
                y_encoded = encoder.fit_transform(y_arr)
                self._label_encoder = encoder
            else:
                self._label_encoder = None
                y_encoded = y_arr
        self._estimator.fit(X, y_encoded)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = self._estimator.predict(X)
        if self._label_encoder is not None:
            preds = self._label_encoder.inverse_transform(np.asarray(preds, dtype=int))
        return preds

    def predict_numeric(self, X: np.ndarray) -> np.ndarray:
        preds = self._estimator.predict(X)
        return np.asarray(preds, dtype=float)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.supports_proba:
            return super().predict_proba(X)
        return self._estimator.predict_proba(X)

    def __getattr__(self, item: str) -> Any:
        if item.startswith("__") or self._estimator is None:
            raise AttributeError(item)
        # Delegate everything else to the underlying estimator (e.g., feature_importances_).
        return getattr(self._estimator, item)


def train_simple_classifier(
    X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42
) -> SklearnModel:
    """Train a default RandomForest and return the wrapped model."""

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=random_state,
    )
    model = SklearnModel(name="random_forest_default", estimator=clf)
    model.fit(X_train, y_train)
    return model


__all__ = ["SklearnModel", "train_simple_classifier"]


def _needs_label_encoding(y: np.ndarray) -> bool:
    if y.dtype.kind in {"O", "U", "S"}:
        return True
    return False
