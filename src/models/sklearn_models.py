"""Wrappers around scikit-learn estimators with a unified interface."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .base import BaseModel


class SklearnModel(BaseModel[Any]):
    """Wrap a scikit-learn estimator to expose the :class:`BaseModel` API."""

    def __init__(self, name: str, estimator: Any):
        super().__init__(name=name, estimator=estimator)
        self.supports_proba = hasattr(estimator, "predict_proba")

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "SklearnModel":
        self._estimator.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._estimator.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.supports_proba:
            return super().predict_proba(X)
        return self._estimator.predict_proba(X)

    def __getattr__(self, item: str) -> Any:
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
