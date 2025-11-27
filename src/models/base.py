"""Common interfaces for model wrappers used in experiments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, TypeVar

import numpy as np

EstimatorT = TypeVar("EstimatorT")


class BaseModel(ABC, Generic[EstimatorT]):
    """Simple interface that wraps estimators with uniform APIs."""

    supports_proba: bool = False

    def __init__(self, name: str, estimator: EstimatorT):
        self.name = name
        self._estimator = estimator

    @property
    def estimator(self) -> EstimatorT:
        """Return the underlying estimator instance."""
        return self._estimator

    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "BaseModel[EstimatorT]":
        """Fit the underlying estimator and return ``self`` for chaining."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions for ``X``."""

    def predict_numeric(self, X: np.ndarray) -> np.ndarray:
        """Numeric predictions if available; defaults to ``predict`` output."""
        return self.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probability predictions when supported by the estimator."""
        raise AttributeError(f"{self.__class__.__name__} does not support predict_proba")

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"{self.__class__.__name__}(name={self.name}, estimator={self._estimator})"
