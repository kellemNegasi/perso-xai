# src/datasets/tabular.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence
import numpy as np

try:
    import pandas as pd
    _HAS_PANDAS = True
except Exception:
    _HAS_PANDAS = False


@dataclass
class TabularDataset:
    X_train: np.ndarray
    y_train: Optional[np.ndarray]
    X_test: np.ndarray
    y_test: Optional[np.ndarray]
    feature_names: Sequence[str]

    def __post_init__(self):
        # For BaseExplainer fallback
        self.feature_means = np.mean(self.X_train, axis=0)

    @classmethod
    def from_arrays(
        cls,
        X_train,
        y_train,
        X_test,
        y_test=None,
        feature_names: Optional[Sequence[str]] = None,
    ):
        X_train = _to_numpy_2d(X_train)
        X_test  = _to_numpy_2d(X_test)
        y_train = _to_numpy_1d(y_train) if y_train is not None else None
        y_test  = _to_numpy_1d(y_test) if y_test is not None else None

        if feature_names is None:
            if _HAS_PANDAS and hasattr(X_train, "columns"):
                feature_names = list(X_train.columns)
            else:
                feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

        return cls(X_train, y_train, X_test, y_test, feature_names)


def _to_numpy_2d(X):
    if _HAS_PANDAS:
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            return X.values
        if isinstance(X, pd.Series):
            return X.values.reshape(1, -1)
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return X


def _to_numpy_1d(y):
    if y is None:
        return None
    if _HAS_PANDAS:
        import pandas as pd
        if isinstance(y, pd.Series):
            return y.values
        if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            return y.values.ravel()
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.ravel()
    return y
