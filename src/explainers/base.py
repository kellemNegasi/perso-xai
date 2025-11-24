"""
Base explainer class for local (per-instance) XAI methods on tabular data.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

try:
    import pandas as pd
    _HAS_PANDAS = True
except Exception:
    logging.getLogger(__name__).debug("pandas not available, skipping related support.")
    _HAS_PANDAS = False


ArrayLike = Union[np.ndarray, "pd.Series", "pd.DataFrame", List[float], Tuple[float, ...]]  # noqa: F821 (pd guarded)
InstanceLike = Union[np.ndarray, "pd.Series", List[float], Tuple[float, ...]]  # noqa: F821


class BaseExplainer(ABC):
    """
    Base class for all local (per-instance) explanation methods on tabular data.

    Subclasses MUST implement:
        - explain_instance(self, instance: InstanceLike) -> Dict[str, Any]

    Optional overrides:
        - fit(self, X: ArrayLike, y: Optional[ArrayLike]) -> None
        - is_compatible(self) -> bool
        - explain_batch(self, X: ArrayLike) -> List[Dict[str, Any]]
        - explain_dataset(self, X: ArrayLike, y: Optional[ArrayLike]) -> Dict[str, Any]
    """

    # Declare support (subclasses should override as needed)
    supported_data_types: List[str] = ["tabular"]
    supported_model_types: List[str] = ["sklearn", "xgboost", "lightgbm", "catboost", "generic-predict"]

    def __init__(self, config: Dict[str, Any], model: Any, dataset: Any):
        """
        Args:
            config: Explanation method configuration (dict-like; safe-get is used).
            model: Trained model (must implement .predict; .predict_proba optional).
            dataset: Dataset object (used for metadata: feature names, transformers, etc.).
        """
        self.config = config or {}
        self.model = model
        self.dataset = dataset

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.generation_time: float = 0.0

        # Cache commonly-used config paths
        self._exp_cfg = self.config.get("experiment", {}) or {}
        self._expl_cfg = self._exp_cfg.get("explanation", {}) or {}
        self._log_cfg = self._exp_cfg.get("logging", {}) or {}
        self._log_progress = bool(
            self.config.get("log_progress")
            or self._exp_cfg.get("log_progress")
            or self._log_cfg.get("progress")
        )

        # Seed (if provided) for any stochastic explainers
        self.random_state: Optional[int] = self._expl_cfg.get("random_state")

    # -----------------------------
    # Public API (high-level)
    # -----------------------------

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> None:
        """
        Optional: Fit any explainer-specific surrogates/precomputations.
        Default no-op.
        """
        return None

    @abstractmethod
    def explain_instance(self, instance: InstanceLike) -> Dict[str, Any]:
        """
        Generate a local explanation for a single tabular instance.

        Returns a dictionary with at least:
            {
                "method": <str>,
                "prediction": <float|int|np.ndarray>,
                "prediction_proba": <np.ndarray or None>,
                "attributions": <np.ndarray or list>,
                "feature_names": <list[str]> (if available),
                "metadata": {...},
                "generation_time": <float seconds>
            }
        """
        raise NotImplementedError

    def explain_batch(self, X: ArrayLike) -> List[Dict[str, Any]]:
        """
        Default batch implementation loops over instances and calls explain_instance.
        Subclasses can override for vectorized speed-ups.
        """
        X_np, _ = self._coerce_X_y(X, None)
        results: List[Dict[str, Any]] = []
        for i in range(len(X_np)):
            start = time.time()
            res = self.explain_instance(self._row_to_instance(X, i))
            res["generation_time_total"] = time.time() - start
            results.append(res)
        return results

    def explain_dataset(
        self, X: ArrayLike, y: Optional[ArrayLike] = None
    ) -> Dict[str, Any]:
        """
        Orchestrates local explanations over a dataset (test/holdout).
        Applies optional sample limiting and aggregates results.
        """
        start_all = time.time()

        X_np, y_np = self._coerce_X_y(X, y)
        X_np, y_np = self._limit_samples(X_np, y_np)

        if self._log_progress:
            self.logger.info(
                "Running %s explanations on %d instances",
                self.config.get("type", self.__class__.__name__),
                len(X_np),
            )

        explanations = self.explain_batch(X_np)

        total_time = time.time() - start_all
        self.generation_time = total_time

        if self._log_progress:
            self.logger.info(
                "Finished %s explanations in %.2fs",
                self.config.get("type", self.__class__.__name__),
                total_time,
            )

        return {
            "method": self.config.get("type", self.__class__.__name__),
            "explanations": explanations,
            "n_explanations": len(explanations),
            "generation_time": total_time,
            "info": self.get_info(),
        }

    # -----------------------------
    # Capability & info
    # -----------------------------

    def is_compatible(self) -> bool:
        """
        Override to enforce constraints (e.g., requires predict_proba).
        Default: checks model has .predict and data type is tabular.
        """
        has_predict = hasattr(self.model, "predict")
        return bool(has_predict)

    def get_info(self) -> Dict[str, Any]:
        """Basic metadata about the explainer."""
        return {
            "name": self.config.get("name", self.__class__.__name__),
            "type": self.config.get("type", "local"),
            "config": self.config,
            "supported_data_types": getattr(self, "supported_data_types", []),
            "supported_model_types": getattr(self, "supported_model_types", []),
        }

    def empty_result(self, reason: str = "Not compatible or failed.") -> Dict[str, Any]:
        """Standard empty result structure."""
        return {
            "method": self.config.get("type", self.__class__.__name__),
            "explanations": [],
            "n_explanations": 0,
            "generation_time": 0.0,
            "info": {
                "error": reason,
                "supported_data_types": getattr(self, "supported_data_types", []),
                "supported_model_types": getattr(self, "supported_model_types", []),
            },
        }

    # -----------------------------
    # Utilities for tabular data
    # -----------------------------

    def _limit_samples(
        self, X: np.ndarray, y: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Limit samples based on `experiment.explanation.max_instances` (or `max_test_samples` for backward compat).
        """
        max_n = self._expl_cfg.get("max_instances")
        if max_n is None:
            # backward-compat with older config name
            max_n = self._expl_cfg.get("max_test_samples")

        if max_n is not None and len(X) > int(max_n):
            self.logger.info(
                "Limiting dataset from %d to %d instances for explanation generation",
                len(X), int(max_n),
            )
            X = X[: int(max_n)]
            if y is not None:
                y = y[: int(max_n)]
        return X, y

    def _coerce_X_y(
        self, X: ArrayLike, y: Optional[ArrayLike]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Convert X, y to numpy arrays with safe shapes for tabular data.
        """
        X_np = self._to_numpy_2d(X)
        y_np = None
        if y is not None:
            y_np = self._to_numpy_1d(y)
        return X_np, y_np

    def _to_numpy_2d(self, X: ArrayLike) -> np.ndarray:
        if _HAS_PANDAS and isinstance(X, (pd.DataFrame,)):
            return X.values
        if isinstance(X, (list, tuple)):
            X = np.asarray(X)
        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                return X.reshape(1, -1)
            if X.ndim == 2:
                return X
        raise TypeError("X must be a 1D/2D array-like (list, np.ndarray, or pandas DataFrame).")

    def _to_numpy_1d(self, y: ArrayLike) -> np.ndarray:
        if _HAS_PANDAS and isinstance(y, (pd.Series,)):
            return y.values
        if isinstance(y, (list, tuple)):
            y = np.asarray(y)
        if isinstance(y, np.ndarray):
            if y.ndim == 0:
                return y.reshape(1,)
            if y.ndim == 1:
                return y
            if y.ndim == 2 and y.shape[1] == 1:
                return y.ravel()
        raise TypeError("y must be a 1D array-like (list, np.ndarray, or pandas Series).")

    def _row_to_instance(self, X: ArrayLike, idx: int) -> InstanceLike:
        """
        Extract row `idx` from X in a way that preserves original container semantics
        for pandas; otherwise returns 1D numpy array.
        """
        if _HAS_PANDAS and isinstance(X, (pd.DataFrame,)):
            return X.iloc[idx]
        X_np = self._to_numpy_2d(X)
        return X_np[idx, :]

    # -----------------------------
    # Prediction helpers
    # -----------------------------

    def _predict(self, X: ArrayLike) -> np.ndarray:
        """
        Model prediction wrapper, coerces shape and returns 1D or 2D np.ndarray.
        """
        X_np = self._to_numpy_2d(X)
        preds = self.model.predict(X_np)
        return np.asarray(preds)

    def _predict_proba(self, X: ArrayLike) -> Optional[np.ndarray]:
        """
        Optional probability prediction wrapper (classification).
        Returns None if model has no predict_proba.
        """
        if hasattr(self.model, "predict_proba"):
            X_np = self._to_numpy_2d(X)
            try:
                proba = self.model.predict_proba(X_np)
                return np.asarray(proba)
            except Exception as e:
                self.logger.debug("predict_proba failed: %s", e)
                return None
        return None

    def _timeit(self, fn, *args, **kwargs) -> Tuple[Any, float]:
        """
        Utility to time any callable; returns (result, elapsed_seconds).
        """
        t0 = time.time()
        out = fn(*args, **kwargs)
        return out, time.time() - t0

    # -----------------------------
    # Default single-instance scaffold (optional use by subclasses)
    # -----------------------------

    def _standardize_explanation_output(
        self,
        *,
        attributions: Union[np.ndarray, List[float]],
        instance: InstanceLike,
        prediction: Any,
        prediction_proba: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        per_instance_time: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Produce a consistent, schema-like dict for downstream metrics & storage.
        """
        if feature_names is None:
            feature_names = self._infer_feature_names(instance)

        return {
            "method": self.config.get("type", self.__class__.__name__),
            "prediction": prediction,
            "prediction_proba": prediction_proba,
            "attributions": attributions,
            "feature_names": feature_names,
            "metadata": metadata or {},
            "generation_time": per_instance_time,
            "instance": np.asarray(instance).tolist(),
        }

    def _infer_feature_names(self, instance: InstanceLike) -> List[str]:
        """
        Try to get feature names from dataset or pandas; fall back to index list.
        """
        # From dataset object (common pattern: dataset.feature_names)
        names = getattr(self.dataset, "feature_names", None)
        if names is not None:
            return list(names)

        if _HAS_PANDAS and hasattr(instance, "index"):
            try:
                return list(instance.index)
            except Exception:
                pass

        # Fallback to positional names
        instance_array = np.asarray(instance)
        if instance_array.ndim == 0:
            return ["feature_0"]
        if instance_array.ndim == 1:
            return [f"feature_{i}" for i in range(instance_array.shape[0])]
        # 2D edge-case (shouldn't happen for a single instance)
        return [f"feature_{i}" for i in range(instance_array.shape[-1])]
