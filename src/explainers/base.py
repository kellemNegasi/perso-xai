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


def _safe_scalar(value: Any) -> Any:
    """Convert numpy scalars/arrays to Python scalars where possible."""
    if value is None:
        return None
    if np.isscalar(value):
        return value.item() if isinstance(value, np.generic) else value
    arr = np.asarray(value)
    if arr.ndim == 0:
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    return arr.tolist()


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
        log_level = self._log_cfg.get("level")
        if log_level:
            numeric_level = getattr(logging, str(log_level).upper(), None)
            if isinstance(numeric_level, int):
                self.logger.setLevel(numeric_level)
        self._log_progress = bool(
            self.config.get("log_progress")
            or self._exp_cfg.get("log_progress")
            or self._log_cfg.get("progress")
        )

        # Seed (if provided) for any stochastic explainers
        self.random_state: Optional[int] = self._expl_cfg.get("random_state")
        self._sampling_info: Dict[str, Any] = {
            "strategy": self._expl_cfg.get("sampling_strategy", "sequential"),
            "max_instances": self._expl_cfg.get("max_instances")
            or self._expl_cfg.get("max_test_samples"),
            "method_cap": self._expl_cfg.get("method_max_instances"),
            "original_size": None,
            "selected_size": None,
            "problem_type": None,
        }
        self._sample_indices: Optional[np.ndarray] = None

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
        explanations = self._augment_metadata(explanations, y_np)

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
            "sampling": self._sampling_info,
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
        Applies strategy-aware sampling to maintain class/target coverage.
        """
        max_n = self._expl_cfg.get("max_instances")
        if max_n is None:
            # backward-compat with older config name
            max_n = self._expl_cfg.get("max_test_samples")

        method_cap = self._expl_cfg.get("method_max_instances")
        if method_cap is not None:
            try:
                cap_value = int(method_cap)
            except (TypeError, ValueError):
                raise ValueError(
                    f"experiment.explanation.method_max_instances must be an integer. Got {method_cap!r}"
                ) from None
            if cap_value <= 0:
                raise ValueError(
                    f"experiment.explanation.method_max_instances must be > 0. Got {cap_value}"
                )
            if max_n is None:
                max_n = cap_value
            else:
                max_n = min(int(max_n), cap_value)

        original_len = len(X)
        self._sampling_info.setdefault("strategy", "sequential")
        self._sampling_info["original_size"] = original_len
        self._sampling_info["selected_size"] = original_len
        self._sampling_info["max_instances"] = int(max_n) if max_n is not None else None
        base_indices = np.arange(original_len, dtype=int)

        if max_n is None or original_len <= int(max_n):
            self._sample_indices = base_indices
            return X, y

        problem_type = self._infer_problem_type(y)
        self._sampling_info["problem_type"] = problem_type

        strategy_cfg = str(self._expl_cfg.get("sampling_strategy", "sequential")).lower()
        if strategy_cfg == "auto":
            if problem_type == "classification":
                strategy = "balanced"
            elif problem_type == "regression":
                strategy = "quantile"
            else:
                strategy = "random"
        else:
            strategy = strategy_cfg
        self._sampling_info["strategy"] = strategy

        indices = self._select_sample_indices(
            n_samples=original_len,
            y=y,
            max_n=int(max_n),
            strategy=strategy,
            problem_type=problem_type,
        )
        if len(indices) != int(max_n):
            # Fallback to sequential slice if strategy produced unexpected count.
            indices = np.arange(int(max_n))

        X = X[indices]
        if y is not None:
            y = y[indices]
        self._sampling_info["selected_size"] = len(X)
        self._sample_indices = indices
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

    def _predict_numeric(self, X: ArrayLike) -> np.ndarray:
        """Return numeric predictions, delegating to model.predict_numeric when available."""
        X_np = self._to_numpy_2d(X)
        predict_numeric = getattr(self.model, "predict_numeric", None)
        if callable(predict_numeric):
            preds = predict_numeric(X_np)
        else:
            preds = self.model.predict(X_np)
        arr = np.asarray(preds)
        if np.issubdtype(arr.dtype, np.number):
            return arr.astype(float)
        try:
            return arr.astype(float)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "Model predictions are not numeric; implement predict_numeric to continue"
            ) from exc

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

    def _infer_problem_type(self, y: Optional[np.ndarray]) -> str:
        """Best-effort detection of classification vs regression."""
        cfg_override = self._expl_cfg.get("problem_type")
        if isinstance(cfg_override, str):
            override = cfg_override.lower()
            if override in {"classification", "regression"}:
                return override

        dataset_task = getattr(self.dataset, "task", None) or getattr(
            self.dataset, "task_type", None
        )
        if isinstance(dataset_task, str):
            lowered = dataset_task.lower()
            if lowered in {"classification", "regression"}:
                return lowered

        estimator_type = getattr(self.model, "_estimator_type", None)
        if estimator_type in {"classifier", "regressor"}:
            return "classification" if estimator_type == "classifier" else "regression"

        if y is None:
            return "unknown"

        y_arr = np.asarray(y)
        if y_arr.dtype.kind in {"U", "S", "O"}:
            return "classification"
        unique_vals = np.unique(y_arr)
        # Heuristic: if many repeated discrete values, treat as classification.
        if y_arr.dtype.kind in {"b", "i", "u"}:
            if len(unique_vals) <= max(15, int(0.1 * len(y_arr))):
                return "classification"
        return "regression"

    def _select_sample_indices(
        self,
        *,
        n_samples: int,
        y: Optional[np.ndarray],
        max_n: int,
        strategy: str,
        problem_type: str,
    ) -> np.ndarray:
        """Return indices according to the requested sampling strategy."""
        def _finalize(selected: np.ndarray) -> np.ndarray:
            selected = np.asarray(selected, dtype=int)
            return self._ensure_min_class_labels(
                selected,
                y=y,
                problem_type=problem_type,
                min_classes=2,
            )

        if max_n >= n_samples:
            return _finalize(np.arange(n_samples, dtype=int))

        stratified_aliases = {"balanced", "stratified", "class_balanced"}
        quantile_aliases = {"quantile", "diverse", "range"}

        if strategy in stratified_aliases and y is not None and problem_type == "classification":
            return _finalize(self._balanced_class_indices(y, max_n))

        if strategy in quantile_aliases and y is not None and problem_type == "regression":
            return _finalize(self._quantile_sample_indices(y, max_n))

        if strategy in {"random", "shuffle"}:
            rng = np.random.default_rng(self.random_state)
            return _finalize(rng.choice(n_samples, size=max_n, replace=False))

        if strategy in {"sequential", "first"}:
            return _finalize(np.arange(max_n, dtype=int))

        # Fallbacks
        if problem_type == "classification" and y is not None:
            return _finalize(self._balanced_class_indices(y, max_n))
        if problem_type == "regression" and y is not None:
            return _finalize(self._quantile_sample_indices(y, max_n))
        return _finalize(np.arange(max_n, dtype=int))

    def _balanced_class_indices(self, y: np.ndarray, max_n: int) -> np.ndarray:
        """Select up to ``max_n`` indices with roughly balanced class coverage."""
        y_arr = np.asarray(y)
        classes, inverse = np.unique(y_arr, return_inverse=True)
        if len(classes) == 0:
            return np.arange(max_n, dtype=int)

        rng = np.random.default_rng(self.random_state)
        per_class = max(1, max_n // len(classes))
        selected: List[int] = []
        leftovers: List[int] = []

        for class_idx, cls in enumerate(classes):
            class_member_indices = np.where(inverse == class_idx)[0]
            if len(class_member_indices) == 0:
                continue
            perm = rng.permutation(class_member_indices)
            take = min(per_class, len(perm))
            selected.extend(perm[:take].tolist())
            if take < len(perm):
                leftovers.extend(perm[take:].tolist())

        if len(selected) < max_n and leftovers:
            rng.shuffle(leftovers)
            needed = max_n - len(selected)
            selected.extend(leftovers[:needed])

        if len(selected) < max_n:
            remaining = np.setdiff1d(
                np.arange(len(y_arr), dtype=int),
                np.asarray(selected, dtype=int),
                assume_unique=False,
            )
            if len(remaining) > 0:
                remaining = np.asarray(remaining)
                rng.shuffle(remaining)
                needed = max_n - len(selected)
                selected.extend(remaining[:needed].tolist())

        if not selected:
            return np.arange(max_n, dtype=int)

        selected_array = np.asarray(selected, dtype=int)
        if len(selected_array) > max_n:
            selected_array = selected_array[:max_n]
        return selected_array

    def _quantile_sample_indices(self, y: np.ndarray, max_n: int) -> np.ndarray:
        """Select indices spread across the target distribution."""
        y_arr = np.asarray(y, dtype=float).ravel()
        if len(y_arr) == 0:
            return np.arange(max_n, dtype=int)
        valid_mask = ~np.isnan(y_arr)
        if not np.any(valid_mask):
            return np.arange(max_n, dtype=int)
        valid_indices = np.where(valid_mask)[0]
        sorted_order = valid_indices[np.argsort(y_arr[valid_mask], kind="mergesort")]
        if max_n >= len(sorted_order):
            return sorted_order
        positions = np.floor(
            np.linspace(0, len(sorted_order), num=max_n, endpoint=False)
        ).astype(int)
        selected = sorted_order[positions]
        if len(selected) < max_n:
            rng = np.random.default_rng(self.random_state)
            remaining = np.setdiff1d(sorted_order, selected, assume_unique=True)
            if len(remaining) > 0:
                rng.shuffle(remaining)
                selected = np.concatenate([selected, remaining[: max_n - len(selected)]])
        return selected.astype(int)

    def _ensure_min_class_labels(
        self,
        indices: np.ndarray,
        *,
        y: Optional[np.ndarray],
        problem_type: str,
        min_classes: int,
    ) -> np.ndarray:
        """
        Ensure the sampled indices span at least ``min_classes`` unique labels when possible.
        """
        if (
            y is None
            or problem_type != "classification"
            or min_classes <= 1
            or len(indices) < min_classes
        ):
            return indices
        y_arr = np.asarray(y)
        all_classes = np.unique(y_arr)
        if len(all_classes) < min_classes:
            return indices
        selected_classes = np.unique(y_arr[indices])
        if len(selected_classes) >= min_classes:
            return indices
        rng = np.random.default_rng(self.random_state)
        target_class = selected_classes[0]
        # Candidates from other classes not already selected.
        other_candidates = np.where(y_arr != target_class)[0]
        if other_candidates.size == 0:
            return indices
        other_candidates = np.setdiff1d(
            other_candidates,
            indices,
            assume_unique=False,
        )
        if other_candidates.size == 0:
            return indices
        replacement = rng.choice(other_candidates)
        selected_labels = y_arr[indices]
        same_class_positions = np.where(selected_labels == target_class)[0]
        if same_class_positions.size == 0:
            return indices
        drop_pos = same_class_positions[rng.integers(same_class_positions.size)]
        indices = np.array(indices, copy=True)
        indices[drop_pos] = int(replacement)
        return indices

    def _augment_metadata(
        self,
        explanations: List[Dict[str, Any]],
        y: Optional[np.ndarray],
    ) -> List[Dict[str, Any]]:
        """
        Inject structured metadata (instance index, true label, sampling info) into
        explanation dictionaries. Subclasses can rely on this being called by
        explain_dataset regardless of how explain_batch is implemented.
        """
        sampling_meta = dict(self._sampling_info) if self._sampling_info else {}
        dataset_indices: Optional[np.ndarray] = None
        if self._sample_indices is not None:
            dataset_indices = np.asarray(self._sample_indices, dtype=int)
        else:
            dataset_indices = np.arange(len(explanations), dtype=int)
        for idx, record in enumerate(explanations):
            metadata = record.setdefault("metadata", {})
            metadata.setdefault("instance_index", idx)
            metadata.setdefault("dataset_index", int(dataset_indices[idx]))
            if y is not None and idx < len(y):
                metadata.setdefault("true_label", _safe_scalar(y[idx]))
            if sampling_meta and "sampling" not in metadata:
                metadata["sampling"] = sampling_meta
        return explanations
