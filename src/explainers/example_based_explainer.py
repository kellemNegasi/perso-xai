"""
Example-based explainers for tabular data (prototype and counterfactual).
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .base import ArrayLike, BaseExplainer, InstanceLike


class _BaseExampleExplainer(BaseExplainer):
    """Shared utilities for example-based explainers."""

    supported_data_types = ["tabular"]
    supported_model_types = ["sklearn", "xgboost", "lightgbm", "catboost", "generic-predict"]

    def __init__(self, config: Dict[str, Any], model: Any, dataset: Any):
        super().__init__(config=config, model=model, dataset=dataset)
        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._train_predictions: Optional[np.ndarray] = None

        ds_X = getattr(dataset, "X_train", None)
        ds_y = getattr(dataset, "y_train", None)
        if ds_X is not None:
            self._cache_training_data(ds_X, ds_y)

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> None:
        self._cache_training_data(X, y)

    def is_compatible(self) -> bool:
        if not super().is_compatible():
            return False
        if self._X_train is None or len(self._X_train) == 0:
            self.logger.warning("Example-based explainer requires training data.")
            return False
        return True

    def _cache_training_data(self, X: ArrayLike, y: Optional[ArrayLike]) -> None:
        if X is None:
            self._X_train = None
            self._y_train = None
            self._train_predictions = None
            return
        X_np, y_np = self._coerce_X_y(X, y)
        self._X_train = X_np
        self._y_train = y_np
        self._train_predictions = None

    def _ensure_training_predictions(self) -> np.ndarray:
        if self._train_predictions is None:
            if self._X_train is None:
                raise ValueError("Training data unavailable for example-based explainer.")
            preds = np.asarray(self._predict(self._X_train))
            if preds.ndim > 1 and preds.shape[1] == 1:
                preds = preds.ravel()
            self._train_predictions = preds
        return self._train_predictions

    def _reference_labels(self) -> Optional[np.ndarray]:
        if self._y_train is not None:
            return self._y_train
        try:
            return self._ensure_training_predictions()
        except Exception:
            return None

    def _prepare_instance(self, instance: InstanceLike) -> np.ndarray:
        inst2d = self._to_numpy_2d(instance)
        return inst2d[0]

    def _prediction_value(self, predictions: np.ndarray, index: int = 0) -> Any:
        arr = np.asarray(predictions)
        if arr.ndim == 0:
            return _to_scalar(arr)
        row = arr[index]
        if isinstance(row, np.ndarray):
            row = row.ravel()[0]
        return _to_scalar(row)

    def _reference_prediction(self, reference: np.ndarray) -> Any:
        try:
            ref_pred = self._predict(reference.reshape(1, -1))
            return self._prediction_value(ref_pred, 0)
        except Exception:
            return None

    def _find_reference(
        self,
        instance: np.ndarray,
        target_label: Any,
        *,
        same_class: bool,
    ) -> Optional[Tuple[np.ndarray, float, Any]]:
        if self._X_train is None:
            return None
        labels = self._reference_labels()
        if labels is None:
            return None

        candidate_mask = labels == target_label if same_class else labels != target_label
        if not np.any(candidate_mask):
            return None

        candidates = self._X_train[candidate_mask]
        diffs = candidates - instance
        dists = np.linalg.norm(diffs, axis=1)
        best_idx = int(np.argmin(dists))
        ref = candidates[best_idx]
        distance = float(dists[best_idx])

        label_subset = np.asarray(labels)[candidate_mask]
        ref_label = label_subset[best_idx]
        return ref, distance, ref_label


class PrototypeExplainer(_BaseExampleExplainer):
    """Finds nearest prototype of the predicted class and reports differences."""

    def explain_instance(self, instance: InstanceLike) -> Dict[str, Any]:
        inst_vec = self._prepare_instance(instance)
        if self._X_train is None:
            return self._standardize_explanation_output(
                attributions=np.zeros_like(inst_vec).tolist(),
                instance=inst_vec,
                prediction=float("nan"),
                metadata={"error": "No training data available."},
            )

        inst2d = inst_vec.reshape(1, -1)
        prediction, t_pred = self._timeit(self._predict, inst2d)
        prediction_proba = self._predict_proba(inst2d)
        proba_value = prediction_proba[0] if prediction_proba is not None else None
        pred_value = self._prediction_value(prediction)

        search_start = time.time()
        reference = self._find_reference(inst_vec, pred_value, same_class=True)
        search_time = time.time() - search_start

        if reference is None:
            metadata = {"reference_type": "prototype", "warning": "No reference in same class."}
            return self._standardize_explanation_output(
                attributions=np.zeros_like(inst_vec).tolist(),
                instance=inst_vec,
                prediction=pred_value,
                prediction_proba=proba_value,
                metadata=metadata,
                per_instance_time=t_pred + search_time,
            )

        ref_instance, distance, ref_label = reference
        attributions = np.abs(inst_vec - ref_instance)
        ref_prediction = self._reference_prediction(ref_instance)

        metadata = {
            "reference_type": "prototype",
            "reference_instance": ref_instance.tolist(),
            "reference_distance": distance,
            "reference_label": _to_scalar(ref_label),
            "reference_prediction": ref_prediction,
        }

        return self._standardize_explanation_output(
            attributions=attributions.tolist(),
            instance=inst_vec,
            prediction=pred_value,
            prediction_proba=proba_value,
            metadata=metadata,
            per_instance_time=t_pred + search_time,
        )


class CounterfactualExplainer(_BaseExampleExplainer):
    """Finds closest training example with a different predicted label."""

    def explain_instance(self, instance: InstanceLike) -> Dict[str, Any]:
        inst_vec = self._prepare_instance(instance)
        if self._X_train is None:
            return self._standardize_explanation_output(
                attributions=np.zeros_like(inst_vec).tolist(),
                instance=inst_vec,
                prediction=float("nan"),
                metadata={"error": "No training data available."},
            )

        inst2d = inst_vec.reshape(1, -1)
        prediction, t_pred = self._timeit(self._predict, inst2d)
        prediction_proba = self._predict_proba(inst2d)
        proba_value = prediction_proba[0] if prediction_proba is not None else None
        pred_value = self._prediction_value(prediction)

        search_start = time.time()
        reference = self._find_reference(inst_vec, pred_value, same_class=False)
        search_time = time.time() - search_start

        if reference is None:
            metadata = {"reference_type": "counterfactual", "warning": "No opposite class found."}
            return self._standardize_explanation_output(
                attributions=np.zeros_like(inst_vec).tolist(),
                instance=inst_vec,
                prediction=pred_value,
                prediction_proba=proba_value,
                metadata=metadata,
                per_instance_time=t_pred + search_time,
            )

        ref_instance, distance, ref_label = reference
        attributions = np.abs(inst_vec - ref_instance)
        ref_prediction = self._reference_prediction(ref_instance)

        metadata = {
            "reference_type": "counterfactual",
            "reference_instance": ref_instance.tolist(),
            "reference_distance": distance,
            "reference_label": _to_scalar(ref_label),
            "reference_prediction": ref_prediction,
        }

        return self._standardize_explanation_output(
            attributions=attributions.tolist(),
            instance=inst_vec,
            prediction=pred_value,
            prediction_proba=proba_value,
            metadata=metadata,
            per_instance_time=t_pred + search_time,
        )


def _to_scalar(value: Any) -> Any:
    arr = np.asarray(value)
    if arr.ndim == 0:
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    return value
