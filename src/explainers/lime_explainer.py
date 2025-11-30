"""
LIME explainer specialized for local explanations on tabular data.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
from sklearn.linear_model import Ridge

from .base import ArrayLike, BaseExplainer, InstanceLike


class LIMEExplainer(BaseExplainer):
    """Local Interpretable Model-agnostic Explanations (tabular-only)."""

    supported_data_types = ["tabular"]
    supported_model_types = [
        "sklearn",
        "xgboost",
        "lightgbm",
        "catboost",
        "generic-predict",
    ]

    def __init__(self, config: Dict[str, Any], model: Any, dataset: Any):
        super().__init__(config=config, model=model, dataset=dataset)
        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._train_mean: Optional[np.ndarray] = None
        self._train_std: Optional[np.ndarray] = None
        self._rng = np.random.default_rng(self.random_state)

        ds_X = getattr(self.dataset, "X_train", None)
        ds_y = getattr(self.dataset, "y_train", None)
        if ds_X is not None:
            self._cache_training_stats(ds_X, ds_y)

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> None:
        """Cache training arrays/statistics for later perturbations."""
        self._cache_training_stats(X, y)

    def explain_instance(self, instance: InstanceLike) -> Dict[str, Any]:
        """Generate a local linear surrogate around a single instance."""
        inst2d = self._to_numpy_2d(instance)
        inst_vec = inst2d[0]
        self._ensure_training_cache(inst_vec)

        (attributions, info), t_lime = self._timeit(
            self._generate_local_explanation, inst_vec
        )
        prediction, t_pred = self._timeit(self._predict_numeric, inst2d)
        prediction_proba = self._predict_proba(inst2d)

        baseline_prediction = None
        if self._train_mean is not None:
            baseline_prediction = float(
                np.asarray(self._predict_numeric(self._train_mean.reshape(1, -1))).ravel()[0]
            )

        metadata = {
            "baseline_prediction": baseline_prediction,
            "num_samples": info["num_samples"],
            "kernel_width": info["kernel_width"],
            "noise_scale": info["noise_scale"],
        }

        pred_array = np.asarray(prediction)
        raw_pred = pred_array[0] if pred_array.ndim > 0 else pred_array
        pred_value = float(raw_pred)

        proba_value = None
        if prediction_proba is not None:
            proba_value = np.asarray(prediction_proba)[0]

        return self._standardize_explanation_output(
            attributions=attributions.tolist(),
            instance=inst_vec,
            prediction=pred_value,
            prediction_proba=proba_value,
            metadata=metadata,
            per_instance_time=t_lime + t_pred,
        )
    def explain_batch(self, X: ArrayLike) -> List[Dict[str, Any]]:
        """
        Runs LIME over a batch by sharing preprocessing/prediction while still
        generating per-instance explanations.
        """
        X_np, _ = self._coerce_X_y(X, None)

        if len(X_np) == 0:
            return []

        batch_start = time.time()
        preds = np.asarray(self._predict_numeric(X_np))
        proba = self._predict_proba(X_np)
        baseline_prediction = None
        if self._train_mean is not None:
            baseline_prediction = float(
                np.asarray(self._predict_numeric(self._train_mean.reshape(1, -1))).ravel()[0]
            )
        results: List[Dict[str, Any]] = []
        for idx, inst_vec in enumerate(X_np):
            self._ensure_training_cache(inst_vec)
            attributions, info = self._generate_local_explanation(inst_vec)

            pred_row = np.asarray(preds[idx]).ravel()
            base_value = pred_row[0] if pred_row.size else pred_row
            pred_value = float(base_value)

            proba_value = None
            if proba is not None:
                proba_value = np.asarray(proba[idx])

            metadata = {
                "baseline_prediction": baseline_prediction,
                "num_samples": info["num_samples"],
                "kernel_width": info["kernel_width"],
                "noise_scale": info["noise_scale"],
            }

            results.append(
                self._standardize_explanation_output(
                    attributions=attributions.tolist(),
                    instance=inst_vec,
                    prediction=pred_value,
                    prediction_proba=proba_value,
                    metadata=metadata,
                    per_instance_time=0.0,
                )
            )
        total_time = time.time() - batch_start
        avg_time = total_time / len(results) if results else 0.0
        for record in results:
            record["generation_time"] = avg_time
        return results

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _cache_training_stats(
        self, X: ArrayLike, y: Optional[ArrayLike]
    ) -> None:
        """Store training arrays plus mean/std statistics."""
        if X is None:
            return
        X_np, y_np = self._coerce_X_y(X, y)
        self._X_train = X_np
        self._y_train = y_np
        self._train_mean = np.mean(X_np, axis=0)
        std = np.std(X_np, axis=0)
        std[std == 0.0] = 1e-6
        self._train_std = std

    def _ensure_training_cache(self, fallback_instance: np.ndarray) -> None:
        """Guarantee that perturbation stats exist, even if fit() was skipped."""
        if self._X_train is not None:
            return
        self._cache_training_stats(fallback_instance.reshape(1, -1), None)

    def _generate_local_explanation(
        self, instance: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Perturb the instance, fit weighted linear model, return coefficients."""
        n_features = instance.shape[0]
        n_samples = int(self._expl_cfg.get("lime_num_samples", 100))
        kernel_width = float(
            self._expl_cfg.get("lime_kernel_width", np.sqrt(n_features) * 0.75)
        )
        noise_scale = float(self._expl_cfg.get("lime_noise_scale", 0.1))
        alpha = float(self._expl_cfg.get("lime_alpha", 1e-2))

        std = (
            self._train_std
            if self._train_std is not None
            else np.ones_like(instance)
        )
        perturbations = instance + self._rng.normal(
            0.0, std * noise_scale, size=(n_samples, n_features)
        )
        perturbations = np.vstack([instance, perturbations])

        preds = np.asarray(self._predict_numeric(perturbations))
        target = self._local_target_vector(perturbations, preds)

        distances = np.linalg.norm(perturbations - instance, axis=1)
        weights = np.exp(-(distances ** 2) / (kernel_width ** 2 + 1e-12))
        weights[0] = weights.max()

        linear_model = Ridge(alpha=alpha)
        linear_model.fit(perturbations, target, sample_weight=weights)
        importance = np.abs(linear_model.coef_)

        info = {
            "num_samples": n_samples,
            "kernel_width": kernel_width,
            "noise_scale": noise_scale,
        }
        return importance, info

    def _local_target_vector(self, perturbations: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        """Return numeric targets for the local surrogate regression."""
        proba = self._predict_proba(perturbations)
        if proba is not None:
            proba_arr = np.asarray(proba)
            if proba_arr.ndim == 1:
                return proba_arr.reshape(-1)
            if proba_arr.ndim == 2:
                if proba_arr.shape[1] == 1:
                    return proba_arr.ravel()
                if proba_arr.shape[1] == 2:
                    return proba_arr[:, 1]
                idx = self._prediction_indices(predictions)
                rows = np.arange(len(idx))
                idx = np.clip(idx, 0, proba_arr.shape[1] - 1)
                return proba_arr[rows, idx]
            flat = proba_arr.reshape(proba_arr.shape[0], -1)
            return flat[:, 0]
        return self._encode_prediction_labels(predictions)

    def _prediction_indices(self, predictions: np.ndarray) -> np.ndarray:
        preds = np.asarray(predictions)
        if preds.ndim > 1 and preds.shape[1] == 1:
            preds = preds.ravel()
        classes = getattr(self.model, "classes_", None)
        if classes is not None:
            mapping = {cls: idx for idx, cls in enumerate(list(classes))}
            return np.array([mapping.get(val, 0) for val in preds], dtype=int)
        if np.issubdtype(preds.dtype, np.number):
            return preds.astype(int).reshape(-1)
        _, inverse = np.unique(preds, return_inverse=True)
        return inverse.astype(int)

    def _encode_prediction_labels(self, predictions: np.ndarray) -> np.ndarray:
        preds = np.asarray(predictions)
        if preds.ndim > 1 and preds.shape[1] == 1:
            preds = preds.ravel()
        if np.issubdtype(preds.dtype, np.number):
            return preds.astype(float)
        indices = self._prediction_indices(predictions)
        return indices.astype(float)
