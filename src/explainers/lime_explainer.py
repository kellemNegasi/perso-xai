"""
LIME explainer specialized for local explanations on tabular data.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

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
        prediction, t_pred = self._timeit(self._predict, inst2d)
        prediction_proba = self._predict_proba(inst2d)

        baseline_prediction = None
        if self._train_mean is not None:
            baseline_prediction = float(
                np.asarray(self._predict(self._train_mean.reshape(1, -1))).ravel()[0]
            )

        metadata = {
            "baseline_prediction": baseline_prediction,
            "num_samples": info["num_samples"],
            "kernel_width": info["kernel_width"],
            "noise_scale": info["noise_scale"],
        }

        pred_array = np.asarray(prediction)
        pred_value = pred_array[0] if pred_array.ndim > 0 else float(pred_array)

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
        n_samples = int(self._expl_cfg.get("lime_num_samples", 500))
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
        rng = np.random.default_rng(self.random_state)
        perturbations = instance + rng.normal(
            0.0, std * noise_scale, size=(n_samples, n_features)
        )
        perturbations = np.vstack([instance, perturbations])

        preds = np.asarray(self._predict(perturbations))
        if preds.ndim == 1:
            target = preds
        elif preds.ndim == 2 and preds.shape[1] == 1:
            target = preds.ravel()
        else:
            # Multi-output fallback: explain first column
            target = preds[:, 0]

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
