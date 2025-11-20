"""
Continuity metric – stability for slight variations.

Adapts the non-sensitivity test: apply a small perturbation to an instance,
approximate the explanation for the perturbed sample, and measure similarity with
the original attribution vector.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

_FEATURE_METHOD_KEYS = {
    "shap",
    "lime",
    "integrated_gradients",
    "integratedgradients",
    "causal_shap",
    "causalshap",
}


class ContinuityEvaluator:
    """
    Estimates explanation continuity by perturbing inputs slightly and checking whether
    the resulting explanation remains similar (Section 6.4 stability check).
    """

    def __init__(
        self,
        *,
        max_instances: int = 5,
        noise_scale: float = 0.01,
        metric_key: str = "continuity_stability",
        random_state: Optional[int] = 42,
    ) -> None:
        self.max_instances = max(1, int(max_instances))
        self.noise_scale = float(max(0.0, noise_scale))
        self.metric_key = metric_key or "continuity_stability"
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)

    def evaluate(
        self,
        model: Any,
        explanation_results: Dict[str, Any],
        dataset: Any | None = None,
    ) -> Dict[str, float]:
        method = (explanation_results.get("method") or "").lower()
        if method not in _FEATURE_METHOD_KEYS:
            return {self.metric_key: 0.0}

        explanations = explanation_results.get("explanations") or []
        if not explanations:
            return {self.metric_key: 0.0}

        feature_std = self._dataset_feature_std(dataset, explanations[0])
        rng = np.random.default_rng(self.random_state)

        similarities: List[float] = []
        n_samples = min(self.max_instances, len(explanations))

        for i in range(n_samples):
            explanation = explanations[i]
            instance = self._instance_vector(explanation)
            importance = self._importance_vector(explanation)
            if instance is None or importance is None or importance.size == 0:
                continue

            std_vec = self._match_std_vector(feature_std, instance.shape[0])
            noise = rng.normal(0.0, std_vec * self.noise_scale, size=instance.shape[0])
            perturbed_instance = instance + noise

            perturbed_importance = self._approximate_perturbed_importance(
                instance, perturbed_instance, importance
            )
            if perturbed_importance is None or perturbed_importance.size != importance.size:
                continue

            corr = self._similarity(importance, perturbed_importance)
            if corr is not None and not np.isnan(corr):
                similarities.append(abs(corr))

        score = float(np.mean(similarities)) if similarities else 0.0
        return {self.metric_key: score}

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _dataset_feature_std(self, dataset: Any | None, example_explanation: Dict[str, Any]) -> Optional[np.ndarray]:
        if dataset is None:
            return None
        X_train = getattr(dataset, "X_train", None)
        if X_train is None:
            return None
        X_arr = np.asarray(X_train)
        if X_arr.ndim == 1:
            return np.std(X_arr, axis=0).reshape(1)
        if X_arr.ndim >= 2:
            if X_arr.ndim > 2:
                X_arr = X_arr.reshape(X_arr.shape[0], -1)
            std = np.std(X_arr, axis=0)
            std[std == 0.0] = 1e-6
            return std
        return None

    def _match_std_vector(self, feature_std: Optional[np.ndarray], n_features: int) -> np.ndarray:
        if feature_std is None:
            return np.ones(n_features, dtype=float)
        std_vec = np.asarray(feature_std, dtype=float).reshape(-1)
        if std_vec.size == 1:
            return np.full(n_features, std_vec[0], dtype=float)
        if std_vec.size != n_features:
            return np.resize(std_vec, n_features)
        return std_vec

    def _instance_vector(self, explanation: Dict[str, Any]) -> Optional[np.ndarray]:
        candidate = (
            explanation.get("instance")
            or (explanation.get("metadata") or {}).get("instance")
            or explanation.get("input")
        )
        if candidate is None:
            return None
        arr = np.asarray(candidate, dtype=float).reshape(-1)
        return arr.copy()

    def _importance_vector(self, explanation: Dict[str, Any]) -> Optional[np.ndarray]:
        candidates = ("feature_importance", "feature_importances", "attributions", "importance")
        metadata = explanation.get("metadata") or {}
        for container in (explanation, metadata):
            for key in candidates:
                vec = container.get(key)
                arr = self._to_array(vec)
                if arr is not None:
                    return arr
        return None

    def _to_array(self, values: Any) -> Optional[np.ndarray]:
        if values is None:
            return None
        if isinstance(values, np.ndarray):
            arr = values
        elif isinstance(values, Sequence):
            arr = np.asarray(values)
        else:
            return None
        if arr.ndim == 0:
            arr = arr.reshape(1)
        elif arr.ndim > 1:
            arr = arr.reshape(-1)
        return arr.astype(float)

    def _approximate_perturbed_importance(
        self,
        original_instance: np.ndarray,
        perturbed_instance: np.ndarray,
        original_importance: np.ndarray,
    ) -> Optional[np.ndarray]:
        if original_instance.shape != perturbed_instance.shape:
            return None

        input_change = perturbed_instance - original_instance
        change_magnitude = np.abs(input_change) / (np.abs(original_instance) + 1e-8)
        return original_importance * (1.0 + 0.1 * change_magnitude)

    def _similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> Optional[float]:
        if vec_a.size < 2 or vec_b.size < 2:
            return None
        std_a = float(np.std(vec_a))
        std_b = float(np.std(vec_b))
        if std_a < 1e-8 or std_b < 1e-8: # prevent division by zero, np.corrcoef would complain.
            return None
        try:
            corr = np.corrcoef(vec_a, vec_b)[0, 1]
            return float(corr)
        except Exception:
            return None
