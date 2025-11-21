"""
Completeness evaluator implementing the deletion-check metric.

Given a feature-attribution explanation, we remove (mask) every feature that the
explanation marks as important and measure how much the model prediction drops.
The score is contrasted against randomly masked feature sets of the same size,
as suggested by the deletion check in the Co-12 survey.
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
}


class CompletenessEvaluator:
    """
    Measure completeness by masking the entire attribution support and comparing
    the resulting prediction drop with random deletions of equal size.
    """

    def __init__(
        self,
        *,
        magnitude_threshold: float = 1e-8,
        min_features: int = 1,
        random_trials: int = 10,
        default_baseline: float = 0.0,
        random_state: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        magnitude_threshold : float, optional
            Absolute attribution magnitude that defines whether a feature belongs
            to the explanation support. If not enough features pass the threshold,
            the highest-magnitude features are used until ``min_features`` is met.
        min_features : int, optional
            Minimum number of features to mask regardless of threshold filtering.
        random_trials : int, optional
            How many random deletion baselines to compute per explanation.
        default_baseline : float, optional
            Fallback value for masked features when the explanation metadata does
            not provide a per-feature ``baseline_instance``.
        random_state : int | None, optional
            Seed for the random baseline sampler.
        """
        self.magnitude_threshold = float(max(0.0, magnitude_threshold))
        self.min_features = max(1, int(min_features))
        self.random_trials = max(0, int(random_trials))
        self.default_baseline = float(default_baseline)
        self._rng = np.random.default_rng(random_state)
        self.logger = logging.getLogger(__name__)

    def evaluate(
        self,
        model: Any,
        explanation_results: Dict[str, Any],
        dataset: Any | None = None,  # unused, kept for evaluator API symmetry
        explainer: Any | None = None,  # unused placeholder
    ) -> Dict[str, float]:
        """
        Compute deletion-check completeness metrics for a batch of explanations.
        """
        method = (explanation_results.get("method") or "").lower()
        if method not in _FEATURE_METHOD_KEYS:
            return self._empty_result()

        explanations = explanation_results.get("explanations") or []
        if not explanations:
            return self._empty_result()

        drops: List[float] = []
        baseline_drops: List[float] = []
        scores: List[float] = []

        for explanation in explanations:
            importance = self._importance_vector(explanation)
            if importance is None:
                continue

            instance = self._extract_instance(explanation)
            if instance is None:
                continue

            baseline = self._baseline_vector(explanation, instance)
            orig_pred = self._prediction_value(explanation)
            if orig_pred is None:
                continue

            mask_indices = self._support_indices(importance)
            if mask_indices.size == 0:
                continue

            target_drop = self._normalized_drop(
                model, instance, baseline, mask_indices, orig_pred
            )
            if target_drop is None:
                continue
            drops.append(target_drop)

            random_values = self._random_baseline_drops(
                model,
                instance,
                baseline,
                len(importance),
                len(mask_indices),
                orig_pred,
            )
            random_mean = float(np.mean(random_values)) if random_values else 0.0
            baseline_drops.append(random_mean)

            score = max(0.0, target_drop - random_mean)
            scores.append(score)

        if not drops:
            return self._empty_result()

        return {
            "completeness_drop": float(np.mean(drops)),
            "completeness_random_drop": float(np.mean(baseline_drops)) if baseline_drops else 0.0,
            "completeness_score": float(np.mean(scores)),
        }

    # ------------------------------------------------------------------ #
    # Helper utilities                                                   #
    # ------------------------------------------------------------------ #

    def _empty_result(self) -> Dict[str, float]:
        return {
            "completeness_drop": 0.0,
            "completeness_random_drop": 0.0,
            "completeness_score": 0.0,
        }

    def _importance_vector(self, explanation: Dict[str, Any]) -> Optional[np.ndarray]:
        candidates = ("feature_importance", "feature_importances", "attributions", "importance")
        metadata = explanation.get("metadata") or {}

        for container in (explanation, metadata):
            for key in candidates:
                vec = container.get(key)
                arr = self._to_array(vec)
                if arr is not None:
                    return arr
        self.logger.debug("CompletenessEvaluator missing attribution vector.")
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

    def _extract_instance(self, explanation: Dict[str, Any]) -> Optional[np.ndarray]:
        candidate = (
            explanation.get("instance")
            or (explanation.get("metadata") or {}).get("instance")
            or explanation.get("input")
        )
        if candidate is None:
            return None
        arr = np.asarray(candidate, dtype=float).reshape(-1)
        return arr.copy()

    def _baseline_vector(self, explanation: Dict[str, Any], instance: np.ndarray) -> np.ndarray:
        metadata = explanation.get("metadata") or {}
        baseline = metadata.get("baseline_instance")
        if baseline is not None:
            base_arr = np.asarray(baseline, dtype=float).reshape(-1)
            if base_arr.shape == instance.shape:
                return base_arr
        self.logger.debug(
            "CompletenessEvaluator using default baseline %.5f", self.default_baseline
        )
        return np.full_like(instance, self.default_baseline, dtype=float)

    def _prediction_value(self, explanation: Dict[str, Any]) -> Optional[float]:
        proba = explanation.get("prediction_proba")
        if proba is not None:
            proba_arr = np.asarray(proba).ravel()
            if proba_arr.size == 0:
                return None
            if proba_arr.size == 2:
                return float(proba_arr[1])
            return float(proba_arr.max())

        prediction = explanation.get("prediction")
        if prediction is None:
            return None
        arr = np.asarray(prediction).ravel()
        if arr.size == 0:
            return None
        try:
            return float(arr[0])
        except Exception:
            return None

    def _support_indices(self, importance: np.ndarray) -> np.ndarray:
        magnitudes = np.abs(importance)
        mask = magnitudes >= self.magnitude_threshold
        indices = np.flatnonzero(mask)
        if indices.size >= self.min_features:
            return indices

        order = np.argsort(-magnitudes)
        needed = max(self.min_features, 1)
        return order[: min(needed, magnitudes.size)]

    def _normalized_drop(
        self,
        model: Any,
        instance: np.ndarray,
        baseline: np.ndarray,
        indices: np.ndarray,
        original_pred: float,
    ) -> Optional[float]:
        if indices.size == 0:
            return None
        perturbed = instance.copy()
        perturbed[indices] = baseline[indices]
        try:
            new_pred = self._model_prediction(model, perturbed)
        except Exception as exc:
            self.logger.debug("CompletenessEvaluator failed to evaluate perturbed instance: %s", exc)
            return None

        denom = abs(original_pred) + 1e-8
        if denom < 1e-12:
            return None
        drop = abs(original_pred - new_pred) / denom
        if np.isnan(drop) or np.isinf(drop):
            return None
        return float(np.clip(drop, 0.0, 1.0))

    def _random_baseline_drops(
        self,
        model: Any,
        instance: np.ndarray,
        baseline: np.ndarray,
        n_features: int,
        mask_size: int,
        original_pred: float,
    ) -> List[float]:
        if self.random_trials <= 0 or mask_size <= 0 or mask_size > n_features:
            return []

        drops: List[float] = []
        for _ in range(self.random_trials):
            indices = self._rng.choice(n_features, size=mask_size, replace=False)
            drop = self._normalized_drop(model, instance, baseline, indices, original_pred)
            if drop is not None:
                drops.append(drop)
        return drops

    def _model_prediction(self, model: Any, instance: np.ndarray) -> float:
        batch = instance.reshape(1, -1)
        if hasattr(model, "predict_proba"):
            proba = np.asarray(model.predict_proba(batch)).ravel()
            if proba.size == 0:
                raise ValueError("Model.predict_proba returned empty output.")
            if proba.size == 2:
                return float(proba[1])
            return float(proba.max())

        if not hasattr(model, "predict"):
            raise AttributeError("Model must expose a predict() method.")

        preds = np.asarray(model.predict(batch)).ravel()
        if preds.size == 0:
            raise ValueError("Model.predict returned empty output.")
        return float(preds[0])

