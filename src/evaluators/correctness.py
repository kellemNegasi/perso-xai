"""
Correctness metric for local tabular explanations.

The metric is tailored to feature-attribution methods (SHAP/LIME/IG/Causal SHAP)
that return predictions, true labels, and per-feature importance values in the
BaseExplainer output schema.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


_FEATURE_METHOD_KEYS = {
    "shap",
    "lime",
    "integrated_gradients",
    "integratedgradients",
    "causal_shap",
    "causalshap",
}


class CorrectnessEvaluator:
    """Computes a scalar correctness score for local tabular explanations."""

    def __init__(self, *, prediction_weight: float = 0.7) -> None:
        """
        Args:
            prediction_weight: Importance given to model prediction accuracy when
                combining with explanation informativeness. Remaining weight is
                assigned to the informativeness component.
        """
        self.prediction_weight = float(prediction_weight)
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def evaluate(self, explanation_results: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate correctness for SHAP/LIME/IG/Causal SHAP explanations.

        Args:
            explanation_results: Output dict from BaseExplainer.explain_dataset.

        Returns:
            {"correctness": score in [0, 1]} – returns 0.0 if the method or
            payload is incompatible with this metric.
        """
        method = (explanation_results.get("method") or "").lower()
        if method not in _FEATURE_METHOD_KEYS:
            self.logger.info(
                "CorrectnessEvaluator skipped: method '%s' not a feature-attribution explainer",
                method,
            )
            return {"correctness": 0.0}

        explanations = explanation_results.get("explanations") or []
        if not explanations:
            return {"correctness": 0.0}

        scores: List[float] = []
        for explanation in explanations:
            score = self._per_instance_score(explanation)
            if score is not None:
                scores.append(score)

        correctness = float(np.mean(scores)) if scores else 0.0
        return {"correctness": correctness}

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _per_instance_score(self, explanation: Dict[str, Any]) -> float | None:
        """Combine prediction correctness and importance informativeness."""
        prediction = explanation.get("prediction")
        true_label = explanation.get("true_label")
        importances = explanation.get("feature_importance")

        if prediction is None or true_label is None or importances is None:
            return None

        importance_vec = self._importance_to_array(importances)
        if importance_vec is None or importance_vec.size == 0:
            return None

        pred_correctness = 1.0 if self._prediction_matches(prediction, true_label) else 0.0
        informativeness = self._importance_informativeness(importance_vec)

        weight = self.prediction_weight
        return weight * pred_correctness + (1.0 - weight) * informativeness

    def _importance_to_array(self, importances: Any) -> np.ndarray | None:
        """Convert feature importance into a 1-D numpy array."""
        if isinstance(importances, np.ndarray):
            if importances.ndim == 1:
                return importances.astype(float)
            if importances.ndim > 1:
                return importances.reshape(-1).astype(float)
            return importances.astype(float)

        if isinstance(importances, Sequence):
            arr = np.asarray(importances, dtype=float)
            if arr.ndim == 1:
                return arr
            if arr.ndim > 1:
                return arr.reshape(-1)
        return None

    def _prediction_matches(self, prediction: Any, true_label: Any) -> bool:
        """Gracefully compare model prediction and true label."""
        try:
            pred_scalar = float(np.asarray(prediction).ravel()[0])
            true_scalar = float(np.asarray(true_label).ravel()[0])
            return abs(pred_scalar - true_scalar) < 0.5
        except Exception:
            return prediction == true_label

    def _importance_informativeness(self, importance_vec: np.ndarray) -> float:
        """
        Estimate informativeness via normalized variance of absolute importance.
        Returns values in [0, 1]; high variance => concentrated attribution.
        """
        importance_abs = np.abs(importance_vec)
        if importance_abs.size <= 1:
            return 0.0
        if not np.isfinite(importance_abs).all():
            return 0.0

        variance = float(np.var(importance_abs))
        max_variance = float(np.var(np.concatenate(([1.0], np.zeros(importance_abs.size - 1)))))
        if max_variance <= 0:
            return 0.0
        return float(np.clip(variance / max_variance, 0.0, 1.0))
