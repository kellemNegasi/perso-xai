"""
Correctness (faithfulness) metric for local tabular explanations.

Implements a feature-removal test: the more the model prediction changes after
masking the most important features (according to the explanation), the more
"correct" the explanation is with respect to the black-box.
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


class CorrectnessEvaluator:
    """
    Computes a correctness score via a feature-removal test.

    Parameters
    ----------
    removal_fraction : float
        Fraction of top-ranked features (by absolute importance) to mask.
    default_baseline : float
        Value used to replace masked features when no baseline vector is provided
        in the explanation metadata (via ``baseline_instance``).
    min_features : int
        Minimum number of features to mask, regardless of ``removal_fraction``.
    """

    def __init__(
        self,
        *,
        removal_fraction: float = 0.1,
        default_baseline: float = 0.0,
        min_features: int = 1,
    ) -> None:
        self.removal_fraction = float(np.clip(removal_fraction, 0.0, 1.0))
        self.default_baseline = float(default_baseline)
        self.min_features = max(1, int(min_features))
        self.logger = logging.getLogger(__name__)

    def evaluate(
        self,
        model: Any,
        explanation_results: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Evaluate correctness for SHAP/LIME/IG/Causal SHAP explanations.

        Parameters
        ----------
        model : Any
            Trained model that produced the explanations (must expose ``predict``).
        explanation_results : Dict[str, Any]
            Output dict from ``BaseExplainer.explain_dataset``.

        Returns
        -------
        Dict[str, float]
            {"correctness": score in [0, 1]} – returns 0.0 if inputs are incompatible.
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
            score = self._feature_removal_score(model, explanation)
            if score is not None:
                scores.append(score)

        correctness = float(np.mean(scores)) if scores else 0.0
        return {"correctness": correctness}

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _feature_removal_score(self, model: Any, explanation: Dict[str, Any]) -> Optional[float]:
        importance_vec = self._feature_importance_vector(explanation)
        if importance_vec is None or importance_vec.size == 0:
            return None

        instance = self._extract_instance(explanation)
        if instance is None:
            return None

        baseline = self._baseline_vector(explanation, instance)
        k = max(self.min_features, int(np.ceil(self.removal_fraction * len(importance_vec))))
        k = min(k, len(importance_vec))
        top_indices = np.argsort(-np.abs(importance_vec))[:k]

        perturbed = instance.copy()
        perturbed[top_indices] = baseline[top_indices]

        orig_pred = self._prediction_value(explanation.get("prediction"))
        if orig_pred is None:
            return None

        try:
            new_pred = self._model_prediction(model, perturbed)
        except Exception as exc:
            self.logger.debug("CorrectnessEvaluator failed to perturb instance: %s", exc)
            return None

        change = abs(orig_pred - new_pred)
        denom = abs(orig_pred) + 1e-8
        return float(np.clip(change / denom, 0.0, 1.0))
    
    def _feature_importance_vector(self, explanation: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Extract feature-importance/attribution vector from a standardized explanation dict.
        Looks in both the root of the explanation and inside metadata so we can support
        multiple explainer schemas.
        """
        candidates = ("feature_importance", "feature_importances", "attributions", "importance")
        metadata = explanation.get("metadata") or {}

        for container in (explanation, metadata):
            for key in candidates:
                vec = container.get(key)
                arr = self._importance_to_array(vec)
                if arr is not None:
                    return arr
        return None
    def _importance_to_array(self, importances: Any) -> Optional[np.ndarray]:
        if importances is None:
            return None
        if isinstance(importances, np.ndarray):
            vec = importances
        elif isinstance(importances, Sequence):
            vec = np.asarray(importances)
        else:
            return None
        if vec.ndim == 0:
            vec = vec.reshape(1)
        elif vec.ndim > 1:
            vec = vec.reshape(-1)
        return vec.astype(float)

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
        return np.full_like(instance, self.default_baseline, dtype=float)

    def _prediction_value(self, prediction: Any) -> Optional[float]:
        if prediction is None:
            return None
        arr = np.asarray(prediction).ravel()
        if arr.size == 0:
            return None
        try:
            return float(arr[0])
        except Exception:
            return None

    def _model_prediction(self, model: Any, instance: np.ndarray) -> float:
        if not hasattr(model, "predict"):
            raise AttributeError("Model must expose a predict() method.")
        batch = instance.reshape(1, -1)
        preds = model.predict(batch)
        preds_arr = np.asarray(preds).ravel()
        if preds_arr.size == 0:
            raise ValueError("Model.predict returned empty output.")
        return float(preds_arr[0])
