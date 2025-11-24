"""
Non-sensitivity evaluator mirrored after Nguyen & Rodriguez Martinez (2020).

Checks whether features assigned near-zero attribution truly have negligible
influence by perturbing them and measuring the resulting change in model output.
Adapted from the Quantus `NonSensitivity` metric (https://github.com/understandable-machine-intelligence-lab/Quantus).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .base_metric import MetricCapabilities, MetricInput

_FEATURE_METHOD_KEYS = {
    "shap",
    "lime",
    "integrated_gradients",
    "integratedgradients",
    "causal_shap",
    "causalshap",
}


class NonSensitivityEvaluator(MetricCapabilities):
    """
    Flag situations where a feature receives (near) zero attribution yet materially
    changes the prediction when perturbed.

    Metrics
    -------
    non_sensitivity_violation_fraction
        Fraction of zero-attribution features that caused prediction changes larger
        than ``delta_tolerance`` when perturbed (lower is better).
    non_sensitivity_safe_fraction
        Complementary fraction of zero-attribution features whose perturbations
        stayed below the tolerance threshold.
    non_sensitivity_zero_features
        Average count of features flagged as zero attribution per explanation.
    non_sensitivity_delta_mean
        Mean absolute prediction change observed while probing zero-attribution
        features, indicating violation severity.

    Parameters
    ----------
    zero_threshold : float, optional
        Absolute attribution magnitude treated as zero importance.
    delta_tolerance : float, optional
        Maximum prediction delta tolerated when perturbing zero-attribution features.
        Deltas above this tolerance are counted as violations.
    features_per_step : int, optional
        Number of zero-attribution features to perturb simultaneously. When
        greater than one, all features in the group inherit the same verdict.
    default_baseline : float, optional
        Value used to fill masked features when explanations do not provide a
        ``baseline_instance`` in the metadata.
    """

    per_instance = True
    requires_full_batch = False
    metric_names = (
        "non_sensitivity_violation_fraction",
        "non_sensitivity_safe_fraction",
        "non_sensitivity_zero_features",
        "non_sensitivity_delta_mean",
    )
    supported_methods = tuple(_FEATURE_METHOD_KEYS)

    def __init__(
        self,
        *,
        zero_threshold: float = 1e-5,
        delta_tolerance: float = 1e-4,
        features_per_step: int = 1,
        default_baseline: float = 0.0,
    ) -> None:
        self.zero_threshold = float(max(0.0, zero_threshold))
        self.delta_tolerance = float(max(0.0, delta_tolerance))
        self.features_per_step = max(1, int(features_per_step))
        self.default_baseline = float(default_baseline)
        self.logger = logging.getLogger(__name__)

    def evaluate(
        self,
        model: Any,
        explanation_results: Dict[str, Any],
        dataset: Any | None = None,
        explainer: Any | None = None,
    ) -> Dict[str, float]:
        metric_input = MetricInput.from_results(
            model=model,
            explanation_results=explanation_results,
            dataset=dataset,
            explainer=explainer,
        )
        return self._evaluate(metric_input)

    def _evaluate(self, metric_input: MetricInput) -> Dict[str, float]:
        if metric_input.method not in self.supported_methods:
            return self._empty_result()

        explanations = metric_input.explanations
        if not explanations:
            return self._empty_result()

        if metric_input.explanation_idx is not None:
            idx = metric_input.explanation_idx
            if not (0 <= idx < len(explanations)):
                return self._empty_result()
            metrics = self._metrics_for_explanation(metric_input.model, explanations[idx])
            return metrics or self._empty_result()

        violation_rates: List[float] = []
        safe_rates: List[float] = []
        zero_counts: List[float] = []
        delta_means: List[float] = []

        for explanation in explanations:
            metrics = self._metrics_for_explanation(metric_input.model, explanation)
            if not metrics:
                continue
            violation_rates.append(metrics["non_sensitivity_violation_fraction"])
            safe_rates.append(metrics["non_sensitivity_safe_fraction"])
            zero_counts.append(metrics["non_sensitivity_zero_features"])
            delta_means.append(metrics["non_sensitivity_delta_mean"])

        if not violation_rates:
            return self._empty_result()

        return {
            "non_sensitivity_violation_fraction": float(np.mean(violation_rates)),
            "non_sensitivity_safe_fraction": float(np.mean(safe_rates)),
            "non_sensitivity_zero_features": float(np.mean(zero_counts)),
            "non_sensitivity_delta_mean": float(np.mean(delta_means)),
        }

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _empty_result(self) -> Dict[str, float]:
        return {key: 0.0 for key in self.metric_names}

    def _metrics_for_explanation(self, model: Any, explanation: Dict[str, Any]) -> Optional[Dict[str, float]]:
        importance = self._importance_vector(explanation)
        if importance is None or importance.size == 0:
            return None

        zero_indices = np.flatnonzero(np.abs(importance) <= self.zero_threshold)
        if zero_indices.size == 0:
            return {
                "non_sensitivity_violation_fraction": 0.0,
                "non_sensitivity_safe_fraction": 0.0,
                "non_sensitivity_zero_features": 0.0,
                "non_sensitivity_delta_mean": 0.0,
            }

        instance = self._extract_instance(explanation)
        if instance is None:
            return None

        baseline = self._baseline_vector(explanation, instance)
        original_pred = self._prediction_value(explanation)
        if original_pred is None:
            return None

        deltas: List[float] = []
        violations = 0
        safe = 0

        for group in self._chunk_indices(zero_indices):
            perturbed = instance.copy()
            perturbed[group] = baseline[group]
            try:
                perturbed_pred = self._model_prediction(model, perturbed)
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.debug("NonSensitivityEvaluator failed to score perturbation: %s", exc)
                continue
            delta = float(abs(original_pred - perturbed_pred))
            deltas.append(delta)
            if delta > self.delta_tolerance:
                violations += len(group)
            else:
                safe += len(group)

        total = violations + safe
        if total == 0:
            return {
                "non_sensitivity_violation_fraction": 0.0,
                "non_sensitivity_safe_fraction": 0.0,
                "non_sensitivity_zero_features": float(zero_indices.size),
                "non_sensitivity_delta_mean": 0.0,
            }

        violation_fraction = float(violations / total)
        safe_fraction = float(safe / total)
        avg_delta = float(np.mean(deltas)) if deltas else 0.0

        return {
            "non_sensitivity_violation_fraction": violation_fraction,
            "non_sensitivity_safe_fraction": safe_fraction,
            "non_sensitivity_zero_features": float(zero_indices.size),
            "non_sensitivity_delta_mean": avg_delta,
        }

    def _chunk_indices(self, indices: np.ndarray) -> List[np.ndarray]:
        groups: List[np.ndarray] = []
        step = self.features_per_step
        for start in range(0, len(indices), step):
            groups.append(indices[start : start + step])
        return groups

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
            "NonSensitivityEvaluator using default baseline %.5f for instance",
            self.default_baseline,
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
