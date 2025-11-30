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

from .base_metric import MetricCapabilities, MetricInput

_FEATURE_METHOD_KEYS = {
    "shap",
    "lime",
    "integrated_gradients",
    "integratedgradients",
    "causal_shap",
}


class CompletenessEvaluator(MetricCapabilities):
    """
    Measure completeness by masking the entire attribution support and comparing
    the resulting prediction drop with random deletions of equal size.

    Parameters
    ----------
    magnitude_threshold : float, optional
        Absolute attribution magnitude threshold defining the support.
    min_features : int, optional
        Minimum number of features to mask regardless of threshold.
    random_trials : int, optional
        Number of random deletion baselines to compute per explanation.
    default_baseline : float, optional
        Value used to fill masked features when no explainer baseline is given.
    random_state : int | None, optional
        Seed for the random baseline sampler.
    """

    def __init__(
        self,
        *,
        magnitude_threshold: float = 1e-8,
        min_features: int = 1,
        random_trials: int = 5,
        default_baseline: float = 0.0,
        fast_mode: bool = True,
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
        self.fast_mode = bool(fast_mode)
        self._rng = np.random.default_rng(random_state)
        self.logger = logging.getLogger(__name__)

    def evaluate(
        self,
        model: Any,
        explanation_results: Dict[str, Any],
        dataset: Any | None = None,
        explainer: Any | None = None,
    ) -> Dict[str, float]:
        """
        Compute deletion-check completeness metrics for a batch of explanations.

        Parameters
        ----------
        model : Any
            Trained model that produced the explanations.
        explanation_results : Dict[str, Any]
            Output dict from ``BaseExplainer.explain_dataset``.
        dataset : Any | None, optional
            Dataset reference (unused placeholder).
        explainer : Any | None, optional
            Explainer instance (unused placeholder).

        Returns
        -------
        Dict[str, float]
            Averaged completeness metrics (drop, random drop, score).
        """
        metric_input = MetricInput.from_results(
            model=model,
            explanation_results=explanation_results,
            dataset=dataset,
            explainer=explainer,
        )
        return self._evaluate(metric_input)

    def _evaluate(self, metric_input: MetricInput) -> Dict[str, float]:
        """
        Internal helper operating directly on MetricInput.

        Parameters
        ----------
        metric_input : MetricInput
            Standardized evaluator payload.

        Returns
        -------
        Dict[str, float]
            Aggregated completeness metrics or zeros if inputs are invalid.
        """
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

        drops: List[float] = []
        baseline_drops: List[float] = []
        scores: List[float] = []

        for explanation in explanations:
            metrics = self._metrics_for_explanation(metric_input.model, explanation)
            if not metrics:
                continue
            drops.append(metrics["completeness_drop"])
            baseline_drops.append(metrics["completeness_random_drop"])
            scores.append(metrics["completeness_score"])

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
        return {key: 0.0 for key in self.metric_names}

    def _metrics_for_explanation(self, model: Any, explanation: Dict[str, Any]) -> Optional[Dict[str, float]]:
        if self.fast_mode:
            metrics = self._fast_metrics_for_explanation(explanation)
            if metrics is not None:
                return metrics
            # fall back to exact computation if heuristic failed

        importance = self._importance_vector(explanation)
        if importance is None:
            return None

        instance = self._extract_instance(explanation)
        if instance is None:
            return None

        baseline = self._baseline_vector(explanation, instance)
        orig_pred = self._prediction_value(explanation)
        if orig_pred is None:
            return None

        mask_indices = self._support_indices(importance)
        if mask_indices.size == 0:
            return None

        target_drop = self._normalized_drop(
            model, instance, baseline, mask_indices, orig_pred
        )
        if target_drop is None:
            return None

        random_values = self._random_baseline_drops(
            model,
            instance,
            baseline,
            len(importance),
            len(mask_indices),
            orig_pred,
        )
        random_mean = float(np.mean(random_values)) if random_values else 0.0
        score = max(0.0, target_drop - random_mean)
        return {
            "completeness_drop": target_drop,
            "completeness_random_drop": random_mean,
            "completeness_score": score,
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

        trials = self.random_trials
        batch = np.repeat(instance[np.newaxis, :], trials, axis=0)
        for row in range(trials):
            indices = self._rng.permutation(n_features)[:mask_size]
            batch[row, indices] = baseline[indices]

        try:
            preds = self._model_predictions(model, batch)
        except Exception as exc:
            self.logger.debug("CompletenessEvaluator random baseline failed: %s", exc)
            return []

        denom = abs(original_pred) + 1e-8
        if denom < 1e-12:
            return []
        drops = np.abs(original_pred - preds) / denom
        drops = drops[np.isfinite(drops)]
        valid = np.clip(drops, 0.0, 1.0)
        return valid.tolist()

    def _model_predictions(self, model: Any, instances: np.ndarray) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            proba = np.asarray(model.predict_proba(instances))
            if proba.ndim == 1:
                return proba.astype(float)
            if proba.shape[1] == 2:
                return proba[:, 1].astype(float)
            return np.max(proba, axis=1).astype(float)
        if hasattr(model, "predict"):
            preds = np.asarray(model.predict(instances)).reshape(instances.shape[0])
            return preds.astype(float)
        raise AttributeError("Model must expose predict() or predict_proba().")

    def _fast_metrics_for_explanation(
        self, explanation: Dict[str, Any]
    ) -> Optional[Dict[str, float]]:
        importance = self._importance_vector(explanation)
        if importance is None or importance.size == 0:
            return None

        metadata = explanation.get("metadata") or {}
        text_content = explanation.get("text_content") or metadata.get("text_content")

        if text_content:
            score = self._text_completeness_score(text_content, importance)
            return {
                "completeness_drop": score,
                "completeness_random_drop": 0.0,
                "completeness_score": score,
            }

        prediction_value = self._prediction_value(explanation)
        if prediction_value is None:
            return None

        baseline_prediction = metadata.get("baseline_prediction")
        if baseline_prediction is None:
            baseline_prediction = metadata.get("expected_value")
        if isinstance(baseline_prediction, (list, tuple, np.ndarray)):
            baseline_arr = np.asarray(baseline_prediction).ravel()
            baseline_prediction = float(baseline_arr[0]) if baseline_arr.size else 0.0
        if baseline_prediction is None:
            baseline_prediction = 0.0

        score = self._tabular_completeness_score(
            importance, prediction_value, float(baseline_prediction)
        )
        return {
            "completeness_drop": score,
            "completeness_random_drop": 0.0,
            "completeness_score": score,
        }

    def _text_completeness_score(self, text: str, importance: np.ndarray) -> float:
        words = text.split()
        if not words:
            return 0.0
        vec = np.abs(importance[: len(words)])
        if vec.size == 0:
            return 0.0

        percentile_threshold = np.percentile(vec, 80)
        important_words = np.count_nonzero(vec >= percentile_threshold)
        coverage = important_words / len(words)

        total = float(np.sum(vec))
        if total > 1e-12:
            normalized = vec / total
            entropy = -np.sum(normalized * np.log(normalized + 1e-10))
            max_entropy = np.log(len(words)) if len(words) > 1 else 0.0
            entropy_score = entropy / max_entropy if max_entropy > 0 else 0.0
            simpson = np.sum(normalized**2)
            effective_words = 1.0 / simpson if simpson > 0 else len(words)
            effective_score = effective_words / len(words)
        else:
            entropy_score = 0.0
            effective_score = 0.0

        score = np.mean([coverage, entropy_score, effective_score])
        return float(np.clip(score, 0.0, 1.0))

    def _tabular_completeness_score(
        self,
        importance: np.ndarray,
        prediction_value: float,
        baseline_prediction: float,
    ) -> float:
        sum_attributions = float(np.sum(importance))
        output_diff = float(prediction_value - baseline_prediction)
        if abs(output_diff) > 1e-8:
            score = 1.0 - abs(sum_attributions - output_diff) / abs(output_diff)
        else:
            score = 1.0 if abs(sum_attributions) < 1e-8 else 0.0
        return float(np.clip(score, 0.0, 1.0))

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
    per_instance = True
    requires_full_batch = False
    metric_names = (
        "completeness_drop",
        "completeness_random_drop",
        "completeness_score",
    )
    supported_methods = tuple(_FEATURE_METHOD_KEYS)
