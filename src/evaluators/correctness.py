"""
Correctness (faithfulness) metric for local tabular explanations.

Implements a feature-removal test: the more the model prediction changes after
masking the most important features (according to the explanation), the more
"correct" the explanation is with respect to the black-box.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
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


@dataclass
class _CorrectnessContext:
    """Cached payload with everything needed to score an explanation."""

    importance: np.ndarray
    instance: np.ndarray
    baseline: Optional[np.ndarray]
    top_indices: np.ndarray
    original_prediction: float


class CorrectnessEvaluator(MetricCapabilities):
    """
    Computes a correctness score via a feature-removal test.

    Parameters
    ----------
    removal_fraction : float | int
        Fraction of top-ranked features (by absolute importance) to mask when given
        as a float in [0, 1]. If an integer is provided, it is interpreted as the
        absolute number of top features to delete (use 1 for single-feature deletion).
    default_baseline : float
        Value used to replace masked features when no baseline vector is provided
        in the explanation metadata (via ``baseline_instance``).
    min_features : int
        Minimum number of features to mask, regardless of ``removal_fraction``.
    """

    per_instance = True
    requires_full_batch = False
    metric_names = ("correctness",)
    supported_methods = tuple(_FEATURE_METHOD_KEYS)

    def __init__(
        self,
        *,
        removal_fraction: float = 0.1,
        default_baseline: float = 0.0,
        min_features: int = 1,
        fast_mode: bool = True,
        cache_context: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        removal_fraction : float | int, optional
            Fraction (or absolute count) of top-ranked features to mask when computing
            the deletion score.
        default_baseline : float, optional
            Value substituted for masked features when an explanation does not provide
            its own ``baseline_instance`` metadata.
        min_features : int, optional
            Enforce a minimum number of masked features even if ``removal_fraction``
            would suggest fewer.
        """
        if isinstance(removal_fraction, bool):
            # avoid treating booleans as integers; coerce to float fraction
            removal_fraction = float(removal_fraction)

        if isinstance(removal_fraction, (int, np.integer)):
            self._removal_mode = "count"
            self._removal_count = max(1, int(removal_fraction))
            self.removal_fraction = None
        else:
            self._removal_mode = "fraction"
            self.removal_fraction = float(np.clip(float(removal_fraction), 0.0, 1.0))
            self._removal_count = None
        self.default_baseline = float(default_baseline)
        self.min_features = max(1, int(min_features))
        self.fast_mode = bool(fast_mode)
        self.cache_context = bool(cache_context)
        self.logger = logging.getLogger(__name__)

    def evaluate(
        self,
        model: Any,
        explanation_results: Dict[str, Any],
        dataset: Any | None = None,
        explainer: Any | None = None,
    ) -> Dict[str, float]:
        """
        Evaluate correctness for SHAP/LIME/IG/Causal SHAP explanations.

        Parameters
        ----------
        model : Any
            Trained model associated with the explanations (must expose predict /
            predict_proba used during feature masking).
        explanation_results : Dict[str, Any]
            Output of ``BaseExplainer.explain_dataset`` (or compatible structure
            containing ``method`` and ``explanations`` entries).
        dataset : Any | None, optional
            Dataset object (unused currently but accepted for interface parity).
        explainer : Any | None, optional
            Explainer instance (unused, kept for symmetry with other evaluators).

        Returns
        -------
        Dict[str, float]
            Mapping with a single ``"correctness"`` entry in [0, 1].
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
        Return correctness score using a fully prepared MetricInput.

        Parameters
        ----------
        metric_input : MetricInput
            Unified payload describing the model/dataset/explanations context.

        Returns
        -------
        Dict[str, float]
            Dictionary with the averaged correctness score (or per-instance value
            when ``explanation_idx`` is provided).
        """
        if metric_input.method not in self.supported_methods:
            self.logger.info(
                "CorrectnessEvaluator skipped: method '%s' not a feature-attribution explainer",
                metric_input.method,
            )
            return {"correctness": 0.0}

        explanations = metric_input.explanations
        if not explanations:
            return {"correctness": 0.0}

        context_cache: Optional[Dict[int, _CorrectnessContext]] = (
            {} if self.cache_context else None
        )

        if metric_input.explanation_idx is not None:
            idx = metric_input.explanation_idx
            if not (0 <= idx < len(explanations)):
                return {"correctness": 0.0}
            score = self._feature_removal_score(
                metric_input.model,
                explanations[idx],
                context_cache,
            )
            return {"correctness": float(score) if score is not None else 0.0}

        scores: List[float] = []
        for explanation in explanations:
            score = self._feature_removal_score(
                metric_input.model,
                explanation,
                context_cache,
            )
            if score is not None:
                scores.append(score)

        correctness = float(np.mean(scores)) if scores else 0.0
        return {"correctness": correctness}

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _feature_removal_score(
        self,
        model: Any,
        explanation: Dict[str, Any],
        context_cache: Optional[Dict[int, _CorrectnessContext]] = None,
    ) -> Optional[float]:
        """Mask the most important features and return the resulting normalized drop."""
        context = self._prepare_context(explanation, context_cache)
        if context is None:
            return None

        if self.fast_mode:
            return self._fast_feature_removal_score(model, explanation, context)
        return self._slow_feature_removal_score(model, context)

    def _prepare_context(
        self,
        explanation: Dict[str, Any],
        context_cache: Optional[Dict[int, _CorrectnessContext]],
    ) -> Optional[_CorrectnessContext]:
        cache_key = id(explanation)
        if context_cache is not None and cache_key in context_cache:
            return context_cache[cache_key]

        importance_vec = self._feature_importance_vector(explanation)
        if importance_vec is None or importance_vec.size == 0:
            return None

        instance = self._extract_instance(explanation)
        if instance is None:
            return None

        k = self._num_features_to_mask(len(importance_vec))
        top_indices = np.argsort(-np.abs(importance_vec))[:k]

        baseline = None
        if not self.fast_mode:
            baseline = self._baseline_vector(explanation, instance)

        orig_pred = self._prediction_value(explanation)
        if orig_pred is None:
            return None

        context = _CorrectnessContext(
            importance=importance_vec,
            instance=instance,
            baseline=baseline,
            top_indices=top_indices,
            original_prediction=orig_pred,
        )
        if context_cache is not None:
            context_cache[cache_key] = context
        return context

    def _slow_feature_removal_score(
        self,
        model: Any,
        context: _CorrectnessContext,
    ) -> Optional[float]:
        if context.baseline is None:
            return None
        perturbed = context.instance.copy()
        perturbed[context.top_indices] = context.baseline[context.top_indices]

        try:
            new_pred = self._model_prediction(model, perturbed)
        except Exception as exc:
            self.logger.debug("CorrectnessEvaluator failed to perturb instance: %s", exc)
            return None

        return self._normalised_change(context.original_prediction, new_pred)

    def _fast_feature_removal_score(
        self,
        model: Any,
        explanation: Dict[str, Any],
        context: _CorrectnessContext,
    ) -> Optional[float]:
        baseline = self._fast_baseline_vector(explanation, context.instance)
        perturbed = context.instance.copy()
        perturbed[context.top_indices] = baseline[context.top_indices]

        orig_pred = self._fast_original_prediction(model, explanation, context)
        if orig_pred is None:
            return None

        try:
            new_pred = self._fast_model_prediction(model, perturbed)
        except Exception as exc:
            self.logger.debug("CorrectnessEvaluator fast mode failed: %s", exc)
            return None

        return self._normalised_change(orig_pred, new_pred)

    def _normalised_change(self, orig_pred: float, new_pred: float) -> Optional[float]:
        change = abs(orig_pred - new_pred)
        denom = abs(orig_pred) + 1e-8
        if denom < 1e-12:
            self.logger.debug(
                "CorrectnessEvaluator denominator nearly zero (orig=%s); skipping instance",
                orig_pred,
            )
            return None
        score = float(np.clip(change / denom, 0.0, 1.0))
        if np.isnan(score):
            self.logger.debug(
                "CorrectnessEvaluator produced NaN score (orig=%s, new=%s, denom=%s)",
                orig_pred,
                new_pred,
                denom,
            )
            return None
        return score

    def _fast_original_prediction(
        self,
        model: Any,
        explanation: Dict[str, Any],
        context: _CorrectnessContext,
    ) -> Optional[float]:
        prediction = explanation.get("prediction")
        if prediction is not None:
            arr = np.asarray(prediction).ravel()
            if arr.size:
                try:
                    return float(arr[0])
                except Exception:
                    pass
        # Fallback to the richer probability-aware estimate prepared for the slow path.
        if context.original_prediction is not None:
            return float(context.original_prediction)
        try:
            return self._fast_model_prediction(model, context.instance)
        except Exception:
            return None

    def _fast_model_prediction(self, model: Any, instance: np.ndarray) -> float:
        batch = instance.reshape(1, -1)
        if hasattr(model, "predict"):
            preds = np.asarray(model.predict(batch)).ravel()
            if preds.size == 0:
                raise ValueError("Model.predict returned empty output.")
            return float(preds[0])
        return self._model_prediction(model, instance)

    def _fast_baseline_vector(
        self,
        explanation: Dict[str, Any],
        instance: np.ndarray,
    ) -> np.ndarray:
        metadata = explanation.get("metadata") or {}
        baseline = metadata.get("baseline_instance")
        if baseline is not None:
            arr = np.asarray(baseline, dtype=float).reshape(-1)
            if arr.shape == instance.shape:
                return arr
        return np.zeros_like(instance)

    def _num_features_to_mask(self, n_features: int) -> int:
        """
        Determine how many top-ranked features to mask based on the evaluator
        configuration (fractional removal or fixed count).
        """
        if self._removal_mode == "count":
            k = max(self.min_features, self._removal_count)
        else:
            frac = self.removal_fraction if self.removal_fraction is not None else 0.0
            k = max(self.min_features, int(np.ceil(frac * n_features)))
        return max(1, min(k, n_features))
    
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
        """Coerce importance values into a 1-D float numpy array."""
        if importances is None:
            self.logger.debug("CorrectnessEvaluator missing importance vector in explanation")
            return None
        if isinstance(importances, np.ndarray):
            vec = importances
        elif isinstance(importances, Sequence):
            vec = np.asarray(importances)
        else:
            self.logger.debug("CorrectnessEvaluator missing instance vector in explanation")
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
        """Return explainer-provided baseline if present, otherwise fill with default."""
        metadata = explanation.get("metadata") or {}
        baseline = metadata.get("baseline_instance")
        if baseline is not None:
            base_arr = np.asarray(baseline, dtype=float).reshape(-1)
            if base_arr.shape == instance.shape:
                return base_arr
        self.logger.debug(
            "CorrectnessEvaluator using default baseline %.5f for instance", self.default_baseline
        )
        return np.full_like(instance, self.default_baseline, dtype=float)

    def _prediction_value(self, explanation: Dict[str, Any]) -> Optional[float]:
        """
        Scalar value to track under feature removal.
        Prefer class probability if available; otherwise use the raw prediction.
        """
        proba = explanation.get("prediction_proba")
        if proba is not None:
            proba_arr = np.asarray(proba).ravel()
            if proba_arr.size == 0:
                return None
            # Binary: use positive class (index 1)
            if proba_arr.size == 2:
                return float(proba_arr[1])
            # Multiclass: use max probability
            return float(proba_arr.max())

        prediction = explanation.get("prediction")
        if prediction is None:
            self.logger.debug("CorrectnessEvaluator missing prediction value in explanation")
            return None
        arr = np.asarray(prediction).ravel()
        if arr.size == 0:
            return None
        try:
            return float(arr[0])
        except Exception:
            return None


    def _model_prediction(self, model: Any, instance: np.ndarray) -> float:
        """
        Compute scalar prediction for a perturbed instance, aligned with _prediction_value:
        prefer class probability if available; otherwise use raw prediction.
        """
        batch = instance.reshape(1, -1)
        if hasattr(model, "predict_proba"):
            proba = np.asarray(model.predict_proba(batch)).ravel()
            if proba.size == 0:
                raise ValueError("Model.predict_proba returned empty output.")
            if proba.size == 2:
                return float(proba[1])          # positive class for binary
            return float(proba.max())           # max prob for multiclass

        if not hasattr(model, "predict"):
            raise AttributeError("Model must expose a predict() method.")

        preds = np.asarray(model.predict(batch)).ravel()
        if preds.size == 0:
            raise ValueError("Model.predict returned empty output.")
        return float(preds[0])
