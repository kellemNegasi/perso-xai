"""
Infidelity metric adapted from Quantus / Yeh et al. (2019).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .base_metric import MetricCapabilities, MetricInput
from .correctness import _FEATURE_METHOD_KEYS


class InfidelityEvaluator(MetricCapabilities):
    """
    Measures explanation infidelity following Yeh et al. (2019).

    For each instance and perturbation sample, we:
      1. Select a subset of features to perturb (replace with a baseline).
      2. Compute the input delta (original minus perturbed) and take the dot
         product with the attribution vector.
      3. Compare that estimate against the actual prediction change induced by
         the perturbation; the squared difference is the infidelity loss.
    Lower scores indicate the attribution vector accurately predicts the model's
    behaviour when the input is perturbed.
    """

    per_instance = True
    requires_full_batch = False
    metric_names = ("infidelity",)
    supported_methods = tuple(_FEATURE_METHOD_KEYS)

    def __init__(
        self,
        *,
        n_perturb_samples: int = 16,
        features_per_sample: int = 1,
        default_baseline: float = 0.0,
        abs_attributions: bool = False,
        normalise: bool = False,
        noise_scale: float = 0.0,
        random_state: Optional[int] = None,
    ) -> None:
        if n_perturb_samples < 1:
            raise ValueError("n_perturb_samples must be >= 1.")
        if features_per_sample < 1:
            raise ValueError("features_per_sample must be >= 1.")
        if noise_scale < 0.0:
            raise ValueError("noise_scale must be >= 0.")

        self.n_perturb_samples = int(n_perturb_samples)
        self.features_per_sample = int(features_per_sample)
        self.default_baseline = float(default_baseline)
        self.abs_attributions = bool(abs_attributions)
        self.normalise = bool(normalise)
        self.noise_scale = float(noise_scale)
        self.random_state = random_state

        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

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
            self.logger.info(
                "InfidelityEvaluator skipped: method '%s' not in feature-attribution set",
                metric_input.method,
            )
            return {"infidelity": 0.0}

        explanations = metric_input.explanations
        if not explanations:
            return {"infidelity": 0.0}

        rng = np.random.default_rng(self.random_state)

        if metric_input.explanation_idx is not None:
            idx = metric_input.explanation_idx
            if not (0 <= idx < len(explanations)):
                return {"infidelity": 0.0}
            score = self._infidelity_score(metric_input.model, explanations[idx], rng)
            return {"infidelity": float(score) if score is not None else 0.0}

        scores: List[float] = []
        for explanation in explanations:
            score = self._infidelity_score(metric_input.model, explanation, rng)
            if score is not None:
                scores.append(score)

        infidelity = float(np.mean(scores)) if scores else 0.0
        return {"infidelity": infidelity}

    # ------------------------------------------------------------------ #
    # Core metric
    # ------------------------------------------------------------------ #

    def _infidelity_score(
        self,
        model: Any,
        explanation: Dict[str, Any],
        rng: np.random.Generator,
    ) -> Optional[float]:
        attrs = self._feature_importance_vector(explanation)
        if attrs is None or attrs.size == 0:
            return None
        attrs = self._prepare_attributions(attrs)

        instance = self._extract_instance(explanation)
        if instance is None:
            return None

        if instance.size != attrs.size:
            self.logger.debug(
                "InfidelityEvaluator length mismatch: instance=%s, attrs=%s",
                instance.size,
                attrs.size,
            )
            return None

        baseline = self._baseline_vector(explanation, instance)

        try:
            orig_pred = self._model_prediction(model, instance)
        except Exception as exc:
            self.logger.debug("InfidelityEvaluator failed to score instance: %s", exc)
            return None

        n_features = instance.size
        k = min(self.features_per_sample, n_features)
        errors: List[float] = []
        feature_scale = self._feature_scale(instance)

        for _ in range(self.n_perturb_samples):
            if k == n_features:
                chosen = np.arange(n_features)
            else:
                chosen = rng.choice(n_features, size=k, replace=False)

            perturbed = instance.copy()
            replacement = baseline[chosen]
            if self.noise_scale > 0.0:
                noise = rng.normal(
                    loc=0.0,
                    scale=self.noise_scale * feature_scale[chosen],
                    size=chosen.size,
                )
                replacement = replacement + noise
            perturbed[chosen] = replacement

            delta = instance - perturbed
            dot = float(np.dot(attrs, delta))

            try:
                new_pred = self._model_prediction(model, perturbed)
            except Exception:
                continue

            pred_change = float(orig_pred - new_pred)
            errors.append((dot - pred_change) ** 2)

        if not errors:
            return None
        return float(np.mean(errors))

    # ------------------------------------------------------------------ #
    # Shared helpers (mirrors other evaluators)
    # ------------------------------------------------------------------ #

    def _feature_importance_vector(self, explanation: Dict[str, Any]) -> Optional[np.ndarray]:
        candidates = ("feature_importance", "feature_importances", "attributions", "importance")
        metadata = explanation.get("metadata") or {}
        containers: Sequence[Dict[str, Any]] = (explanation, metadata)

        for container in containers:
            for key in candidates:
                vec = container.get(key)
                arr = self._importance_to_array(vec)
                if arr is not None:
                    return arr
        self.logger.debug("InfidelityEvaluator missing importance vector.")
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

    def _prepare_attributions(self, vec: np.ndarray) -> np.ndarray:
        arr = vec.astype(float).ravel()
        if self.abs_attributions:
            arr = np.abs(arr)
        if self.normalise:
            denom = np.max(np.abs(arr))
            if denom > 0:
                arr = arr / denom
        return arr

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

    def _feature_scale(self, instance: np.ndarray) -> np.ndarray:
        return np.maximum(np.abs(instance), 1e-3)

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
            raise AttributeError("Model must expose predict() or predict_proba().")

        preds = np.asarray(model.predict(batch)).ravel()
        if preds.size == 0:
            raise ValueError("Model.predict returned empty output.")
        return float(preds[0])
