"""
Relative stability metrics adapted from Quantus
(https://github.com/understandable-machine-intelligence-lab/Quantus).

This module currently implements Relative Input Stability (RIS) from Agarwal
et al. (2022): perturb the input slightly, recompute attributions, and report
the maximum ratio between relative attribution change and relative input change.
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


class RelativeInputStabilityEvaluator(MetricCapabilities):
    """
    Estimate how sensitive an explanation is relative to input perturbations.

    For each instance, RIS samples small input perturbations, reruns the
    explainer, and computes:

        || (e(x) - e(x')) / e(x) || / max(|| (x - x') / x ||, eps_min)

    Higher scores imply the attribution changes faster than the input itself.
    """
    supported_methods = tuple(_FEATURE_METHOD_KEYS)
    metric_names = ("relative_input_stability",)
    per_instance = True
    requires_full_batch = False

    def __init__(
        self,
        *,
        metric_key: str = "relative_input_stability",
        max_instances: int = 5,
        num_samples: int = 10,
        noise_scale: float = 0.01,
        eps_min: float = 1e-6,
        random_state: Optional[int] = 42,
    ) -> None:
        self.metric_key = metric_key or "relative_input_stability"
        self.max_instances = max(1, int(max_instances))
        self.num_samples = max(1, int(num_samples))
        self.noise_scale = float(max(0.0, noise_scale))
        self.eps_min = float(max(1e-12, eps_min))
        self.random_state = random_state
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
            return {self.metric_key: 0.0}

        explanations = metric_input.explanations
        if not explanations:
            return {self.metric_key: 0.0}

        if metric_input.explainer is None or not hasattr(metric_input.explainer, "explain_instance"):
            self.logger.debug(
                "RelativeInputStabilityEvaluator requires an explainer instance to rerun perturbations."
            )
            return {self.metric_key: 0.0}

        std_vec = self._dataset_feature_std(metric_input.dataset, explanations[0])

        if metric_input.explanation_idx is not None:
            idx = metric_input.explanation_idx
            if not (0 <= idx < len(explanations)):
                return {self.metric_key: 0.0}
            seed = None if self.random_state is None else self.random_state + int(idx)
            rng = np.random.default_rng(seed)
            score = self._ris_score(
                explanation=explanations[idx],
                explainer=metric_input.explainer,
                feature_std=std_vec,
                rng=rng,
            )
            return {self.metric_key: float(score) if score is not None else 0.0}

        rng = np.random.default_rng(self.random_state)
        scores: List[float] = []
        n_eval = min(self.max_instances, len(explanations))

        for i in range(n_eval):
            score = self._ris_score(
                explanation=explanations[i],
                explainer=metric_input.explainer,
                feature_std=std_vec,
                rng=rng,
            )
            if score is not None and np.isfinite(score):
                scores.append(float(score))

        return {self.metric_key: float(np.mean(scores)) if scores else 0.0}

    # ------------------------------------------------------------------ #
    # RIS helpers                                                        #
    # ------------------------------------------------------------------ #

    def _ris_score(
        self,
        explanation: Dict[str, Any],
        explainer: Any,
        feature_std: Optional[np.ndarray],
        rng: np.random.Generator,
    ) -> Optional[float]:
        instance = self._instance_vector(explanation)
        importance = self._importance_vector(explanation)
        if instance is None or importance is None or importance.size == 0:
            return None

        std_vec = self._match_std_vector(feature_std, instance.size)
        baseline = abs(instance) + self.eps_min
        attr_base = abs(importance) + self.eps_min
        ratios: List[float] = []

        for _ in range(self.num_samples):
            noise = rng.normal(0.0, std_vec * self.noise_scale, size=instance.shape[0])
            perturbed_instance = instance + noise

            perturbed_importance = self._rerun_explainer(
                explainer=explainer,
                template=explanation,
                perturbed_instance=perturbed_instance,
            )
            if perturbed_importance is None or perturbed_importance.size != importance.size:
                continue

            rel_attr = np.linalg.norm((importance - perturbed_importance) / attr_base, ord=2)
            denom = np.linalg.norm(noise / baseline, ord=2)
            denom = max(denom, self.eps_min)
            ratios.append(float(rel_attr / denom))

        return max(ratios) if ratios else None

    def _rerun_explainer(
        self,
        explainer: Any,
        template: Dict[str, Any],
        perturbed_instance: np.ndarray,
    ) -> Optional[np.ndarray]:
        try:
            result = explainer.explain_instance(perturbed_instance)
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.debug("RelativeInputStabilityEvaluator failed to re-explain perturbed sample: %s", exc)
            return None
        importance = self._importance_vector(result)
        if importance is None:
            self.logger.debug("RelativeInputStabilityEvaluator explanation missing attributions after rerun.")
        return importance

    def _dataset_feature_std(self, dataset: Any | None, example_explanation: Dict[str, Any]) -> Optional[np.ndarray]:
        if dataset is None:
            return None
        X_train = getattr(dataset, "X_train", None)
        if X_train is None:
            return None
        X_arr = np.asarray(X_train)
        if X_arr.ndim == 1:
            std = np.std(X_arr, axis=0)
            return std.reshape(1)
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
        vec = np.asarray(feature_std, dtype=float).reshape(-1)
        if vec.size == 1:
            return np.full(n_features, vec[0], dtype=float)
        if vec.size != n_features:
            return np.resize(vec, n_features)
        return vec

    def _instance_vector(self, explanation: Dict[str, Any]) -> Optional[np.ndarray]:
        candidate = explanation.get("instance")
        if candidate is None:
            metadata = explanation.get("metadata") or {}
            candidate = metadata.get("instance")
        if candidate is None:
            candidate = explanation.get("input")
        if candidate is None:
            self.logger.debug("RelativeInputStabilityEvaluator missing instance in explanation.")
            return None
        return np.asarray(candidate, dtype=float).reshape(-1)

    def _importance_vector(self, explanation: Dict[str, Any]) -> Optional[np.ndarray]:
        candidates = ("feature_importance", "feature_importances", "attributions", "importance")
        metadata = explanation.get("metadata") or {}
        for container in (explanation, metadata):
            for key in candidates:
                if key in container:
                    arr = self._to_array(container.get(key))
                    if arr is not None:
                        return arr
        self.logger.debug("RelativeInputStabilityEvaluator missing attribution vector in explanation.")
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
