"""
Covariate complexity / regularity metric inspired by Co-12 (Sec. 6.6).

Computes the Shannon entropy of each explanation's attribution vector to quantify
how noisy the feature-importance distribution is. Lower entropy (higher
regularity) indicates concentrated, easier-to-memorize explanations.
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


class CovariateComplexityEvaluator:
    """Evaluate average Shannon entropy (and its complement) over local explanations."""

    def __init__(
        self,
        *,
        metric_key: str = "covariate_complexity",
        regularity_key: str = "covariate_regularity",
    ) -> None:
        """
        Parameters
        ----------
        metric_key : str, optional
            Dictionary key used for the normalized entropy output.
        regularity_key : str, optional
            Dictionary key for the 1 - entropy complement (higher is better).
        """
        self.metric_key = metric_key or "covariate_complexity"
        self.regularity_key = regularity_key or f"{self.metric_key}_regularity"
        self.logger = logging.getLogger(__name__)

    def evaluate(
        self,
        model: Any,
        explanation_results: Dict[str, Any],
        dataset: Any | None = None,
        explainer: Any | None = None,
    ) -> Dict[str, float]:
        """
        Compute the mean normalized entropy of feature-importance vectors.

        Parameters
        ----------
        model : Any
            Present for API symmetry; not used by this metric.
        explanation_results : Dict[str, Any]
            Output dict from ``BaseExplainer.explain_dataset`` containing per-instance
            explanations with attribution scores.
        dataset : Any | None, optional
            Accepted for interface compatibility; unused.
        explainer : Any | None, optional
            Accepted for interface compatibility; unused.
        """
        del model, dataset, explainer  # unused but kept for API parity

        method = (explanation_results.get("method") or "").lower()
        if method not in _FEATURE_METHOD_KEYS:
            return {
                self.metric_key: 0.0,
                self.regularity_key: 0.0,
            }

        explanations = explanation_results.get("explanations") or []
        if not explanations:
            return {
                self.metric_key: 0.0,
                self.regularity_key: 0.0,
            }

        entropies: List[float] = []
        regularities: List[float] = []

        for explanation in explanations:
            importance = self._importance_vector(explanation)
            if importance is None or importance.size == 0:
                continue

            if not np.all(np.isfinite(importance)):
                continue

            importance = np.abs(importance)
            total = float(np.sum(importance))
            if total <= 0.0:
                entropies.append(0.0)
                regularities.append(1.0)
                continue

            prob = importance / total
            entropy = self._shannon_entropy(prob)
            max_entropy = np.log2(prob.size) if prob.size > 1 else 0.0
            normalized = float(entropy / max_entropy) if max_entropy > 0 else 0.0
            normalized = float(np.clip(normalized, 0.0, 1.0))

            entropies.append(normalized)
            regularities.append(1.0 - normalized)

        complexity = float(np.mean(entropies)) if entropies else 0.0
        regularity = float(np.mean(regularities)) if regularities else 0.0
        return {
            self.metric_key: complexity,
            self.regularity_key: regularity,
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _importance_vector(self, explanation: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Extract feature-importance values regardless of explainer schema by checking
        both the explanation root and its metadata.
        """
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
        """Convert a supported sequence into a flattened float array."""
        if values is None:
            return None
        if isinstance(values, np.ndarray):
            arr = values.astype(float, copy=False).reshape(-1)
        elif isinstance(values, Sequence):
            arr = np.asarray(values, dtype=float).reshape(-1)
        else:
            return None
        if arr.ndim != 1:
            arr = arr.reshape(-1)
        return arr

    def _shannon_entropy(self, prob: np.ndarray) -> float:
        """Return Shannon entropy in bits for a probability vector."""
        safe_prob = np.clip(prob, 1e-12, 1.0)
        entropy = -float(np.sum(safe_prob * np.log2(safe_prob)))
        return entropy
