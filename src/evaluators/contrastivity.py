"""
Contrastivity metric for local feature-attribution explanations on tabular data.

Measures how dissimilar attribution patterns are across predictions for different
classes using a Structural Similarity Index Measure (SSIM) variant. Inspired by
the Random Logit / target-sensitivity checks described by Sixt et al. (2020).
"""

from __future__ import annotations

import logging
from importlib import import_module
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from .utils import structural_similarity as default_structural_similarity

_FEATURE_METHOD_KEYS = {
    "shap",
    "lime",
    "integrated_gradients",
    "integratedgradients",
    "causal_shap",
    "causalshap",
}


class ContrastivityEvaluator:
    """
    Estimate target contrastivity by comparing attributions across classes.

    For each explanation we repeatedly sample a reference explanation predicted
    for a different class and compute SSIM similarity between the attribution
    vectors. Scores are inverted (1 - SSIM) so higher values indicate that
    explanations differ strongly across classes (i.e., high contrastivity).
    """

    def __init__(
        self,
        *,
        pairs_per_instance: int = 3,
        normalise: bool = True,
        similarity_func: Optional[Union[str, Callable[[np.ndarray, np.ndarray], Optional[float]]]] = None,
        random_state: Optional[int] = 42,
    ) -> None:
        """
        Parameters
        ----------
        pairs_per_instance : int, optional
            How many off-class comparisons to sample per explanation.
        normalise : bool, optional
            Whether to L1-normalise attribution vectors before similarity.
        similarity_func : callable | str, optional
            Callable returning a similarity score given two vectors or a fully-qualified
            import path to such a function (default: SSIM util).
        random_state : int | None, optional
            Random seed for pairing instances across classes.
        """
        self.pairs_per_instance = max(1, int(pairs_per_instance))
        self.normalise = bool(normalise)
        self.similarity_func = self._resolve_similarity_func(similarity_func)
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)

    def evaluate(
        self,
        model: Any,
        explanation_results: Dict[str, Any],
        dataset: Any | None = None,
        explainer: Any | None = None,
    ) -> Dict[str, float]:
        """
        Compute the average contrastivity score over a batch of explanations.
        """
        method = (explanation_results.get("method") or "").lower()
        if method not in _FEATURE_METHOD_KEYS:
            return self._empty_result()

        explanations = explanation_results.get("explanations") or []
        if len(explanations) < 2:
            return self._empty_result()

        labeled_importance: List[tuple[Any, np.ndarray]] = []
        for explanation in explanations:
            importance = self._importance_vector(explanation)
            label = self._prediction_label(explanation)
            if importance is None or label is None:
                continue
            labeled_importance.append((label, importance))

        if len(labeled_importance) < 2:
            return self._empty_result()

        labels = [label for label, _ in labeled_importance]
        unique_labels = list({label for label in labels})
        if len(unique_labels) < 2:
            self.logger.info("ContrastivityEvaluator skipped: only one label present.")
            return self._empty_result()

        label_indices: Dict[Any, List[int]] = {}
        for idx, label in enumerate(labels):
            label_indices.setdefault(label, []).append(idx)

        rng = np.random.default_rng(self.random_state)
        contrastive_scores: List[float] = []

        for idx, (label, importance) in enumerate(labeled_importance):
            candidate_labels = [lbl for lbl in unique_labels if lbl != label and label_indices.get(lbl)]
            if not candidate_labels:
                continue
            for _ in range(self.pairs_per_instance):
                ref_label = rng.choice(candidate_labels)
                ref_idx = int(rng.choice(label_indices[ref_label]))
                ref_importance = labeled_importance[ref_idx][1]
                score = self._contrastive_score(importance, ref_importance)
                if score is not None:
                    contrastive_scores.append(score)

        if not contrastive_scores:
            return self._empty_result()

        return {
            "contrastivity": float(np.mean(contrastive_scores)),
            "contrastivity_pairs": float(len(contrastive_scores)),
        }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _contrastive_score(self, vec_a: np.ndarray, vec_b: np.ndarray) -> Optional[float]:
        """Return contrastivity score given two attribution vectors."""
        norm_a = self._normalise(vec_a)
        norm_b = self._normalise(vec_b)
        similarity = self.similarity_func(norm_a, norm_b)
        if similarity is None or np.isnan(similarity):
            return None
        score = 1.0 - float(similarity)
        return float(np.clip(score, 0.0, 1.0))

    def _normalise(self, vec: np.ndarray) -> np.ndarray:
        """Optional L1-normalisation for comparability across instances."""
        arr = vec.astype(float).reshape(-1)
        if not self.normalise:
            return arr
        norm = np.sum(np.abs(arr))
        if norm < 1e-12:
            return arr
        return arr / norm

    def _importance_vector(self, explanation: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract attribution vector from explanation or metadata."""
        candidates = ("feature_importance", "feature_importances", "attributions", "importance")
        metadata = explanation.get("metadata") or {}
        for container in (explanation, metadata):
            for key in candidates:
                vec = container.get(key)
                arr = self._to_array(vec)
                if arr is not None:
                    return arr
        return None

    def _prediction_label(self, explanation: Dict[str, Any]) -> Any:
        """Derive the predicted label from explanation (uses prediction or proba)."""
        prediction = explanation.get("prediction")
        if prediction is not None:
            if isinstance(prediction, (str, bytes)):
                return prediction
            pred_arr = np.asarray(prediction)
            if pred_arr.ndim == 0:
                value = pred_arr.item()
            else:
                if pred_arr.size == 0:
                    value = None
                else:
                    value = pred_arr.ravel()[0]
            if value is None:
                return None
            if isinstance(value, (str, bytes)):
                return value
            if isinstance(value, (np.integer, int)):
                return int(value)
            if isinstance(value, (np.floating, float)):
                rounded = int(round(float(value)))
                if abs(float(value) - rounded) < 1e-6:
                    return rounded
                return None

        proba = explanation.get("prediction_proba")
        if proba is None:
            return None
        proba_arr = np.asarray(proba).ravel()
        if proba_arr.size == 0:
            return None
        return int(np.argmax(proba_arr))

    def _to_array(self, values: Any) -> Optional[np.ndarray]:
        """Convert values into a 1-D numpy array when possible."""
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

    def _empty_result(self) -> Dict[str, float]:
        return {"contrastivity": 0.0, "contrastivity_pairs": 0.0}

    def _resolve_similarity_func(
        self,
        func: Optional[Union[str, Callable[[np.ndarray, np.ndarray], Optional[float]]]],
    ) -> Callable[[np.ndarray, np.ndarray], Optional[float]]:
        """Return a callable similarity function for attribution vectors."""
        if func is None:
            return default_structural_similarity
        if callable(func):
            return func
        if isinstance(func, str):
            module_path, _, attr = func.rpartition(".")
            if not module_path:
                raise ValueError(
                    f"similarity_func must be a callable or module path, got '{func}'."
                )
            module = import_module(module_path)
            attr_obj = getattr(module, attr)
            if not callable(attr_obj):
                raise TypeError(f"Resolved object '{func}' is not callable.")
            return attr_obj
        raise TypeError("similarity_func must be None, callable, or module path string.")
