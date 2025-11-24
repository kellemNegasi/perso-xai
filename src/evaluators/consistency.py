"""
Consistency evaluator adapted from the Quantus metric
(https://github.com/understandable-machine-intelligence-lab/Quantus).

Discretises attribution vectors and measures how often instances that receive
the same discretised explanation share the same predicted class, mirroring
Dasgupta et al. (ICML 2022).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence

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


def _top_n_sign(values: np.ndarray, *, n: int = 5) -> int:
    """
    Default discretiser: keep the sign of the ``n`` largest-magnitude attribution
    components and hash them into a comparable token. This prefers comparing the
    most influential features first rather than the first ``n`` indices.
    """
    if values.size == 0:
        return 0
    n = max(1, min(n, values.size))
    order = np.argsort(-np.abs(values))
    top_indices = order[:n]
    signs = np.sign(values[top_indices]).astype(int)
    payload = np.column_stack((top_indices, signs)).astype(np.int64)
    return hash(payload.tobytes())


class ConsistencyEvaluator(MetricCapabilities):
    """
    Probability that two instances sharing the same discretised explanation also
    share the same predicted class (higher is better).

    Parameters
    ----------
    discretise_func : callable, optional
        Function that maps an attribution vector to a hashable label. Defaults
        to ``top_n_sign`` from Quantus.
    discretise_kwargs : dict, optional
        Extra keyword arguments forwarded to ``discretise_func``.
    """

    per_instance = True
    requires_full_batch = False
    metric_names = ("consistency",)
    supported_methods = tuple(_FEATURE_METHOD_KEYS)

    def __init__(
        self,
        *,
        discretise_func: Optional[Callable[[np.ndarray], Any]] = None,
        discretise_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.discretise_func = discretise_func
        self.discretise_kwargs = discretise_kwargs or {}
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
            return {"consistency": 0.0}

        explanations = metric_input.explanations
        if not explanations:
            return {"consistency": 0.0}

        scores = self._consistency_scores(explanations)
        if scores is None:
            return {"consistency": 0.0}

        if metric_input.explanation_idx is not None:
            idx = metric_input.explanation_idx
            if 0 <= idx < len(scores):
                return {"consistency": float(scores[idx])}
            return {"consistency": 0.0}

        valid = scores[np.isfinite(scores)]
        if valid.size == 0:
            return {"consistency": 0.0}

        return {"consistency": float(np.mean(valid))}

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _consistency_scores(self, explanations: Sequence[Dict[str, Any]]) -> Optional[np.ndarray]:
        labels: List[Any] = []
        pred_classes: List[Any] = []
        for explanation in explanations:
            importance = self._importance_vector(explanation)
            label = self._prediction_label(explanation)
            if importance is None or label is None:
                labels.append(None)
                pred_classes.append(None)
                continue
            discretised = self._discretise(importance)
            labels.append(discretised)
            pred_classes.append(label)

        if not any(label is not None for label in labels):
            return None

        n = len(explanations)
        scores = np.zeros(n, dtype=float)
        token_array = np.asarray(labels, dtype=object)
        class_array = np.asarray(pred_classes, dtype=object)

        for i in range(n):
            if token_array[i] is None or class_array[i] is None:
                scores[i] = 0.0
                continue
            same_expl = token_array == token_array[i]
            same_expl[i] = False
            if not np.any(same_expl):
                scores[i] = 0.0
                continue
            same_pred = class_array == class_array[i]
            matches = np.logical_and(same_expl, same_pred)
            denom = np.count_nonzero(same_expl)
            scores[i] = float(np.sum(matches) / denom) if denom > 0 else 0.0
        return scores

    def _discretise(self, importance: np.ndarray) -> Any:
        func = self.discretise_func or _top_n_sign
        kwargs = dict(self.discretise_kwargs)
        if "n" in kwargs:
            original = kwargs["n"]
            kwargs["n"] = max(1, min(int(kwargs["n"]), importance.size))
            if kwargs["n"] != original:
                self.logger.debug(
                    "ConsistencyEvaluator clamped discretiser 'n' from %s to %s to match feature count.",
                    original,
                    kwargs["n"],
                )
        try:
            return func(importance, **kwargs)
        except TypeError:
            self.logger.debug(
                "ConsistencyEvaluator discretise_func did not accept kwargs %s; retrying without.",
                kwargs,
            )
            return func(importance)
        except Exception as exc:
            self.logger.debug("ConsistencyEvaluator discretise_func failed: %s", exc)
            return 0

    def _importance_vector(self, explanation: Dict[str, Any]) -> Optional[np.ndarray]:
        candidates = ("feature_importance", "feature_importances", "attributions", "importance")
        metadata = explanation.get("metadata") or {}
        for container in (explanation, metadata):
            for key in candidates:
                vec = container.get(key)
                arr = self._to_array(vec)
                if arr is not None:
                    return arr
        self.logger.debug("ConsistencyEvaluator missing importance vector in explanation")
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

    def _prediction_label(self, explanation: Dict[str, Any]) -> Optional[int]:
        label = explanation.get("prediction")
        if label is not None:
            try:
                return int(label)
            except Exception:
                try:
                    return int(np.asarray(label).ravel()[0])
                except Exception:
                    self.logger.debug("ConsistencyEvaluator could not coerce prediction to int: %s", label)
                    return None
        proba = explanation.get("prediction_proba")
        if proba is None:
            self.logger.debug("ConsistencyEvaluator missing prediction/prediction_proba in explanation")
            return None
        arr = np.asarray(proba).ravel()
        if arr.size == 0:
            self.logger.debug("ConsistencyEvaluator received empty prediction_proba array")
            return None
        return int(np.argmax(arr))
