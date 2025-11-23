"""
Shared helpers for evaluator metadata and unified inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Sequence


@dataclass(frozen=True)
class MetricInput:
    """
    Canonical payload handed to every evaluator.

    Parameters
    ----------
    model : Any
        Trained model used for metrics that re-query predictions.
    dataset : Any | None
        Optional dataset reference for metadata (feature names, stats, etc.).
    explainer : Any | None
        Explainer instance that generated the explanations.
    explanations : Sequence[Dict[str, Any]]
        Full list of per-instance explanations returned by ``explain_dataset``.
    method : str
        Lowercased method identifier provided by the explainer.
    raw_results : Dict[str, Any]
        Original results dictionary for any additional metadata.
    explanation_idx : Optional[int]
        Index of the explanation under inspection (None when aggregating batches).
    """

    model: Any
    dataset: Any | None
    explainer: Any | None
    explanations: Sequence[Dict[str, Any]]
    method: str
    raw_results: Dict[str, Any]
    explanation_idx: Optional[int] = None

    @classmethod
    def from_results(
        cls,
        model: Any,
        explanation_results: Dict[str, Any],
        *,
        dataset: Any | None = None,
        explainer: Any | None = None,
        explanation_idx: Optional[int] = None,
    ) -> "MetricInput":
        """
        Build a MetricInput from an ``explain_dataset``-style output.

        Parameters
        ----------
        model : Any
            Trained model associated with the explanations.
        explanation_results : Dict[str, Any]
            Dict containing at least ``method`` and ``explanations`` fields.
        dataset : Any | None, optional
            Dataset reference (optional).
        explainer : Any | None, optional
            Explainer instance (optional).
        explanation_idx : Optional[int], optional
            Index to target for per-instance metrics. If omitted, falls back to
            ``explanation_results["current_index"]`` when available.
        """
        idx = explanation_idx
        if idx is None:
            idx = explanation_results.get("current_index")
        method = (explanation_results.get("method") or "").lower()
        explanations = explanation_results.get("explanations") or []
        return cls(
            model=model,
            dataset=dataset,
            explainer=explainer,
            explanations=explanations,
            method=method,
            raw_results=explanation_results,
            explanation_idx=idx,
        )

    def with_index(self, index: int) -> "MetricInput":
        """Return a copy targeting a specific explanation index."""
        return replace(self, explanation_idx=index)


class MetricCapabilities:
    """Mixin describing evaluator invocation expectations."""

    per_instance: bool = True
    requires_full_batch: bool = False
    metric_names: tuple[str, ...] = ()
    supported_methods: tuple[str, ...] = ()
