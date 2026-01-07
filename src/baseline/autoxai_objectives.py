from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from .autoxai_types import ObjectiveTerm
from .autoxai_utils import safe_float


def parse_objective_terms(tokens: Sequence[str]) -> List[ObjectiveTerm]:
    """
    Parse objective terms in the form:
      name[:direction]:metric_key[:weight]

    Examples:
      robustness:min:relative_input_stability:1
      fidelity:min:infidelity:2
      conciseness:max:compactness_effective_features:0.5
    """
    terms: List[ObjectiveTerm] = []
    for raw in tokens:
        parts = [part.strip() for part in raw.split(":") if part.strip()]
        if len(parts) not in (3, 4):
            raise ValueError(
                f"Invalid objective term {raw!r}. Expected name:direction:metric_key[:weight]."
            )
        name, direction, metric_key = parts[:3]
        weight = float(parts[3]) if len(parts) == 4 else 1.0
        terms.append(ObjectiveTerm(name=name, metric_key=metric_key, direction=direction, weight=weight))
    if not terms:
        raise ValueError("Objective must contain at least one term.")
    return terms


def _metric_fetchers() -> Dict[str, Any]:
    compactness_keys = (
        "compactness_effective_features",
        "compactness_sparsity",
        "compactness_top10_coverage",
        "compactness_top5_coverage",
    )
    correctness_keys = (
        "correctness",
        "infidelity",
        "non_sensitivity_violation_fraction",
        "non_sensitivity_safe_fraction",
        "non_sensitivity_delta_mean",
        "monotonicity",
    )

    def mean_metric(metrics: Mapping[str, float], keys: Sequence[str]) -> Optional[float]:
        collected: List[float] = []
        for key in keys:
            value = safe_float(metrics.get(key))
            if value is not None:
                collected.append(value)
        if not collected:
            return None
        return sum(collected) / len(collected)

    return {
        "compactness": lambda metrics: mean_metric(metrics, compactness_keys),
        "correctness_group": lambda metrics: mean_metric(metrics, correctness_keys),
        "contrastivity": lambda metrics: safe_float(metrics.get("contrastivity")),
        "stability": lambda metrics: safe_float(metrics.get("relative_input_stability")),
        "faithfulness": lambda metrics: safe_float(metrics.get("correctness")),
        "completeness": lambda metrics: safe_float(metrics.get("completeness_score")),
        "consistency": lambda metrics: safe_float(metrics.get("consistency")),
    }


def fetch_metric(metrics: Mapping[str, float], metric_key: str) -> Optional[float]:
    fetcher = _metric_fetchers().get(metric_key)
    if fetcher is not None:
        return fetcher(metrics)
    return safe_float(metrics.get(metric_key))


def default_objective_terms() -> List[ObjectiveTerm]:
    """
    AutoXAI-paper-aligned objective (same for all personas).

    Notes:
    - Matches AutoXAI (CIKM'22) diabetes scenario aggregation:
      robustness=1, infidelity=2, number_of_features=0.5.
    - In this codebase, we map:
        - robustness/continuity -> relative_input_stability (lower is better)
        - infidelity -> infidelity (lower is better)
        - number_of_features -> compactness_effective_features (higher is better)
    """
    return [
        ObjectiveTerm(
            name="robustness",
            metric_key="relative_input_stability",
            direction="min",
            weight=1.0,
        ),
        ObjectiveTerm(
            name="infidelity",
            metric_key="infidelity",
            direction="min",
            weight=2.0,
        ),
        ObjectiveTerm(
            name="compactness",
            metric_key="compactness_effective_features",
            direction="max",
            weight=0.5,
        ),
    ]


def persona_objective_terms(persona: str) -> List[ObjectiveTerm]:
    """
    Persona-aligned objectives: paper-aligned metrics for every persona.
    """
    if persona == "autoxai":
        return default_objective_terms()
    if persona == "layperson":
        return [
            ObjectiveTerm(
                name="compactness",
                metric_key="compactness_effective_features",
                direction="max",
                weight=1.0,
            ),
            ObjectiveTerm(
                name="contrastivity",
                metric_key="contrastivity",
                direction="max",
                weight=1.0,
            ),
            ObjectiveTerm(
                name="stability",
                metric_key="relative_input_stability",
                direction="min",
                weight=1.0,
            ),
        ]
    if persona == "regulator":
        return [
            ObjectiveTerm(
                name="faithfulness",
                metric_key="correctness",
                direction="max",
                weight=1.0,
            ),
            ObjectiveTerm(
                name="completeness",
                metric_key="completeness_score",
                direction="max",
                weight=1.0,
            ),
            ObjectiveTerm(
                name="consistency",
                metric_key="consistency",
                direction="max",
                weight=1.0,
            ),
            ObjectiveTerm(
                name="compactness",
                metric_key="compactness_effective_features",
                direction="max",
                weight=1.0,
            ),
        ]
    raise ValueError(f"Unknown persona preset: {persona!r}")
