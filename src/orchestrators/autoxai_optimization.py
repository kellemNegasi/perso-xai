"""
AutoXAI-style online optimization helpers.

This module is used by the orchestrator (`metrics_runner.py`) when explainer hyperparameter
spaces include `randint` ranges and are optimized sequentially (Bayesian optimization).
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.baseline.autoxai_objectives import fetch_metric, persona_objective_terms
from src.baseline.autoxai_param_optimizer import RandIntSpec, SearchSpace
from src.baseline.autoxai_scoring import compute_scores, _trial_objective_value


def build_method_label(base: str, params: Mapping[str, Any]) -> str:
    if not params:
        return base
    parts = [f"{k}-{str(v).replace(' ', '')}" for k, v in sorted(params.items())]
    return f"{base}__{'__'.join(parts)}"


def _resolve_randint_bound(value: object, *, n_features: int) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"d", "n_features", "nfeatures", "num_features"}:
            return int(n_features)
    raise ValueError(f"Unsupported randint bound: {value!r}")


def parse_explainer_space(grid: Mapping[str, Any], *, n_features: int) -> SearchSpace:
    categorical: Dict[str, List[object]] = {}
    randint: Dict[str, RandIntSpec] = {}
    for key, value in grid.items():
        if isinstance(value, list):
            categorical[str(key)] = list(value)
            continue
        if isinstance(value, dict) and "randint" in value:
            bounds = value.get("randint")
            if not isinstance(bounds, list) or len(bounds) != 2:
                raise ValueError(f"{key}: randint must be a 2-item list, got {bounds!r}")
            low = int(bounds[0])
            high = _resolve_randint_bound(bounds[1], n_features=n_features)
            if low > high:
                raise ValueError(f"{key}: randint low must be <= high, got {low}..{high}")
            randint[str(key)] = RandIntSpec(low=low, high=high)
            continue
        raise ValueError(
            "Explainer hyperparameter values must be either a list (categorical grid) "
            f"or a dict with `randint: [low, high]`. Got {key}={value!r}."
        )
    return SearchSpace(categorical=categorical, randint=randint)


def aggregate_trial_metrics(
    *,
    batch_metrics: Mapping[str, float],
    instance_metrics: Mapping[int, Mapping[int, Mapping[str, float]]],
) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for _dataset_idx, by_local in instance_metrics.items():
        for _local_idx, metrics_vals in by_local.items():
            for key, value in metrics_vals.items():
                try:
                    v = float(value)
                except (TypeError, ValueError):
                    continue
                totals[key] = totals.get(key, 0.0) + v
                counts[key] = counts.get(key, 0) + 1
    means = {key: (totals[key] / counts[key]) for key in totals.keys() if counts.get(key, 0)}
    means.update({str(k): float(v) for k, v in batch_metrics.items()})
    return means


def compute_objective_term_values(*, metrics: Mapping[str, float], persona: str) -> Dict[str, float]:
    """
    Return objective-term values with direction applied (all terms maximized).
    """
    objective = persona_objective_terms(persona)
    out: Dict[str, float] = {}
    for term in objective:
        value = fetch_metric(metrics, term.metric_key)
        if value is None:
            return {}
        out[term.name] = term.apply_direction(float(value))
    return out


def trial_history_objective_score(
    *,
    history: Sequence[str],
    candidate: str,
    variant_term_means: Mapping[str, Mapping[str, float]],
    persona: str,
    scaling: str,
) -> Optional[float]:
    return _trial_objective_value(
        history=history,
        candidate=candidate,
        variant_term_means=variant_term_means,
        objective=persona_objective_terms(persona),
        scaling=scaling,
    )


def _to_candidate_metrics(
    combined_metric_records: Sequence[Mapping[str, Any]],
) -> Dict[int, Dict[str, Dict[str, float]]]:
    candidate_metrics: Dict[int, Dict[str, Dict[str, float]]] = {}
    for record in combined_metric_records:
        dataset_idx = record.get("dataset_index")
        variant = record.get("method_variant")
        metrics_map = record.get("metrics")
        if not isinstance(dataset_idx, int) or not isinstance(variant, str) or not isinstance(metrics_map, dict):
            continue
        cleaned: Dict[str, float] = {}
        for k, v in metrics_map.items():
            try:
                cleaned[str(k)] = float(v)
            except (TypeError, ValueError):
                continue
        candidate_metrics.setdefault(dataset_idx, {})[variant] = cleaned
    return candidate_metrics


def serialize_candidate_scores(scores: Sequence[Any]) -> List[Dict[str, Any]]:
    return [
        {
            "dataset_index": int(score.dataset_index),
            "method_variant": str(score.method_variant),
            "aggregated_score": float(score.aggregated_score),
            "raw_terms": dict(score.raw_terms),
            "scaled_terms": dict(score.scaled_terms),
        }
        for score in scores
    ]


def build_candidate_scores_reports(
    *,
    combined_metric_records: Sequence[Mapping[str, Any]],
    method_label: str,
    persona: str,
    scaling: str,
    trial_history: Sequence[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns (overall_trial_scope, per_instance_trial_scope) candidate score lists.

    - overall_trial_scope: scaling_scope="trial" over per-variant means across ALL instances.
    - per_instance_trial_scope: scaling_scope="trial" but computed per instance by restricting
      candidates to a single dataset_index at a time (so "variant means" are that instance’s values).
    """
    candidate_metrics = _to_candidate_metrics(combined_metric_records)
    variant_to_method = {variant: method_label for variant in trial_history}
    objective = persona_objective_terms(persona)

    overall = compute_scores(
        candidate_metrics=candidate_metrics,
        variant_to_method=variant_to_method,
        objective=objective,
        scaling=scaling,
        scaling_scope="trial",
        trial_history_for_scaling=trial_history,
    )

    per_instance_scores: List[Any] = []
    for dataset_idx, per_variant in candidate_metrics.items():
        one = compute_scores(
            candidate_metrics={dataset_idx: per_variant},
            variant_to_method=variant_to_method,
            objective=objective,
            scaling=scaling,
            scaling_scope="trial",
            trial_history_for_scaling=trial_history,
        )
        per_instance_scores.extend(one)

    return serialize_candidate_scores(overall), serialize_candidate_scores(per_instance_scores)

