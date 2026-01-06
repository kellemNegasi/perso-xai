from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .autoxai_objectives import fetch_metric
from .autoxai_types import CandidateScore, ObjectiveTerm
from .autoxai_utils import minmax_scale, scalarize_terms, standard_scale


def compute_scores(
    *,
    candidate_metrics: Mapping[int, Mapping[str, Mapping[str, float]]],
    variant_to_method: Mapping[str, str],
    objective: Sequence[ObjectiveTerm],
    scaling: str = "Std",
    scaling_scope: str = "trial",
    trial_history_for_scaling: Optional[Sequence[str]] = None,
    apply_direction: bool = True,
) -> List[CandidateScore]:
    """
    Produces AutoXAI-style standardized + weighted aggregated scores.

    scaling: "Std" or "MinMax" (mirrors AutoXAI).

    scaling_scope:
      - "trial": scale each objective term over per-variant means (matches AutoXAI's trial-history scaling).
      - "global": scale each objective term over all (instance, variant) candidates.
      - "instance": scale each objective term within each instance across variants.

    trial_history_for_scaling:
      When scaling_scope="trial", optionally provide the sequence of evaluated trial variants to fit
      the scalers on (supports duplicates). This matches AutoXAI's sequential scaling behaviour for
      random/BO runs where scaling only uses the evaluated trial history.

    apply_direction:
      When True (default), applies each ObjectiveTerm's direction ("min" terms are negated) so that
      all terms are maximized. Set False when the upstream metrics are already oriented (e.g. Pareto
      files that already negated lower-is-better metrics).
    """

    if scaling not in {"Std", "MinMax"}:
        raise ValueError("scaling must be 'Std' or 'MinMax'.")
    if scaling_scope not in {"trial", "global", "instance"}:
        raise ValueError("scaling_scope must be 'trial', 'global' or 'instance'.")

    weight_map = {term.name: term.weight for term in objective}
    term_names = [term.name for term in objective]

    raw: List[Tuple[int, str, str, Dict[str, float]]] = []
    for dataset_index, variants in candidate_metrics.items():
        for variant, metrics in variants.items():
            term_values: Dict[str, float] = {}
            missing = False
            for term in objective:
                value = fetch_metric(metrics, term.metric_key)
                if value is None:
                    missing = True
                    break
                term_values[term.name] = term.apply_direction(value) if apply_direction else float(value)
            if missing:
                continue
            method = variant_to_method.get(variant, "unknown")
            raw.append((dataset_index, variant, method, term_values))

    def _scale_value(value: float, *, params: Tuple[float, float], mode: str) -> float:
        a, b = params
        if mode == "Std":
            mean, std = a, b
            if std == 0.0:
                return 0.0
            return (value - mean) / std
        lo, hi = a, b
        if hi == lo:
            return 0.0
        return (value - lo) / (hi - lo)

    if scaling_scope == "trial":
        per_variant_values: Dict[str, Dict[str, List[float]]] = {}
        for _, variant, _, term_values in raw:
            bucket = per_variant_values.setdefault(variant, {name: [] for name in term_names})
            for name in term_names:
                bucket[name].append(term_values[name])

        per_variant_means: Dict[str, Dict[str, float]] = {}
        for variant, buckets in per_variant_values.items():
            if any(not buckets[name] for name in term_names):
                continue
            per_variant_means[variant] = {name: (sum(buckets[name]) / len(buckets[name])) for name in term_names}

        term_scalers: Dict[str, Tuple[float, float]] = {}
        for name in term_names:
            if trial_history_for_scaling is None:
                series = [per_variant_means[variant][name] for variant in sorted(per_variant_means)]
            else:
                series = [
                    per_variant_means[variant][name]
                    for variant in trial_history_for_scaling
                    if variant in per_variant_means
                ]
            if not series:
                term_scalers[name] = (0.0, 0.0)
                continue
            if scaling == "Std":
                mean = sum(series) / len(series)
                var = sum((v - mean) ** 2 for v in series) / len(series)
                std = math.sqrt(var)
                term_scalers[name] = (mean, std)
            else:
                term_scalers[name] = (min(series), max(series))

        scores: List[CandidateScore] = []
        for dataset_index, variant, method, term_values in raw:
            scaled_terms = {name: _scale_value(term_values[name], params=term_scalers[name], mode=scaling) for name in term_names}
            aggregated = scalarize_terms(scaled_terms, weights=weight_map)
            scores.append(
                CandidateScore(
                    dataset_index=dataset_index,
                    method_variant=variant,
                    method=method,
                    raw_terms=dict(term_values),
                    scaled_terms=scaled_terms,
                    aggregated_score=aggregated,
                )
            )
        return scores

    if scaling_scope == "global":
        per_term_series: Dict[str, List[float]] = {name: [] for name in term_names}
        for _, _, _, term_values in raw:
            for name in term_names:
                per_term_series[name].append(term_values[name])

        scaled_series: Dict[str, List[float]] = {}
        for name, values in per_term_series.items():
            scaled_series[name] = standard_scale(values) if scaling == "Std" else minmax_scale(values)

        scores: List[CandidateScore] = []
        for idx, (dataset_index, variant, method, term_values) in enumerate(raw):
            scaled_terms = {name: scaled_series[name][idx] for name in term_names}
            aggregated = scalarize_terms(scaled_terms, weights=weight_map)
            scores.append(
                CandidateScore(
                    dataset_index=dataset_index,
                    method_variant=variant,
                    method=method,
                    raw_terms=dict(term_values),
                    scaled_terms=scaled_terms,
                    aggregated_score=aggregated,
                )
            )
        return scores

    grouped: Dict[int, List[Tuple[str, str, Dict[str, float]]]] = {}
    for dataset_index, variant, method, term_values in raw:
        grouped.setdefault(dataset_index, []).append((variant, method, term_values))

    results: List[CandidateScore] = []
    for dataset_index, entries in grouped.items():
        per_term_series = {name: [] for name in term_names}
        for _, _, term_values in entries:
            for name in term_names:
                per_term_series[name].append(term_values[name])

        scaled_series: Dict[str, List[float]] = {}
        for name, values in per_term_series.items():
            scaled_series[name] = standard_scale(values) if scaling == "Std" else minmax_scale(values)

        for idx, (variant, method, term_values) in enumerate(entries):
            scaled_terms = {name: scaled_series[name][idx] for name in term_names}
            aggregated = scalarize_terms(scaled_terms, weights=weight_map)
            results.append(
                CandidateScore(
                    dataset_index=dataset_index,
                    method_variant=variant,
                    method=method,
                    raw_terms=dict(term_values),
                    scaled_terms=scaled_terms,
                    aggregated_score=aggregated,
                )
            )
    return results


def compute_variant_term_means(
    *,
    candidate_metrics: Mapping[int, Mapping[str, Mapping[str, float]]],
    objective: Sequence[ObjectiveTerm],
    apply_direction: bool = True,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, int]]:
    """
    Compute per-variant mean objective-term values over instances.

    Returns:
      - variant -> term_name -> mean_value
      - variant -> contributing_instance_count
    """
    term_names = [term.name for term in objective]
    sums: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, int] = {}

    for _, variants in candidate_metrics.items():
        for variant, metrics in variants.items():
            term_values: Dict[str, float] = {}
            for term in objective:
                value = fetch_metric(metrics, term.metric_key)
                if value is None:
                    term_values = {}
                    break
                term_values[term.name] = term.apply_direction(value) if apply_direction else float(value)
            if not term_values:
                continue

            sums.setdefault(variant, {name: 0.0 for name in term_names})
            counts[variant] = counts.get(variant, 0) + 1
            for name in term_names:
                sums[variant][name] += term_values[name]

    means: Dict[str, Dict[str, float]] = {}
    for variant, term_sums in sums.items():
        count = counts.get(variant, 0)
        if count <= 0:
            continue
        means[variant] = {name: term_sums[name] / count for name in term_names}
    return means, counts


def _fit_trial_scalers_from_history(
    *,
    trial_history: Sequence[str],
    variant_term_means: Mapping[str, Mapping[str, float]],
    term_names: Sequence[str],
    scaling: str,
) -> Dict[str, Tuple[float, float]]:
    term_scalers: Dict[str, Tuple[float, float]] = {}
    for name in term_names:
        series: List[float] = []
        for variant in trial_history:
            values = variant_term_means.get(variant)
            if values is None:
                continue
            value = values.get(name)
            if value is None:
                continue
            series.append(float(value))

        if not series:
            term_scalers[name] = (0.0, 0.0)
            continue

        if scaling == "Std":
            mean = sum(series) / len(series)
            var = sum((v - mean) ** 2 for v in series) / len(series)
            std = math.sqrt(var)
            term_scalers[name] = (mean, std)
        else:
            term_scalers[name] = (min(series), max(series))
    return term_scalers


def _scale_term(value: float, *, scaling: str, params: Tuple[float, float]) -> float:
    a, b = params
    if scaling == "Std":
        mean, std = a, b
        if std == 0.0:
            return 0.0
        return (value - mean) / std
    lo, hi = a, b
    if hi == lo:
        return 0.0
    return (value - lo) / (hi - lo)


def _score_variant_mean_terms(
    *,
    variant: str,
    variant_term_means: Mapping[str, Mapping[str, float]],
    term_scalers: Mapping[str, Tuple[float, float]],
    objective: Sequence[ObjectiveTerm],
    scaling: str,
) -> Optional[float]:
    values = variant_term_means.get(variant)
    if values is None:
        return None
    scaled_terms: Dict[str, float] = {}
    for term in objective:
        raw_value = values.get(term.name)
        if raw_value is None:
            return None
        params = term_scalers.get(term.name, (0.0, 0.0))
        scaled_terms[term.name] = _scale_term(float(raw_value), scaling=scaling, params=params)
    weight_map = {term.name: term.weight for term in objective}
    return scalarize_terms(scaled_terms, weights=weight_map)


def _trial_objective_value(
    *,
    history: Sequence[str],
    candidate: str,
    variant_term_means: Mapping[str, Mapping[str, float]],
    objective: Sequence[ObjectiveTerm],
    scaling: str,
) -> Optional[float]:
    term_names = [term.name for term in objective]
    augmented = list(history) + [candidate]
    term_scalers = _fit_trial_scalers_from_history(
        trial_history=augmented,
        variant_term_means=variant_term_means,
        term_names=term_names,
        scaling=scaling,
    )
    return _score_variant_mean_terms(
        variant=candidate,
        variant_term_means=variant_term_means,
        term_scalers=term_scalers,
        objective=objective,
        scaling=scaling,
    )


def select_best_variants(
    scores: Sequence[CandidateScore],
    *,
    methods: Sequence[str],
) -> Dict[str, Tuple[str, float]]:
    per_variant: Dict[str, List[float]] = {}
    variant_method: Dict[str, str] = {}
    for score in scores:
        per_variant.setdefault(score.method_variant, []).append(score.aggregated_score)
        variant_method[score.method_variant] = score.method

    per_method_best: Dict[str, Tuple[str, float]] = {}
    for variant, values in per_variant.items():
        method = variant_method.get(variant)
        if method not in methods:
            continue
        mean_score = sum(values) / len(values)
        current = per_method_best.get(method)
        if current is None or mean_score > current[1]:
            per_method_best[method] = (variant, mean_score)
    return per_method_best


def compute_mean_score_by_variant(
    scores: Sequence[CandidateScore],
) -> Dict[str, Tuple[str, float]]:
    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    variant_method: Dict[str, str] = {}
    for score in scores:
        totals[score.method_variant] = totals.get(score.method_variant, 0.0) + score.aggregated_score
        counts[score.method_variant] = counts.get(score.method_variant, 0) + 1
        variant_method[score.method_variant] = score.method
    means: Dict[str, Tuple[str, float]] = {}
    for variant, total in totals.items():
        count = counts.get(variant, 0)
        if count:
            means[variant] = (variant_method.get(variant, "unknown"), total / count)
    return means
