from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml

# A representation of an objective term.
@dataclass(frozen=True)
class ObjectiveTerm:
    name: str
    metric_key: str
    direction: str  # "max" or "min"
    weight: float = 1.0

    def apply_direction(self, value: float) -> float:
        if self.direction == "max":
            return value
        if self.direction == "min":
            return -value
        raise ValueError(f"Unknown direction: {self.direction!r} (expected 'max' or 'min').")


@dataclass(frozen=True)
class CandidateScore:
    dataset_index: int
    method_variant: str
    method: str
    raw_terms: Mapping[str, float]
    scaled_terms: Mapping[str, float]
    aggregated_score: float


@dataclass(frozen=True)
class HPOTrial:
    method: str
    method_variant: str
    mean_score: float


@dataclass(frozen=True)
class HPOResult:
    method: str
    mode: str
    seed: int
    epochs: int
    default_variant: Optional[str]
    default_mean_score: Optional[float]
    trials: Sequence[HPOTrial]
    best_variant: str
    best_mean_score: float


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _standard_scale(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    std = math.sqrt(var)
    if std == 0.0:
        return [0.0 for _ in values]
    return [(v - mean) / std for v in values]


def _minmax_scale(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi == lo:
        return [0.0 for _ in values]
    span = hi - lo
    return [(v - lo) / span for v in values]


def _scalarize_terms(
    term_values: Mapping[str, float],
    *,
    weights: Mapping[str, float],
) -> float:
    if not term_values:
        return float("-inf")
    denom = len(term_values)
    return sum(weights.get(name, 1.0) * value for name, value in term_values.items()) / denom


def _tie_break(seed: int, dataset_index: object, pair_1: str, pair_2: str) -> int:
    token = f"{seed}:{dataset_index}:{pair_1}:{pair_2}".encode("utf-8")
    digest = hashlib.sha256(token).digest()
    return digest[0] % 2


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

    def mean_metric(metrics: Mapping[str, float], keys: Sequence[str]) -> Optional[float]:
        collected: List[float] = []
        for key in keys:
            value = _safe_float(metrics.get(key))
            if value is not None:
                collected.append(value)
        if not collected:
            return None
        return sum(collected) / len(collected)

    return {
        "compactness": lambda metrics: mean_metric(metrics, compactness_keys),
        "contrastivity": lambda metrics: _safe_float(metrics.get("contrastivity")),
        "stability": lambda metrics: _safe_float(metrics.get("relative_input_stability")),
        "faithfulness": lambda metrics: _safe_float(metrics.get("correctness")),
        "completeness": lambda metrics: _safe_float(metrics.get("completeness_score")),
        "consistency": lambda metrics: _safe_float(metrics.get("consistency")),
    }


def fetch_metric(metrics: Mapping[str, float], metric_key: str) -> Optional[float]:
    fetcher = _metric_fetchers().get(metric_key)
    if fetcher is not None:
        return fetcher(metrics)
    return _safe_float(metrics.get(metric_key))


def default_objective_terms() -> List[ObjectiveTerm]:
    """
    A practical AutoXAI-like objective for HC-XAI's per-instance metrics.

    Notes:
    - AutoXAI maximizes -infidelity and -robustness losses.
    - HC-XAI exposes per-instance 'infidelity' and 'relative_input_stability' (lower is better).
    - For conciseness we use compactness_effective_features (higher is more compact).
    """
    return [
        ObjectiveTerm(name="robustness", metric_key="relative_input_stability", direction="min", weight=1.0),
        ObjectiveTerm(name="fidelity", metric_key="infidelity", direction="min", weight=2.0),
        ObjectiveTerm(name="conciseness", metric_key="compactness_effective_features", direction="max", weight=0.5),
    ]


def persona_objective_terms(persona: str) -> List[ObjectiveTerm]:
    """
    Persona-aligned objectives using the same metric groupings as HC-XAI's pair-label generator.

    See `hc-xai/candidates_pair_ranker.py` for the priority definitions:
      - layperson: compactness -> contrastivity -> stability
      - regulator: faithfulness -> completeness -> consistency -> compactness
    """
    if persona == "layperson":
        return [
            ObjectiveTerm(name="compactness", metric_key="compactness", direction="max", weight=1.0),
            ObjectiveTerm(name="contrastivity", metric_key="contrastivity", direction="max", weight=1.0),
            ObjectiveTerm(name="stability", metric_key="stability", direction="min", weight=1.0),
        ]
    if persona == "regulator":
        return [
            ObjectiveTerm(name="faithfulness", metric_key="faithfulness", direction="max", weight=1.0),
            ObjectiveTerm(name="completeness", metric_key="completeness", direction="max", weight=1.0),
            ObjectiveTerm(name="consistency", metric_key="consistency", direction="max", weight=1.0),
            ObjectiveTerm(name="compactness", metric_key="compactness", direction="max", weight=1.0),
        ]
    if persona == "autoxai":
        return default_objective_terms()
    raise ValueError(f"Unknown persona: {persona!r}")


def load_explainer_grid(config_path: Path) -> Dict[str, Dict[str, List[object]]]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    explainers = payload.get("explainers") if isinstance(payload, dict) else None
    if not isinstance(explainers, dict):
        raise ValueError(f"Unexpected grid format in {config_path}.")
    parsed: Dict[str, Dict[str, List[object]]] = {}
    for method, grid in explainers.items():
        if not isinstance(grid, dict):
            continue
        parsed[method] = {key: list(values) for key, values in grid.items() if isinstance(values, list)}
    return parsed


def load_default_variants(explainers_config_path: Path) -> Dict[str, str]:
    """
    Extract default method_variant identifiers from `src/configs/explainers.yml`.

    This matches the naming scheme used in HC-XAI metric artifacts, e.g.:
      - lime__lime_kernel_width-2.0__lime_num_samples-100
      - shap__background_sample_size-100
    """
    payload = yaml.safe_load(explainers_config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected explainers.yml format in {explainers_config_path}.")

    defaults: Dict[str, str] = {}
    lime = payload.get("lime")
    if isinstance(lime, dict):
        params = ((lime.get("params") or {}).get("experiment") or {}).get("explanation") if isinstance(lime.get("params"), dict) else None
        if isinstance(params, dict):
            kw = params.get("lime_kernel_width")
            ns = params.get("lime_num_samples")
            if kw is not None and ns is not None:
                defaults["lime"] = f"lime__lime_kernel_width-{kw}__lime_num_samples-{ns}"

    shap_cfg = payload.get("shap")
    if isinstance(shap_cfg, dict):
        params = ((shap_cfg.get("params") or {}).get("experiment") or {}).get("explanation") if isinstance(shap_cfg.get("params"), dict) else None
        if isinstance(params, dict):
            bg = params.get("background_sample_size")
            if bg is not None:
                defaults["shap"] = f"shap__background_sample_size-{bg}"

    return defaults


def _method_metrics_path(results_root: Path, dataset: str, model: str, method: str) -> Path:
    return results_root / "metrics_results" / dataset / model / f"{method}_metrics.json"


def load_method_instances(
    *,
    results_root: Path,
    dataset: str,
    model: str,
    method: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Returns:
      - instances list (raw JSON)
      - map method_variant -> hyperparameters dict (best-effort)
    """
    path = _method_metrics_path(results_root, dataset, model, method)
    payload = _load_json(path)
    instances = payload.get("instances") or []
    if not isinstance(instances, list):
        raise ValueError(f"Expected 'instances' list in {path}.")
    variant_hparams: Dict[str, Dict[str, Any]] = {}
    for entry in instances:
        if not isinstance(entry, dict):
            continue
        variant = entry.get("method_variant")
        if not isinstance(variant, str):
            continue
        meta = entry.get("explanation_metadata") or {}
        if isinstance(meta, dict):
            hparams = meta.get("hyperparameters")
            if isinstance(hparams, dict) and variant not in variant_hparams:
                variant_hparams[variant] = dict(hparams)
    return instances, variant_hparams


def build_candidate_metrics(
    *,
    instances_by_method: Mapping[str, List[Dict[str, Any]]],
    allowed_variants: Optional[set[str]] = None,
) -> Tuple[Dict[int, Dict[str, Dict[str, float]]], Dict[str, str]]:
    """
    Returns:
      - dataset_index -> method_variant -> metric_key -> value
      - method_variant -> method
    """
    per_instance: Dict[int, Dict[str, Dict[str, float]]] = {}
    variant_to_method: Dict[str, str] = {}
    for method, instances in instances_by_method.items():
        for entry in instances:
            if not isinstance(entry, dict):
                continue
            dataset_index = entry.get("dataset_index")
            if not isinstance(dataset_index, int):
                continue
            variant = entry.get("method_variant")
            if not isinstance(variant, str):
                continue
            if allowed_variants is not None and variant not in allowed_variants:
                continue
            metrics = entry.get("metrics") or {}
            if not isinstance(metrics, dict):
                continue
            per_instance.setdefault(dataset_index, {})
            metric_values: Dict[str, float] = {}
            for key, value in metrics.items():
                parsed = _safe_float(value)
                if parsed is not None:
                    metric_values[str(key)] = parsed
            per_instance[dataset_index][variant] = metric_values
            variant_to_method[variant] = method
    return per_instance, variant_to_method


def compute_scores(
    *,
    candidate_metrics: Mapping[int, Mapping[str, Mapping[str, float]]],
    variant_to_method: Mapping[str, str],
    objective: Sequence[ObjectiveTerm],
    scaling: str = "Std",
    scaling_scope: str = "trial",
    trial_history_for_scaling: Optional[Sequence[str]] = None,
) -> List[CandidateScore]:
    """
    Produces AutoXAI-style standardized + weighted aggregated scores.

    scaling: "Std" or "MinMax" (mirrors AutoXAI).

    scaling_scope:
      - "trial": scale each objective term over per-variant means (matches AutoXAI's trial-history scaling).
      - "global": scale each objective term over all (instance, variant) candidates.
      - "instance": scale each objective term within each instance across variants.

    trial_history_for_scaling:
      When scaling_scope=\"trial\", optionally provide the sequence of evaluated trial variants to fit
      the scalers on (supports duplicates). This matches AutoXAI's sequential scaling behaviour for
      random/BO runs where scaling only uses the evaluated trial history.
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
                term_values[term.name] = term.apply_direction(value)
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
        # mode == "MinMax"
        lo, hi = a, b
        if hi == lo:
            return 0.0
        return (value - lo) / (hi - lo)

    if scaling_scope == "trial":
        # Fit each term's scaler on per-variant means (trial-history population).
        per_variant_values: Dict[str, Dict[str, List[float]]] = {}
        for _, variant, _, term_values in raw:
            bucket = per_variant_values.setdefault(variant, {name: [] for name in term_names})
            for name in term_names:
                bucket[name].append(term_values[name])

        per_variant_means: Dict[str, Dict[str, float]] = {}
        for variant, buckets in per_variant_values.items():
            if any(not buckets[name] for name in term_names):
                continue
            per_variant_means[variant] = {
                name: (sum(buckets[name]) / len(buckets[name]))
                for name in term_names
            }

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
            scaled_terms = {
                name: _scale_value(term_values[name], params=term_scalers[name], mode=scaling)
                for name in term_names
            }
            aggregated = _scalarize_terms(scaled_terms, weights=weight_map)
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
            scaled_series[name] = _standard_scale(values) if scaling == "Std" else _minmax_scale(values)

        scores: List[CandidateScore] = []
        for idx, (dataset_index, variant, method, term_values) in enumerate(raw):
            scaled_terms = {name: scaled_series[name][idx] for name in term_names}
            aggregated = _scalarize_terms(scaled_terms, weights=weight_map)
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

    # scaling_scope == "instance"
    grouped: Dict[int, List[Tuple[str, str, Dict[str, float]]]] = {}
    for dataset_index, variant, method, term_values in raw:
        grouped.setdefault(dataset_index, []).append((variant, method, term_values))

    results: List[CandidateScore] = []
    for dataset_index, entries in grouped.items():
        per_term_series: Dict[str, List[float]] = {name: [] for name in term_names}
        for _, _, term_values in entries:
            for name in term_names:
                per_term_series[name].append(term_values[name])
        scaled_series: Dict[str, List[float]] = {}
        for name, values in per_term_series.items():
            scaled_series[name] = _standard_scale(values) if scaling == "Std" else _minmax_scale(values)
        for idx, (variant, method, term_values) in enumerate(entries):
            scaled_terms = {name: scaled_series[name][idx] for name in term_names}
            aggregated = _scalarize_terms(scaled_terms, weights=weight_map)
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
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, int]]:
    """
    Compute per-variant mean objective-term values over instances.

    This mirrors AutoXAI's evaluation measures, which return dataset-level scores per trial.
    Values are direction-adjusted (so higher is always better).

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
                term_values[term.name] = term.apply_direction(value)
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
    return _scalarize_terms(scaled_terms, weights=weight_map)


def _trial_objective_value(
    *,
    history: Sequence[str],
    candidate: str,
    variant_term_means: Mapping[str, Mapping[str, float]],
    objective: Sequence[ObjectiveTerm],
    scaling: str,
) -> Optional[float]:
    """
    Compute the aggregated score for a candidate trial, using scalers fit on (history + candidate).

    This matches AutoXAI's pattern of scaling over the score history observed so far.
    """
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
    """
    Returns method -> (best_variant, best_mean_score) based on mean score over instances.
    """
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
    """
    Returns method_variant -> (method, mean_score).
    """
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


def run_hpo(
    *,
    method: str,
    variant_means: Mapping[str, Tuple[str, float]],
    variants: Sequence[str],
    mode: str,
    epochs: int,
    seed: int,
    default_variant: Optional[str],
) -> HPOResult:
    """
    Hyperparameter selection over precomputed variants.

    This mirrors AutoXAI's *outer loop* (evaluate default, then search) but uses cached
    HC-XAI artifacts instead of rerunning explainers during optimization.

    mode:
      - "grid": evaluate all variants (deterministic; effectively exhaustive search)
      - "random": sample `epochs` variants uniformly at random (with replacement)
      - "gp": optional Bayesian optimization using scikit-optimize over a categorical space
    """
    if mode not in {"grid", "random", "gp"}:
        raise ValueError("hpo mode must be one of: grid, random, gp.")
    if epochs <= 0:
        raise ValueError("epochs must be positive.")

    method_variants = [v for v in variants if variant_means.get(v, ("", 0.0))[0] == method]
    if not method_variants:
        raise ValueError(f"No variants found for method {method!r}.")

    default_mean: Optional[float] = None
    if default_variant is not None:
        record = variant_means.get(default_variant)
        if record is not None and record[0] == method:
            default_mean = record[1]

    trials: List[HPOTrial] = []
    best_variant = method_variants[0]
    best_mean = variant_means.get(best_variant, (method, float("-inf")))[1]

    if mode == "grid":
        for variant in method_variants:
            mean_score = variant_means[variant][1]
            trials.append(HPOTrial(method=method, method_variant=variant, mean_score=mean_score))
            if mean_score > best_mean:
                best_variant = variant
                best_mean = mean_score

    elif mode == "random":
        import random

        rng = random.Random(seed)
        for _ in range(epochs):
            variant = rng.choice(method_variants)
            mean_score = variant_means[variant][1]
            trials.append(HPOTrial(method=method, method_variant=variant, mean_score=mean_score))
            if mean_score > best_mean:
                best_variant = variant
                best_mean = mean_score

    else:  # mode == "gp"
        try:
            from skopt import gp_minimize  # type: ignore
            from skopt.space import Categorical  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "hpo=gp requires scikit-optimize. Install it with `pip install scikit-optimize`."
            ) from exc

        def objective_fn(params: List[str]) -> float:
            variant = params[0]
            mean_score = variant_means[variant][1]
            trials.append(HPOTrial(method=method, method_variant=variant, mean_score=mean_score))
            return -mean_score

        space = [Categorical(method_variants, name="variant")]
        result = gp_minimize(
            objective_fn,
            space,
            n_calls=epochs,
            random_state=seed,
            n_initial_points=min(5, epochs),
        )
        # gp_minimize minimizes, so the best is the smallest function value.
        best_variant = str(result.x[0])
        best_mean = variant_means[best_variant][1]

    return HPOResult(
        method=method,
        mode=mode,
        seed=seed,
        epochs=epochs,
        default_variant=default_variant,
        default_mean_score=default_mean,
        trials=trials,
        best_variant=best_variant,
        best_mean_score=best_mean,
    )


def schedule_autoxai_trials(
    *,
    method: str,
    method_variants: Sequence[str],
    mode: str,
    epochs: int,
    seed: int,
    default_variant: Optional[str],
    variant_term_means: Mapping[str, Mapping[str, float]],
    objective: Sequence[ObjectiveTerm],
    scaling: str,
    global_history: List[str],
) -> List[str]:
    """
    Create an AutoXAI-like trial schedule for one method while updating a shared global history.

    For random/gp, trials are selected sequentially and the objective value for a candidate is
    computed using scalers fit on the score history observed so far (global_history).
    """
    if mode not in {"grid", "random", "gp"}:
        raise ValueError("hpo mode must be one of: grid, random, gp.")
    if mode in {"random", "gp"} and epochs <= 0:
        raise ValueError("epochs must be positive.")

    available = [v for v in method_variants if v in variant_term_means]
    if not available:
        raise ValueError(f"No variants with term means found for method {method!r}.")

    trials: List[str] = []
    seen: set[str] = set()

    def add_trial(variant: str) -> None:
        trials.append(variant)
        global_history.append(variant)
        seen.add(variant)

    # Evaluate default first (outer-loop style), if available.
    if default_variant is not None and default_variant in available:
        add_trial(default_variant)
        if mode in {"random", "gp"} and len(trials) >= epochs:
            return trials

    if mode == "grid":
        for variant in sorted(available):
            if variant == default_variant:
                continue
            add_trial(variant)
        return trials

    if mode == "random":
        import random

        rng = random.Random(seed)
        while len(trials) < epochs:
            remaining = [v for v in available if v not in seen]
            if remaining:
                add_trial(rng.choice(remaining))
            else:
                # Fall back to sampling with replacement when epochs exceed the discrete space.
                add_trial(rng.choice(available))
        return trials

    # mode == "gp"
    try:
        from skopt import Optimizer  # type: ignore
        from skopt.space import Categorical  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "hpo=gp requires scikit-optimize. Install it with `pip install scikit-optimize`."
        ) from exc

    # Use an iterative ask/tell loop so we can avoid duplicate trials.
    opt = Optimizer(
        [Categorical(list(available), name="variant")],
        random_state=seed,
        n_initial_points=min(5, max(1, epochs)),
    )

    # Prime with default if it was evaluated.
    if trials:
        variant = trials[-1]
        score = _trial_objective_value(
            history=global_history[:-1],
            candidate=variant,
            variant_term_means=variant_term_means,
            objective=objective,
            scaling=scaling,
        )
        opt.tell([variant], -(score if score is not None else float("-inf")))

    while len(trials) < epochs:
        # ask() may propose duplicates; skip them deterministically.
        variant = None
        for _ in range(len(available) + 5):
            proposed = opt.ask()[0]
            proposed = str(proposed)
            if proposed not in seen:
                variant = proposed
                break
        if variant is None:
            # No unseen points remain; fall back to sampling with replacement.
            variant = str(opt.ask()[0])

        score = _trial_objective_value(
            history=global_history,
            candidate=variant,
            variant_term_means=variant_term_means,
            objective=objective,
            scaling=scaling,
        )
        opt.tell([variant], -(score if score is not None else float("-inf")))
        add_trial(variant)

    return trials


def compare_against_pair_labels(
    *,
    pair_labels_path: Path,
    scores: Sequence[CandidateScore],
    tie_breaker_seed: int = 13,
) -> Dict[str, Any]:
    """
    Compare per-instance candidate scores against HC-XAI pairwise labels.

    Pair label convention matches `hc-xai/candidates_pair_ranker.py`:
      - label 0 means pair_1 is preferred
      - label 1 means pair_2 is preferred
    """
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "pandas is required to read the parquet pair label files. "
            "Install the hc-xai dependencies (or convert the parquet to CSV)."
        ) from exc

    df = pd.read_parquet(pair_labels_path)
    needed = {"dataset_index", "pair_1", "pair_2", "label"}
    if not needed.issubset(set(df.columns)):
        raise ValueError(f"Pair labels file missing required columns: {sorted(needed)}")

    score_lookup: Dict[Tuple[int, str], float] = {
        (score.dataset_index, score.method_variant): score.aggregated_score for score in scores
    }

    total = 0
    correct = 0
    skipped_missing = 0
    wins: Dict[str, int] = {}
    losses: Dict[str, int] = {}
    for row in df.itertuples(index=False):
        dataset_index = getattr(row, "dataset_index")
        pair_1 = getattr(row, "pair_1")
        pair_2 = getattr(row, "pair_2")
        label = getattr(row, "label")

        key_a = (int(dataset_index), str(pair_1))
        key_b = (int(dataset_index), str(pair_2))
        score_a = score_lookup.get(key_a)
        score_b = score_lookup.get(key_b)
        if score_a is None or score_b is None:
            skipped_missing += 1
            continue

        if int(label) == 0:
            winner = str(pair_1)
            loser = str(pair_2)
        else:
            winner = str(pair_2)
            loser = str(pair_1)
        wins[winner] = wins.get(winner, 0) + 1
        losses[loser] = losses.get(loser, 0) + 1

        if score_a == score_b:
            pred = _tie_break(tie_breaker_seed, dataset_index, str(pair_1), str(pair_2))
        else:
            pred = 0 if score_a > score_b else 1

        total += 1
        if int(label) == int(pred):
            correct += 1

    accuracy = correct / total if total else 0.0
    variant_ranking = []
    for variant in sorted(set(wins) | set(losses)):
        w = wins.get(variant, 0)
        l = losses.get(variant, 0)
        n = w + l
        rate = (w / n) if n else 0.0
        variant_ranking.append(
            {"method_variant": variant, "wins": w, "losses": l, "games": n, "win_rate": rate}
        )
    variant_ranking.sort(key=lambda item: (item["win_rate"], item["games"]), reverse=True)
    return {
        "pairs_evaluated": total,
        "pairs_correct": correct,
        "pairwise_accuracy": accuracy,
        "pairs_skipped_missing_candidates": skipped_missing,
        "hc_xai_pair_label_ranking": variant_ranking,
    }


def run(
    *,
    results_root: Path,
    dataset: str,
    model: str,
    methods: Sequence[str],
    objective: Sequence[ObjectiveTerm],
    scaling: str,
    scaling_scope: str,
    grid_config_path: Optional[Path] = None,
    explainers_config_path: Optional[Path] = None,
    hpo_mode: str = "grid",
    hpo_epochs: int = 20,
    hpo_seed: int = 0,
    pair_labels_path: Optional[Path] = None,
    tie_breaker_seed: int = 13,
) -> Dict[str, Any]:
    grid_variants: Optional[set[str]] = None
    if grid_config_path is not None:
        grid = load_explainer_grid(grid_config_path)
        allowed_methods = set(methods)
        grid_variants = set()
        for method, grid_params in grid.items():
            if method not in allowed_methods:
                continue
            if method == "lime":
                for num_samples in grid_params.get("lime_num_samples", []):
                    for kernel_width in grid_params.get("lime_kernel_width", []):
                        grid_variants.add(
                            f"lime__lime_kernel_width-{kernel_width}__lime_num_samples-{num_samples}"
                        )
            if method == "shap":
                for size in grid_params.get("background_sample_size", []):
                    grid_variants.add(f"shap__background_sample_size-{size}")

    instances_by_method: Dict[str, List[Dict[str, Any]]] = {}
    variant_hparams_by_method: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for method in methods:
        instances, variant_hparams = load_method_instances(
            results_root=results_root, dataset=dataset, model=model, method=method
        )
        instances_by_method[method] = instances
        variant_hparams_by_method[method] = variant_hparams

    candidate_metrics, variant_to_method = build_candidate_metrics(
        instances_by_method=instances_by_method, allowed_variants=grid_variants
    )
    defaults = (
        load_default_variants(explainers_config_path)
        if explainers_config_path is not None
        else {}
    )

    hpo_results: Dict[str, HPOResult] = {}
    if scaling_scope == "trial":
        # AutoXAI-like: scale over evaluated trial history (random/gp) or full set (grid).
        variant_term_means, _variant_counts = compute_variant_term_means(
            candidate_metrics=candidate_metrics, objective=objective
        )
        global_history: List[str] = []
        trials_by_method: Dict[str, List[str]] = {}

        for method in methods:
            method_variants = sorted(
                variant
                for variant, mapped_method in variant_to_method.items()
                if mapped_method == method and variant in variant_term_means
            )
            trials_by_method[method] = schedule_autoxai_trials(
                method=method,
                method_variants=method_variants,
                mode=hpo_mode,
                epochs=hpo_epochs,
                seed=hpo_seed,
                default_variant=defaults.get(method),
                variant_term_means=variant_term_means,
                objective=objective,
                scaling=scaling,
                global_history=global_history,
            )

        evaluated_variants = set(global_history)
        # Restrict per-instance candidate scores to evaluated variants (AutoXAI only ranks tried trials).
        candidate_metrics_scored, variant_to_method_scored = build_candidate_metrics(
            instances_by_method=instances_by_method, allowed_variants=evaluated_variants
        )
        scores = compute_scores(
            candidate_metrics=candidate_metrics_scored,
            variant_to_method=variant_to_method_scored,
            objective=objective,
            scaling=scaling,
            scaling_scope="trial",
            trial_history_for_scaling=global_history,
        )

        term_names = [term.name for term in objective]
        final_scalers = _fit_trial_scalers_from_history(
            trial_history=global_history,
            variant_term_means=variant_term_means,
            term_names=term_names,
            scaling=scaling,
        )

        score_lookup: Dict[str, float] = {}
        for variant in evaluated_variants | set(defaults.values()):
            maybe = _score_variant_mean_terms(
                variant=variant,
                variant_term_means=variant_term_means,
                term_scalers=final_scalers,
                objective=objective,
                scaling=scaling,
            )
            if maybe is not None:
                score_lookup[variant] = maybe

        for method in methods:
            default_variant = defaults.get(method)
            default_score = score_lookup.get(default_variant) if default_variant else None
            method_trials = trials_by_method.get(method, [])
            trials: List[HPOTrial] = []
            for variant in method_trials:
                trials.append(
                    HPOTrial(
                        method=method,
                        method_variant=variant,
                        mean_score=score_lookup.get(variant, float("-inf")),
                    )
                )
            if not trials:
                raise ValueError(f"No HPO trials scheduled for method {method!r}.")
            best_trial = max(trials, key=lambda trial: trial.mean_score)
            epochs_used = len(method_trials) if hpo_mode == "grid" else hpo_epochs
            hpo_results[method] = HPOResult(
                method=method,
                mode=hpo_mode,
                seed=hpo_seed,
                epochs=epochs_used,
                default_variant=default_variant,
                default_mean_score=default_score,
                trials=trials,
                best_variant=best_trial.method_variant,
                best_mean_score=best_trial.mean_score,
            )

    else:
        scores = compute_scores(
            candidate_metrics=candidate_metrics,
            variant_to_method=variant_to_method,
            objective=objective,
            scaling=scaling,
            scaling_scope=scaling_scope,
        )
        variant_means = compute_mean_score_by_variant(scores)
        for method in methods:
            hpo_results[method] = run_hpo(
                method=method,
                variant_means=variant_means,
                variants=sorted(variant_means.keys()),
                mode=hpo_mode,
                epochs=hpo_epochs,
                seed=hpo_seed,
                default_variant=defaults.get(method),
            )

    best_by_method = {
        method: (result.best_variant, result.best_mean_score) for method, result in hpo_results.items()
    }
    method_ranking = sorted(best_by_method.items(), key=lambda item: item[1][1], reverse=True)

    comparison: Optional[Dict[str, Any]] = None
    if pair_labels_path is not None:
        comparison = compare_against_pair_labels(
            pair_labels_path=pair_labels_path, scores=scores, tie_breaker_seed=tie_breaker_seed
        )

    return {
        "dataset": dataset,
        "model": model,
        "results_root": str(results_root),
        "methods": list(methods),
        "objective": [
            {"name": term.name, "metric_key": term.metric_key, "direction": term.direction, "weight": term.weight}
            for term in objective
        ],
        "scaling": scaling,
        "scaling_scope": scaling_scope,
        "grid_filter": str(grid_config_path) if grid_config_path else None,
        "num_scored_candidates": len(scores),
        "hpo": {
            "mode": hpo_mode,
            "epochs": hpo_epochs,
            "seed": hpo_seed,
            "defaults": defaults,
            "results": {
                method: {
                    "default_variant": result.default_variant,
                    "default_mean_score": result.default_mean_score,
                    "best_variant": result.best_variant,
                    "best_mean_score": result.best_mean_score,
                    "trials": [
                        {
                            "method_variant": trial.method_variant,
                            "mean_score": trial.mean_score,
                        }
                        for trial in result.trials
                    ],
                }
                for method, result in hpo_results.items()
            },
        },
        "best_variant_by_method": {
            method: {"method_variant": variant, "mean_score": mean_score}
            for method, (variant, mean_score) in best_by_method.items()
        },
        "method_ranking": [
            {"method": method, "method_variant": variant, "mean_score": mean_score}
            for method, (variant, mean_score) in method_ranking
        ],
        "comparison": comparison,
        "variant_hyperparameters": variant_hparams_by_method,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AutoXAI-style baseline ranking over cached HC-XAI metric artifacts.",
    )
    parser.add_argument("--results-root", type=Path, required=True, help="HC-XAI run directory (contains metrics_results/).")
    parser.add_argument("--dataset", required=True, help="Dataset key (e.g. open_compas).")
    parser.add_argument("--model", required=True, help="Model key (e.g. mlp_classifier).")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["lime", "shap"],
        help="Which methods to include (default: lime shap).",
    )
    parser.add_argument(
        "--objective",
        nargs="*",
        default=None,
        help="Objective terms: name:direction:metric_key[:weight]. If omitted, uses a robust default.",
    )
    parser.add_argument(
        "--persona",
        choices=["autoxai", "layperson", "regulator"],
        default="autoxai",
        help="Select a preset objective aligned with an HC-XAI persona (default: autoxai).",
    )
    parser.add_argument(
        "--scaling",
        choices=["Std", "MinMax"],
        default="Std",
        help="Scaling used before scalarization (default: Std).",
    )
    parser.add_argument(
        "--scaling-scope",
        choices=["trial", "global", "instance"],
        default="trial",
        help=(
            "Scaling population for objective terms (default: trial). "
            "trial matches AutoXAI by scaling over per-variant means; "
            "global scales over all (instance,variant) candidates; "
            "instance scales within each instance across variants."
        ),
    )
    parser.add_argument(
        "--grid-config",
        type=Path,
        default=Path("src/configs/explainer_hyperparameters.yml"),
        help="Filter to variants defined in this hyperparameter grid (default: src/configs/explainer_hyperparameters.yml).",
    )
    parser.add_argument(
        "--explainer-config",
        type=Path,
        default=Path("src/configs/explainers.yml"),
        help="Explainer defaults file used to report the default variant score (default: src/configs/explainers.yml).",
    )
    parser.add_argument(
        "--hpo",
        choices=["grid", "random", "gp"],
        default="grid",
        help="Hyperparameter optimization mode over cached variants (default: grid).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of HPO trials when --hpo is random/gp (default: 20).",
    )
    parser.add_argument(
        "--hpo-seed",
        type=int,
        default=0,
        help="Random seed for HPO (default: 0).",
    )
    parser.add_argument(
        "--pair-labels",
        type=Path,
        default=None,
        help="Optional parquet file with HC-XAI pair labels to score against.",
    )
    parser.add_argument(
        "--tie-breaker-seed",
        type=int,
        default=13,
        help="Tie breaker seed used for pairwise comparisons (default: 13).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional path to write a JSON report. "
            "When omitted, writes to <results-root>/autoxai_baseline__<dataset>__<model>__<persona>.json."
        ),
    )
    parser.add_argument(
        "--require-write",
        action="store_true",
        help="Fail if the JSON report cannot be written (default: print JSON to stdout and continue).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.objective:
        objective = parse_objective_terms(args.objective)
    else:
        objective = persona_objective_terms(args.persona)

    grid_path = args.grid_config if args.grid_config and args.grid_config.exists() else None
    explainer_path = (
        args.explainer_config if args.explainer_config and args.explainer_config.exists() else None
    )
    report = run(
        results_root=args.results_root,
        dataset=args.dataset,
        model=args.model,
        methods=args.methods,
        objective=objective,
        scaling=args.scaling,
        scaling_scope=args.scaling_scope,
        grid_config_path=grid_path,
        explainers_config_path=explainer_path,
        hpo_mode=args.hpo,
        hpo_epochs=args.epochs,
        hpo_seed=args.hpo_seed,
        pair_labels_path=args.pair_labels,
        tie_breaker_seed=args.tie_breaker_seed,
    )

    payload = json.dumps(report, indent=2, sort_keys=True)
    output_path = args.output
    if output_path is None:
        output_path = args.results_root / f"autoxai_baseline__{args.dataset}__{args.model}__{args.persona}.json"
    wrote_report = False
    write_error: Optional[Exception] = None
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
        wrote_report = True
    except Exception as exc:  # pragma: no cover - depends on runtime FS permissions
        write_error = exc
        if args.require_write:
            raise

    # Keep stdout JSON for convenience; file write is best-effort unless --require-write.
    print(payload)
    if wrote_report:
        print(f"[autoxai-baseline] wrote report to {output_path}", file=sys.stderr)
    elif write_error is not None:
        print(
            f"[autoxai-baseline] WARNING: failed to write report to {output_path}: {write_error}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
