from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .autoxai_config import load_default_variants, load_explainer_grid
from .autoxai_data import build_candidate_metrics, load_method_instances
from .autoxai_evaluation import (
    compare_against_pair_labels,
    evaluate_topk_against_pair_labels,
    load_hc_xai_splits,
)
from .autoxai_hpo import run_hpo, schedule_autoxai_trials
from .autoxai_scoring import (
    _fit_trial_scalers_from_history,
    _score_variant_mean_terms,
    compute_mean_score_by_variant,
    compute_scores,
    compute_variant_term_means,
)
from .autoxai_types import HPOResult, HPOTrial, ObjectiveTerm


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
    hc_xai_split_json: Optional[Path] = None,
    split_set: str = "test",
    top_k: Iterable[int] = (3, 5),
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
            if method == "integrated_gradients":
                for steps in grid_params.get("ig_steps", []):
                    grid_variants.add(f"integrated_gradients__ig_steps-{steps}")
            if method == "causal_shap":
                for coalitions in grid_params.get("causal_shap_coalitions", []):
                    grid_variants.add(f"causal_shap__causal_shap_coalitions-{coalitions}")

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

    split_payload: Optional[Dict[str, Any]] = None
    if hc_xai_split_json is not None:
        split_payload = load_hc_xai_splits(hc_xai_split_json)
        if split_set == "train":
            instance_filter = set(split_payload["train_instances"])
        elif split_set == "test":
            instance_filter = set(split_payload["test_instances"])
        elif split_set == "all":
            instance_filter = set(split_payload["train_instances"]) | set(split_payload["test_instances"])
        else:
            raise ValueError("split_set must be 'train', 'test', or 'all'.")
        candidate_metrics = {idx: blob for idx, blob in candidate_metrics.items() if int(idx) in instance_filter}

    defaults = load_default_variants(explainers_config_path) if explainers_config_path is not None else {}

    hpo_results: Dict[str, HPOResult] = {}
    if scaling_scope == "trial":
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

    best_by_method: Dict[str, Tuple[str, float]] = {
        method: (result.best_variant, result.best_mean_score) for method, result in hpo_results.items()
    }
    method_ranking = sorted(best_by_method.items(), key=lambda item: item[1][1], reverse=True)

    comparison: Optional[Dict[str, Any]] = None
    if pair_labels_path is not None:
        comparison = compare_against_pair_labels(
            pair_labels_path=pair_labels_path, scores=scores, tie_breaker_seed=tie_breaker_seed
        )

    topk_eval: Optional[Dict[str, Any]] = None
    if pair_labels_path is not None:
        topk_eval = evaluate_topk_against_pair_labels(
            pair_labels_path=pair_labels_path,
            scores=scores,
            k_values=top_k,
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
                        {"method_variant": trial.method_variant, "mean_score": trial.mean_score}
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
        "top_k_evaluation": topk_eval,
        "hc_xai_split": (
            {
                "split_json": str(hc_xai_split_json),
                "split_set": split_set,
                "dataset": (split_payload or {}).get("dataset"),
                "model": (split_payload or {}).get("model"),
                "train_instances": (split_payload or {}).get("train_instances"),
                "test_instances": (split_payload or {}).get("test_instances"),
                "instances_scored": sorted({int(score.dataset_index) for score in scores}),
            }
            if hc_xai_split_json is not None
            else None
        ),
        "variant_hyperparameters": variant_hparams_by_method,
    }

