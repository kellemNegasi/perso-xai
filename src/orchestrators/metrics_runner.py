"""
Experiment runner that instantiates datasets/models/explainers and computes metrics.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from src.baseline.autoxai_param_optimizer import BayesRangeOptimizer
from src.validators import TabularDataValidator
from src.utils.hyperparameter_tuning import HyperparameterTuner
from src.utils.model_persistence import ModelPersistence

from .artifact_resolver import resolve_artifact_key, resolve_artifact_list
from .dataset_utils import resolve_tuning_subset, run_dataset_validation
from .metrics_utils import (
    evaluate_metrics_for_method,
    extract_metric_parameters,
    safe_scalar,
    value_at,
)
from .persistence import (
    checkpoint_explanations,
    ensure_dataset_metadata,
    load_cached_explanations,
    load_cached_metrics,
    load_completion_flag,
    write_completion_flag,
    write_metric_results as persist_metric_results,
)
from .utils import (
    DATASET_REGISTRY,
    EXPLAINER_REGISTRY,
    EXPLAINER_HPARAM_CFG,
    EXPERIMENT_CFG,
    MODEL_REGISTRY,
    VALIDATION_CFG,
    instantiate_dataset,
    instantiate_explainer,
    instantiate_metric,
    instantiate_model,
    metric_capabilities,
    to_serializable,
)
from .validation import validate_artifact_compatibility
from .autoxai_optimization import (
    aggregate_trial_metrics,
    build_candidate_scores_reports,
    build_method_label,
    compute_objective_term_values,
    parse_explainer_space,
    trial_history_objective_score,
)

LOGGER = logging.getLogger(__name__)
TABULAR_VALIDATOR = TabularDataValidator(VALIDATION_CFG.get("tabular", {}))


@dataclass
class MethodArtifact:
    explainer_key: str
    method_label: str
    detail_path: Path
    metrics_path: Optional[Path]
    reused: bool = False


def run_experiment(
    experiment_name: str,
    *,
    max_instances: Optional[int] = None,
    output_path: Optional[str | Path] = None,
    model_override: Optional[str] = None,
    tune_models: bool = False,
    use_tuned_params: bool = False,
    reuse_trained_models: bool = False,
    tuning_output_dir: Optional[str | Path] = None,
    model_store_dir: Optional[str | Path] = None,
    stop_after_training: bool = False,
    stop_after_explanations: bool = False,
    write_detailed_explanations: bool = False,
    detailed_output_dir: Optional[str | Path] = None,
    reuse_detailed_explanations: bool = False,
    write_metric_results: bool = False,
    metrics_output_dir: Optional[str | Path] = None,
    skip_existing_methods: bool = False,
    skip_if_output_exists: bool = False,
    return_summary_only: bool = False,
) -> Dict[str, Any]:
    """
    Execute a configured experiment (dataset/model/explainers/metrics).

    Parameters
    ----------
    experiment_name : str
        Key defined in ``configs/experiments.yml``.
    max_instances : int | None, optional
        Optional cap on the number of evaluation instances.
    output_path : str | Path | None, optional
        If provided, the serialized results are written to this path.
    tune_models : bool, optional
        Run hyperparameter tuning before fitting any missing models.
    use_tuned_params : bool, optional
        Reuse previously tuned parameters (if available) during instantiation.
    reuse_trained_models : bool, optional
        Load/supply persisted models stored under ``saved_models``.
    tuning_output_dir : str | Path | None, optional
        Custom directory for hyperparameter tuning artifacts.
    model_store_dir : str | Path | None, optional
        Directory for persisted trained models.
    stop_after_training : bool, optional
        Halt the pipeline after tuning/training (no explanations or metrics).
    stop_after_explanations : bool, optional
        Generate explanations but skip metric computation/evaluation.
    write_detailed_explanations : bool, optional
        Persist per-explainer instance-level outputs to disk.
    detailed_output_dir : str | Path | None, optional
        Base directory where detailed explanation JSON files are written.
    reuse_detailed_explanations : bool, optional
        Load cached detailed explanation files when present instead of recomputing.
    write_metric_results : bool, optional
        Persist per-method metric artifacts to disk.
    metrics_output_dir : str | Path | None, optional
        Base directory for structured metric outputs.
    skip_existing_methods : bool, optional
        Skip explainer runs when cached detailed and metric artifacts already exist.
    skip_if_output_exists : bool, optional
        Return the on-disk experiment result if output_path already exists.
    return_summary_only : bool, optional
        When True, return only a compact summary dict after persisting experiment artifacts.

    Returns
    -------
    Dict[str, Any]
        Nested experiment result or a compact summary depending on ``return_summary_only``.
    """
    if stop_after_training and stop_after_explanations:
        raise ValueError(
            "stop_after_training and stop_after_explanations cannot both be True."
        )

    LOGGER.info("=== Running experiment '%s' ===", experiment_name)
    exp_cfg = EXPERIMENT_CFG[experiment_name]
    logging_cfg = exp_cfg.get("logging", {}) or {}
    log_progress = bool(logging_cfg.get("progress"))

    dataset_name = resolve_artifact_key(exp_cfg.get("dataset"), "dataset")
    configured_models = resolve_artifact_list(exp_cfg.get("models"), "model")
    model_entry = resolve_artifact_key(
        exp_cfg.get("model"), "model", required=False
    )
    if model_override is not None:
        model_name = model_override
    elif configured_models:
        if len(configured_models) != 1:
            raise ValueError(
                "Experiment defines multiple models; provide model_override or use run_experiments."
            )
        model_name = configured_models[0]
    elif model_entry:
        model_name = model_entry
    else:
        raise ValueError(f"Experiment '{experiment_name}' is missing a model reference.")
    if skip_if_output_exists and output_path is not None:
        result_path = Path(output_path)
        if result_path.exists():
            LOGGER.info(
                "Skipping experiment '%s' (model=%s); found existing result at %s",
                experiment_name,
                model_name,
                result_path,
            )
            try:
                return json.loads(result_path.read_text(encoding="utf-8"))
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.warning(
                    "Failed to load cached experiment result from %s (%s); rerunning.",
                    result_path,
                    exc,
                )
    explainer_configs = exp_cfg.get("explainers")
    if not explainer_configs and exp_cfg.get("explainer"):
        explainer_configs = [exp_cfg["explainer"]]
    explainer_names = resolve_artifact_list(explainer_configs, "explainer")
    expl_hparam_grids = EXPLAINER_HPARAM_CFG.get("explainers") or {}
    expl_hpo_cfg = exp_cfg.get("explainer_hpo") or {}
    metric_names = exp_cfg.get("metrics", [])

    dataset_spec = DATASET_REGISTRY.get(dataset_name)
    model_spec = MODEL_REGISTRY.get(model_name)
    explainer_specs = [(name, EXPLAINER_REGISTRY.get(name)) for name in explainer_names]
    dataset_type = validate_artifact_compatibility(
        dataset=(dataset_name, dataset_spec),
        models=[(model_name, model_spec)],
        explainers=explainer_specs,
        scope=f"experiment '{experiment_name}'",
    )

    LOGGER.info("Loading dataset '%s' (type=%s)", dataset_name, dataset_type)
    dataset = instantiate_dataset(dataset_name, data_type=dataset_type)
    run_dataset_validation(
        dataset_name=dataset_name,
        dataset_type=dataset_type,
        dataset_spec=dataset_spec,
        dataset=dataset,
        experiment_name=experiment_name,
        validator=TABULAR_VALIDATOR,
    )
    feature_names = list(getattr(dataset, "feature_names", []) or [])

    def _finalize(
        *,
        instances_data: List[Dict[str, Any]],
        batch_metrics_data: Dict[str, Dict[str, float]],
        stage_completed: str,
        detailed_paths: Optional[Dict[str, str]] = None,
        metric_paths: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        # Minimal payload to avoid duplicating heavy per-instance content; artifacts hold full data.
        result_payload = {
            "experiment": experiment_name,
            "dataset": dataset_name,
            "model": model_name,
            "stage_completed": stage_completed,
            "counts": {
                "instances": len(instances_data),
                "base_explainers": len(explainer_names),
                "explainer_variants": int(total_variants),
            },
            "artifacts": {
                "detailed_explanations": detailed_paths or {},
                "per_method_metric_files": metric_paths or {},
                "batch_metrics": batch_metrics_data,
            },
        }
        if output_path is not None:
            path = Path(output_path)
            path.write_text(
                json.dumps(to_serializable(result_payload), indent=2),
                encoding="utf-8",
            )
            LOGGER.info("Wrote experiment result to %s", path)
        LOGGER.info(
            "Completed experiment '%s' (stage=%s, %d instances, %d explainers)",
            experiment_name,
            stage_completed,
            len(instances_data),
            len(explainer_names),
        )
        return result_payload

    def _build_summary(instances_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "experiment": experiment_name,
            "dataset": dataset_name,
            "model": model_name,
            "instances": len(instances_data),
        }
    LOGGER.info("Instantiating model '%s'", model_name)
    tuner: Optional[HyperparameterTuner] = None
    if tune_models or use_tuned_params:
        tuner = HyperparameterTuner(
            output_dir=tuning_output_dir,
            model_registry=MODEL_REGISTRY,
        )
    persistence: Optional[ModelPersistence] = None
    if reuse_trained_models:
        persistence = ModelPersistence(base_dir=model_store_dir or Path("saved_models"))

    model = None
    if persistence:
        cached_model = persistence.load(dataset_name, model_name)
        if cached_model is not None:
            model = cached_model

    if model is None:
        params_override: Optional[Dict[str, Any]] = None
        if tuner:
            if tune_models:
                X_tune, y_tune = resolve_tuning_subset(
                    dataset_name=dataset_name,
                    dataset_spec=dataset_spec,
                    dataset=dataset,
                )
                params_override = tuner.ensure_best_parameters(
                    dataset_name=dataset_name,
                    model_name=model_name,
                    X=X_tune,
                    y=y_tune,
                    dataset_type=dataset_type,
                )
            elif use_tuned_params:
                params_override = tuner.load_best_parameters(dataset_name, model_name)
        model = instantiate_model(
            model_name, data_type=dataset_type, params_override=params_override
        )
        LOGGER.info("Training model '%s' on dataset '%s'", model_name, dataset_name)
        model.fit(dataset.X_train, dataset.y_train)
        if persistence:
            persistence.save(dataset_name, model_name, model)

    if stop_after_training:
        LOGGER.info(
            "Stopping experiment '%s' after training stage (per CLI flag).",
            experiment_name,
        )
        return _finalize(
            instances_data=[],
            batch_metrics_data={},
            stage_completed="training",
            detailed_paths={},
        )

    X_eval = dataset.X_test
    y_eval = dataset.y_test
    if max_instances is not None and max_instances < len(X_eval):
        # TODO this should be sampled randomly rather than just truncating
        X_eval = X_eval[: max_instances] 
        if y_eval is not None:
            y_eval = y_eval[: max_instances]

    y_pred = model.predict(X_eval)
    supports_proba = getattr(model, "supports_proba", hasattr(model, "predict_proba"))
    y_proba = model.predict_proba(X_eval) if supports_proba else None
    if log_progress:
        LOGGER.info(
            "Progress enabled: %d explainers, %d metrics, %d evaluation instances",
            len(explainer_names),
            len(metric_names),
            len(X_eval),
        )

    metric_objs: Dict[str, Any] = {}
    metric_caps: Dict[str, Dict[str, Any]] = {}
    if stop_after_explanations and metric_names:
        LOGGER.info(
            "Skipping metric evaluation for experiment '%s' (stop-after-explanations flag).",
            experiment_name,
        )
    else:
        metric_objs = {name: instantiate_metric(name) for name in metric_names}
        metric_caps = {
            name: metric_capabilities(metric) for name, metric in metric_objs.items()
        }
    metric_metadata: Dict[str, Dict[str, Any]] = {}
    if metric_objs:
        for name, metric in metric_objs.items():
            metric_metadata[name] = {
                "class": metric.__class__.__name__,
                "per_instance": metric_caps[name]["per_instance"],
                "requires_full_batch": metric_caps[name]["requires_full_batch"],
                "parameters": extract_metric_parameters(metric),
            }

    # Prepare on-disk directories for per-method artifacts.
    detailed_paths: Dict[str, str] = {}
    resolved_detailed = Path(detailed_output_dir or Path("saved_models") / "detailed_explanations")
    dataset_detail_dir = resolved_detailed / dataset_name
    dataset_detail_dir.mkdir(parents=True, exist_ok=True)
    ensure_dataset_metadata(dataset_detail_dir, dataset_name, dataset_type, feature_names)
    detailed_dir: Path = dataset_detail_dir / model_name
    detailed_dir.mkdir(parents=True, exist_ok=True)
    status_dir = detailed_dir / "_status"
    status_dir.mkdir(parents=True, exist_ok=True)

    metric_paths: Dict[str, str] = {}
    metrics_dir: Optional[Path] = None
    if metric_names:
        resolved_metrics = Path(metrics_output_dir or Path("saved_models") / "metrics_results")
        dataset_metrics_dir = resolved_metrics / dataset_name
        dataset_metrics_dir.mkdir(parents=True, exist_ok=True)
        ensure_dataset_metadata(dataset_metrics_dir, dataset_name, dataset_type, feature_names)
        metrics_dir = dataset_metrics_dir / model_name
        metrics_dir.mkdir(parents=True, exist_ok=True)

    # Expand explainer names into variants using hyperparameter grids (one variant per combination).
    # NOTE: this only handles finite categorical grids (lists). Randint-based BO is handled in the main loop.
    def _expand_variants(name: str) -> List[Tuple[str, Dict[str, Any]]]:
        grid = expl_hparam_grids.get(name) or {}
        if not grid:
            return [(name, {})]
        if any(isinstance(v, dict) and "randint" in v for v in grid.values()):
            raise ValueError(
                f"Explainer '{name}' uses randint ranges; enable explainer_hpo and run sequential optimization."
            )
        from itertools import product

        keys = sorted(grid.keys())
        values = [grid[k] for k in keys]
        variants: List[Tuple[str, Dict[str, Any]]] = []
        for combo in product(*values):
            override = {k: v for k, v in zip(keys, combo)}
            variants.append((build_method_label(name, override), override))
        return variants

    total_explainers = len(explainer_names)
    total_variants = 0
    for name in explainer_names:
        grid = expl_hparam_grids.get(name) or {}
        has_randint = any(isinstance(v, dict) and "randint" in v for v in grid.values())
        if has_randint:
            total_variants += int(expl_hpo_cfg.get("epochs") or 20)
        else:
            total_variants += len(_expand_variants(name))
    if log_progress:
        LOGGER.info(
            "Progress: prepared %d explainers (%d total variants)",
            total_explainers,
            total_variants,
        )

    method_artifacts: List[MethodArtifact] = []
    all_dataset_indices: Set[int] = set()
    for expl_index, expl_name in enumerate(explainer_names, start=1):
        grid = expl_hparam_grids.get(expl_name) or {}
        has_randint = any(isinstance(v, dict) and "randint" in v for v in grid.values())
        if not has_randint:
            variants = _expand_variants(expl_name)
            hpo_optimizer: Optional[BayesRangeOptimizer] = None
            hpo_persona = ""
            hpo_scaling = ""
            hpo_trial_history: List[str] = []
            hpo_variant_term_means: Dict[str, Dict[str, float]] = {}
            hpo_trials: List[Dict[str, Any]] = []
            hpo_epochs = len(variants)
        else:
            mode = str(expl_hpo_cfg.get("mode") or "gp").lower()
            hpo_epochs = int(expl_hpo_cfg.get("epochs") or 20)
            seed = int(expl_hpo_cfg.get("seed") or 0)
            init_points = int(expl_hpo_cfg.get("init_points") or 5)
            if mode != "gp":
                raise ValueError(
                    f"explainer_hpo.mode must be 'gp' when using randint ranges; got {mode!r}."
                )
            if hpo_epochs <= 0:
                raise ValueError("explainer_hpo.epochs must be positive.")
            if not metric_objs:
                raise ValueError(
                    "explainer_hpo requires metrics to be enabled so an objective can be evaluated."
                )
            hpo_persona = str(expl_hpo_cfg.get("persona") or "autoxai").strip().lower()
            hpo_scaling = str(expl_hpo_cfg.get("scaling") or "Std")
            space = parse_explainer_space(
                grid, n_features=len(feature_names) or int(dataset.X_train.shape[1])
            )
            hpo_optimizer = BayesRangeOptimizer(
                space=space,
                seed=seed,
                n_initial_points=init_points,
            )
            hpo_trial_history = []
            hpo_variant_term_means = {}
            hpo_trials = []
            variants = []  # handled sequentially below

        base_label = expl_name

        if log_progress:
            percent = 100.0 * expl_index / total_explainers if total_explainers else 100.0
            LOGGER.info(
                "[progress] Starting explainer %d/%d (%.1f%%): %s (%d variants)",
                expl_index,
                total_explainers,
                percent,
                expl_name,
                len(variants),
            )

        status_path = status_dir / f"{base_label}_status.json"
        status_info = load_completion_flag(status_path)
        cached_detail_path = (
            Path(status_info["detail_path"])
            if status_info and status_info.get("detail_path")
            else None
        )
        cached_metrics_path = (
            Path(status_info["metrics_path"])
            if status_info and status_info.get("metrics_path")
            else None
        )

        can_reuse_method = (
            status_info is not None
            and cached_detail_path is not None
            and cached_detail_path.exists()
        )
        if metric_objs:
            can_reuse_method = can_reuse_method and (
                cached_metrics_path is not None and cached_metrics_path.exists()
            )

        if skip_existing_methods and can_reuse_method:
            method_label = status_info.get("method_label", base_label)
            dataset_indices = {int(idx) for idx in status_info.get("dataset_indices", [])}
            all_dataset_indices.update(dataset_indices)
            method_artifacts.append(
                MethodArtifact(
                    explainer_key=expl_name,
                    method_label=method_label,
                    detail_path=cached_detail_path,
                    metrics_path=cached_metrics_path,
                    reused=True,
                )
            )
            if write_detailed_explanations:
                detailed_paths[method_label] = str(cached_detail_path)
            if write_metric_results and cached_metrics_path is not None:
                metric_paths[method_label] = str(cached_metrics_path)
            LOGGER.info(
                "Skipping '%s' for model '%s'; using cached artifacts.",
                method_label,
                model_name,
            )
            if log_progress:
                LOGGER.info(
                    "[progress] Completed %s (%d/%d explainers) via cache reuse",
                    base_label,
                    expl_index,
                    total_explainers,
                )
            continue

        # Optional reuse of cached explanations to compute metrics without recomputation.
        cached_expl: Optional[Dict[str, Any]] = None
        cached_variants: Dict[str, List[Dict[str, Any]]] = {}
        max_cached_idx = -1
        if reuse_detailed_explanations and cached_detail_path is not None and cached_detail_path.exists():
            cached_expl = load_cached_explanations(cached_detail_path, base_label)
            if cached_expl:
                for expl in cached_expl.get("explanations", []):
                    meta = expl.get("metadata") or {}
                    mv = meta.get("method_variant")
                    cached_variants.setdefault(mv, []).append(expl)
                    idx_val = meta.get("instance_index")
                    try:
                        idx_int = int(idx_val)
                        if idx_int > max_cached_idx:
                            max_cached_idx = idx_int
                    except (TypeError, ValueError):
                        continue

        LOGGER.info(
            "Generating explanations with '%s' (%d variants)",
            expl_name,
            hpo_epochs if hpo_optimizer is not None else len(variants),
        )

        combined_dataset_mapping: Dict[int, List[Tuple[int, Dict[str, Any]]]] = {}
        combined_metric_records: List[Dict[str, Any]] = []
        combined_batch_metrics: Dict[str, float] = {}
        batch_metrics_by_variant: Dict[str, Dict[str, float]] = {}
        next_global_idx = max_cached_idx + 1 if max_cached_idx >= 0 else 0
        combined_dataset_indices: Set[int] = set()

        combined_detail_path: Optional[Path] = None
        combined_metrics_path: Optional[Path] = None

        def _iter_variants():
            if hpo_optimizer is None:
                for item in variants:
                    yield item
                return
            for _ in range(hpo_epochs):
                params = hpo_optimizer.ask()
                yield build_method_label(expl_name, params), params

        for variant_index, (method_label, param_override) in enumerate(_iter_variants(), start=1):
            if log_progress and (hpo_epochs > 1):
                LOGGER.info(
                    "[progress] %s variant %d/%d (%.1f%%)",
                    base_label,
                    variant_index,
                    hpo_epochs,
                    100.0 * variant_index / max(1, hpo_epochs),
                )
            # Gather explanations: reuse cached for this variant when available, otherwise compute.
            variant_explanations: List[Dict[str, Any]] = []
            if cached_variants:
                variant_explanations = cached_variants.get(method_label, [])

            explainer = instantiate_explainer(
                expl_name,
                model,
                dataset,
                data_type=dataset_type,
                logging_cfg=logging_cfg,
                params_override=param_override,
            )

            if not variant_explanations:
                expl_results = explainer.explain_dataset(X_eval, y_eval)
                variant_explanations = expl_results.get("explanations", [])
                method_label_final = expl_results.get("method", expl_name)
            else:
                expl_results = {
                    "method": expl_name,
                    "explanations": variant_explanations,
                    "n_explanations": len(variant_explanations),
                    "info": {"source": "cached_reuse"},
                }
                method_label_final = expl_name

            if param_override:
                for explanation in variant_explanations:
                    meta = explanation.setdefault("metadata", {})
                    meta.setdefault("hyperparameters", param_override)
                    meta.setdefault("method_variant", method_label)

            variant_mapping_local: Dict[int, List[Tuple[int, Dict[str, Any]]]] = {}
            local_to_global: Dict[int, int] = {}
            for local_idx, explanation in enumerate(variant_explanations):
                metadata = explanation.get("metadata") or {}
                dataset_idx = metadata.get("dataset_index", local_idx)
                try:
                    dataset_idx_int = int(dataset_idx)
                except (TypeError, ValueError):
                    dataset_idx_int = int(local_idx)
                idx_val = metadata.get("instance_index")
                try:
                    global_idx = int(idx_val) if idx_val is not None else None
                except (TypeError, ValueError):
                    global_idx = None
                if global_idx is None:
                    global_idx = next_global_idx
                    next_global_idx += 1
                    metadata["instance_index"] = global_idx
                local_to_global[local_idx] = global_idx
                variant_mapping_local.setdefault(dataset_idx_int, []).append((local_idx, explanation))
                combined_dataset_mapping.setdefault(dataset_idx_int, []).append((global_idx, explanation))
                combined_dataset_indices.add(dataset_idx_int)

            if metric_objs:
                (
                    batch_metrics,
                    instance_metrics,
                ) = evaluate_metrics_for_method(
                    metric_objs=metric_objs,
                    metric_caps=metric_caps,
                    explainer=explainer,
                    expl_results=expl_results,
                    dataset_mapping=variant_mapping_local,
                    model=model,
                    dataset=dataset,
                    method_label=method_label_final,
                    log_progress=log_progress,
                )
                if batch_metrics:
                    combined_batch_metrics.update(batch_metrics)
                    batch_metrics_by_variant[method_label] = batch_metrics
                for dataset_idx_int, metrics_by_local in instance_metrics.items():
                    for local_idx, metrics_vals in metrics_by_local.items():
                        global_idx = local_to_global.get(local_idx, local_idx)
                        meta_lookup = {}
                        entries = variant_mapping_local.get(dataset_idx_int, [])
                        for entry_local_idx, explanation in entries:
                            meta_lookup[int(entry_local_idx)] = explanation.get("metadata") or {}
                        metric_entry = {
                            "instance_id": int(dataset_idx_int),
                            "dataset_index": int(dataset_idx_int),
                            "explanation_index": int(global_idx),
                            "true_label": safe_scalar(value_at(y_eval, dataset_idx_int)),
                            "prediction": safe_scalar(value_at(y_pred, dataset_idx_int)),
                            "metrics": metrics_vals,
                        }
                        if meta_lookup.get(int(local_idx)):
                            metric_entry["explanation_metadata"] = meta_lookup[int(local_idx)]
                        metric_entry["method_variant"] = method_label
                        combined_metric_records.append(metric_entry)

                if hpo_optimizer is not None:
                    trial_metrics = aggregate_trial_metrics(
                        batch_metrics=batch_metrics, instance_metrics=instance_metrics
                    )
                    persona = hpo_persona or "autoxai"
                    scaling = hpo_scaling or "Std"
                    term_values = compute_objective_term_values(metrics=trial_metrics, persona=persona)
                    if not term_values:
                        score = float("-inf")
                    else:
                        # Track per-variant mean term values and evaluate AutoXAI-like trial-history objective.
                        hpo_variant_term_means[method_label] = term_values
                        score = trial_history_objective_score(
                            history=hpo_trial_history,
                            candidate=method_label,
                            variant_term_means=hpo_variant_term_means,
                            persona=persona,
                            scaling=scaling,
                        )
                        score = float(score) if score is not None else float("-inf")
                    hpo_trial_history.append(method_label)
                    hpo_trials.append(
                        {
                            "trial_index": int(variant_index),
                            "method_variant": method_label,
                            "hyperparameters": dict(param_override),
                            "objective_terms": term_values,
                            "aggregated_score": float(score) if np.isfinite(score) else float("-inf"),
                        }
                    )
                    if not np.isfinite(score):
                        score = -1e9
                    hpo_optimizer.tell(param_override, float(score))

        combined_detail_path = detailed_dir / f"{base_label}_detailed_explanations.json"
        checkpoint_explanations(
            method_label=base_label,
            path=combined_detail_path,
            dataset_mapping=combined_dataset_mapping,
            feature_names=feature_names,
            y_pred=y_pred,
            y_true=y_eval,
            y_proba=y_proba,
        )

        if metric_objs and metrics_dir is not None:
            if hpo_optimizer is not None:
                overall_scores, per_instance_scores = build_candidate_scores_reports(
                    combined_metric_records=combined_metric_records,
                    method_label=base_label,
                    persona=hpo_persona or "autoxai",
                    scaling=hpo_scaling or "Std",
                    trial_history=hpo_trial_history,
                )
                hpo_report = {
                    "mode": "gp",
                    "epochs": int(hpo_epochs),
                    "seed": int(expl_hpo_cfg.get("seed") or 0),
                    "init_points": int(expl_hpo_cfg.get("init_points") or 5),
                    "persona": (hpo_persona or "autoxai"),
                    "scaling": (hpo_scaling or "Std"),
                    "trial_history": list(hpo_trial_history),
                    "trials": list(hpo_trials),
                    "candidate_scores_overall_trial_scope": overall_scores,
                    "candidate_scores_per_instance_trial_scope": per_instance_scores,
                }
            else:
                hpo_report = None

            metrics_cache_str = persist_metric_results(
                metrics_dir=metrics_dir,
                dataset_name=dataset_name,
                model_name=model_name,
                method_label=base_label,
                instances=combined_metric_records,
                batch_metrics=combined_batch_metrics,
                metric_metadata=metric_metadata,
                batch_metrics_by_variant=batch_metrics_by_variant,
                extra={"hpo": hpo_report} if hpo_report is not None else None,
            )
            if metrics_cache_str:
                combined_metrics_path = Path(metrics_cache_str)
                if write_metric_results:
                    metric_paths[base_label] = metrics_cache_str
        else:
            combined_metrics_path = None

        if write_detailed_explanations:
            detailed_paths[base_label] = str(combined_detail_path)

        method_artifacts.append(
            MethodArtifact(
                explainer_key=expl_name,
                method_label=base_label,
                detail_path=combined_detail_path,
                metrics_path=combined_metrics_path,
                reused=False,
            )
        )

        write_completion_flag(
            status_path=status_path,
            explainer_key=expl_name,
            method_label=base_label,
            dataset_name=dataset_name,
            model_name=model_name,
            detail_path=combined_detail_path,
            metrics_path=combined_metrics_path,
            dataset_indices=sorted(combined_dataset_indices),
        )
        if log_progress:
            LOGGER.info(
                "[progress] Completed explainer %d/%d: %s",
                expl_index,
                total_explainers,
                base_label,
            )

    # Collect per-instance metrics and assemble the final output structure.
    explanation_metadata: Dict[str, Dict[int, Any]] = {}
    instances_lookup: Dict[int, Dict[str, Any]] = {}
    sorted_dataset_indices = sorted(all_dataset_indices)
    for position, dataset_idx in enumerate(sorted_dataset_indices):
        inst_record: Dict[str, Any] = {
            "index": int(position),
            "dataset_index": int(dataset_idx),
        }
        if 0 <= dataset_idx < len(y_pred):
            inst_record["predicted_label"] = safe_scalar(y_pred[dataset_idx])
        else:
            inst_record["predicted_label"] = None
        if y_eval is not None and 0 <= dataset_idx < len(y_eval):
            inst_record["true_label"] = safe_scalar(y_eval[dataset_idx])
        else:
            inst_record["true_label"] = None
        if y_proba is not None and 0 <= dataset_idx < len(y_proba):
            inst_record["predicted_proba"] = np.asarray(y_proba[dataset_idx]).tolist()
        instances_lookup[int(dataset_idx)] = inst_record

    batch_metrics_result: Dict[str, Dict[str, float]] = {}
    for artifact in method_artifacts:
        cached = load_cached_explanations(artifact.detail_path, artifact.method_label)
        if cached is None:
            LOGGER.warning(
                "Missing cached explanations for %s at %s; skipping attachment.",
                artifact.method_label,
                artifact.detail_path,
            )
            continue

        instance_metrics: Dict[int, Dict[int, Dict[str, float]]] = {}
        method_batch_metrics: Dict[str, float] = {}
        method_batch_by_variant: Dict[str, Dict[str, float]] = {}
        if artifact.metrics_path is not None and artifact.metrics_path.exists():
            (
                instance_metrics,
                method_batch_metrics,
                method_batch_by_variant,
            ) = load_cached_metrics(artifact.metrics_path, artifact.method_label)
        batch_metrics_result[artifact.method_label] = method_batch_metrics

        for explanation in cached.get("explanations", []):
            metadata = dict(explanation.get("metadata") or {})
            recorded_dataset_idx = metadata.pop("dataset_index", None)
            fallback_idx = explanation.get("metadata_key")
            try:
                dataset_idx_int = (
                    int(recorded_dataset_idx)
                    if recorded_dataset_idx is not None
                    else int(fallback_idx)
                )
            except (TypeError, ValueError):
                continue

            inst_record = instances_lookup.get(dataset_idx_int)
            if inst_record is None:
                continue

            metrics_for_explainer: Dict[str, float] = {}
            metrics_bucket = instance_metrics.get(dataset_idx_int, {})
            local_idx = metadata.get("instance_index")
            selected_metrics: Optional[Dict[str, float]] = None
            if isinstance(metrics_bucket, dict):
                if local_idx is not None:
                    try:
                        selected_metrics = metrics_bucket.get(int(local_idx))
                    except Exception:
                        selected_metrics = None
                if not selected_metrics:
                    # Legacy shape (flat metrics dict) or fallback to first entry.
                    all_values = list(metrics_bucket.values())
                    if all_values and isinstance(all_values[0], dict):
                        selected_metrics = all_values[0]
                    elif metrics_bucket and all(
                        isinstance(v, (int, float)) for v in metrics_bucket.values()
                    ):
                        selected_metrics = metrics_bucket  # legacy single metrics dict
            if selected_metrics:
                metrics_for_explainer.update(selected_metrics)
            for key, value in method_batch_metrics.items():
                metrics_for_explainer.setdefault(key, value)

            if metadata:
                metadata_bucket = explanation_metadata.setdefault(artifact.method_label, {})
                metadata_bucket[dataset_idx_int] = to_serializable(metadata)

            explanation_entry: Dict[str, Any] = {
                "method": artifact.method_label,
                "metrics": metrics_for_explainer,
                "attributions": np.asarray(explanation.get("attributions", [])).tolist(),
                "metadata_key": dataset_idx_int,
            }
            if "instance_index" in metadata:
                try:
                    explanation_entry["explanation_index"] = int(metadata.get("instance_index"))
                except Exception:
                    pass
            gen_time = explanation.get("generation_time")
            if gen_time is not None:
                explanation_entry["generation_time"] = float(gen_time)

            inst_record.setdefault("explanations", []).append(explanation_entry)

    instances: List[Dict[str, Any]] = []
    for dataset_idx in sorted_dataset_indices:
        record = instances_lookup.get(int(dataset_idx))
        if record and record.get("explanations"):
            instances.append(record)

    stage_label = "explanations" if stop_after_explanations else "metrics"
    result_payload = _finalize(
        instances_data=instances,
        batch_metrics_data=batch_metrics_result,
        stage_completed=stage_label,
        detailed_paths=detailed_paths,
        metric_paths=metric_paths,
    )
    if return_summary_only:
        return _build_summary(instances)
    return result_payload


def run_experiments(
    experiment_names: Iterable[str],
    *,
    max_instances: Optional[int] = None,
    output_dir: Optional[str | Path] = None,
    tune_models: bool = False,
    use_tuned_params: bool = False,
    reuse_trained_models: bool = False,
    tuning_output_dir: Optional[str | Path] = None,
    model_store_dir: Optional[str | Path] = None,
    stop_after_training: bool = False,
    stop_after_explanations: bool = False,
    write_detailed_explanations: bool = False,
    detailed_output_dir: Optional[str | Path] = None,
    reuse_detailed_explanations: bool = False,
    write_metric_results: bool = False,
    metrics_output_dir: Optional[str | Path] = None,
    skip_existing_methods: bool = False,
    skip_existing_experiments: bool = False,
    return_summary_only: bool = False,
) -> List[Dict[str, Any]]:
    """
    Run multiple experiments sequentially.

    Parameters
    ----------
    experiment_names : Iterable[str]
        Names defined in the experiments configuration file.
    max_instances : int | None, optional
        Optional evaluation cap shared across experiments.
    output_dir : str | Path | None, optional
        If provided, each experiment result is written to ``output_dir/<name>.json``.
    tune_models : bool, optional
        Whether to run hyperparameter tuning for every dataset/model pair.
    use_tuned_params : bool, optional
        Reuse persisted tuned hyperparameters (if available).
    reuse_trained_models : bool, optional
        Load/save trained estimators for reuse across experiments.
    tuning_output_dir : str | Path | None, optional
        Directory where tuning artifacts will be persisted.
    model_store_dir : str | Path | None, optional
        Directory for serialized model checkpoints.
    stop_after_training : bool, optional
        Halt each experiment after training completes.
    stop_after_explanations : bool, optional
        Run explainers but skip metrics for each experiment.
    write_detailed_explanations : bool, optional
        Persist per-explainer JSON files for every dataset/model pair.
    detailed_output_dir : str | Path | None, optional
        Base directory for detailed explanation artifacts.
    reuse_detailed_explanations : bool, optional
        Load cached detailed explanation files when available.
    write_metric_results : bool, optional
        Persist per-method metric outputs alongside explanations.
    metrics_output_dir : str | Path | None, optional
        Base directory for structured metric results.
    skip_existing_methods : bool, optional
        Skip explainer runs when cached artifacts exist.
    skip_existing_experiments : bool, optional
        Skip entire experiment when the destination output already exists.
    return_summary_only : bool, optional
        When True, each experiment returns a compact summary dict instead of the full payload.

    Returns
    -------
    list[dict]
        List of experiment payloads or summaries depending on ``return_summary_only``.
    """
    results: List[Dict[str, Any]] = []
    output_path = Path(output_dir) if output_dir is not None else None

    for name in experiment_names:
        exp_cfg = EXPERIMENT_CFG[name]
        model_list = resolve_artifact_list(exp_cfg.get("models"), "model")
        if not model_list:
            fallback = resolve_artifact_key(
                exp_cfg.get("model"), "model", required=False
            )
            if fallback:
                model_list = [fallback]
        if not model_list:
            raise ValueError(f"Experiment '{name}' does not define any models.")

        for model_name in model_list:
            file_path = None
            if output_path is not None:
                suffix = f"{name}__{model_name}" if len(model_list) > 1 else name
                file_path = output_path / f"{suffix}.json"
            experiment_result = run_experiment(
                name,
                max_instances=max_instances,
                output_path=file_path,
                model_override=model_name,
                tune_models=tune_models,
                use_tuned_params=use_tuned_params,
                reuse_trained_models=reuse_trained_models,
                tuning_output_dir=tuning_output_dir,
                model_store_dir=model_store_dir,
                stop_after_training=stop_after_training,
                stop_after_explanations=stop_after_explanations,
                write_detailed_explanations=write_detailed_explanations,
                detailed_output_dir=detailed_output_dir,
                reuse_detailed_explanations=reuse_detailed_explanations,
                write_metric_results=write_metric_results,
                metrics_output_dir=metrics_output_dir,
                skip_existing_methods=skip_existing_methods,
                skip_if_output_exists=skip_existing_experiments,
                return_summary_only=return_summary_only,
            )
            results.append(experiment_result)
    return results
