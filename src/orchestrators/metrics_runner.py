"""
Experiment runner that instantiates datasets/models/explainers and computes metrics.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from src.validators import TabularDataValidator
from src.utils.hyperparameter_tuning import HyperparameterTuner
from src.utils.model_persistence import ModelPersistence

from .utils import (
    DATASET_REGISTRY,
    EXPLAINER_REGISTRY,
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

    dataset_name = _resolve_artifact_key(exp_cfg.get("dataset"), "dataset")
    configured_models = _resolve_artifact_list(exp_cfg.get("models"), "model")
    model_entry = _resolve_artifact_key(
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
    explainer_names = _resolve_artifact_list(explainer_configs, "explainer")
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
    _run_dataset_validation(
        dataset_name=dataset_name,
        dataset_type=dataset_type,
        dataset_spec=dataset_spec,
        dataset=dataset,
        experiment_name=experiment_name,
    )
    feature_names = list(getattr(dataset, "feature_names", []) or [])

    def _finalize(
        *,
        instances_data: List[Dict[str, Any]],
        batch_metrics_data: Dict[str, Dict[str, float]],
        metadata_data: Dict[str, Dict[int, Any]],
        stage_completed: str,
        detailed_paths: Optional[Dict[str, str]] = None,
        metric_paths: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        metadata_serialized = {
            method: meta for method, meta in metadata_data.items() if meta
        }
        result_payload = {
            "experiment": experiment_name,
            "dataset": dataset_name,
            "model": model_name,
            "instances": instances_data,
            "feature_names": feature_names,
            "batch_metrics": batch_metrics_data,
            "explanation_metadata": metadata_serialized,
            "stage_completed": stage_completed,
            "detailed_explanations": detailed_paths or {},
            "per_method_metric_files": metric_paths or {},
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
                X_tune, y_tune = _resolve_tuning_subset(
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
            metadata_data={},
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
                "parameters": _extract_metric_parameters(metric),
            }

    # Prepare on-disk directories for per-method artifacts.
    detailed_paths: Dict[str, str] = {}
    resolved_detailed = Path(detailed_output_dir or Path("saved_models") / "detailed_explanations")
    dataset_detail_dir = resolved_detailed / dataset_name
    dataset_detail_dir.mkdir(parents=True, exist_ok=True)
    _ensure_dataset_metadata(dataset_detail_dir, dataset_name, dataset_type, feature_names)
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
        _ensure_dataset_metadata(dataset_metrics_dir, dataset_name, dataset_type, feature_names)
        metrics_dir = dataset_metrics_dir / model_name
        metrics_dir.mkdir(parents=True, exist_ok=True)

    method_artifacts: List[MethodArtifact] = []
    all_dataset_indices: Set[int] = set()
    for expl_name in explainer_names:
        status_path = status_dir / f"{expl_name}_status.json"
        status_info = _load_completion_flag(status_path)
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
            method_label = status_info.get("method_label", expl_name)
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
                expl_name,
                model_name,
            )
            continue

        LOGGER.info("Generating explanations with '%s'", expl_name)
        explainer = instantiate_explainer(
            expl_name, model, dataset, data_type=dataset_type, logging_cfg=logging_cfg
        )

        expl_results = None
        cached_path: Optional[Path] = None
        method_label = expl_name

        if reuse_detailed_explanations:
            candidate_paths: List[Path] = []
            if cached_detail_path is not None:
                candidate_paths.append(cached_detail_path)
            default_path = detailed_dir / f"{expl_name}_detailed_explanations.json"
            if default_path not in candidate_paths:
                candidate_paths.append(default_path)
            for candidate in candidate_paths:
                if not candidate.exists():
                    continue
                cached = _load_cached_explanations(candidate, expl_name)
                if cached is None:
                    continue
                expl_results = cached
                cached_path = candidate
                method_label = cached.get("method", expl_name)
                break

        if expl_results is None:
            expl_results = explainer.explain_dataset(X_eval, y_eval)
            method_label = expl_results.get("method", expl_name)
        else:
            method_label = expl_results.get("method", expl_name)

        dataset_mapping: Dict[int, Tuple[int, Dict[str, Any]]] = {}
        for local_idx, explanation in enumerate(expl_results.get("explanations", [])):
            metadata = explanation.get("metadata") or {}
            dataset_idx = metadata.get("dataset_index", local_idx)
            try:
                dataset_idx_int = int(dataset_idx)
            except (TypeError, ValueError):
                dataset_idx_int = int(local_idx)
            dataset_mapping[dataset_idx_int] = (local_idx, explanation)
        dataset_indices = set(dataset_mapping.keys())
        all_dataset_indices.update(dataset_indices)

        batch_metrics: Dict[str, float] = {}
        metric_records: List[Dict[str, Any]] = []
        instance_metrics: Dict[int, Dict[str, float]] = {}
        if metric_objs:
            (
                batch_metrics,
                instance_metrics,
            ) = _evaluate_metrics_for_method(
                metric_objs=metric_objs,
                metric_caps=metric_caps,
                explainer=explainer,
                expl_results=expl_results,
                dataset_mapping=dataset_mapping,
                model=model,
                dataset=dataset,
                method_label=method_label,
                log_progress=log_progress,
            )
            for dataset_idx_int in sorted(instance_metrics.keys()):
                metric_entry = {
                    "instance_id": int(dataset_idx_int),
                    "dataset_index": int(dataset_idx_int),
                    "true_label": _safe_scalar(_value_at(y_eval, dataset_idx_int)),
                    "prediction": _safe_scalar(_value_at(y_pred, dataset_idx_int)),
                    "metrics": instance_metrics[dataset_idx_int],
                }
                metric_records.append(metric_entry)

        detail_path = cached_path
        if detail_path is None:
            detail_path = detailed_dir / f"{method_label}_detailed_explanations.json"
            _checkpoint_explanations(
                method_label=method_label,
                path=detail_path,
                dataset_mapping=dataset_mapping,
                feature_names=feature_names,
                y_pred=y_pred,
                y_true=y_eval,
                y_proba=y_proba,
            )

        metrics_cache_path: Optional[Path] = None
        if metric_objs and metrics_dir is not None:
            metrics_cache_str = _write_metric_results(
                metrics_dir=metrics_dir,
                dataset_name=dataset_name,
                model_name=model_name,
                method_label=method_label,
                instances=metric_records,
                batch_metrics=batch_metrics,
                metric_metadata=metric_metadata,
            )
            if metrics_cache_str:
                metrics_cache_path = Path(metrics_cache_str)
                if write_metric_results:
                    metric_paths[method_label] = metrics_cache_str

        if write_detailed_explanations:
            detailed_paths[method_label] = str(detail_path)

        method_artifacts.append(
            MethodArtifact(
                explainer_key=expl_name,
                method_label=method_label,
                detail_path=detail_path,
                metrics_path=metrics_cache_path,
                reused=cached_path is not None,
            )
        )

        _write_completion_flag(
            status_path=status_path,
            explainer_key=expl_name,
            method_label=method_label,
            dataset_name=dataset_name,
            model_name=model_name,
            detail_path=detail_path,
            metrics_path=metrics_cache_path,
            dataset_indices=sorted(dataset_indices),
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
            inst_record["predicted_label"] = _safe_scalar(y_pred[dataset_idx])
        else:
            inst_record["predicted_label"] = None
        if y_eval is not None and 0 <= dataset_idx < len(y_eval):
            inst_record["true_label"] = _safe_scalar(y_eval[dataset_idx])
        else:
            inst_record["true_label"] = None
        if y_proba is not None and 0 <= dataset_idx < len(y_proba):
            inst_record["predicted_proba"] = np.asarray(y_proba[dataset_idx]).tolist()
        instances_lookup[int(dataset_idx)] = inst_record

    batch_metrics_result: Dict[str, Dict[str, float]] = {}
    for artifact in method_artifacts:
        cached = _load_cached_explanations(artifact.detail_path, artifact.method_label)
        if cached is None:
            LOGGER.warning(
                "Missing cached explanations for %s at %s; skipping attachment.",
                artifact.method_label,
                artifact.detail_path,
            )
            continue

        instance_metrics: Dict[int, Dict[str, float]] = {}
        method_batch_metrics: Dict[str, float] = {}
        if artifact.metrics_path is not None and artifact.metrics_path.exists():
            instance_metrics, method_batch_metrics = _load_cached_metrics(
                artifact.metrics_path, artifact.method_label
            )
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

            metrics_for_explainer: Dict[str, float] = dict(
                instance_metrics.get(dataset_idx_int, {})
            )
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
        metadata_data=explanation_metadata,
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
        model_list = _resolve_artifact_list(exp_cfg.get("models"), "model")
        if not model_list:
            fallback = _resolve_artifact_key(
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


def _resolve_artifact_key(
    entry: Any,
    kind: str,
    *,
    required: bool = True,
) -> Optional[str]:
    if entry is None:
        if required:
            raise ValueError(f"Experiment is missing a '{kind}' reference.")
        return None
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        key = entry.get("key") or entry.get("name")
        if key:
            return str(key)
    if required:
        raise ValueError(
            f"Experiment {kind} entries must be strings or mappings with a 'key'. Got: {entry!r}"
        )
    return None


def _resolve_artifact_list(entries: Optional[Iterable[Any]], kind: str) -> List[str]:
    if not entries:
        return []
    names: List[str] = []
    for entry in entries:
        key = _resolve_artifact_key(entry, kind)
        if key:
            names.append(key)
    return names


def _run_dataset_validation(
    *,
    dataset_name: str,
    dataset_type: str,
    dataset_spec: Dict[str, Any],
    dataset,
    experiment_name: str,
):
    if dataset_type != "tabular":
        return
    validation_cfg = (dataset_spec.get("validation") or {}).get("overrides") or {}
    result = TABULAR_VALIDATOR.validate(
        dataset=dataset,
        dataset_name=dataset_name,
        overrides=validation_cfg,
    )
    LOGGER.info(
        "Validation summary for %s: %s warnings, %s errors",
        dataset_name,
        len(result.warnings),
        len(result.errors),
    )
    for warning in result.warnings:
        LOGGER.warning(
            "Dataset validation warning for %s (%s): %s",
            dataset_name,
            experiment_name,
            warning,
        )
    if not result.is_valid:
        details = "; ".join(result.errors)
        raise ValueError(
            f"Experiment '{experiment_name}' failed dataset validation for '{dataset_name}': {details}"
        )


def _resolve_tuning_subset(
    *,
    dataset_name: str,
    dataset_spec: Dict[str, Any],
    dataset,
):
    tuning_cfg = (dataset_spec.get("tuning") or {}) if dataset_spec else {}
    sample_fraction = tuning_cfg.get("sample_fraction")
    max_samples = tuning_cfg.get("max_samples")
    if sample_fraction is None and max_samples is None:
        return dataset.X_train, dataset.y_train

    try:
        fraction = float(sample_fraction) if sample_fraction is not None else None
    except (TypeError, ValueError):
        raise ValueError(
            f"Dataset '{dataset_name}' tuning.sample_fraction must be numeric. Got {sample_fraction!r}"
        )
    if fraction is not None:
        if fraction <= 0 or fraction > 1:
            raise ValueError(
                f"Dataset '{dataset_name}' tuning.sample_fraction must be in (0, 1]. Got {fraction}"
            )

    try:
        max_count = int(max_samples) if max_samples is not None else None
    except (TypeError, ValueError):
        raise ValueError(
            f"Dataset '{dataset_name}' tuning.max_samples must be an integer. Got {max_samples!r}"
        )
    if max_count is not None and max_count <= 0:
        raise ValueError(
            f"Dataset '{dataset_name}' tuning.max_samples must be > 0. Got {max_count}"
        )

    X_train = dataset.X_train
    y_train = dataset.y_train
    n_train = len(X_train)
    target = n_train
    if fraction is not None:
        target = min(target, max(1, int(round(n_train * fraction))))
    if max_count is not None:
        target = min(target, max_count)
    if target >= n_train:
        return X_train, y_train

    seed = tuning_cfg.get("random_state")
    rng = np.random.default_rng(seed)
    indices = rng.choice(n_train, size=target, replace=False)
    LOGGER.info(
        "Using %d/%d samples from '%s' for tuning (sample_fraction=%s, max_samples=%s)",
        target,
        n_train,
        dataset_name,
        fraction if fraction is not None else "full",
        max_count if max_count is not None else "full",
    )
    return X_train[indices], None if y_train is None else y_train[indices]


def _ensure_dataset_metadata(
    dataset_dir: Path,
    dataset_name: str,
    dataset_type: str,
    feature_names: List[str],
) -> None:
    meta_path = dataset_dir / "metadata.json"
    if meta_path.exists():
        return
    payload = {
        "dataset": dataset_name,
        "dataset_type": dataset_type,
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "generated_at": datetime.utcnow().isoformat(),
    }
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _checkpoint_explanations(
    *,
    method_label: str,
    path: Path,
    dataset_mapping: Dict[int, Tuple[int, Dict[str, Any]]],
    feature_names: List[str],
    y_pred,
    y_true,
    y_proba,
) -> None:
    records: List[Dict[str, Any]] = []
    for dataset_idx in sorted(dataset_mapping.keys()):
        _, explanation = dataset_mapping[dataset_idx]
        metadata = dict(explanation.get("metadata") or {})
        metadata.pop("dataset_index", None)
        predicted_label = _safe_scalar(_value_at(y_pred, dataset_idx))
        true_label = _safe_scalar(_value_at(y_true, dataset_idx))
        proba_raw = _value_at(y_proba, dataset_idx)
        predicted_proba = (
            None if proba_raw is None else np.asarray(proba_raw).tolist()
        )
        correct_prediction = (
            true_label is not None and predicted_label == true_label
        )
        record: Dict[str, Any] = {
            "instance_id": dataset_idx,
            "dataset_index": dataset_idx,
            "true_label": true_label,
            "prediction": predicted_label,
            "prediction_proba": predicted_proba,
            "correct_prediction": correct_prediction,
            "feature_importance": np.asarray(
                explanation.get("attributions", [])
            ).tolist(),
            "metadata": to_serializable(metadata) if metadata else {},
        }
        gen_time = explanation.get("generation_time")
        if gen_time is not None:
            record["generation_time"] = float(gen_time)
        records.append(record)
    payload = {
        "metadata": {
            "feature_names": feature_names,
            "n_features": len(feature_names),
        },
        "records": records,
    }
    serialized = json.dumps(to_serializable(payload), indent=2)
    path.write_text(serialized, encoding="utf-8")
    LOGGER.debug(
        "Checkpointed %d explanations for %s at %s",
        len(records),
        method_label,
        path,
    )


def _write_completion_flag(
    *,
    status_path: Path,
    explainer_key: str,
    method_label: str,
    dataset_name: str,
    model_name: str,
    detail_path: Path,
    metrics_path: Optional[Path],
    dataset_indices: Iterable[int],
) -> None:
    payload = {
        "explainer": explainer_key,
        "method_label": method_label,
        "dataset": dataset_name,
        "model": model_name,
        "detail_path": str(detail_path),
        "metrics_path": str(metrics_path) if metrics_path else None,
        "dataset_indices": [int(idx) for idx in dataset_indices],
        "completed_at": datetime.utcnow().isoformat(),
    }
    status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_completion_flag(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Failed to read completion flag from %s: %s", path, exc)
        return None


def _load_cached_explanations(file_path: Path, method_label: str) -> Optional[Dict[str, Any]]:
    if not file_path.exists():
        return None
    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.warning("Failed to load cached explanations from %s: %s", file_path, exc)
        return None

    feature_names = None
    if isinstance(raw, dict):
        raw_records = raw.get("records") or raw.get("explanations") or []
        meta_block = raw.get("metadata") or {}
        feature_names = meta_block.get("feature_names") or raw.get("feature_names")
    else:
        raw_records = raw

    explanations: List[Dict[str, Any]] = []
    for record in raw_records:
        metadata = dict(record.get("metadata") or {})
        dataset_idx = record.get("dataset_index")
        if dataset_idx is not None:
            metadata.setdefault("dataset_index", dataset_idx)
        explanation_entry: Dict[str, Any] = {
            "method": method_label,
            "attributions": record.get("feature_importance") or [],
            "metadata": metadata,
            "metadata_key": dataset_idx if dataset_idx is not None else record.get("instance_id"),
        }
        gen_time = record.get("generation_time")
        if gen_time is not None:
            explanation_entry["generation_time"] = float(gen_time)
        explanations.append(explanation_entry)

    return {
        "method": method_label,
        "explanations": explanations,
        "n_explanations": len(explanations),
        "info": {
            "source": "cached_file",
            "path": str(file_path),
            "feature_names": feature_names,
        },
    }


def _load_cached_metrics(
    file_path: Path, method_label: str
) -> Tuple[Dict[int, Dict[str, float]], Dict[str, float]]:
    if not file_path.exists():
        return {}, {}
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.warning("Failed to load cached metrics from %s: %s", file_path, exc)
        return {}, {}
    file_method = payload.get("method", method_label)
    if file_method != method_label:
        LOGGER.warning(
            "Cached metrics file %s method=%s does not match expected label %s",
            file_path,
            file_method,
            method_label,
        )
    instances: Dict[int, Dict[str, float]] = {}
    for record in payload.get("instances", []):
        dataset_idx = record.get("dataset_index", record.get("instance_id"))
        if dataset_idx is None:
            continue
        try:
            dataset_idx_int = int(dataset_idx)
        except (TypeError, ValueError):
            continue
        metrics = _coerce_metric_dict(record.get("metrics") or {})
        if metrics:
            instances[dataset_idx_int] = metrics
    batch_metrics = _coerce_metric_dict(payload.get("batch_metrics") or {})
    return instances, batch_metrics


def _evaluate_metrics_for_method(
    *,
    metric_objs: Dict[str, Any],
    metric_caps: Dict[str, Dict[str, Any]],
    explainer,
    expl_results: Dict[str, Any],
    dataset_mapping: Dict[int, Tuple[int, Dict[str, Any]]],
    model,
    dataset,
    method_label: str,
    log_progress: bool,
) -> Tuple[Dict[str, float], Dict[int, Dict[str, float]]]:
    batch_metrics: Dict[str, float] = {}
    instance_metrics: Dict[int, Dict[str, float]] = {}
    if not metric_objs:
        return batch_metrics, instance_metrics

    for metric_name, metric in metric_objs.items():
        caps = metric_caps[metric_name]
        if caps["per_instance"]:
            if log_progress:
                LOGGER.info(
                    "Running %s metric (per-instance) for %s", metric_name, method_label
                )
            for dataset_idx, (local_idx, _) in dataset_mapping.items():
                payload = dict(expl_results)
                payload["current_index"] = local_idx
                out = metric.evaluate(
                    model=model,
                    explanation_results=payload,
                    dataset=dataset,
                    explainer=explainer,
                )
                values = _coerce_metric_dict(out)
                if not values:
                    continue
                instance_entry = instance_metrics.setdefault(int(dataset_idx), {})
                instance_entry.update(values)
            continue

        if not caps["requires_full_batch"]:
            continue
        if log_progress:
            LOGGER.info("Running %s metric (batch) for %s", metric_name, method_label)
        out = metric.evaluate(
            model=model,
            explanation_results=expl_results,
            dataset=dataset,
            explainer=explainer,
        )
        batch_metrics.update(_coerce_metric_dict(out))

    return batch_metrics, instance_metrics


def _extract_metric_parameters(metric: Any) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    attr_dict = getattr(metric, "__dict__", {})
    for key, value in attr_dict.items():
        if key.startswith("_"):
            continue
        if _is_jsonable(value):
            params[key] = value
    return params


def _is_jsonable(value: Any) -> bool:
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_jsonable(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) for k in value.keys()) and all(
            _is_jsonable(v) for v in value.values()
        )
    return False


def _write_metric_results(
    *,
    metrics_dir: Path,
    dataset_name: str,
    model_name: str,
    method_label: str,
    instances: List[Dict[str, Any]],
    batch_metrics: Dict[str, float],
    metric_metadata: Dict[str, Dict[str, Any]],
) -> Optional[str]:
    if not instances and not batch_metrics:
        return None
    payload = {
        "dataset": dataset_name,
        "model": model_name,
        "method": method_label,
        "generated_at": datetime.utcnow().isoformat(),
        "metric_metadata": metric_metadata,
        "instances": instances,
        "batch_metrics": batch_metrics,
    }
    file_path = metrics_dir / f"{method_label}_metrics.json"
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(to_serializable(payload), handle, indent=2)
    return str(file_path)


def _coerce_metric_dict(values: Optional[Dict[str, Any]]) -> Dict[str, float]:
    # Sanitize metric output to ensure all values are floats.
    # If there are nones or non-coercible values, they are skipped.
    if not values:
        return {}
    coerced: Dict[str, float] = {}
    for key, value in values.items():
        if value is None:
            continue
        try:
            coerced[key] = float(value)
        except (TypeError, ValueError):
            continue
    return coerced


def _value_at(sequence, index: int):
    if sequence is None or index < 0:
        return None
    try:
        length = len(sequence)
    except TypeError:
        return None
    if index >= length:
        return None
    try:
        return sequence[index]
    except (IndexError, TypeError):
        return None


def _safe_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = value.ravel()[0]
    if isinstance(value, np.generic):
        return value.item()
    return value
