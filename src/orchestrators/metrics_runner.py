"""
Experiment runner that instantiates datasets/models/explainers and computes metrics.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from .utils import (
    EXPERIMENT_CFG,
    instantiate_dataset,
    instantiate_explainer,
    instantiate_metric,
    instantiate_model,
    make_serializable_explanation,
    metric_capabilities,
    to_serializable,
)

LOGGER = logging.getLogger(__name__)


def run_experiment(
    experiment_name: str,
    *,
    max_instances: Optional[int] = None,
    output_path: Optional[str | Path] = None,
    model_override: Optional[str] = None,
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

    Returns
    -------
    Dict[str, Any]
        Nested experiment result ready for JSON serialization.
    """
    exp_cfg = EXPERIMENT_CFG[experiment_name]
    logging_cfg = exp_cfg.get("logging", {}) or {}
    log_progress = bool(logging_cfg.get("progress"))

    dataset_name = exp_cfg["dataset"]
    configured_models = exp_cfg.get("models")
    if model_override is not None:
        model_name = model_override
    elif configured_models:
        if len(configured_models) != 1:
            raise ValueError(
                "Experiment defines multiple models; provide model_override or use run_experiments."
            )
        model_name = configured_models[0]
    else:
        model_name = exp_cfg["model"]
    explainer_names = exp_cfg.get("explainers") or [exp_cfg["explainer"]]
    metric_names = exp_cfg.get("metrics", [])

    dataset = instantiate_dataset(dataset_name)
    model = instantiate_model(model_name)
    model.fit(dataset.X_train, dataset.y_train)

    X_eval = dataset.X_test
    y_eval = dataset.y_test
    if max_instances is not None and max_instances < len(X_eval):
        X_eval = X_eval[: max_instances]
        if y_eval is not None:
            y_eval = y_eval[: max_instances]

    y_pred = model.predict(X_eval)
    supports_proba = getattr(model, "supports_proba", hasattr(model, "predict_proba"))
    y_proba = model.predict_proba(X_eval) if supports_proba else None

    metric_objs = {name: instantiate_metric(name) for name in metric_names}
    metric_caps = {name: metric_capabilities(metric) for name, metric in metric_objs.items()}

    # Compute the explanations and batch-level metrics first.
    explainer_outputs: Dict[str, Dict[str, Any]] = {}
    for expl_name in explainer_names:
        if log_progress:
            LOGGER.info("Running %s explainer", expl_name)
        explainer = instantiate_explainer(
            expl_name, model, dataset, logging_cfg=logging_cfg
        )
        expl_results = explainer.explain_dataset(X_eval, y_eval)

        batch_metrics: Dict[str, float] = {}
        for metric_name, metric in metric_objs.items():
            caps = metric_caps[metric_name]
            if caps["per_instance"] or not caps["requires_full_batch"]:
                continue
            if log_progress:
                LOGGER.info("Running %s metric (batch) for %s", metric_name, expl_name)
            out = metric.evaluate(
                model=model,
                explanation_results=expl_results,
                dataset=dataset,
                explainer=explainer,
            )
            batch_metrics.update(_coerce_metric_dict(out))

        explainer_outputs[expl_name] = {
            "explainer": explainer,
            "results": expl_results,
            "batch_metrics": batch_metrics,
        }

    # Collect per-instance metrics and assemble the final output structure.
    instances: List[Dict[str, Any]] = []
    n_instances = len(X_eval)
    announced_metrics: Set[Tuple[str, str]] = set()
    for idx in range(n_instances):
        inst_record: Dict[str, Any] = {
            "index": int(idx),
            "true_label": _safe_scalar(y_eval[idx]) if y_eval is not None else None,
            "predicted_label": _safe_scalar(y_pred[idx]),
        }
        if y_proba is not None:
            inst_record["predicted_proba"] = np.asarray(y_proba[idx]).tolist()

        explainer_records: List[Dict[str, Any]] = []
        for expl_name, data in explainer_outputs.items():
            expl_results = data["results"]
            explainer_obj = data["explainer"]
            explanation_i = expl_results["explanations"][idx]

            metrics_for_explainer: Dict[str, float] = {}
            for metric_name, metric in metric_objs.items():
                caps = metric_caps[metric_name]
                if not caps["per_instance"]:
                    continue
                key = (metric_name, expl_name)
                if log_progress and key not in announced_metrics:
                    LOGGER.info(
                        "Running %s metric (per-instance) for %s",
                        metric_name,
                        expl_name,
                    )
                    announced_metrics.add(key)
                payload = dict(expl_results)
                payload["current_index"] = idx
                out = metric.evaluate(
                    model=model,
                    explanation_results=payload,
                    dataset=dataset,
                    explainer=explainer_obj,
                )
                metrics_for_explainer.update(_coerce_metric_dict(out))

            # Include batch-level metrics so consumers find them alongside per-instance ones.
            for key, value in data["batch_metrics"].items():
                metrics_for_explainer.setdefault(key, value)

            explainer_records.append(
                {
                    "explainer": expl_name,
                    "metrics": metrics_for_explainer,
                    "explanation": make_serializable_explanation(explanation_i),
                }
            )

        inst_record["explanations"] = explainer_records
        instances.append(inst_record)

    result = {
        "experiment": experiment_name,
        "dataset": dataset_name,
        "model": model_name,
        "instances": instances,
        "batch_metrics": {
            expl_name: data["batch_metrics"] for expl_name, data in explainer_outputs.items()
        },
    }

    if output_path is not None:
        path = Path(output_path)
        path.write_text(json.dumps(to_serializable(result), indent=2), encoding="utf-8")

    return result


def run_experiments(
    experiment_names: Iterable[str],
    *,
    max_instances: Optional[int] = None,
    output_dir: Optional[str | Path] = None,
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

    Returns
    -------
    list[dict]
        List of experiment result dictionaries.
    """
    results: List[Dict[str, Any]] = []
    output_path = Path(output_dir) if output_dir is not None else None

    for name in experiment_names:
        exp_cfg = EXPERIMENT_CFG[name]
        model_list = list(exp_cfg.get("models") or [])
        if not model_list:
            model_list = [exp_cfg["model"]]

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
            )
            results.append(experiment_result)
    return results


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


def _safe_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = value.ravel()[0]
    if isinstance(value, np.generic):
        return value.item()
    return value
