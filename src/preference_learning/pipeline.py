"""High-level pipeline to train and evaluate preference-learning models."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.baseline.autoxai_objectives import default_objective_terms
from src.baseline.autoxai_scoring import compute_scores

from .config import ExperimentConfig
from .data import (
    PairwisePreferenceData,
    PreferenceDatasetBuilder,
    _differences_for_instance,
    _infer_feature_columns,
)
from .evaluation import build_ground_truth_order, evaluate_topk
from .models import LinearSVCConfig, LinearSVCPreferenceModel
from .persona import HierarchicalDirichletUser, load_persona_config
from .ranker import PersonaPairwiseRanker

DEFAULT_RESULTS_ROOT = Path("results") / "full_run_dec8"
DEFAULT_PROCESSED_DIR = DEFAULT_RESULTS_ROOT / "preference_learning"


def _slug(value: object) -> str:
    if not isinstance(value, str) or not value:
        return "unknown"
    cleaned = []
    for ch in value.strip():
        if ch.isalnum() or ch in {"_", "-"}:
            cleaned.append(ch)
        else:
            cleaned.append("_")
    return "".join(cleaned).lower()


def _md5_prefix(payload: object, *, length: int = 8) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.md5(encoded).hexdigest()[:length]


def run_linear_svc_experiment(
    *,
    encoded_path: Path,
    pair_labels_dir: Path,
    persona: str,
    output_dir: Path | None = None,
    experiment_config: ExperimentConfig | None = None,
    model_config: LinearSVCConfig | None = None,
) -> dict:
    """Train + evaluate a LinearSVC on pairwise difference features."""
    config = experiment_config or ExperimentConfig()
    builder = PreferenceDatasetBuilder(encoded_path, pair_labels_dir)
    dataset = builder.build(
        test_size=config.test_size,
        random_state=config.random_state,
        excluded_feature_groups=config.exclude_feature_groups,
    )
    output_root = output_dir or DEFAULT_PROCESSED_DIR
    experiment_dir = output_root / persona / encoded_path.stem.replace("_encoded", "")
    _persist_processed_data(dataset, experiment_dir)

    model_conf = model_config or LinearSVCConfig(random_state=config.random_state)
    model = LinearSVCPreferenceModel(model_conf)
    tuning_summary = None
    if getattr(model_conf, "tune", False):
        tuning_summary = model.tune_and_fit(dataset.train_features, dataset.train_labels)
    else:
        model.fit(dataset.train_features, dataset.train_labels)

    metrics = _evaluate_model(
        model=model,
        dataset=dataset,
        experiment_dir=experiment_dir,
        feature_columns=dataset.feature_columns,
        top_k=config.top_k,
    )
    summary = {
        "dataset": dataset.dataset_name,
        "model": dataset.model_name,
        "persona": persona,
        "train_instances": dataset.train_instances,
        "test_instances": dataset.test_instances,
        "train_rows": int(len(dataset.train_features)),
        "test_instances_evaluated": list(metrics.keys()),
        "svc_tuning": tuning_summary,
        "experiment_config": {
            "test_size": config.test_size,
            "random_state": config.random_state,
            "top_k": list(config.top_k),
            "exclude_feature_groups": list(config.exclude_feature_groups),
        },
    }
    (experiment_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (experiment_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return metrics


def run_persona_linear_svc_simulation(
    *,
    encoded_path: Path,
    persona_config_path: Path,
    output_dir: Path | None = None,
    experiment_config: ExperimentConfig | None = None,
    model_config: LinearSVCConfig | None = None,
) -> dict:
    """Sample multiple users, train/evaluate, then aggregate top-k without persisting pair labels."""
    config = experiment_config or ExperimentConfig()
    if config.num_users < 1:
        raise ValueError("num_users must be >= 1.")

    encoded_df = pd.read_parquet(encoded_path)
    if encoded_df.empty:
        raise ValueError(f"No rows found in encoded file {encoded_path}")

    dataset_name = encoded_df["dataset"].iloc[0]
    model_name = encoded_df["model"].iloc[0]
    feature_columns = _infer_feature_columns(
        encoded_df,
        excluded_feature_groups=config.exclude_feature_groups,
    )

    instance_ids = encoded_df["instance_index"].unique()
    if len(instance_ids) < 2:
        raise ValueError("At least two instances are required to perform a split.")
    train_ids, test_ids = train_test_split(
        instance_ids,
        test_size=config.test_size,
        random_state=config.random_state,
        shuffle=True,
    )
    train_ids = sorted(int(idx) for idx in train_ids)
    test_ids = sorted(int(idx) for idx in test_ids)

    persona_config = load_persona_config(persona_config_path)
    persona_config_raw: dict[str, Any] | None
    try:
        persona_config_raw = json.loads(persona_config_path.read_text(encoding="utf-8"))
    except Exception:
        persona_config_raw = None
    pareto_metrics, variant_to_method = _load_pareto_metrics_for_encoded(encoded_path)
    autoxai_objective = default_objective_terms()

    per_user_summaries: list[dict] = []
    per_user_topk: list[Dict[str, Dict[str, float]]] = []
    per_user_autoxai_topk: list[Dict[str, Dict[str, float]]] = []
    for user_idx in range(config.num_users):
        user = HierarchicalDirichletUser(
            persona_config,
            seed=config.persona_seed + user_idx,
            tau=config.tau,
        )
        ranker = PersonaPairwiseRanker(
            user=user,
            rng=np.random.default_rng(config.label_seed + user_idx),
        )

        train_rows: list[np.ndarray] = []
        train_labels: list[int] = []
        for instance_id in train_ids:
            instance_df = encoded_df.loc[encoded_df["instance_index"] == instance_id].copy()
            if instance_df.empty:
                continue
            pair_df = ranker.label_instance(
                dataset_index=int(instance_id),
                candidates=instance_df,
            )
            diff_matrix, diff_labels = _differences_for_instance(instance_df, pair_df, feature_columns)
            if diff_matrix.size == 0:
                continue
            train_rows.extend(diff_matrix)
            train_labels.extend(diff_labels)

        if not train_rows:
            raise ValueError("Training set is empty after generating persona labels.")

        X_train = pd.DataFrame(train_rows, columns=feature_columns)
        y_train = pd.Series(train_labels, name="label")

        model_conf = model_config or LinearSVCConfig(random_state=config.random_state)
        model = LinearSVCPreferenceModel(model_conf)
        svc_tuning = None
        if getattr(model_conf, "tune", False):
            svc_tuning = model.tune_and_fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)

        per_instance_topk: list[Dict[str, Dict[str, float]]] = []
        per_instance_autoxai_topk: list[Dict[str, Dict[str, float]]] = []
        instances_evaluated = 0
        for instance_id in test_ids:
            instance_df = encoded_df.loc[encoded_df["instance_index"] == instance_id].copy()
            if instance_df.empty:
                continue
            pair_df = ranker.label_instance(
                dataset_index=int(instance_id),
                candidates=instance_df,
            )
            ground_truth = build_ground_truth_order(pair_df)
            if not ground_truth:
                continue
            scores = model.score_candidates(instance_df, feature_columns)
            topk_metrics = evaluate_topk(dict(scores.items()), ground_truth, k_values=config.top_k)
            if topk_metrics:
                per_instance_topk.append(topk_metrics)
                instances_evaluated += 1

            autoxai_metrics = pareto_metrics.get(int(instance_id))
            if autoxai_metrics:
                variants_present = set(instance_df["method_variant"].astype(str).tolist())
                instance_candidate_metrics = {
                    variant: metrics
                    for variant, metrics in autoxai_metrics.items()
                    if variant in variants_present
                }
                instance_variant_to_method = {
                    variant: variant_to_method.get(variant, "unknown") for variant in instance_candidate_metrics
                }
                if instance_candidate_metrics:
                    scored = compute_scores(
                        candidate_metrics={int(instance_id): instance_candidate_metrics},
                        variant_to_method=instance_variant_to_method,
                        objective=autoxai_objective,
                        scaling="Std",
                        scaling_scope="instance",
                        # Pareto-front metrics are already oriented as "higher is better".
                        apply_direction=False,
                    )
                    predicted = {
                        score.method_variant: float(score.aggregated_score)
                        for score in scored
                        if score.dataset_index == int(instance_id)
                    }
                    autoxai_topk = evaluate_topk(predicted, ground_truth, k_values=config.top_k)
                    if autoxai_topk:
                        per_instance_autoxai_topk.append(autoxai_topk)

        user_topk_mean = _average_topk(per_instance_topk)
        user_autoxai_topk_mean = _average_topk(per_instance_autoxai_topk)
        per_user_topk.append(user_topk_mean)
        per_user_autoxai_topk.append(user_autoxai_topk_mean)
        per_user_summaries.append(
            {
                "user_index": user_idx,
                "tau": user.tau,
                "persona_seed": int(config.persona_seed + user_idx),
                "label_seed": int(config.label_seed + user_idx),
                "group_weights": dict(user.group_weights),
                "metric_weights": dict(user.metric_weights),
                "train_rows": int(len(X_train)),
                "test_instances_evaluated": int(instances_evaluated),
                "top_k_mean": user_topk_mean,
                "svc_top_k_mean": user_topk_mean,
                "svc_tuning": svc_tuning,
                "autoxai_top_k_mean": user_autoxai_topk_mean,
            }
        )

    aggregate = _average_topk(per_user_topk)
    aggregate_autoxai = _average_topk(per_user_autoxai_topk)
    pareto_path = _default_pareto_path(encoded_path)
    model_conf = model_config or LinearSVCConfig(random_state=config.random_state)
    persona_config_fingerprint: object
    if persona_config_raw is not None:
        persona_config_fingerprint = persona_config_raw
    else:
        persona_config_fingerprint = {"path": str(persona_config_path)}
    config_fingerprint = {
        "persona_config": persona_config_fingerprint,
        "experiment_config": asdict(config),
        "model_config": asdict(model_conf),
    }
    config_md5 = _md5_prefix(config_fingerprint)
    result = {
        "dataset": dataset_name,
        "model": model_name,
        "persona": persona_config.persona,
        "encoded_path": str(encoded_path),
        "persona_config_path": str(persona_config_path),
        "persona_config": persona_config_raw,
        "persona_sampling": {
            "metric_order": list(persona_config.metric_names()),
            "default_tau": float(persona_config.tau) if persona_config.tau is not None else None,
            "tau_override": float(config.tau) if config.tau is not None else None,
        },
        "experiment_config": {
            "test_size": config.test_size,
            "random_state": config.random_state,
            "top_k": list(config.top_k),
            "num_users": config.num_users,
            "persona_seed": config.persona_seed,
            "label_seed": config.label_seed,
            "tau": float(config.tau) if config.tau is not None else None,
            "exclude_feature_groups": list(config.exclude_feature_groups),
        },
        "config_md5": config_md5,
        "autoxai_baseline": {
            "scaling": "Std",
            "scaling_scope": "instance",
            "objective": [
                {
                    "name": term.name,
                    "metric_key": term.metric_key,
                    "direction": term.direction,
                    "weight": term.weight,
                }
                for term in autoxai_objective
            ],
            "pareto_json_used": str(pareto_path) if pareto_path.exists() else None,
        },
        "train_instances": train_ids,
        "test_instances": test_ids,
        "per_user": per_user_summaries,
        "aggregate_top_k_mean": aggregate,
        "aggregate_autoxai_top_k_mean": aggregate_autoxai,
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_name = "__".join(
            [
                "persona_simulation_summary",
                _slug(dataset_name),
                _slug(model_name),
                _slug(persona_config.persona),
                config_md5,
            ]
        )
        output_path = output_dir / f"{output_name}.json"
        output_path.write_text(json.dumps(result, indent=2))

        manifest_path = output_dir / "persona_simulation_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        except Exception:
            manifest = {}
        if not isinstance(manifest, dict):
            manifest = {}
        manifest[output_path.name] = {
            "dataset": dataset_name,
            "model": model_name,
            "persona": persona_config.persona,
            "config_md5": config_md5,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return result


def _average_topk(items: Sequence[Mapping[str, Mapping[str, object]]]) -> Dict[str, Dict[str, float]]:
    """Average nested top-k dicts: k -> metric_name[.subkey] -> value."""
    if not items:
        return {}
    sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for item in items:
        for k, metrics in item.items():
            for name, value in metrics.items():
                if isinstance(value, Mapping):
                    for sub_name, sub_value in value.items():
                        if isinstance(sub_value, (int, float)):
                            key = f"{name}.{sub_name}"
                            sums[str(k)][key] += float(sub_value)
                            counts[str(k)][key] += 1
                    continue
                if isinstance(value, (int, float)):
                    sums[str(k)][str(name)] += float(value)
                    counts[str(k)][str(name)] += 1
    means: Dict[str, Dict[str, float]] = {}
    for k, metric_sums in sums.items():
        means[k] = {}
        for name, total in metric_sums.items():
            denom = counts[k].get(name, 0)
            means[k][name] = total / denom if denom else 0.0
    return means


def _default_pareto_path(encoded_path: Path) -> Path:
    base = encoded_path.stem.replace("_encoded", "")
    return DEFAULT_RESULTS_ROOT / "pareto_fronts" / f"{base}.json"



def _load_pareto_metrics_for_encoded(encoded_path: Path) -> Tuple[Dict[int, Dict[str, Dict[str, float]]], Dict[str, str]]:
    """
    Load raw per-instance Pareto-front metrics for AutoXAI baseline scoring.

    Returns:
      - dataset_index -> method_variant -> metrics dict
      - method_variant -> method
    """
    # We separately load the raw Pareto metrics for the AutoXAI baseline scoring.
    # The baseline scorer expects raw per-instance metric dicts keyed by method_variant.
    path = _default_pareto_path(encoded_path)
    if not path.exists():
        return {}, {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    instances = payload.get("instances") or []
    candidate_metrics: Dict[int, Dict[str, Dict[str, float]]] = {}
    variant_to_method: Dict[str, str] = {}
    for instance in instances:
        dataset_index = instance.get("dataset_index")
        if not isinstance(dataset_index, (int, float)):
            continue
        idx = int(dataset_index)
        variants: Dict[str, Dict[str, float]] = {}
        for entry in instance.get("pareto_front") or []:
            variant = entry.get("method_variant")
            method = entry.get("method")
            metrics = entry.get("metrics") or {}
            if not isinstance(variant, str) or not variant:
                continue
            if isinstance(method, str) and method:
                variant_to_method[variant] = method
            if not isinstance(metrics, Mapping):
                continue
            cleaned: Dict[str, float] = {}
            for k, v in metrics.items():
                if isinstance(k, str) and isinstance(v, (int, float)):
                    cleaned[k] = float(v)
            if cleaned:
                variants[variant] = cleaned
        if variants:
            candidate_metrics[idx] = variants
    return candidate_metrics, variant_to_method


def _persist_processed_data(dataset: PairwisePreferenceData, experiment_dir: Path) -> None:
    processed_dir = experiment_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    dataset.train_features.to_parquet(processed_dir / "train_features.parquet", index=False)
    dataset.train_labels.to_frame().to_parquet(processed_dir / "train_labels.parquet", index=False)
    splits_payload = {
        "dataset": dataset.dataset_name,
        "model": dataset.model_name,
        "encoded_path": str(dataset.encoded_path),
        "pair_labels_path": str(dataset.pair_labels_path),
        "feature_columns": dataset.feature_columns,
        "train_instances": dataset.train_instances,
        "test_instances": dataset.test_instances,
    }
    (processed_dir / "splits.json").write_text(json.dumps(splits_payload, indent=2))

    test_dir = processed_dir / "test_instances"
    test_dir.mkdir(exist_ok=True)
    for instance in dataset.test_data:
        instance_dir = test_dir / f"instance_{instance.instance_index}"
        instance_dir.mkdir(exist_ok=True)
        instance.candidates.to_parquet(instance_dir / "candidates.parquet", index=False)
        instance.pair_labels.to_parquet(instance_dir / "pair_labels.parquet", index=False)


def _evaluate_model(
    *,
    model: LinearSVCPreferenceModel,
    dataset: PairwisePreferenceData,
    experiment_dir: Path,
    feature_columns: Sequence[str],
    top_k: Iterable[int],
) -> dict:
    predictions_dir = experiment_dir / "predictions"
    predictions_dir.mkdir(exist_ok=True)
    metrics: dict = {}
    for instance in dataset.test_data:
        scores = model.score_candidates(instance.candidates, feature_columns)
        scores_df = (
            scores.reset_index()
            .rename(columns={"index": "method_variant"})
            .sort_values(by="score", ascending=False)
        )
        scores_df.to_parquet(
            predictions_dir / f"instance_{instance.instance_index}_scores.parquet",
            index=False,
        )
        ground_truth = build_ground_truth_order(instance.pair_labels)
        metric = evaluate_topk(
            dict(scores.items()),
            ground_truth,
            k_values=top_k,
        )
        metrics[str(instance.instance_index)] = {
            "ground_truth": ground_truth,
            "top_k": metric,
        }
    return metrics
