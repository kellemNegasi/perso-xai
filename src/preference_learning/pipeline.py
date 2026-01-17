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
import yaml

from src.baseline.autoxai_scoring import compute_scores
from src.baseline.autoxai_types import ObjectiveTerm

from .config import ExperimentConfig
from .data import (
    PairwisePreferenceData,
    PreferenceDatasetBuilder,
    _differences_for_instance,
    _dedupe_candidates_by_variant,
    _infer_feature_columns,
)
from .evaluation import build_ground_truth_order, evaluate_topk
from .models import LinearSVCConfig, LinearSVCPreferenceModel
from .persona import HierarchicalDirichletUser, load_persona_config
from .ranker import PersonaPairwiseRanker
from .auto_score_utils import (
    build_autoxai_hpo_per_instance_scores,
    load_autoxai_hpo_candidate_scores,
)

DEFAULT_RESULTS_ROOT = Path("results") / "full_run_dec8"
DEFAULT_PROCESSED_DIR = DEFAULT_RESULTS_ROOT / "preference_learning"

# Some Pareto-front generators persist "lower is better" metrics already negated so the
# Pareto optimisation is a pure maximisation problem (see `generate_pareto_fronts.py`).
# Other runs persist raw (non-negated) metrics. We detect this at load time.
PARETO_MINIMIZE_METRICS: frozenset[str] = frozenset(
    {
        "infidelity",
        "non_sensitivity_violation_fraction",
        "non_sensitivity_delta_mean",
        "relative_input_stability",
        "covariate_complexity",
    }
)


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
        persona_config_raw = yaml.safe_load(persona_config_path.read_text(encoding="utf-8"))
    except Exception:
        persona_config_raw = None
    pareto_metrics, variant_to_method, pareto_metrics_already_oriented = _load_pareto_metrics_for_encoded(
        encoded_path
    )
    (
        hpo_overall_entries,
        hpo_per_instance_entries,
        hpo_trial_variant_scores,
        hpo_metrics_dir_used,
    ) = load_autoxai_hpo_candidate_scores(
        encoded_path,
        dataset_name=str(dataset_name),
        model_name=str(model_name),
        default_results_root=DEFAULT_RESULTS_ROOT,
    )
    allowed_test_instances = set(int(idx) for idx in test_ids)
    allowed_test_variants = set(
        encoded_df.loc[encoded_df["instance_index"].isin(test_ids), "method_variant"].astype(str).tolist()
    )
    autoxai_hpo_overall_variant_scores = {
        variant: float(score)
        for variant, score in hpo_trial_variant_scores.items()
        if variant in allowed_test_variants
    }
    autoxai_hpo_per_instance_scores = build_autoxai_hpo_per_instance_scores(
        hpo_per_instance_entries,
        allowed_instances=allowed_test_instances,
        allowed_variants=allowed_test_variants,
    )
    autoxai_objective: list[ObjectiveTerm] = []
    if config.autoxai_enabled:
        # AutoXAI-style baseline objective uses the paper's three metrics.
        # We preserve the top-level weighting scheme:
        #   robustness = 1, correctness = 2 (infidelity), compactness = 0.5
        autoxai_objective = [
            ObjectiveTerm(name="robustness", metric_key="relative_input_stability", direction="min", weight=1.0),
            ObjectiveTerm(name="infidelity", metric_key="infidelity", direction="min", weight=2.0),
            ObjectiveTerm(
                name="compactness",
                metric_key="compactness_effective_features",
                direction="max",
                weight=0.5,
            ),
        ]
        if config.autoxai_include_all_metrics:
            existing_keys = {term.metric_key for term in autoxai_objective}
            all_metric_keys: set[str] = set()
            for variants in pareto_metrics.values():
                for metric_blob in variants.values():
                    all_metric_keys.update(metric_blob.keys())
            remaining = sorted(key for key in all_metric_keys if key not in existing_keys)
            boosted_weight_keys = {
                # Deletion-check completeness metrics.
                "completeness_drop",
                "completeness_random_drop",
                "completeness_score",
                # Non-sensitivity metrics.
                "non_sensitivity_violation_fraction",
                "non_sensitivity_safe_fraction",
                "non_sensitivity_zero_features",
                "non_sensitivity_delta_mean",
                # Monotonicity metric.
                "monotonicity",
            }
            autoxai_objective.extend(
                ObjectiveTerm(
                    name=key,
                    metric_key=key,
                    direction="max",
                    weight=2.0 if key in boosted_weight_keys else 0.3,
                )
                for key in remaining
            )

    per_user_summaries: list[dict] = []
    per_user_topk: list[Dict[str, Dict[str, float]]] = []
    per_user_autoxai_topk: list[Dict[str, Dict[str, float]]] = []
    per_user_autoxai_hpo_overall_topk: list[Dict[str, Dict[str, float]]] = []
    per_user_autoxai_hpo_per_instance_topk: list[Dict[str, Dict[str, float]]] = []

    # === USER SIMULATION START ===
    # We simulate `num_users` independent users by (1) sampling a persona weight vector (seeded by
    # persona_seed + user_idx) and (2) sampling pairwise preference labels (seeded by label_seed + user_idx).
    # For each simulated user, we train a separate LinearSVC preference model.
    for user_idx in range(config.num_users):
        user = HierarchicalDirichletUser(
            persona_config,
            seed=config.persona_seed + user_idx,
            tau=config.tau,
            concentration_c=config.concentration_c,
        )
        ranker = PersonaPairwiseRanker(
            user=user,
            rng=np.random.default_rng(config.label_seed + user_idx),
        )

        train_rows: list[np.ndarray] = []
        train_labels: list[int] = []
        for instance_id in train_ids:
            instance_df = encoded_df.loc[encoded_df["instance_index"] == instance_id].copy()
            instance_df = _dedupe_candidates_by_variant(instance_df)
            if instance_df.empty:
                continue
            # === CANDIDATES LABELED (SIMULATED USER PREFERENCES) ===
            # Produces a DataFrame of pairwise preferences with label encoding:
            #   label=0 => pair_1 preferred, label=1 => pair_2 preferred.
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
        if len(train_rows) != len(train_labels):
            raise ValueError(
                "Inconsistent training data lengths: "
                f"features={len(train_rows)} labels={len(train_labels)}. "
                "This can happen if encoded candidates contain duplicate method_variant rows."
            )

        X_train = pd.DataFrame(train_rows, columns=feature_columns)
        y_train = pd.Series(train_labels, name="label")

        # === MODEL TRAINED (ONE SVM PER SIMULATED USER) ===
        # The LinearSVC is trained on pairwise difference vectors with balanced labels in {-1, +1}
        # (see `src/preference_learning/data.py::_differences_for_instance` for the +/- construction).
        model_conf = model_config or LinearSVCConfig(random_state=config.random_state)
        model = LinearSVCPreferenceModel(model_conf)
        svc_tuning = None
        if getattr(model_conf, "tune", False):
            svc_tuning = model.tune_and_fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)

        per_instance_topk: list[Dict[str, Dict[str, float]]] = []
        per_instance_autoxai_topk: list[Dict[str, Dict[str, float]]] = []
        per_instance_autoxai_hpo_overall_topk: list[Dict[str, Dict[str, float]]] = []
        per_instance_autoxai_hpo_per_instance_topk: list[Dict[str, Dict[str, float]]] = []
        instances_evaluated = 0
        for instance_id in test_ids:
            instance_df = encoded_df.loc[encoded_df["instance_index"] == instance_id].copy()
            instance_df = _dedupe_candidates_by_variant(instance_df)
            if instance_df.empty:
                continue
            # === CANDIDATES LABELED (SIMULATED TEST-TIME PREFERENCES) ===
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

            variants_present = set(instance_df["method_variant"].astype(str).tolist())
            if autoxai_hpo_overall_variant_scores:
                predicted_overall = {
                    variant: float(score)
                    for variant, score in autoxai_hpo_overall_variant_scores.items()
                    if variant in variants_present
                }
                if predicted_overall:
                    topk = evaluate_topk(predicted_overall, ground_truth, k_values=config.top_k)
                    if topk:
                        per_instance_autoxai_hpo_overall_topk.append(topk)
            if autoxai_hpo_per_instance_scores:
                instance_scores = autoxai_hpo_per_instance_scores.get(int(instance_id)) or {}
                predicted_per_instance = {
                    variant: float(score) for variant, score in instance_scores.items() if variant in variants_present
                }
                if predicted_per_instance:
                    topk = evaluate_topk(predicted_per_instance, ground_truth, k_values=config.top_k)
                    if topk:
                        per_instance_autoxai_hpo_per_instance_topk.append(topk)

            if config.autoxai_enabled:
                autoxai_metrics = pareto_metrics.get(int(instance_id))
                if autoxai_metrics:
                    instance_candidate_metrics = {
                        variant: metrics
                        for variant, metrics in autoxai_metrics.items()
                        if variant in variants_present
                    }
                    instance_variant_to_method = {
                        variant: variant_to_method.get(variant, "unknown")
                        for variant in instance_candidate_metrics
                    }
                    if instance_candidate_metrics:
                        scored = compute_scores(
                            candidate_metrics={int(instance_id): instance_candidate_metrics},
                            variant_to_method=instance_variant_to_method,
                            objective=autoxai_objective,
                            scaling="Std",
                            scaling_scope="instance",
                            apply_direction=not pareto_metrics_already_oriented,
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
        user_autoxai_topk_mean = _average_topk(per_instance_autoxai_topk) if config.autoxai_enabled else {}
        user_autoxai_hpo_overall_topk_mean = _average_topk(per_instance_autoxai_hpo_overall_topk)
        user_autoxai_hpo_per_instance_topk_mean = _average_topk(per_instance_autoxai_hpo_per_instance_topk)
        per_user_topk.append(user_topk_mean)
        if config.autoxai_enabled:
            per_user_autoxai_topk.append(user_autoxai_topk_mean)
        per_user_autoxai_hpo_overall_topk.append(user_autoxai_hpo_overall_topk_mean)
        per_user_autoxai_hpo_per_instance_topk.append(user_autoxai_hpo_per_instance_topk_mean)
        per_user_summaries.append(
            {
                "user_index": user_idx,
                "tau": user.tau,
                "concentration_c": user.concentration_c,
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
                "autoxai_hpo_overall_top_k_mean": user_autoxai_hpo_overall_topk_mean,
                "autoxai_hpo_per_instance_top_k_mean": user_autoxai_hpo_per_instance_topk_mean,
            }
        )

    aggregate = _average_topk(per_user_topk)
    aggregate_autoxai = _average_topk(per_user_autoxai_topk) if config.autoxai_enabled else {}
    aggregate_autoxai_hpo_overall = _average_topk(per_user_autoxai_hpo_overall_topk)
    aggregate_autoxai_hpo_per_instance = _average_topk(per_user_autoxai_hpo_per_instance_topk)
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
            "concentration_c_override": float(config.concentration_c) if config.concentration_c is not None else None,
        },
        "experiment_config": {
            "test_size": config.test_size,
            "random_state": config.random_state,
            "top_k": list(config.top_k),
            "num_users": config.num_users,
            "persona_seed": config.persona_seed,
            "label_seed": config.label_seed,
            "tau": float(config.tau) if config.tau is not None else None,
            "concentration_c": float(config.concentration_c) if config.concentration_c is not None else None,
            "exclude_feature_groups": list(config.exclude_feature_groups),
            "autoxai_include_all_metrics": bool(config.autoxai_include_all_metrics),
            "autoxai_enabled": bool(config.autoxai_enabled),
        },
        "config_md5": config_md5,
        "autoxai_baseline": (
            {
                "scaling": "Std",
                "scaling_scope": "instance",
                "pareto_metrics_already_oriented": bool(pareto_metrics_already_oriented),
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
            }
            if config.autoxai_enabled
            else None
        ),
        "train_instances": train_ids,
        "test_instances": test_ids,
        "per_user": per_user_summaries,
        "aggregate_top_k_mean": aggregate,
        "aggregate_autoxai_top_k_mean": aggregate_autoxai,
        "aggregate_autoxai_hpo_overall_top_k_mean": aggregate_autoxai_hpo_overall,
        "aggregate_autoxai_hpo_per_instance_top_k_mean": aggregate_autoxai_hpo_per_instance,
        "autoxai_hpo_source": (
            {
                "metrics_results_dir": str(hpo_metrics_dir_used) if hpo_metrics_dir_used is not None else None,
                "trial_variants": int(len(hpo_trial_variant_scores)),
                "overall_entries": int(len(hpo_overall_entries)),
                "per_instance_entries": int(len(hpo_per_instance_entries)),
            }
            if (hpo_trial_variant_scores or hpo_overall_entries or hpo_per_instance_entries)
            else None
        ),
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
    # Prefer a Pareto-front JSON that lives alongside the encoded parquet (e.g. under
    # `results/<run_id>/pareto_fronts/<dataset__model>.json`). Fall back to the
    # historical `DEFAULT_RESULTS_ROOT` when no adjacent Pareto front exists.
    for parent in encoded_path.parents:
        candidate = parent / "pareto_fronts" / f"{base}.json"
        if candidate.exists():
            return candidate
    return DEFAULT_RESULTS_ROOT / "pareto_fronts" / f"{base}.json"


def _load_pareto_metrics_for_encoded(
    encoded_path: Path,
) -> Tuple[Dict[int, Dict[str, Dict[str, float]]], Dict[str, str], bool]:
    """
    Load raw per-instance Pareto-front metrics for AutoXAI baseline scoring.

    Returns:
      - dataset_index -> method_variant -> metrics dict
      - method_variant -> method
      - whether metrics are already oriented ("higher is better") for minimize metrics
    """
    # We separately load the raw Pareto metrics for the AutoXAI baseline scoring.
    # The baseline scorer expects raw per-instance metric dicts keyed by method_variant.
    path = _default_pareto_path(encoded_path)
    if not path.exists():
        return {}, {}, False
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

    values_checked: list[float] = []
    for variants in candidate_metrics.values():
        for metrics in variants.values():
            for key in PARETO_MINIMIZE_METRICS:
                value = metrics.get(key)
                if isinstance(value, (int, float)):
                    values_checked.append(float(value))
    # If most values for known non-negative metrics are negative, we assume the Pareto file
    # already negated lower-is-better metrics.
    already_oriented = False
    if values_checked:
        negative = sum(1 for v in values_checked if v < 0)
        already_oriented = (negative / len(values_checked)) > 0.5

    return candidate_metrics, variant_to_method, already_oriented


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
    allowed_instances = set(int(idx) for idx in dataset.test_instances)
    allowed_variants: set[str] = set()
    for instance in dataset.test_data:
        allowed_variants.update(instance.candidates["method_variant"].astype(str).tolist())
    hpo_overall_entries, hpo_per_instance_entries, hpo_trial_variant_scores, _ = load_autoxai_hpo_candidate_scores(
        dataset.encoded_path,
        dataset_name=str(dataset.dataset_name),
        model_name=str(dataset.model_name),
        default_results_root=DEFAULT_RESULTS_ROOT,
    )
    autoxai_hpo_overall_variant_scores = {
        variant: float(score) for variant, score in hpo_trial_variant_scores.items() if variant in allowed_variants
    }
    autoxai_hpo_per_instance_scores = build_autoxai_hpo_per_instance_scores(
        hpo_per_instance_entries,
        allowed_instances=allowed_instances,
        allowed_variants=allowed_variants,
    )
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
        autoxai_hpo_overall_topk: Dict[str, Dict[str, float]] = {}
        autoxai_hpo_per_instance_topk: Dict[str, Dict[str, float]] = {}
        if ground_truth:
            variants_present = set(instance.candidates["method_variant"].astype(str).tolist())
            if autoxai_hpo_overall_variant_scores:
                predicted_overall = {
                    variant: float(score)
                    for variant, score in autoxai_hpo_overall_variant_scores.items()
                    if variant in variants_present
                }
                autoxai_hpo_overall_topk = evaluate_topk(predicted_overall, ground_truth, k_values=top_k)
            if autoxai_hpo_per_instance_scores:
                instance_scores = autoxai_hpo_per_instance_scores.get(int(instance.instance_index)) or {}
                predicted_per_instance = {
                    variant: float(score) for variant, score in instance_scores.items() if variant in variants_present
                }
                autoxai_hpo_per_instance_topk = evaluate_topk(predicted_per_instance, ground_truth, k_values=top_k)
        metrics[str(instance.instance_index)] = {
            "ground_truth": ground_truth,
            "top_k": metric,
            "autoxai_hpo_overall_top_k": autoxai_hpo_overall_topk,
            "autoxai_hpo_per_instance_top_k": autoxai_hpo_per_instance_topk,
        }
    return metrics
