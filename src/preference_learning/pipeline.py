"""High-level pipeline to train and evaluate preference-learning models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

from .config import ExperimentConfig
from .data import PairwisePreferenceData, PreferenceDatasetBuilder
from .evaluation import build_ground_truth_order, evaluate_topk
from .models import LinearSVCConfig, LinearSVCPreferenceModel

DEFAULT_RESULTS_ROOT = Path("results") / "full_run_dec8"
DEFAULT_PROCESSED_DIR = DEFAULT_RESULTS_ROOT / "preference_learning"


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
    dataset = builder.build(test_size=config.test_size, random_state=config.random_state)
    output_root = output_dir or DEFAULT_PROCESSED_DIR
    experiment_dir = output_root / persona / encoded_path.stem.replace("_encoded", "")
    _persist_processed_data(dataset, experiment_dir)

    model_conf = model_config or LinearSVCConfig(random_state=config.random_state)
    model = LinearSVCPreferenceModel(model_conf)
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
        "experiment_config": {
            "test_size": config.test_size,
            "random_state": config.random_state,
            "top_k": list(config.top_k),
        },
    }
    (experiment_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (experiment_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return metrics


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
