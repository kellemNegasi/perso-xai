#!/usr/bin/env python3
"""
Encode Pareto-front metrics into ranker-friendly tabular features.

For every Pareto summary JSON, this script builds a pandas DataFrame where each
row corresponds to a specific explanation candidate (method variant) evaluated
for a dataset instance. The DataFrame captures dataset metadata encodings,
explainer metadata (deterministic IDs + family flags), normalized
hyperparameter embeddings, and z-normalised metric scores (with
lower-is-better metrics negated before scaling). Each encoded DataFrame is
persisted alongside the Pareto files so downstream ranking pipelines can load
them directly.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import yaml

DEFAULT_RESULTS_ROOT = Path("results") / "full_run_dec8"
DEFAULT_PARETO_DIR = DEFAULT_RESULTS_ROOT / "pareto_fronts"
DEFAULT_METADATA_DIR = DEFAULT_RESULTS_ROOT / "metadata"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_ROOT / "encoded_pareto_fronts"
DEFAULT_HPARAM_CONFIG = Path("src") / "configs" / "explainer_hyperparameters.yml"

DATASET_META_FIELDS = (
    "log_feature_count_z",
    "class_entropy_z",
    "categorical_to_numerical_ratio_z",
    "has_sensitive_attributes",
    "high_stakes_domain",
    "log_dataset_size_z",
    "mean_of_means_z",
    "std_of_means_z",
    "mean_variance_z",
    "max_variance_z",
    "mean_skewness_z",
    "std_skewness_z",
    "max_kurtosis_z",
    "mean_std_z",
    "std_std_z",
    "max_std_z",
    "mean_range_z",
    "max_range_z",
    "mean_cardinality_z",
    "max_cardinality_z",
    "mean_cat_entropy_z",
    "std_cat_entropy_z",
    "mean_top_freq_z",
    "max_top_freq_z",
    "landmark_acc_knn1_z",
    "landmark_acc_gaussian_nb_z",
    "landmark_acc_decision_stump_z",
    "landmark_acc_logreg_z",
)

EXPLAINER_META_FIELDS = (
    "type",
    "is_additive_attribution",
    "is_gradient_based",
    "is_causal",
    "is_perturbation_based",
)

LOG_SCALED_HPARAMS = {
    "background_sample_size",
    "lime_num_samples",
    "causal_shap_coalitions",
    "ig_steps",
}

NEGATE_METRICS = {
    "infidelity",
    "non_sensitivity_violation_fraction",
    "non_sensitivity_delta_mean",
    "covariate_complexity",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for encoding Pareto fronts into tabular features."""
    parser = argparse.ArgumentParser(
        description="Encode Pareto-front metrics into feature-rich DataFrames.",
    )
    parser.add_argument(
        "--pareto-dir",
        type=Path,
        default=DEFAULT_PARETO_DIR,
        help=f"Directory containing Pareto-front JSON files (default: {DEFAULT_PARETO_DIR}).",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=DEFAULT_METADATA_DIR,
        help=f"Directory holding dataset/explainer metadata JSONs (default: {DEFAULT_METADATA_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory where encoded DataFrames will be stored (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--hyperparameters",
        type=Path,
        default=DEFAULT_HPARAM_CONFIG,
        help="Path to explainer hyperparameter grid config (default: src/configs/explainer_hyperparameters.yml).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point that loads metadata/stats and writes encoded parquet files."""
    args = parse_args(argv)
    dataset_meta = load_dataset_metadata(args.metadata_dir / "dataset_metadata.json")
    explainer_meta = load_explainer_metadata(args.metadata_dir / "explainers_metadata.json")
    hyper_cfg = load_hyperparameter_config(args.hyperparameters)
    hyper_stats, hyperparam_universe = build_hyperparameter_stats(hyper_cfg)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pareto_files = sorted(args.pareto_dir.glob("*_pareto.json"))
    if not pareto_files:
        raise FileNotFoundError(f"No pareto JSON files found under {args.pareto_dir}")

    for pareto_path in pareto_files:
        df = encode_pareto_file(
            pareto_path,
            dataset_meta=dataset_meta,
            explainer_meta=explainer_meta,
            hyper_stats=hyper_stats,
            hyperparam_universe=hyperparam_universe,
        )
        output_path = args.output_dir / f"{pareto_path.stem}_encoded.parquet"
        df.to_parquet(output_path, index=False)
        print(f"Wrote encoded DataFrame with {len(df)} rows to {output_path}")


def load_dataset_metadata(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load dataset metadata JSON and return the `datasets` mapping."""
    data = _load_json(path)
    datasets = data.get("datasets")
    if not isinstance(datasets, Mapping):
        raise ValueError(f"Malformed dataset metadata at {path}")
    return datasets  # type: ignore[return-value]


def load_explainer_metadata(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load explainer metadata JSON and return the `explainers` mapping."""
    data = _load_json(path)
    explainers = data.get("explainers")
    if not isinstance(explainers, Mapping):
        raise ValueError(f"Malformed explainer metadata at {path}")
    return explainers  # type: ignore[return-value]


def load_hyperparameter_config(path: Path) -> Dict[str, Any]:
    """Load explainer hyperparameter grid config (YAML) into a dict."""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_hyperparameter_stats(
    config: Mapping[str, Any],
) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], List[str]]:
    """Pre-compute mean/std (plus log-scaling flag) for each explainer hyperparameter."""
    stats: Dict[str, Dict[str, Dict[str, float]]] = {}
    universe: List[str] = []
    explainers_cfg = config.get("explainers") or {}
    for explainer_name, grid in explainers_cfg.items():
        method_stats: Dict[str, Dict[str, float]] = {}
        for param_name, values in (grid or {}).items():
            numeric_values = [_coerce_float(value) for value in values]
            clean_values = [value for value in numeric_values if value is not None]
            if not clean_values:
                continue
            transformed = [
                math.log1p(value) if param_name in LOG_SCALED_HPARAMS else value
                for value in clean_values
            ]
            mean = sum(transformed) / len(transformed)
            if len(transformed) > 1:
                variance = sum((value - mean) ** 2 for value in transformed) / len(transformed)
                std = math.sqrt(variance)
            else:
                std = 0.0
            method_stats[param_name] = {
                "mean": mean,
                "std": std,
                "scale": "log" if param_name in LOG_SCALED_HPARAMS else "linear",
            }
            if param_name not in universe:
                universe.append(param_name)
        if method_stats:
            stats[explainer_name] = method_stats
    universe.sort()
    return stats, universe


def encode_pareto_file(
    path: Path,
    *,
    dataset_meta: Mapping[str, Mapping[str, Any]],
    explainer_meta: Mapping[str, Mapping[str, Any]],
    hyper_stats: Mapping[str, Mapping[str, Dict[str, float]]],
    hyperparam_universe: Sequence[str],
) -> pd.DataFrame:
    """
    Convert a single Pareto JSON summary into a feature-rich DataFrame.

    Steps:
    1) Load dataset/model identifiers from the Pareto payload and validate metadata coverage.
    2) Pre-compute one-hot templates for all dataset and explainer IDs so every row has the
       same columns regardless of the active dataset/explainer in the instance.
    3) For each instance in the payload, iterate through Pareto candidates (method variants):
         - Parse and sign-flip raw metrics so larger-is-better.
         - Pull dataset-level meta features and explainer family flags/IDs.
         - Parse hyperparameters from the method variant name, z-score them within that explainer
           family, and add applicability indicators for parameters irrelevant to the method.
         - Accumulate rows containing metadata, hyperparameters, and raw metrics.
    4) Within each instance, z-normalize metrics across its candidates and emit the completed rows.
    5) Concatenate all instance rows into a DataFrame to be saved to parquet by the caller.
    """
    payload = _load_json(path)
    dataset_name = payload.get("dataset")
    model_name = payload.get("model")
    if dataset_name not in dataset_meta:
        raise KeyError(f"Dataset '{dataset_name}' missing from dataset metadata.")
    dataset_features = dataset_meta[dataset_name]
    dataset_ids = _sorted_unique_ints(
        record.get("dataset_id") for record in dataset_meta.values()
    )
    explainer_ids = _sorted_unique_ints(
        record.get("explainer_id") for record in explainer_meta.values()
    )
    instances = payload.get("instances") or []

    records: List[Dict[str, Any]] = []
    for instance in instances:
        rows: List[Dict[str, Any]] = []
        pareto_metrics: List[str] = list(instance.get("pareto_metrics") or [])
        for entry in instance.get("pareto_front") or []:
            method = entry.get("method")
            if method not in explainer_meta:
                raise KeyError(f"Explainer '{method}' missing from explainer metadata.")
            method_variant = entry.get("method_variant")
            metrics = transform_metrics(entry.get("metrics") or {})
            expl_meta = explainer_meta[method]
            dataset_id = dataset_features.get("dataset_id")
            explainer_id = expl_meta.get("explainer_id")
            hparam_values = parse_variant_hyperparameters(method_variant)
            hp_features, applicability = encode_hyperparameters(
                method,
                hparam_values,
                hyper_stats=hyper_stats,
                hyperparam_universe=hyperparam_universe,
            )
            row: Dict[str, Any] = {
                "dataset": dataset_name,
                "model": model_name,
                "instance_index": instance.get("dataset_index"),
                "method": method,
                "method_variant": method_variant,
            }
            if dataset_id is None:
                raise KeyError(f"Dataset '{dataset_name}' is missing dataset_id in metadata.")
            for value in dataset_ids:
                row[f"dataset_id_oh_{value}"] = 1 if value == dataset_id else 0
            for field in DATASET_META_FIELDS:
                row[f"dataset_{field}"] = dataset_features.get(field)
            if explainer_id is None:
                raise KeyError(f"Explainer '{method}' is missing explainer_id in metadata.")
            for value in explainer_ids:
                row[f"explainer_id_oh_{value}"] = 1 if value == explainer_id else 0
            for field in EXPLAINER_META_FIELDS:
                key = f"explainer_{field}" if field != "type" else "explainer_type"
                row[key] = expl_meta.get(field)
            row.update(hp_features)
            row.update(applicability)
            row["_metrics_raw"] = metrics
            rows.append(row)
        records.extend(normalize_instance_metrics(rows, pareto_metrics))
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    return df


def transform_metrics(metrics: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    """Convert metrics to floats and negate lower-is-better metrics."""
    transformed: Dict[str, Optional[float]] = {}
    for key, value in metrics.items():
        numeric = _coerce_float(value)
        if numeric is None:
            transformed[key] = None
            continue
        transformed[key] = -numeric if key in NEGATE_METRICS else numeric
    return transformed


def normalize_instance_metrics(
    rows: List[Dict[str, Any]],
    metric_names: Sequence[str],
) -> List[Dict[str, Any]]:
    """
    Z-normalize metrics across candidates within a single instance.
    Candidates here include hyperparameter variants of the same method and variants of
    other methods for that instance, and normalization uses all of them together.
    """
    if not rows:
        return []
    per_metric_values: Dict[str, List[float]] = {name: [] for name in metric_names}
    per_row_metrics: List[Dict[str, Optional[float]]] = []

    for row in rows:
        metrics = row.pop("_metrics_raw", {}) or {}
        row_metrics: Dict[str, Optional[float]] = {}
        for metric in metric_names:
            value = metrics.get(metric)
            if value is not None:
                per_metric_values[metric].append(value)
            row_metrics[metric] = value
        per_row_metrics.append(row_metrics)

    stats: Dict[str, Tuple[float, float]] = {}
    for metric, values in per_metric_values.items():
        if values:
            mean = sum(values) / len(values)
            if len(values) > 1:
                variance = sum((value - mean) ** 2 for value in values) / len(values)
                std = math.sqrt(variance)
            else:
                std = 0.0
            stats[metric] = (mean, std)
        else:
            stats[metric] = (0.0, 0.0)

    encoded_rows: List[Dict[str, Any]] = []
    for row, row_metrics in zip(rows, per_row_metrics):
        for metric in metric_names:
            value = row_metrics.get(metric)
            mean, std = stats.get(metric, (0.0, 0.0))
            if value is None or std == 0.0:
                row[metric] = 0.0 if std != 0.0 else 0.0
            else:
                row[metric] = (value - mean) / std
        encoded_rows.append(row)
    return encoded_rows


def parse_variant_hyperparameters(variant: Optional[str]) -> Dict[str, float]:
    """Extract numeric hyperparameters from a method_variant string (name-value pairs)."""
    if not variant:
        return {}
    parts = variant.split("__")
    values: Dict[str, float] = {}
    for part in parts[1:]:
        if "-" not in part:
            continue
        name, raw_value = part.split("-", 1)
        numeric = _coerce_float(raw_value)
        if numeric is not None:
            values[name] = numeric
    return values


def encode_hyperparameters(
    method: str,
    values: Mapping[str, float],
    *,
    hyper_stats: Mapping[str, Mapping[str, Dict[str, float]]],
    hyperparam_universe: Sequence[str],
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """Produce z-scored hyperparameter features plus applicability bits for every parameter."""
    features: Dict[str, float] = {}
    applicability: Dict[str, int] = {}
    method_stats = hyper_stats.get(method, {})
    for param in hyperparam_universe:
        applicable = param in method_stats
        applicability[f"is_applicable_{param}"] = 1 if applicable else 0
        if not applicable:
            features[f"hp_{param}"] = 0.0
            continue
        stats = method_stats[param]
        value = values.get(param)
        if value is None:
            features[f"hp_{param}"] = 0.0
            continue
        transformed = math.log1p(value) if stats.get("scale") == "log" else value
        mean = stats.get("mean", 0.0)
        std = stats.get("std", 0.0)
        if std == 0.0:
            features[f"hp_{param}"] = 0.0
        else:
            features[f"hp_{param}"] = (transformed - mean) / std
    return features, applicability


def _coerce_float(value: Any) -> Optional[float]:
    """Best-effort conversion to float; returns None on failure."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _sorted_unique_ints(values: Iterable[Any]) -> List[int]:
    """Convert an iterable to a sorted list of unique ints, skipping invalid entries."""
    unique: set[int] = set()
    for value in values:
        try:
            unique.add(int(value))
        except (TypeError, ValueError):
            continue
    return sorted(unique)


def _load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file, raising if the file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required JSON file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
