#!/usr/bin/env python3
"""
Preprocess dataset- and explainer-level metadata for downstream ranking models.

This utility scans the dataset registry entries, loads the requested datasets,
and derives normalized metadata signals that can be shared across every
explanation candidate originating from the same dataset. It also inspects the
explainer registry to emit deterministic explainer IDs alongside hand-crafted
family flags (additive, gradient-based, causal, perturbation-based). The
resulting payloads are written under ``results/.../metadata`` so they can be
co-located with Pareto fronts and reused by ranker training pipelines.
"""

from __future__ import annotations

import argparse
import gzip
import importlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    import pandas as pd

    _HAS_PANDAS = True
except Exception:  # pragma: no cover - defensive import
    pd = None  # type: ignore
    _HAS_PANDAS = False

from src.datasets.adapters import LoaderDatasetAdapter
from src.datasets.tabular import TabularDataset
from src.orchestrators.registry import DatasetRegistry, ExplainerRegistry

DEFAULT_RESULTS_ROOT = Path("results") / "full_run_dec8"
DEFAULT_METADATA_DIR = DEFAULT_RESULTS_ROOT / "metadata"
DEFAULT_DATASETS = (
    "open_compas",
    "openml_bank_marketing",
    "openml_german_credit",
)

SENSITIVE_KEYWORDS = (
    "race",
    "sex",
    "gender",
    "age",
    "nationality",
    "marital",
)
HIGH_STAKES_KEYWORDS = (
    "medical",
    "clinic",
    "health",
    "legal",
    "law",
    "justice",
    "court",
    "financial",
    "finance",
    "credit",
    "bank",
    "loan",
)

OPENML_CACHE_ROOT = Path("data") / "cache" / "openml" / "openml" / "openml.org" / "api" / "v1" / "json" / "data"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate normalized dataset metadata encodings for ranker inputs.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help=(
            "Datasets to process (defaults to open_compas, openml_bank_marketing, "
            "openml_german_credit). Use '*' to process every tabular dataset from the registry."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_METADATA_DIR,
        help=f"Directory where dataset metadata will be stored (default: {DEFAULT_METADATA_DIR}).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    dataset_registry = DatasetRegistry()
    dataset_names = normalise_dataset_names(args.datasets, dataset_registry)
    if not dataset_names:
        raise ValueError("No datasets resolved from the registry.")

    dataset_payload = build_dataset_metadata(dataset_registry, dataset_names)
    explainer_payload = build_explainer_metadata()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "dataset_metadata.json"
    dataset_path.write_text(json.dumps(dataset_payload, indent=2), encoding="utf-8")
    explainer_path = output_dir / "explainers_metadata.json"
    explainer_path.write_text(json.dumps(explainer_payload, indent=2), encoding="utf-8")
    print(
        f"Wrote dataset metadata for {len(dataset_payload['datasets'])} datasets to {dataset_path}",
    )
    print(
        f"Wrote explainer metadata for {len(explainer_payload['explainers'])} explainers to {explainer_path}",
    )


def build_dataset_metadata(
    registry: DatasetRegistry,
    dataset_names: Sequence[str],
) -> Dict[str, Any]:
    dataset_records: Dict[str, Dict[str, Any]] = {}
    log_feature_values: Dict[str, float] = {}
    entropy_values: Dict[str, float] = {}
    categorical_ratio_values: Dict[str, float] = {}
    log_dataset_size_values: Dict[str, float] = {}
    mean_of_means_values: Dict[str, float] = {}
    std_of_means_values: Dict[str, float] = {}
    mean_variance_values: Dict[str, float] = {}
    max_variance_values: Dict[str, float] = {}
    mean_skewness_values: Dict[str, float] = {}
    std_skewness_values: Dict[str, float] = {}
    max_kurtosis_values: Dict[str, float] = {}
    mean_std_values: Dict[str, float] = {}
    std_std_values: Dict[str, float] = {}
    max_std_values: Dict[str, float] = {}
    mean_range_values: Dict[str, float] = {}
    max_range_values: Dict[str, float] = {}
    mean_cardinality_values: Dict[str, float] = {}
    max_cardinality_values: Dict[str, float] = {}
    mean_cat_entropy_values: Dict[str, float] = {}
    std_cat_entropy_values: Dict[str, float] = {}
    mean_top_freq_values: Dict[str, float] = {}
    max_top_freq_values: Dict[str, float] = {}

    for dataset_id, dataset_name in enumerate(dataset_names):
        spec = registry.get(dataset_name)
        openml_meta = load_openml_metadata(spec)
        dataset = load_dataset(dataset_name, spec)
        raw_feature_frame = extract_raw_feature_frame(dataset_name, spec)

        n_features = len(dataset.feature_names)
        log_feature_count = math.log1p(n_features)
        log_feature_values[dataset_name] = log_feature_count

        normalized_entropy = compute_normalized_entropy(dataset)
        entropy_values[dataset_name] = normalized_entropy

        cat_count, numeric_count = compute_feature_type_counts(
            raw_feature_frame,
            fallback_total=n_features,
            openml_meta=openml_meta,
        )
        categorical_ratio = compute_categorical_ratio(cat_count, numeric_count)
        categorical_ratio_values[dataset_name] = categorical_ratio

        has_sensitive = detect_sensitive_attributes(
            raw_feature_frame.columns if raw_feature_frame is not None else None,
            dataset.feature_names,
        )
        high_stakes = detect_high_stakes_domain(dataset_name, spec)
        numeric_stats = compute_numeric_feature_statistics(raw_feature_frame, dataset)
        categorical_stats = compute_categorical_feature_statistics(raw_feature_frame)

        record = {
            "dataset_id": dataset_id,
            "n_features": n_features,
            "log_feature_count": log_feature_count,
            "class_entropy": normalized_entropy,
            "categorical_to_numerical_ratio": categorical_ratio,
            "n_categorical_features": cat_count,
            "n_numeric_features": numeric_count,
            "has_sensitive_attributes": has_sensitive,
            "high_stakes_domain": high_stakes,
            **numeric_stats,
            **categorical_stats,
        }
        if openml_meta and openml_meta.get("data_id"):
            record["openml_data_id"] = openml_meta["data_id"]
        dataset_records[dataset_name] = record
        log_dataset_size_values[dataset_name] = numeric_stats["log_dataset_size"]
        mean_of_means_values[dataset_name] = numeric_stats["mean_of_means"]
        std_of_means_values[dataset_name] = numeric_stats["std_of_means"]
        mean_variance_values[dataset_name] = numeric_stats["mean_variance"]
        max_variance_values[dataset_name] = numeric_stats["max_variance"]
        mean_skewness_values[dataset_name] = numeric_stats["mean_skewness"]
        std_skewness_values[dataset_name] = numeric_stats["std_skewness"]
        max_kurtosis_values[dataset_name] = numeric_stats["max_kurtosis"]
        mean_std_values[dataset_name] = numeric_stats["mean_std"]
        std_std_values[dataset_name] = numeric_stats["std_std"]
        max_std_values[dataset_name] = numeric_stats["max_std"]
        mean_range_values[dataset_name] = numeric_stats["mean_range"]
        max_range_values[dataset_name] = numeric_stats["max_range"]
        mean_cardinality_values[dataset_name] = categorical_stats["mean_cardinality"]
        max_cardinality_values[dataset_name] = categorical_stats["max_cardinality"]
        mean_cat_entropy_values[dataset_name] = categorical_stats["mean_cat_entropy"]
        std_cat_entropy_values[dataset_name] = categorical_stats["std_cat_entropy"]
        mean_top_freq_values[dataset_name] = categorical_stats["mean_top_freq"]
        max_top_freq_values[dataset_name] = categorical_stats["max_top_freq"]

    log_stats, log_feature_z = compute_z_scores(log_feature_values)
    entropy_stats, entropy_z = compute_z_scores(entropy_values)
    ratio_stats, ratio_z = compute_z_scores(categorical_ratio_values)
    dataset_size_stats, log_dataset_size_z = compute_z_scores(log_dataset_size_values)
    mean_of_means_stats, mean_of_means_z = compute_z_scores(mean_of_means_values)
    std_of_means_stats, std_of_means_z = compute_z_scores(std_of_means_values)
    mean_variance_stats, mean_variance_z = compute_z_scores(mean_variance_values)
    max_variance_stats, max_variance_z = compute_z_scores(max_variance_values)
    mean_skewness_stats, mean_skewness_z = compute_z_scores(mean_skewness_values)
    std_skewness_stats, std_skewness_z = compute_z_scores(std_skewness_values)
    max_kurtosis_stats, max_kurtosis_z = compute_z_scores(max_kurtosis_values)
    mean_std_stats, mean_std_z = compute_z_scores(mean_std_values)
    std_std_stats, std_std_z = compute_z_scores(std_std_values)
    max_std_stats, max_std_z = compute_z_scores(max_std_values)
    mean_range_stats, mean_range_z = compute_z_scores(mean_range_values)
    max_range_stats, max_range_z = compute_z_scores(max_range_values)
    mean_cardinality_stats, mean_cardinality_z = compute_z_scores(mean_cardinality_values)
    max_cardinality_stats, max_cardinality_z = compute_z_scores(max_cardinality_values)
    mean_cat_entropy_stats, mean_cat_entropy_z = compute_z_scores(mean_cat_entropy_values)
    std_cat_entropy_stats, std_cat_entropy_z = compute_z_scores(std_cat_entropy_values)
    mean_top_freq_stats, mean_top_freq_z = compute_z_scores(mean_top_freq_values)
    max_top_freq_stats, max_top_freq_z = compute_z_scores(max_top_freq_values)

    for dataset_name, record in dataset_records.items():
        record["log_feature_count_z"] = log_feature_z.get(dataset_name, 0.0)
        record["class_entropy_z"] = entropy_z.get(dataset_name, 0.0)
        record["categorical_to_numerical_ratio_z"] = ratio_z.get(dataset_name, 0.0)
        record["log_dataset_size_z"] = log_dataset_size_z.get(dataset_name, 0.0)
        record["mean_of_means_z"] = mean_of_means_z.get(dataset_name, 0.0)
        record["std_of_means_z"] = std_of_means_z.get(dataset_name, 0.0)
        record["mean_variance_z"] = mean_variance_z.get(dataset_name, 0.0)
        record["max_variance_z"] = max_variance_z.get(dataset_name, 0.0)
        record["mean_skewness_z"] = mean_skewness_z.get(dataset_name, 0.0)
        record["std_skewness_z"] = std_skewness_z.get(dataset_name, 0.0)
        record["max_kurtosis_z"] = max_kurtosis_z.get(dataset_name, 0.0)
        record["mean_std_z"] = mean_std_z.get(dataset_name, 0.0)
        record["std_std_z"] = std_std_z.get(dataset_name, 0.0)
        record["max_std_z"] = max_std_z.get(dataset_name, 0.0)
        record["mean_range_z"] = mean_range_z.get(dataset_name, 0.0)
        record["max_range_z"] = max_range_z.get(dataset_name, 0.0)
        record["mean_cardinality_z"] = mean_cardinality_z.get(dataset_name, 0.0)
        record["max_cardinality_z"] = max_cardinality_z.get(dataset_name, 0.0)
        record["mean_cat_entropy_z"] = mean_cat_entropy_z.get(dataset_name, 0.0)
        record["std_cat_entropy_z"] = std_cat_entropy_z.get(dataset_name, 0.0)
        record["mean_top_freq_z"] = mean_top_freq_z.get(dataset_name, 0.0)
        record["max_top_freq_z"] = max_top_freq_z.get(dataset_name, 0.0)

    payload = {
        "generated_at": datetime.utcnow().isoformat(),
        "datasets": dataset_records,
        "normalization": {
            "log_feature_count": log_stats,
            "class_entropy": entropy_stats,
            "categorical_to_numerical_ratio": ratio_stats,
            "log_dataset_size": dataset_size_stats,
            "mean_of_means": mean_of_means_stats,
            "std_of_means": std_of_means_stats,
            "mean_variance": mean_variance_stats,
            "max_variance": max_variance_stats,
            "mean_skewness": mean_skewness_stats,
            "std_skewness": std_skewness_stats,
            "max_kurtosis": max_kurtosis_stats,
            "mean_std": mean_std_stats,
            "std_std": std_std_stats,
            "max_std": max_std_stats,
            "mean_range": mean_range_stats,
            "max_range": max_range_stats,
            "mean_cardinality": mean_cardinality_stats,
            "max_cardinality": max_cardinality_stats,
            "mean_cat_entropy": mean_cat_entropy_stats,
            "std_cat_entropy": std_cat_entropy_stats,
            "mean_top_freq": mean_top_freq_stats,
            "max_top_freq": max_top_freq_stats,
        },
    }

    return payload


def build_explainer_metadata() -> Dict[str, Any]:
    registry = ExplainerRegistry()
    records: Dict[str, Dict[str, Any]] = {}
    for explainer_id, name in enumerate(sorted(registry.names())):
        spec = registry.get(name)
        family_token = str(spec.get("type") or name).lower()
        flags = compute_explainer_flags(name, family_token)
        records[name] = {
            "explainer_id": explainer_id,
            "type": spec.get("type"),
            "description": spec.get("description"),
            "supported_data_types": spec.get("supported_data_types"),
            **flags,
        }
    return {
        "generated_at": datetime.utcnow().isoformat(),
        "explainers": records,
    }


def normalise_dataset_names(
    requested: Sequence[str],
    registry: DatasetRegistry,
) -> List[str]:
    if not requested:
        return []
    if any(entry == "*" for entry in requested):
        names = sorted(name for name in registry.names())
    else:
        unknown = [name for name in requested if name not in registry.names()]
        if unknown:
            raise ValueError(f"Unknown dataset(s) requested: {unknown}")
        names = sorted(set(requested))
    return names


def load_dataset(dataset_name: str, spec: Mapping[str, Any]) -> TabularDataset:
    adapter_spec = spec.get("adapter") or {}
    adapter_module = adapter_spec.get("module", "src.datasets.adapters")
    adapter_class = adapter_spec.get("class", "LoaderDatasetAdapter")
    adapter_cls = _import_object(adapter_module, adapter_class)
    if not issubclass(adapter_cls, LoaderDatasetAdapter):
        raise TypeError(f"Adapter {adapter_cls} for dataset '{dataset_name}' is unsupported.")
    adapter = adapter_cls(dataset_name, spec)
    return adapter.load()


def extract_raw_feature_frame(
    dataset_name: str,
    spec: Mapping[str, Any],
) -> Optional["pd.DataFrame"]:
    if not _HAS_PANDAS or pd is None:
        return None

    loader_spec = spec.get("loader")
    if not loader_spec:
        return None
    loader_fn = _import_object(loader_spec["module"], loader_spec["factory"])
    loader_params = dict(spec.get("loader_params") or spec.get("params") or {})
    obj = loader_fn(**loader_params)

    frame = getattr(obj, "frame", None)
    if frame is None or not isinstance(frame, pd.DataFrame):
        data = getattr(obj, "data", None)
        feature_names = getattr(obj, "feature_names", None)
        if data is None:
            return None
        if feature_names is None:
            feature_names = [f"feature_{idx}" for idx in range(data.shape[1])]
        return pd.DataFrame(data, columns=list(feature_names))

    target_column = resolve_target_column(
        dataset_name,
        frame,
        loader_params,
        spec,
        getattr(obj, "target", None),
    )
    feature_frame = frame.drop(columns=[target_column], errors="ignore")
    return feature_frame


def resolve_target_column(
    dataset_name: str,
    frame: "pd.DataFrame",
    loader_params: Mapping[str, Any],
    spec: Mapping[str, Any],
    target: Any,
) -> str:
    target_column = (
        loader_params.get("target_column")
        or spec.get("target_column")
        or getattr(target, "name", None)
    )
    if target_column is None and "target" in frame.columns:
        target_column = "target"
    if target_column is None:
        target_column = frame.columns[-1]
    if target_column not in frame.columns:
        raise ValueError(
            f"Failed to resolve target column for dataset '{dataset_name}'.",
        )
    return target_column


def compute_feature_type_counts(
    frame: Optional["pd.DataFrame"],
    *,
    fallback_total: int,
    openml_meta: Optional[Dict[str, Any]],
) -> Tuple[int, int]:
    counts = counts_from_openml_meta(openml_meta)
    if counts is not None:
        cat_count, numeric_count = counts
        if cat_count is None and numeric_count is None:
            pass
        else:
            if cat_count is None:
                cat_count = max(fallback_total - (numeric_count or 0), 0)
            if numeric_count is None:
                numeric_count = max(fallback_total - cat_count, 0)
            return cat_count, numeric_count

    if frame is None or not _HAS_PANDAS or pd is None:
        return 0, fallback_total
    categorical = frame.select_dtypes(include=["object", "category", "bool"]).columns
    numeric = frame.select_dtypes(exclude=["object", "category", "bool"]).columns
    return len(categorical), len(numeric)


def compute_categorical_ratio(cat_count: int, numeric_count: int) -> float:
    if numeric_count <= 0:
        return float(cat_count)
    return cat_count / float(numeric_count)


def compute_normalized_entropy(dataset: TabularDataset) -> float:
    labels: List[np.ndarray] = []
    if dataset.y_train is not None:
        labels.append(dataset.y_train)
    if dataset.y_test is not None:
        labels.append(dataset.y_test)
    if not labels:
        return 0.0
    combined = np.concatenate(labels)
    if combined.size == 0:
        return 0.0
    values, counts = np.unique(combined, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = float(-np.sum(probabilities * np.log(probabilities + 1e-12)))
    num_classes = len(values)
    if num_classes <= 1:
        return 0.0
    return entropy / math.log(num_classes)


def compute_numeric_feature_statistics(
    frame: Optional["pd.DataFrame"],
    dataset: TabularDataset,
) -> Dict[str, float]:
    """Compute aggregate statistics over numeric feature columns for a dataset."""
    size = estimate_dataset_size(frame, dataset)
    summary: Dict[str, float] = {
        "log_dataset_size": math.log1p(max(size, 0)),
        "mean_of_means": 0.0,
        "std_of_means": 0.0,
        "mean_variance": 0.0,
        "max_variance": 0.0,
        "mean_skewness": 0.0,
        "std_skewness": 0.0,
        "max_kurtosis": 0.0,
        "mean_std": 0.0,
        "std_std": 0.0,
        "max_std": 0.0,
        "mean_range": 0.0,
        "max_range": 0.0,
    }
    if frame is None or not _HAS_PANDAS or pd is None or frame.empty:
        return summary

    numeric = frame.select_dtypes(exclude=["object", "category", "bool"])
    if numeric.empty:
        return summary

    means = numeric.mean()
    variances = numeric.var(ddof=0)
    stds = numeric.std(ddof=0)
    skewness = numeric.skew()
    kurtosis = numeric.kurtosis()
    mins = numeric.min()
    maxs = numeric.max()
    ranges = maxs - mins

    summary.update(
        mean_of_means=_safe_mean(means),
        std_of_means=_safe_std(means),
        mean_variance=_safe_mean(variances),
        max_variance=_safe_max(variances),
        mean_skewness=_safe_mean(skewness),
        std_skewness=_safe_std(skewness),
        max_kurtosis=_safe_max(kurtosis),
        mean_std=_safe_mean(stds),
        std_std=_safe_std(stds),
        max_std=_safe_max(stds),
        mean_range=_safe_mean(ranges),
        max_range=_safe_max(ranges),
    )
    return summary


def estimate_dataset_size(
    frame: Optional["pd.DataFrame"],
    dataset: TabularDataset,
) -> int:
    if frame is not None:
        return int(frame.shape[0])
    total = 0
    for arr in (dataset.X_train, dataset.X_test):
        if arr is not None:
            try:
                total += int(arr.shape[0])
            except Exception:
                continue
    return total


def compute_categorical_feature_statistics(
    frame: Optional["pd.DataFrame"],
) -> Dict[str, float]:
    """Compute aggregate statistics over categorical feature columns for a dataset."""
    summary: Dict[str, float] = {
        "mean_cardinality": 0.0,
        "max_cardinality": 0.0,
        "mean_cat_entropy": 0.0,
        "std_cat_entropy": 0.0,
        "mean_top_freq": 0.0,
        "max_top_freq": 0.0,
    }
    if frame is None or not _HAS_PANDAS or pd is None or frame.empty:
        return summary

    categorical = frame.select_dtypes(include=["object", "category", "bool"])
    if categorical.empty:
        return summary

    cardinality = categorical.nunique(dropna=True)
    entropy = categorical.apply(_entropy_from_series)
    top_freq = categorical.apply(_top_frequency_from_series)

    summary.update(
        mean_cardinality=_safe_mean(cardinality),
        max_cardinality=_safe_max(cardinality),
        mean_cat_entropy=_safe_mean(entropy),
        std_cat_entropy=_safe_std(entropy),
        mean_top_freq=_safe_mean(top_freq),
        max_top_freq=_safe_max(top_freq),
    )
    return summary


def _safe_mean(series: "pd.Series") -> float:
    values = series.dropna().to_numpy()
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


def _safe_std(series: "pd.Series") -> float:
    values = series.dropna().to_numpy()
    if values.size == 0:
        return 0.0
    return float(np.std(values))


def _safe_max(series: "pd.Series") -> float:
    values = series.dropna().to_numpy()
    if values.size == 0:
        return 0.0
    return float(np.max(values))


def _entropy_from_series(series: "pd.Series") -> float:
    counts = series.value_counts(dropna=True)
    if counts.empty:
        return 0.0
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def _top_frequency_from_series(series: "pd.Series") -> float:
    counts = series.value_counts(dropna=True, normalize=True)
    if counts.empty:
        return 0.0
    return float(counts.iloc[0])


def detect_sensitive_attributes(
    raw_columns: Optional[Iterable[str]],
    encoded_feature_names: Sequence[str],
) -> bool:
    names: List[str] = []
    if raw_columns is not None:
        names.extend(list(raw_columns))
    names.extend(encoded_feature_names)
    for name in names:
        if not name:
            continue
        normalized = name.lower()
        for keyword in SENSITIVE_KEYWORDS:
            if keyword in normalized:
                return True
    return False


def detect_high_stakes_domain(dataset_name: str, spec: Mapping[str, Any]) -> bool:
    text_blobs = [
        dataset_name,
        spec.get("description", ""),
        spec.get("source", ""),
    ]
    combined = " ".join(blob for blob in text_blobs if blob).lower()
    return any(keyword in combined for keyword in HIGH_STAKES_KEYWORDS)


def compute_z_scores(values: Mapping[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
    if not values:
        return {"mean": 0.0, "std": 0.0}, {}
    data = list(values.values())
    mean = float(sum(data) / len(data))
    if len(data) > 1:
        variance = float(sum((value - mean) ** 2 for value in data) / len(data))
        std = math.sqrt(variance)
    else:
        std = 0.0
    normalized = {
        key: ((value - mean) / std) if std > 0 else 0.0
        for key, value in values.items()
    }
    return {"mean": mean, "std": std}, normalized


def _import_object(module_name: str, attr: str) -> Any:
    module = importlib.import_module(module_name)
    obj = module
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def load_openml_metadata(spec: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    loader_params = spec.get("loader_params") or {}
    data_id = loader_params.get("data_id")
    if not data_id:
        return None
    try:
        data_id_int = int(data_id)
    except (TypeError, ValueError):
        return None
    features = load_openml_features(data_id_int)
    qualities = load_openml_qualities(data_id_int)
    if not features and not qualities:
        return None
    return {"data_id": data_id_int, "features": features, "qualities": qualities}


def load_openml_features(data_id: int) -> Optional[List[Dict[str, Any]]]:
    path = OPENML_CACHE_ROOT / "features" / f"{data_id}.gz"
    if not path.exists():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            raw = json.load(handle)
    except Exception:
        return None
    features = raw.get("data_features", {}).get("feature")
    if features is None:
        return None
    if isinstance(features, dict):
        return [features]
    return list(features)


def load_openml_qualities(data_id: int) -> Optional[Dict[str, float]]:
    path = OPENML_CACHE_ROOT / "qualities" / f"{data_id}.gz"
    if not path.exists():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            raw = json.load(handle)
    except Exception:
        return None
    qualities_raw = raw.get("data_qualities", {}).get("quality")
    if not qualities_raw:
        return None
    qualities: Dict[str, float] = {}
    for entry in qualities_raw:
        name = entry.get("name")
        value = _coerce_float(entry.get("value"))
        if name and value is not None:
            qualities[name] = value
    return qualities or None


def counts_from_openml_meta(
    openml_meta: Optional[Dict[str, Any]],
) -> Optional[Tuple[Optional[int], Optional[int]]]:
    if not openml_meta:
        return None
    features = openml_meta.get("features") or []
    cat_count: Optional[int] = None
    numeric_count: Optional[int] = None
    if features:
        cat = 0
        num = 0
        for feature in features:
            if feature.get("is_target", "false") == "true":
                continue
            data_type = str(feature.get("data_type", "")).lower()
            if data_type in {"nominal", "string"}:
                cat += 1
            elif data_type in {"numeric", "real", "integer"}:
                num += 1
        cat_count = cat
        numeric_count = num
    if (cat_count is None or numeric_count is None) and openml_meta.get("qualities"):
        qualities = openml_meta["qualities"]
        if cat_count is None:
            cat_val = qualities.get("NumberOfSymbolicFeatures")
            if cat_val is not None:
                cat_count = int(round(cat_val))
        if numeric_count is None:
            num_val = qualities.get("NumberOfNumericFeatures")
            if num_val is not None:
                numeric_count = int(round(num_val))
    return (cat_count, numeric_count)


def _coerce_float(value: Any) -> Optional[float]:
    if value in (None, "", []):
        return None
    if isinstance(value, list):
        if not value:
            return None
        value = value[0]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_explainer_flags(name: str, family_token: str) -> Dict[str, bool]:
    lower_name = name.lower()
    additive = any(token in family_token for token in ("shap", "lime"))
    perturbation = family_token in {"lime", "shap"}
    gradient = "gradient" in family_token
    causal = "causal" in family_token or "causal" in lower_name
    return {
        "is_additive_attribution": additive,
        "is_gradient_based": gradient,
        "is_causal": causal,
        "is_perturbation_based": perturbation,
    }


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
