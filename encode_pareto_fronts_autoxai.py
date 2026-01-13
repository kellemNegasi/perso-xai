#!/usr/bin/env python3
"""
Encode Pareto-front metrics into ranker-friendly tabular features (AutoXAI-aware).

This is a sibling of `encode_pareto_fronts.py` that understands the AutoXAI
hyperparameter search-space encoding in `src/configs/explainer_hyperparameters.yml`:

- `randint: [low, high]` parameter specs (including dataset-dependent `high: n_features`)
- categorical hyperparameters (one-hot encoded)
- conditional SHAP hyperparameters (`shap_l1_reg*` only applicable for kernel SHAP)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import yaml

DEFAULT_RESULTS_ROOT = Path("results") / "hc_combo_20251228_050331"
DEFAULT_PARETO_DIR = DEFAULT_RESULTS_ROOT / "pareto_fronts"
DEFAULT_METADATA_DIR = DEFAULT_RESULTS_ROOT / "metadata"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_ROOT / "encoded_pareto_fronts" / "features_full_lm_stats_autoxai"
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

# Count-like / scale-like hyperparameters that are best encoded in log-space.
LOG_SCALED_HPARAMS = {
    "background_sample_size",
    "lime_num_samples",
    "causal_shap_coalitions",
    "ig_steps",
    "shap_nsamples",
    "shap_l1_reg_k",
}

NEGATE_METRICS = {
    "infidelity",
    "non_sensitivity_violation_fraction",
    "non_sensitivity_delta_mean",
    "relative_input_stability",
    "covariate_complexity",
}

COUNT_LIKE_METRICS = (
    "non_sensitivity_zero_features",
    "contrastivity_pairs",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode Pareto-front metrics into feature-rich DataFrames (AutoXAI-aware).",
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
    parser.add_argument(
        "--metric-value-transform",
        choices=("none", "signed_log1p"),
        default="none",
        help=(
            "Optional transform applied to metric values before within-instance z-normalization. "
            "Use signed_log1p to dampen heavy-tailed/outlier metrics while preserving ordering "
            "(default: none)."
        ),
    )
    parser.add_argument(
        "--transform-metrics",
        nargs="+",
        default=("infidelity", "relative_input_stability"),
        help=(
            "Metric names to transform when --metric-value-transform is enabled "
            "(default: infidelity relative_input_stability)."
        ),
    )
    parser.add_argument(
        "--transform-count-metrics",
        action="store_true",
        help=(
            "When set, also applies the metric-value transform to count-like metrics "
            f"(currently: {', '.join(COUNT_LIKE_METRICS)})."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    dataset_meta = load_dataset_metadata(args.metadata_dir / "dataset_metadata.json")
    explainer_meta = load_explainer_metadata(args.metadata_dir / "explainers_metadata.json")
    hyper_cfg = load_hyperparameter_config(args.hyperparameters)
    schema = build_hyperparameter_schema(hyper_cfg)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pareto_files = sorted(args.pareto_dir.glob("*_pareto.json"))
    if not pareto_files:
        raise FileNotFoundError(f"No pareto JSON files found under {args.pareto_dir}")

    for pareto_path in pareto_files:
        transform_metric_names = list(args.transform_metrics or ())
        if args.transform_count_metrics:
            for name in COUNT_LIKE_METRICS:
                if name not in transform_metric_names:
                    transform_metric_names.append(name)
        df = encode_pareto_file(
            pareto_path,
            dataset_meta=dataset_meta,
            explainer_meta=explainer_meta,
            schema=schema,
            metric_value_transform=str(args.metric_value_transform),
            transform_metric_names=tuple(transform_metric_names),
        )
        output_path = args.output_dir / f"{pareto_path.stem}_encoded.parquet"
        df.to_parquet(output_path, index=False)
        print(f"Wrote encoded DataFrame with {len(df)} rows to {output_path}")


def load_dataset_metadata(path: Path) -> Dict[str, Dict[str, Any]]:
    data = _load_json(path)
    datasets = data.get("datasets")
    if not isinstance(datasets, Mapping):
        raise ValueError(f"Malformed dataset metadata at {path}")
    return datasets  # type: ignore[return-value]


def load_explainer_metadata(path: Path) -> Dict[str, Dict[str, Any]]:
    data = _load_json(path)
    explainers = data.get("explainers")
    if not isinstance(explainers, Mapping):
        raise ValueError(f"Malformed explainer metadata at {path}")
    return explainers  # type: ignore[return-value]


def load_hyperparameter_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def encode_pareto_file(
    path: Path,
    *,
    dataset_meta: Mapping[str, Mapping[str, Any]],
    explainer_meta: Mapping[str, Mapping[str, Any]],
    schema: "HyperparameterSchema",
    metric_value_transform: str = "none",
    transform_metric_names: Sequence[str] = ("infidelity", "relative_input_stability"),
) -> pd.DataFrame:
    payload = _load_json(path)
    dataset_name = payload.get("dataset")
    model_name = payload.get("model")
    if dataset_name not in dataset_meta:
        raise KeyError(f"Dataset '{dataset_name}' missing from dataset metadata.")
    dataset_features = dataset_meta[dataset_name]
    dataset_ids = _sorted_unique_ints(record.get("dataset_id") for record in dataset_meta.values())
    explainer_ids = _sorted_unique_ints(record.get("explainer_id") for record in explainer_meta.values())

    instances = payload.get("instances") or []
    records: List[Dict[str, Any]] = []
    for instance in instances:
        rows: List[Dict[str, Any]] = []
        pareto_metrics: List[str] = list(instance.get("pareto_metrics") or [])
        if not pareto_metrics:
            pareto_metrics = sorted(
                {
                    k
                    for entry in (instance.get("pareto_front") or [])
                    for k in (entry.get("metrics") or {}).keys()
                }
            )
        for entry in instance.get("pareto_front") or []:
            method = entry.get("method")
            if method not in explainer_meta:
                raise KeyError(f"Explainer '{method}' missing from explainer metadata.")
            method_variant = entry.get("method_variant")
            metrics = transform_metrics(entry.get("metrics") or {})
            expl_meta = explainer_meta[method]
            dataset_id = dataset_features.get("dataset_id")
            explainer_id = expl_meta.get("explainer_id")
            dataset_n_features = dataset_features.get("n_features")

            parsed_params = parse_variant_params(method_variant)
            hp_features, applicability = encode_hyperparameters(
                method,
                parsed_params,
                schema=schema,
                dataset_n_features=_coerce_int(dataset_n_features),
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

        records.extend(
            normalize_instance_metrics(
                rows,
                pareto_metrics,
                metric_value_transform=metric_value_transform,
                transform_metric_names=transform_metric_names,
            )
        )
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def transform_metrics(metrics: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    transformed: Dict[str, Optional[float]] = {}
    for key, value in metrics.items():
        transformed[key] = _coerce_float(value)
    return transformed


def normalize_instance_metrics(
    rows: List[Dict[str, Any]],
    metric_names: Sequence[str],
    *,
    metric_value_transform: str = "none",
    transform_metric_names: Sequence[str] = ("infidelity", "relative_input_stability"),
) -> List[Dict[str, Any]]:
    if not rows:
        return []
    transform_set = set(transform_metric_names or ())
    transform_mode = str(metric_value_transform or "none").lower()
    if transform_mode not in {"none", "signed_log1p"}:
        raise ValueError(f"Unknown metric_value_transform={metric_value_transform!r}")

    def _transform(metric: str, value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        if metric not in transform_set or transform_mode == "none":
            return float(value)
        if transform_mode == "signed_log1p":
            return math.copysign(math.log1p(abs(float(value))), float(value))
        return float(value)

    per_metric_values: Dict[str, List[float]] = {name: [] for name in metric_names}
    per_row_metrics: List[Dict[str, Optional[float]]] = []

    for row in rows:
        metrics = row.pop("_metrics_raw", {}) or {}
        row_metrics: Dict[str, Optional[float]] = {}
        for metric in metric_names:
            raw_value = metrics.get(metric)
            value = _transform(metric, raw_value)
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
                row[metric] = 0.0
            else:
                row[metric] = (value - mean) / std
        encoded_rows.append(row)
    return encoded_rows


def parse_variant_params(variant: Optional[str]) -> Dict[str, str]:
    if not variant:
        return {}
    parts = variant.split("__")
    params: Dict[str, str] = {}
    for part in parts[1:]:
        if "-" not in part:
            continue
        name, raw_value = part.split("-", 1)
        params[str(name)] = str(raw_value)
    return params


def _canonical_choice(value: object) -> str:
    return str(value).strip().lower()


def _resolve_randint_bound(value: object, *, dataset_n_features: Optional[int]) -> Optional[int]:
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"d", "n_features", "nfeatures", "num_features"}:
            if dataset_n_features is None:
                return None
            return int(dataset_n_features)
    return None


def _randint_linear_stats(low: int, high: int) -> Tuple[float, float]:
    n = high - low + 1
    if n <= 1:
        return float(low), 0.0
    mean = (low + high) / 2.0
    variance = (n * n - 1.0) / 12.0
    return mean, math.sqrt(variance)


def _randint_log_stats(low: int, high: int, *, samples: int = 2048) -> Tuple[float, float]:
    if high <= low:
        value = math.log1p(float(low))
        return value, 0.0
    n = high - low + 1
    take = min(samples, n)
    if take <= 1:
        value = math.log1p(float(low))
        return value, 0.0
    step = max(1, (n - 1) // (take - 1))
    values = [low + i * step for i in range(take - 1)]
    if values[-1] != high:
        values.append(high)
    transformed = [math.log1p(float(v)) for v in values]
    mean = sum(transformed) / len(transformed)
    if len(transformed) > 1:
        variance = sum((v - mean) ** 2 for v in transformed) / len(transformed)
        std = math.sqrt(variance)
    else:
        std = 0.0
    return mean, std


class HyperparameterSchema:
    def __init__(
        self,
        *,
        numeric_params: Sequence[str],
        categorical_params: Mapping[str, Sequence[str]],
        numeric_specs: Mapping[str, Mapping[str, Mapping[str, Any]]],
        categorical_specs: Mapping[str, Mapping[str, Sequence[str]]],
    ) -> None:
        self.numeric_params = list(numeric_params)
        self.categorical_params = {k: list(v) for k, v in categorical_params.items()}
        self.numeric_specs = {m: {p: dict(s) for p, s in ps.items()} for m, ps in numeric_specs.items()}
        self.categorical_specs = {m: {p: list(v) for p, v in ps.items()} for m, ps in categorical_specs.items()}
        self._dynamic_cache: Dict[Tuple[int, int, str], Tuple[float, float]] = {}

    def numeric_stats(
        self,
        method: str,
        param: str,
        *,
        dataset_n_features: Optional[int],
    ) -> Optional[Tuple[float, float, str]]:
        spec = (self.numeric_specs.get(method) or {}).get(param)
        if not spec:
            return None
        scale = str(spec.get("scale") or "linear")
        if spec.get("kind") == "randint":
            low = int(spec["low"])
            high_raw = spec.get("high")
            high_token = spec.get("high_token")
            if high_raw is not None:
                high = int(high_raw)
            else:
                resolved = _resolve_randint_bound(high_token, dataset_n_features=dataset_n_features)
                if resolved is None:
                    return None
                high = int(resolved)
            high = max(low, high)
            cache_key = (low, high, scale)
            if cache_key in self._dynamic_cache:
                mean, std = self._dynamic_cache[cache_key]
            else:
                if scale == "log":
                    mean, std = _randint_log_stats(low, high)
                else:
                    mean, std = _randint_linear_stats(low, high)
                self._dynamic_cache[cache_key] = (mean, std)
            return mean, std, scale
        return float(spec.get("mean", 0.0)), float(spec.get("std", 0.0)), scale


def build_hyperparameter_schema(config: Mapping[str, Any]) -> HyperparameterSchema:
    explainers_cfg = config.get("explainers") or {}
    numeric_universe: List[str] = []
    categorical_universe: Dict[str, List[str]] = {}
    numeric_specs: Dict[str, Dict[str, Dict[str, Any]]] = {}
    categorical_specs: Dict[str, Dict[str, List[str]]] = {}

    for explainer_name, grid in explainers_cfg.items():
        if not isinstance(grid, Mapping):
            continue
        for param_name, spec in (grid or {}).items():
            param = str(param_name)
            if isinstance(spec, list):
                numeric_values = [_coerce_float(value) for value in spec]
                clean_values = [value for value in numeric_values if value is not None]
                if clean_values and len(clean_values) == len(spec):
                    transformed = [
                        math.log1p(value) if param in LOG_SCALED_HPARAMS else value for value in clean_values
                    ]
                    mean = sum(transformed) / len(transformed)
                    if len(transformed) > 1:
                        variance = sum((value - mean) ** 2 for value in transformed) / len(transformed)
                        std = math.sqrt(variance)
                    else:
                        std = 0.0
                    numeric_specs.setdefault(str(explainer_name), {})[param] = {
                        "kind": "grid",
                        "mean": mean,
                        "std": std,
                        "scale": "log" if param in LOG_SCALED_HPARAMS else "linear",
                    }
                    if param not in numeric_universe:
                        numeric_universe.append(param)
                    continue

                # Categorical list.
                choices = [_canonical_choice(v) for v in spec]
                categorical_specs.setdefault(str(explainer_name), {})[param] = choices
                seen = categorical_universe.setdefault(param, [])
                for choice in choices:
                    if choice not in seen:
                        seen.append(choice)
                continue

            if isinstance(spec, Mapping) and "randint" in spec:
                bounds = spec.get("randint")
                if not isinstance(bounds, list) or len(bounds) != 2:
                    raise ValueError(f"{explainer_name}.{param}: randint must be a 2-item list, got {bounds!r}")
                low = int(bounds[0])
                high_raw = bounds[1]
                scale = "log" if param in LOG_SCALED_HPARAMS else "linear"
                high_int = _resolve_randint_bound(high_raw, dataset_n_features=10)
                if isinstance(high_raw, str) and high_int is None:
                    numeric_specs.setdefault(str(explainer_name), {})[param] = {
                        "kind": "randint",
                        "low": low,
                        "high": None,
                        "high_token": str(high_raw),
                        "scale": scale,
                    }
                else:
                    high = int(high_int) if high_int is not None else int(high_raw)
                    if high < low:
                        raise ValueError(f"{explainer_name}.{param}: randint low must be <= high, got {low}..{high}")
                    if scale == "log":
                        mean, std = _randint_log_stats(low, high)
                    else:
                        mean, std = _randint_linear_stats(low, high)
                    numeric_specs.setdefault(str(explainer_name), {})[param] = {
                        "kind": "randint",
                        "low": low,
                        "high": high,
                        "scale": scale,
                        "mean": mean,
                        "std": std,
                    }
                if param not in numeric_universe:
                    numeric_universe.append(param)
                continue

            raise ValueError(
                "Explainer hyperparameter values must be either a list (categorical/numeric grid) "
                f"or a dict with `randint: [low, high]`. Got {explainer_name}.{param}={spec!r}."
            )

    numeric_universe.sort()
    categorical_universe = {k: v for k, v in sorted(categorical_universe.items(), key=lambda kv: kv[0])}
    return HyperparameterSchema(
        numeric_params=numeric_universe,
        categorical_params=categorical_universe,
        numeric_specs=numeric_specs,
        categorical_specs=categorical_specs,
    )


def _autoxai_shap_param_applicable(param: str, params: Mapping[str, str]) -> bool:
    explainer_type = _canonical_choice(params.get("shap_explainer_type", ""))
    if param in {"shap_l1_reg", "shap_l1_reg_k"} and explainer_type != "kernel":
        return False
    if param == "shap_l1_reg_k":
        l1_reg = _canonical_choice(params.get("shap_l1_reg", ""))
        if l1_reg != "num_features":
            return False
    return True


def encode_hyperparameters(
    method: str,
    params: Mapping[str, str],
    *,
    schema: HyperparameterSchema,
    dataset_n_features: Optional[int] = None,
) -> Tuple[Dict[str, float], Dict[str, int]]:
    features: Dict[str, float] = {}
    applicability: Dict[str, int] = {}

    categorical_defined = schema.categorical_specs.get(method, {})
    numeric_defined = schema.numeric_specs.get(method, {})

    # Applicability bits for all params (numeric + categorical).
    all_param_names = set(schema.numeric_params) | set(schema.categorical_params.keys())
    for param in sorted(all_param_names):
        defined = param in numeric_defined or param in categorical_defined
        if not defined:
            applicability[f"is_applicable_{param}"] = 0
            continue
        if method == "autoxai_shap" and not _autoxai_shap_param_applicable(param, params):
            applicability[f"is_applicable_{param}"] = 0
            continue
        applicability[f"is_applicable_{param}"] = 1

    # Numeric z-scored features.
    for param in schema.numeric_params:
        if applicability.get(f"is_applicable_{param}", 0) != 1:
            features[f"hp_{param}"] = 0.0
            continue
        raw = params.get(param)
        value = _coerce_float(raw)
        if value is None:
            features[f"hp_{param}"] = 0.0
            continue
        stats = schema.numeric_stats(method, param, dataset_n_features=dataset_n_features)
        if stats is None:
            features[f"hp_{param}"] = 0.0
            continue
        mean, std, scale = stats
        transformed = math.log1p(value) if scale == "log" else float(value)
        if std == 0.0:
            features[f"hp_{param}"] = 0.0
        else:
            features[f"hp_{param}"] = (transformed - mean) / std

    # Categorical one-hot features.
    for param, choices in schema.categorical_params.items():
        is_applicable = applicability.get(f"is_applicable_{param}", 0) == 1
        selected = _canonical_choice(params.get(param, "")) if is_applicable else ""
        for choice in choices:
            features[f"hp_{param}_oh_{choice}"] = 1.0 if is_applicable and selected == choice else 0.0

    return features, applicability


def _coerce_float(value: Any) -> Optional[float]:
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


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sorted_unique_ints(values: Iterable[Any]) -> List[int]:
    unique: set[int] = set()
    for value in values:
        try:
            unique.add(int(value))
        except (TypeError, ValueError):
            continue
    return sorted(unique)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required JSON file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":  # pragma: no cover
    main()

