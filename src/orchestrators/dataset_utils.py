"""Dataset-related helpers for experiment orchestration."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


def run_dataset_validation(
    *,
    dataset_name: str,
    dataset_type: str,
    dataset_spec: Dict[str, Any],
    dataset,
    experiment_name: str,
    validator,
) -> None:
    """Run dataset-specific validation checks and raise on fatal issues."""
    if dataset_type != "tabular":
        return
    validation_cfg = (dataset_spec.get("validation") or {}).get("overrides") or {}
    result = validator.validate(
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


def resolve_tuning_subset(
    *,
    dataset_name: str,
    dataset_spec: Dict[str, Any],
    dataset,
) -> Tuple[Any, Any]:
    """Select a subset of training data for tuning based on config sampling rules."""
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


__all__ = ["run_dataset_validation", "resolve_tuning_subset"]
