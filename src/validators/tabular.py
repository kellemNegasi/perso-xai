"""Data validation utilities for tabular datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from src.datasets import TabularDataset


@dataclass
class ValidationResult:
    dataset_name: str
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, float] = field(default_factory=dict)


class TabularDataValidator:
    """Validate TabularDataset instances using configurable thresholds."""

    DEFAULTS: Dict[str, float] = {
        "min_classes": 2,
        "class_imbalance_ratio": 0.2,
        "min_dataset_size": 100,
        "min_test_size": 10,
        "max_missing_ratio": 0.3,
        "feature_correlation_threshold": 0.95,
        "min_features": 2,
        "max_features": 2000,
        "outlier_threshold": 3.0,
        "max_outlier_ratio": 0.1,
        "max_skewness": 3.0,
    }

    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        self.thresholds = dict(self.DEFAULTS)
        if thresholds:
            self.thresholds.update(thresholds)

    def validate(
        self,
        dataset: TabularDataset,
        dataset_name: str,
        overrides: Optional[Dict[str, float]] = None,
    ) -> ValidationResult:
        thresholds = dict(self.thresholds)
        if overrides:
            thresholds.update(overrides)

        result = ValidationResult(dataset_name=dataset_name, is_valid=True)
        X_train = np.asarray(dataset.X_train)
        X_test = np.asarray(dataset.X_test)
        y_train = self._safe_targets(dataset.y_train)
        y_test = self._safe_targets(dataset.y_test)

        self._check_dataset_size(X_train, X_test, thresholds, result)
        self._check_class_cardinality(y_train, y_test, thresholds, result)
        self._check_class_balance(y_train, y_test, thresholds, result)
        self._check_feature_counts(X_train, thresholds, result)
        self._check_missing_values(X_train, X_test, thresholds, result)
        self._check_feature_correlations(X_train, thresholds, result)
        self._check_outliers(X_train, thresholds, result)
        self._check_skewness(X_train, thresholds, result)

        result.is_valid = not result.errors
        return result

    def _check_dataset_size(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        thresholds: Dict[str, float],
        result: ValidationResult,
    ):
        n_train = len(X_train)
        n_test = len(X_test)
        n_total = n_train + n_test
        result.details["n_train"] = n_train
        result.details["n_test"] = n_test
        result.details["n_total"] = n_total

        if n_total < thresholds["min_dataset_size"]:
            result.errors.append(
                f"Dataset too small: {n_total} samples < {thresholds['min_dataset_size']} minimum."
            )
        elif n_total < thresholds["min_dataset_size"] * 1.2:
            result.warnings.append(
                f"Dataset size {n_total} is close to the minimum threshold {thresholds['min_dataset_size']}."
            )

        if n_test < thresholds["min_test_size"]:
            result.warnings.append(
                f"Test split has only {n_test} samples (< {thresholds['min_test_size']})."
            )

    def _check_class_balance(
        self,
        y_train: Optional[np.ndarray],
        y_test: Optional[np.ndarray],
        thresholds: Dict[str, float],
        result: ValidationResult,
    ):
        if y_train is None or y_test is None:
            return

        def _imbalance_ratio(values: np.ndarray) -> float:
            unique, counts = np.unique(values, return_counts=True)
            if len(unique) < 2:
                return 0.0
            return counts.min() / counts.max()

        train_ratio = _imbalance_ratio(y_train)
        test_ratio = _imbalance_ratio(y_test)
        result.details["train_class_ratio"] = float(train_ratio)
        result.details["test_class_ratio"] = float(test_ratio)

        if train_ratio < thresholds["class_imbalance_ratio"]:
            result.warnings.append(
                f"Severe class imbalance in training set (ratio={train_ratio:.3f})."
            )
        if test_ratio < thresholds["class_imbalance_ratio"]:
            result.warnings.append(
                f"Severe class imbalance in test set (ratio={test_ratio:.3f})."
            )

        missing_classes = set(np.unique(y_train)) - set(np.unique(y_test))
        if missing_classes:
            result.errors.append(
                f"Test split missing classes present in training data: {sorted(missing_classes)}."
            )

    def _check_class_cardinality(
        self,
        y_train: Optional[np.ndarray],
        y_test: Optional[np.ndarray],
        thresholds: Dict[str, float],
        result: ValidationResult,
    ):
        min_classes = int(thresholds.get("min_classes", 0))
        if min_classes <= 0 or y_train is None:
            return

        train_unique = np.unique(y_train)
        n_train_classes = len(train_unique)
        result.details["train_unique_classes"] = n_train_classes
        if y_test is not None:
            n_test_classes = len(np.unique(y_test))
            result.details["test_unique_classes"] = n_test_classes
        else:
            n_test_classes = 0

        if n_train_classes < min_classes:
            result.errors.append(
                f"Training labels contain only {n_train_classes} class(es); "
                f"need at least {min_classes}."
            )
        if y_test is not None and n_test_classes < min_classes:
            result.errors.append(
                f"Test labels contain only {n_test_classes} class(es); "
                f"need at least {min_classes}."
            )

    def _check_feature_counts(
        self,
        X_train: np.ndarray,
        thresholds: Dict[str, float],
        result: ValidationResult,
    ):
        n_features = X_train.shape[1]
        result.details["n_features"] = n_features
        if n_features < thresholds["min_features"]:
            result.errors.append(
                f"Dataset has {n_features} features (< {thresholds['min_features']})."
            )
        if n_features > thresholds["max_features"]:
            result.warnings.append(
                f"Dataset has {n_features} features (> {thresholds['max_features']})."
            )

    def _check_missing_values(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        thresholds: Dict[str, float],
        result: ValidationResult,
    ):
        train_missing = float(np.isnan(X_train).mean()) if X_train.size else 0.0
        test_missing = float(np.isnan(X_test).mean()) if X_test.size else 0.0
        result.details["train_missing_ratio"] = train_missing
        result.details["test_missing_ratio"] = test_missing
        max_ratio = thresholds["max_missing_ratio"]
        if train_missing > max_ratio or test_missing > max_ratio:
            result.errors.append(
                f"Missing value ratio exceeds {max_ratio:.2f} "
                f"(train={train_missing:.3f}, test={test_missing:.3f})."
            )
        elif max(train_missing, test_missing) > max_ratio * 0.5:
            result.warnings.append(
                f"High missing value ratio observed "
                f"(train={train_missing:.3f}, test={test_missing:.3f})."
            )

    def _check_feature_correlations(
        self,
        X_train: np.ndarray,
        thresholds: Dict[str, float],
        result: ValidationResult,
    ):
        if X_train.shape[1] < 2:
            return
        with np.errstate(invalid="ignore"):
            corr = np.corrcoef(X_train, rowvar=False)
        if not np.isfinite(corr).any():
            return
        upper = corr[np.triu_indices_from(corr, k=1)]
        if not upper.size:
            return
        max_corr = np.nanmax(np.abs(upper))
        result.details["max_feature_correlation"] = float(max_corr)
        if max_corr > thresholds["feature_correlation_threshold"]:
            result.warnings.append(
                f"Highly correlated features detected (max |rho|={max_corr:.3f})."
            )

    def _check_outliers(
        self,
        X_train: np.ndarray,
        thresholds: Dict[str, float],
        result: ValidationResult,
    ):
        if not X_train.size:
            return
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        std[std == 0] = 1.0
        z_scores = np.abs((X_train - mean) / std)
        outlier_mask = z_scores > thresholds["outlier_threshold"]
        outlier_ratio = float(outlier_mask.any(axis=1).mean())
        result.details["outlier_ratio"] = outlier_ratio
        if outlier_ratio > thresholds["max_outlier_ratio"]:
            result.warnings.append(
                f"Outlier ratio {outlier_ratio:.3f} exceeds "
                f"{thresholds['max_outlier_ratio']:.3f}."
            )

    def _check_skewness(
        self,
        X_train: np.ndarray,
        thresholds: Dict[str, float],
        result: ValidationResult,
    ):
        if not X_train.size:
            return
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        std[std == 0] = 1.0
        centered = X_train - mean
        skewness = np.mean((centered / std) ** 3, axis=0)
        max_skew = float(np.nanmax(np.abs(skewness)))
        result.details["max_skewness"] = max_skew
        if max_skew > thresholds["max_skewness"]:
            result.warnings.append(
                f"Highly skewed feature detected (max skew={max_skew:.3f})."
            )

    @staticmethod
    def _safe_targets(values: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if values is None:
            return None
        array = np.asarray(values)
        if array.ndim > 1:
            array = array.ravel()
        return array
