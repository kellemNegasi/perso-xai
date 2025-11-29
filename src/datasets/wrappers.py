"""Helper wrappers for cleaning TabularDataset targets from common loaders."""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import numpy as np

from src.datasets.tabular import TabularDataset


def _map_labels(values: Optional[Sequence[Any]], transform: Callable[[Any], int]):
    if values is None:
        return None
    array = np.asarray(values)
    return np.asarray([transform(v) for v in array], dtype=int)


def _normalize_text(value: Any) -> str:
    text = str(value).strip().lower()
    text = text.replace(".", "").replace(" ", "")
    return text


def _adult_income_transform(value: Any) -> int:
    text = _normalize_text(value)
    return 1 if ">" in text else 0


def _bank_marketing_transform(value: Any) -> int:
    text = _normalize_text(value)
    if text in {"yes", "1", "true"}:
        return 1
    return 0


def _german_credit_transform(value: Any) -> int:
    text = _normalize_text(value)
    if text in {"good", "1", "true"}:
        return 1
    return 0


def _compas_transform(value: Any) -> int:
    text = _normalize_text(value)
    if text in {"1", "yes", "true"}:
        return 1
    return 0


def wrap_openml_adult_income(
    *,
    X_train,
    y_train,
    X_test,
    y_test,
    feature_names,
):
    y_train_clean = _map_labels(y_train, _adult_income_transform)
    y_test_clean = _map_labels(y_test, _adult_income_transform)
    return TabularDataset.from_arrays(
        X_train=X_train,
        y_train=y_train_clean,
        X_test=X_test,
        y_test=y_test_clean,
        feature_names=feature_names,
    )


def wrap_openml_bank_marketing(
    *,
    X_train,
    y_train,
    X_test,
    y_test,
    feature_names,
):
    y_train_clean = _map_labels(y_train, _bank_marketing_transform)
    y_test_clean = _map_labels(y_test, _bank_marketing_transform)
    return TabularDataset.from_arrays(
        X_train=X_train,
        y_train=y_train_clean,
        X_test=X_test,
        y_test=y_test_clean,
        feature_names=feature_names,
    )


def wrap_openml_german_credit(
    *,
    X_train,
    y_train,
    X_test,
    y_test,
    feature_names,
):
    y_train_clean = _map_labels(y_train, _german_credit_transform)
    y_test_clean = _map_labels(y_test, _german_credit_transform)
    return TabularDataset.from_arrays(
        X_train=X_train,
        y_train=y_train_clean,
        X_test=X_test,
        y_test=y_test_clean,
        feature_names=feature_names,
    )


def wrap_openml_compas(
    *,
    X_train,
    y_train,
    X_test,
    y_test,
    feature_names,
):
    y_train_clean = _map_labels(y_train, _compas_transform)
    y_test_clean = _map_labels(y_test, _compas_transform)
    return TabularDataset.from_arrays(
        X_train=X_train,
        y_train=y_train_clean,
        X_test=X_test,
        y_test=y_test_clean,
        feature_names=feature_names,
    )


__all__ = [
    "wrap_openml_adult_income",
    "wrap_openml_bank_marketing",
    "wrap_openml_german_credit",
    "wrap_openml_compas",
]
