"""Loader for the ProPublica COMPAS recidivism dataset."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

try:  # pragma: no cover - optional dependency
    import pandas as pd
except Exception as exc:  # pragma: no cover
    raise ImportError("pandas is required to load the COMPAS dataset") from exc

from sklearn.model_selection import train_test_split

from src.datasets.tabular import TabularDataset


COMPAS_URL = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
DEFAULT_FEATURES = [
    "age",
    "priors_count",
    "days_b_screening_arrest",
]


def load_compas_dataset(
    *,
    feature_columns: Optional[Sequence[str]] = None,
    target_column: str = "two_year_recid",
    test_size: float = 0.3,
    random_state: Optional[int] = 42,
    stratify: bool = True,
) -> TabularDataset:
    """Load the COMPAS dataset and return a TabularDataset."""
    df = pd.read_csv(COMPAS_URL)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in COMPAS dataset")

    features = list(feature_columns or DEFAULT_FEATURES)
    missing_columns = [col for col in features if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Requested feature columns {missing_columns} not found in COMPAS dataset"
        )

    X = df[features].copy()
    y = df[target_column].copy()

    X = X.apply(_coerce_numeric)
    X = X.fillna(X.mean())
    y = _coerce_numeric_series(y).fillna(0).astype(int)

    stratify_labels = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels,
    )

    return TabularDataset.from_arrays(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=list(X.columns),
    )


def _coerce_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series
    return pd.to_numeric(series, errors="coerce")


def _coerce_numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series
    clean = series.str.strip().str.lower()
    mapping = {"yes": 1, "no": 0, "true": 1, "false": 0}
    return clean.map(mapping).fillna(pd.to_numeric(series, errors="coerce"))


__all__ = ["load_compas_dataset"]
