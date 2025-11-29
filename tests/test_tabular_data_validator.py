from __future__ import annotations

import numpy as np

from src.datasets.tabular import TabularDataset
from src.validators.tabular import TabularDataValidator


def _make_dataset(n_samples: int = 120, n_features: int = 4) -> TabularDataset:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n_samples, n_features))
    y = rng.integers(0, 2, size=n_samples)
    split = int(n_samples * 0.7)
    return TabularDataset.from_arrays(
        X_train=X[:split],
        y_train=y[:split],
        X_test=X[split:],
        y_test=y[split:],
        feature_names=[f"feat_{i}" for i in range(n_features)],
    )


def test_validator_accepts_clean_dataset():
    dataset = _make_dataset()
    validator = TabularDataValidator({"min_dataset_size": 50})
    result = validator.validate(dataset=dataset, dataset_name="clean")
    assert result.is_valid
    assert not result.errors


def test_validator_flags_small_dataset():
    dataset = _make_dataset(n_samples=20)
    validator = TabularDataValidator({"min_dataset_size": 40})
    result = validator.validate(dataset=dataset, dataset_name="tiny")
    assert not result.is_valid
    assert any("Dataset too small" in err for err in result.errors)
