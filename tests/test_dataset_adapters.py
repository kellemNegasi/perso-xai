from __future__ import annotations

import numpy as np

from src.datasets import TabularDataset
from src.datasets.adapters import LoaderDatasetAdapter
from src.orchestrators.utils import instantiate_dataset


def test_instantiate_dataset_tabular_toy_returns_tabular_dataset():
    dataset = instantiate_dataset("tabular_toy")
    assert isinstance(dataset, TabularDataset)
    assert dataset.X_train.shape[1] == 8
    assert dataset.y_train is not None


def test_loader_adapter_handles_dataframe_with_categoricals():
    spec = {
        "loader": {
            "module": "tests.helpers.datasets",
            "factory": "dataframe_loader",
        },
        "params": {},
        "target_column": "income",
        "split": {"test_size": 0.5, "random_state": 1, "stratify": True},
    }
    adapter = LoaderDatasetAdapter(name="dummy_frame", spec=spec)
    dataset = adapter.load()

    assert isinstance(dataset, TabularDataset)
    assert dataset.X_train.dtype == np.float64
    # Expect numeric column plus one-hot encoding of categorical feature
    assert dataset.X_train.shape[1] >= 3
    assert all(name.startswith("num") or name.startswith("cat_") for name in dataset.feature_names)


def test_loader_adapter_preserves_array_feature_names():
    spec = {
        "loader": {
            "module": "tests.helpers.datasets",
            "factory": "simple_array_loader",
        },
        "params": {},
        "split": {"test_size": 0.4, "random_state": 0, "stratify": True},
    }
    adapter = LoaderDatasetAdapter(name="dummy_array", spec=spec)
    dataset = adapter.load()

    assert dataset.feature_names == ["feat_a", "feat_b"]
    assert dataset.X_train.shape[1] == 2
