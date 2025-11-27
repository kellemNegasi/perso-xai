"""Helper dataset loaders for adapter tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd


@dataclass
class DummyBunch:
    data: Any = None
    target: Any = None
    feature_names: Any = None
    frame: Any = None

    def get(self, attr: str, default: Any = None) -> Any:
        return getattr(self, attr, default)


def simple_array_loader() -> DummyBunch:
    data = np.arange(20, dtype=float).reshape(10, 2)
    target = np.array([0, 1] * 5)
    feature_names = ["feat_a", "feat_b"]
    return DummyBunch(data=data, target=target, feature_names=feature_names)


def dataframe_loader() -> DummyBunch:
    records: Dict[str, Any] = {
        "num": [1, 2, 3, 4, 5, 6],
        "cat": ["x", "y", "x", "z", "y", "x"],
        "income": ["<=50K", ">50K", "<=50K", "<=50K", ">50K", ">50K"],
    }
    frame = pd.DataFrame(records)
    return DummyBunch(frame=frame)
