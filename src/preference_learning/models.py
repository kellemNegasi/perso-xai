"""Model definitions for preference learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC


@dataclass
class LinearSVCConfig:
    """Configuration for the linear SVC preference model."""

    C: float = 1.0
    max_iter: int = 5000
    random_state: int = 42
    dual: bool = False


class LinearSVCPreferenceModel:
    """Wrapper around sklearn's LinearSVC for pairwise difference features."""

    def __init__(self, config: LinearSVCConfig | None = None) -> None:
        self.config = config or LinearSVCConfig()
        self._model = LinearSVC(
            C=self.config.C,
            max_iter=self.config.max_iter,
            random_state=self.config.random_state,
            dual=self.config.dual,
        )

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> None:
        self._model.fit(
            np.asarray(features, dtype=float),
            np.asarray(labels, dtype=int),
        )

    def decision_function(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self._model.decision_function(np.asarray(features, dtype=float))

    def score_candidates(
        self,
        candidates: pd.DataFrame,
        feature_columns: Sequence[str],
    ) -> pd.Series:
        scores = self.decision_function(candidates[feature_columns])
        return pd.Series(scores, index=candidates["method_variant"], name="score")
