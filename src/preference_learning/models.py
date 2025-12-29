"""Model definitions for preference learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold


@dataclass
class LinearSVCConfig:
    """Configuration for the linear SVC preference model."""

    C: float = 1.0
    max_iter: int = 5000
    random_state: int = 42
    dual: bool = False
    tune: bool = False
    tune_cv_folds: int = 5
    tune_scoring: str = "balanced_accuracy"
    tune_C_grid: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0)


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
        self.tuning_summary: dict[str, Any] | None = None

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> None:
        self._model.fit(
            np.asarray(features, dtype=float),
            np.asarray(labels, dtype=int),
        )

    def tune_and_fit(self, features: pd.DataFrame, labels: pd.Series) -> dict[str, Any]:
        X = np.asarray(features, dtype=float)
        y = np.asarray(labels, dtype=int)

        unique, counts = np.unique(y, return_counts=True)
        if unique.size < 2:
            self.fit(features, labels)
            summary = {
                "best_params": {"C": float(self.config.C)},
                "best_score": None,
                "scoring": str(self.config.tune_scoring or "balanced_accuracy"),
                "cv_folds": 0,
                "C_grid": [float(self.config.C)],
                "note": "Skipped tuning because training labels had <2 classes.",
            }
            self.tuning_summary = summary
            return summary

        c_grid = [float(c) for c in (self.config.tune_C_grid or ())]
        if not c_grid:
            c_grid = [float(self.config.C)]
        c_grid = sorted({c for c in c_grid if c > 0})
        if not c_grid:
            c_grid = [1.0]

        max_folds = int(self.config.tune_cv_folds)
        max_folds = max(2, max_folds)
        cv_folds = min(max_folds, int(counts.min()))
        if cv_folds < 2:
            self.fit(features, labels)
            summary = {
                "best_params": {"C": float(self.config.C)},
                "best_score": None,
                "scoring": str(self.config.tune_scoring or "balanced_accuracy"),
                "cv_folds": 0,
                "C_grid": c_grid,
                "note": "Skipped tuning because not enough samples per class for CV.",
            }
            self.tuning_summary = summary
            return summary

        splitter = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )
        base = LinearSVC(
            max_iter=self.config.max_iter,
            random_state=self.config.random_state,
            dual=self.config.dual,
        )
        search = GridSearchCV(
            estimator=base,
            param_grid={"C": c_grid},
            scoring=str(self.config.tune_scoring or "balanced_accuracy"),
            cv=splitter,
            n_jobs=1,
            refit=True,
        )
        try:
            search.fit(X, y)
            self._model = search.best_estimator_
            summary = {
                "best_params": dict(search.best_params_ or {}),
                "best_score": float(search.best_score_) if search.best_score_ is not None else None,
                "scoring": str(search.scoring),
                "cv_folds": cv_folds,
                "C_grid": c_grid,
            }
        except Exception as exc:
            self.fit(features, labels)
            summary = {
                "best_params": {"C": float(self.config.C)},
                "best_score": None,
                "scoring": str(self.config.tune_scoring or "balanced_accuracy"),
                "cv_folds": 0,
                "C_grid": c_grid,
                "note": f"Tuning failed; fell back to default fit. Error: {exc}",
            }
        self.tuning_summary = summary
        return summary

    def decision_function(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self._model.decision_function(np.asarray(features, dtype=float))

    def score_candidates(
        self,
        candidates: pd.DataFrame,
        feature_columns: Sequence[str],
    ) -> pd.Series:
        scores = self.decision_function(candidates[feature_columns])
        return pd.Series(scores, index=candidates["method_variant"], name="score")
