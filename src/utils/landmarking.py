"""Landmarking feature extraction using cheap, fixed-capacity models."""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from src.datasets.tabular import TabularDataset

LOGGER = logging.getLogger(__name__)


def compute_landmarking_features(dataset: TabularDataset, seed: int = 42) -> Dict[str, float]:
    """
    Train simple models with fixed hyperparameters and return cross-val accuracies.

    Models (all evaluated with 5-fold stratified CV, reduced if data is tiny):
    - kNN (k=1, Euclidean, uniform weights)
    - Gaussian Naive Bayes
    - Decision stump (depth=1 tree)
    - Logistic regression (L2, C=1.0, lbfgs)
    """
    X, y = _stack_features_and_labels(dataset)
    if X is None or y is None:
        LOGGER.warning("Skipping landmarking features because X or y is missing.")
        return _zero_features()

    cv = _make_cv(y, seed=seed)
    if cv is None:
        LOGGER.warning("Insufficient data to run landmarking CV; emitting zeros.")
        return _zero_features()

    models = {
        "landmark_acc_knn1": KNeighborsClassifier(
            n_neighbors=1,
            metric="euclidean",
            weights="uniform",
        ),
        "landmark_acc_gaussian_nb": GaussianNB(),
        "landmark_acc_decision_stump": DecisionTreeClassifier(
            max_depth=1,
            random_state=seed,
        ),
        "landmark_acc_logreg": LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=200,
            random_state=seed,
        ),
    }

    features: Dict[str, float] = {}
    for name, model in models.items():
        score = _cv_accuracy(model, X, y, cv)
        features[name] = score
    return features


def _stack_features_and_labels(dataset: TabularDataset) -> Tuple[np.ndarray | None, np.ndarray | None]:
    arrays = []
    targets = []
    for X, y in (
        (dataset.X_train, dataset.y_train),
        (dataset.X_test, dataset.y_test),
    ):
        if X is None or y is None:
            continue
        if len(X) == 0 or len(y) == 0:
            continue
        arrays.append(np.asarray(X))
        targets.append(np.asarray(y))
    if not arrays:
        return None, None
    X_all = np.vstack(arrays)
    y_all = np.concatenate(targets)
    return X_all, y_all


def _make_cv(y: np.ndarray, *, seed: int) -> StratifiedKFold | None:
    _, counts = np.unique(y, return_counts=True)
    if counts.size < 2:
        return None
    n_samples = len(y)
    min_class = int(counts.min())
    n_splits = min(5, n_samples, min_class)
    if n_splits < 2:
        return None
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def _cv_accuracy(model, X: np.ndarray, y: np.ndarray, cv) -> float:
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy",
        error_score=np.nan,
    )
    valid = scores[~np.isnan(scores)]
    if valid.size == 0:
        return 0.0
    return float(np.mean(valid))


def _zero_features() -> Dict[str, float]:
    return {
        "landmark_acc_knn1": 0.0,
        "landmark_acc_gaussian_nb": 0.0,
        "landmark_acc_decision_stump": 0.0,
        "landmark_acc_logreg": 0.0,
    }
