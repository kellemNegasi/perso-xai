# src/models/sklearn_models.py
from __future__ import annotations
from typing import Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def train_simple_classifier(
    X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42
) -> RandomForestClassifier:
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=random_state,
    )
    clf.fit(X_train, y_train)
    return clf
