"""
Utilities for saving and reloading trained models.
"""

from __future__ import annotations

import logging
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

import joblib

LOGGER = logging.getLogger(__name__)


class ModelPersistence:
    """Store serialized models under ``saved_models/<dataset>/<model>.joblib``."""

    def __init__(
        self,
        base_dir: str | Path = "saved_models",
        recursion_limit: Optional[int] = 1_000_000,
    ):
        self.base_dir = Path(base_dir)
        self._recursion_limit = recursion_limit

    def path_for(self, dataset_name: str, model_name: str) -> Path:
        safe_dataset = dataset_name.replace("/", "_")
        safe_model = model_name.replace("/", "_")
        return self.base_dir / safe_dataset / f"{safe_model}.joblib"

    def load(self, dataset_name: str, model_name: str) -> Optional[Any]:
        path = self.path_for(dataset_name, model_name)
        if not path.exists():
            return None
        try:
            with self._boosted_recursion_limit():
                model = joblib.load(path)
            LOGGER.info("[PERSIST] Loaded model %s/%s from %s", dataset_name, model_name, path)
            return model
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.warning(
                "Failed to load model from %s: %s\n%s",
                path,
                exc,
                traceback.format_exc(),
            )
            return None

    def save(self, dataset_name: str, model_name: str, model: Any) -> None:
        path = self.path_for(dataset_name, model_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self._boosted_recursion_limit():
                joblib.dump(model, path)
            LOGGER.info("[PERSIST] Saved model %s/%s to %s", dataset_name, model_name, path)
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.warning(
                "Failed to save model to %s: %s\n%s",
                path,
                exc,
                traceback.format_exc(),
            )

    @contextmanager
    def _boosted_recursion_limit(self):
        """Temporarily raise the recursion limit for (de)serializing deep estimators."""
        if self._recursion_limit is None:
            yield
            return
        current = sys.getrecursionlimit()
        if self._recursion_limit <= current:
            yield
            return
        sys.setrecursionlimit(self._recursion_limit)
        try:
            yield
        finally:
            sys.setrecursionlimit(current)
