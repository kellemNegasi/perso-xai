"""
Utilities for saving and reloading trained models.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import joblib

LOGGER = logging.getLogger(__name__)


class ModelPersistence:
    """Store serialized models under ``saved_models/<dataset>/<model>.joblib``."""

    def __init__(self, base_dir: str | Path = "saved_models"):
        self.base_dir = Path(base_dir)

    def path_for(self, dataset_name: str, model_name: str) -> Path:
        safe_dataset = dataset_name.replace("/", "_")
        safe_model = model_name.replace("/", "_")
        return self.base_dir / safe_dataset / f"{safe_model}.joblib"

    def load(self, dataset_name: str, model_name: str) -> Optional[Any]:
        path = self.path_for(dataset_name, model_name)
        if not path.exists():
            return None
        try:
            model = joblib.load(path)
            LOGGER.info("[PERSIST] Loaded model %s/%s from %s", dataset_name, model_name, path)
            return model
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.warning("Failed to load model from %s: %s", path, exc)
            return None

    def save(self, dataset_name: str, model_name: str, model: Any) -> None:
        path = self.path_for(dataset_name, model_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            joblib.dump(model, path)
            LOGGER.info("[PERSIST] Saved model %s/%s to %s", dataset_name, model_name, path)
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.warning("Failed to save model to %s: %s", path, exc)
