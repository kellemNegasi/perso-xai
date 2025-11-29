"""
Hyperparameter tuning utilities inspired by the benchmarking project.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml
from sklearn.model_selection import GridSearchCV

from src.models.builder import (
    build_estimator_from_spec,
    prefix_param_grid_for_pipeline,
    strip_pipeline_prefix,
)
from src.orchestrators.registry import ModelRegistry

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
LOGGER = logging.getLogger(__name__)


def _load_config() -> Dict[str, Any]:
    path = CONFIG_DIR / "hyperparameters.yml"
    if not path.exists():
        return {"settings": {}, "grids": {}}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


class HyperparameterTuner:
    """
    Lightweight tuner that mirrors benchmarking's GridSearch-based workflow.
    """

    DEFAULT_SETTINGS: Dict[str, Any] = {
        "cv_folds": 5,
        "scoring": "accuracy",
        "n_jobs": -1,
        "verbose": 1,
        "optimization_method": "grid_search",
        "n_trials": 100,
        "timeout": 3600,
    }

    def __init__(
        self,
        *,
        config: Optional[Dict[str, Any]] = None,
        output_dir: Optional[str | Path] = None,
        model_registry: Optional[ModelRegistry] = None,
    ):
        raw_config = config or _load_config()
        self.settings = dict(self.DEFAULT_SETTINGS)
        self.settings.update(raw_config.get("settings", {}))
        self.grids: Dict[str, Dict[str, Any]] = raw_config.get("grids", {})
        self.output_dir = Path(output_dir or Path("saved_models") / "tuning_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_registry = model_registry or ModelRegistry()

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def ensure_best_parameters(
        self,
        *,
        dataset_name: str,
        model_name: str,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Load tuned parameters if present; otherwise run tuning and persist them.
        """
        LOGGER.info(
            "[TUNER] Ensuring tuned params for dataset=%s model=%s", dataset_name, model_name
        )
        best = self.load_best_parameters(dataset_name, model_name, silent=True)
        if best:
            LOGGER.info("[TUNER] Reusing cached parameters for %s/%s", dataset_name, model_name)
            return best
        LOGGER.info("[TUNER] No cache found. Running grid search for %s/%s", dataset_name, model_name)
        return self.tune_and_save(dataset_name=dataset_name, model_name=model_name, X=X, y=y)

    def load_best_parameters(
        self,
        dataset_name: str,
        model_name: str,
        *,
        silent: bool = False,
    ) -> Dict[str, Any]:
        """Return cached best parameters (if available)."""
        path = self._result_path(dataset_name, model_name)
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not silent:
            LOGGER.info("[TUNER] Loaded cached parameters for %s/%s", dataset_name, model_name)
        return payload.get("best_params", {})

    def tune_and_save(
        self,
        *,
        dataset_name: str,
        model_name: str,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, Any]:
        """Run GridSearch tuning for a model/dataset pair and persist results."""
        grid = self.grids.get(model_name)
        if not grid:
            LOGGER.info("No hyperparameter grid defined for model '%s'. Skipping tuning.", model_name)
            return {}

        spec = self.model_registry.get(model_name)
        estimator = build_estimator_from_spec(spec)
        search_space = prefix_param_grid_for_pipeline(spec, grid)

        y_vector = np.asarray(y)
        if y_vector.ndim > 1:
            y_vector = y_vector.ravel()

        method = self.settings.get("optimization_method", "grid_search")
        if method != "grid_search":
            LOGGER.warning(
                "Optimization method '%s' not implemented. Falling back to grid search.",
                method,
            )

        search = GridSearchCV(
            estimator,
            search_space,
            cv=int(self.settings["cv_folds"]),
            scoring=self.settings.get("scoring"),
            n_jobs=int(self.settings.get("n_jobs", -1)),
            verbose=int(self.settings.get("verbose", 1)),
        )
        search.fit(X, y_vector)

        raw_best = strip_pipeline_prefix(spec, search.best_params_)
        LOGGER.info(
            "[TUNER] Grid search complete for %s/%s (best score=%.4f)",
            dataset_name,
            model_name,
            search.best_score_,
        )
        self._write_results(
            dataset_name=dataset_name,
            model_name=model_name,
            best_params=raw_best,
            best_score=float(search.best_score_),
            cv_results=search.cv_results_,
        )
        return raw_best

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _write_results(
        self,
        *,
        dataset_name: str,
        model_name: str,
        best_params: Dict[str, Any],
        best_score: float,
        cv_results: Dict[str, Any],
    ) -> None:
        path = self._result_path(dataset_name, model_name)
        payload = {
            "dataset": dataset_name,
            "model": model_name,
            "best_params": best_params,
            "best_score": best_score,
            "cv_results": cv_results,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=_json_converter)

    def _result_path(self, dataset_name: str, model_name: str) -> Path:
        safe_dataset = dataset_name.replace("/", "_")
        safe_model = model_name.replace("/", "_")
        return self.output_dir / safe_dataset / f"{safe_model}.json"


def _json_converter(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj
