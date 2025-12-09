"""
Hyperparameter tuning utilities inspired by the benchmarking project.
"""

from __future__ import annotations

import json
import logging
import ast
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from sklearn.model_selection import GridSearchCV, cross_val_score

from src.models.builder import (
    build_estimator_from_spec,
    prefix_param_grid_for_pipeline,
    strip_pipeline_prefix,
)
from src.orchestrators.registry import ModelRegistry

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optuna is optional
    import optuna
except ImportError:  # pragma: no cover - optuna only if installed
    optuna = None

MODEL_ALIASES = {
    "mlp_classifier": "mlp",
}


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
        dataset_type: str = "tabular",
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
        LOGGER.info("[TUNER] No cache found. Running search for %s/%s", dataset_name, model_name)
        return self.tune_and_save(
            dataset_name=dataset_name,
            model_name=model_name,
            X=X,
            y=y,
            dataset_type=dataset_type,
        )

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
        dataset_type: str = "tabular",
    ) -> Dict[str, Any]:
        """Run GridSearch tuning for a model/dataset pair and persist results."""
        grid = self.get_model_param_grid(model_name, dataset_type)
        if not grid:
            LOGGER.info("No hyperparameter grid defined for model '%s'. Skipping tuning.", model_name)
            return {}

        spec = self.model_registry.get(model_name)
        search_space = prefix_param_grid_for_pipeline(spec, grid)

        y_vector = np.asarray(y)
        if y_vector.ndim > 1:
            y_vector = y_vector.ravel()

        method = (self.settings.get("optimization_method") or "grid_search").lower()
        best_score: float
        cv_results: Dict[str, Any]
        if method == "optuna" and optuna is not None:
            best_params, best_score, trial_history = self._run_optuna_search(
                spec,
                search_space,
                X,
                y_vector,
            )
            cv_results = {"optuna_trials": trial_history}
        else:
            if method == "optuna" and optuna is None:
                LOGGER.warning(
                    "Optuna optimization requested but package not installed. Falling back to grid search."
                )
            best_params, best_score, cv_results = self._run_grid_search(
                spec,
                search_space,
                X,
                y_vector,
            )
        LOGGER.info(
            "[TUNER] Search complete for %s/%s (best score=%.4f)",
            dataset_name,
            model_name,
            best_score,
        )
        self._write_results(
            dataset_name=dataset_name,
            model_name=model_name,
            best_params=best_params,
            best_score=float(best_score),
            cv_results=cv_results,
        )
        return best_params

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

    # ------------------------------------------------------------------ #
    # Extended helpers                                                   #
    # ------------------------------------------------------------------ #

    def get_model_param_grid(self, model_name: str, dataset_type: Optional[str]) -> Dict[str, Any]:
        """Return the tuned parameter grid adjusted for dataset type where applicable."""
        dataset_type = dataset_type or "tabular"
        grid = self._resolve_grid(model_name)
        if not grid:
            return {}

        # Mirror benchmarking customization hooks so that tabular MLPs stay lightweight.
        if dataset_type == "tabular" and model_name in {"mlp", "mlp_classifier"}:
            raw_sizes = grid.get("hidden_layer_sizes", [[50], [100], [50, 50]])
            return {
                "hidden_layer_sizes": self._normalize_hidden_layer_sizes(raw_sizes),
                "activation": grid.get("activation", ["relu", "tanh"]),
                "alpha": grid.get("alpha", [0.0001, 0.001, 0.01]),
                "max_iter": grid.get("max_iter", [500, 1000]),
            }

        if dataset_type == "image" and model_name in {"cnn", "vit"}:
            return grid

        if dataset_type == "text" and model_name in {"bert", "lstm"}:
            return grid

        return grid

    def _resolve_grid(self, model_name: str) -> Dict[str, Any]:
        """Return a parameter grid honoring model-name aliases."""
        if model_name in self.grids:
            return self.grids[model_name]
        alias = MODEL_ALIASES.get(model_name)
        if alias and alias in self.grids:
            return self.grids[alias]
        return {}

    @staticmethod
    def _normalize_hidden_layer_sizes(values: List[Any]) -> List[Any]:
        """
        Ensure hidden_layer_sizes values are tuples/ints even when config strings slip through.
        """
        normalized: List[Any] = []
        for raw in values:
            parsed = raw
            if isinstance(raw, str):
                try:
                    parsed = ast.literal_eval(raw)
                except Exception:
                    parsed = raw
            if isinstance(parsed, int):
                normalized.append(int(parsed))
            elif isinstance(parsed, (list, tuple)):
                normalized.append(tuple(int(v) for v in parsed))
            else:
                raise ValueError(
                    "hidden_layer_sizes entries must be ints or sequences "
                    f"(received {raw!r} of type {type(raw).__name__})."
                )
        return normalized

    def _run_grid_search(
        self,
        spec: Dict[str, Any],
        search_space: Dict[str, List[Any]],
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[Dict[str, Any], float, Dict[str, Any]]:
        search = GridSearchCV(
            build_estimator_from_spec(spec),
            search_space,
            cv=int(self.settings["cv_folds"]),
            scoring=self.settings.get("scoring"),
            n_jobs=int(self.settings.get("n_jobs", -1)),
            verbose=int(self.settings.get("verbose", 1)),
        )
        search.fit(X, y)
        best_params = strip_pipeline_prefix(spec, search.best_params_)
        return best_params, float(search.best_score_), dict(search.cv_results_)

    def _run_optuna_search(
        self,
        spec: Dict[str, Any],
        search_space: Dict[str, List[Any]],
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
        if optuna is None:  # pragma: no cover - defensive, handled earlier
            raise RuntimeError("Optuna not available")

        cv = int(self.settings["cv_folds"])
        scoring = self.settings.get("scoring")
        n_jobs = int(self.settings.get("n_jobs", -1))
        n_trials = int(self.settings.get("n_trials", 100))
        timeout = self.settings.get("timeout")

        def objective(trial: "optuna.Trial") -> float:
            estimator = build_estimator_from_spec(spec)
            params: Dict[str, Any] = {}
            for key, values in search_space.items():
                if not values:
                    continue
                params[key] = trial.suggest_categorical(key, list(values))
            if params:
                estimator.set_params(**params)
            scores = cross_val_score(
                estimator,
                X,
                y,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
            )
            return float(np.mean(scores))

        def _log_trial_progress(study, trial) -> None:
            best_value = getattr(study, "best_value", None)
            LOGGER.info(
                "[TUNER][OPTUNA] trial=%s state=%s value=%s best=%s",
                trial.number,
                trial.state.name if hasattr(trial.state, "name") else str(trial.state),
                trial.value,
                f"{best_value:.4f}" if isinstance(best_value, (float, int)) else best_value,
            )

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, timeout=timeout, callbacks=[_log_trial_progress])
        best_params = strip_pipeline_prefix(spec, study.best_params)
        trials_metadata = [
            {
                "number": trial.number,
                "value": trial.value,
                "params": strip_pipeline_prefix(spec, trial.params),
                "state": str(trial.state),
            }
            for trial in study.trials
        ]
        return best_params, float(study.best_value), trials_metadata


def _json_converter(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj
