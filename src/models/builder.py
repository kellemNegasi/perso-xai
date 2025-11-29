"""
Helpers for constructing estimator instances from registry specifications.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Optional

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _import_object(module_name: str, attr: str) -> Any:
    module = import_module(module_name)
    obj = module
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def build_estimator_from_spec(
    spec: Dict[str, Any],
    *,
    params_override: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Instantiate an estimator described in ``models.yml``.

    Parameters
    ----------
    spec : Dict[str, Any]
        Model specification entry.
    params_override : Dict[str, Any] | None
        Optional dictionary of parameter overrides (applied after defaults).
    """
    params = dict(spec.get("params", {}) or {})
    if params_override:
        params.update(params_override)

    model_cls = _import_object(spec["module"], spec["class"])
    estimator = model_cls(**params)

    if spec.get("fit", {}).get("requires_scaler", False):
        estimator = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("estimator", estimator),
            ]
        )
    return estimator


def prefix_param_grid_for_pipeline(
    spec: Dict[str, Any],
    grid: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Adjust parameter grids when the estimator is wrapped in a scaler pipeline.
    """
    if not spec.get("fit", {}).get("requires_scaler", False):
        return dict(grid)
    return {f"estimator__{key}": value for key, value in grid.items()}


def strip_pipeline_prefix(
    spec: Dict[str, Any],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Remove pipeline prefixes from GridSearch results for downstream consumption.
    """
    if not spec.get("fit", {}).get("requires_scaler", False):
        return dict(params)
    prefix = "estimator__"
    cleaned: Dict[str, Any] = {}
    for key, value in params.items():
        if key.startswith(prefix):
            cleaned[key[len(prefix) :]] = value
        else:
            cleaned[key] = value
    return cleaned
