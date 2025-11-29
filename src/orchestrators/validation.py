"""Reusable validation helpers for orchestrator configuration."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple


def validate_artifact_compatibility(
    *,
    dataset: Tuple[str, Dict[str, object]],
    models: Iterable[Tuple[str, Dict[str, object]]],
    explainers: Iterable[Tuple[str, Dict[str, object]]],
    scope: str,
) -> str:
    """
    Ensure that datasets, models, and explainers all agree on the target data type.

    Returns the resolved dataset type so callers can route to the appropriate handlers.
    """

    dataset_name, dataset_spec = dataset
    dataset_type = dataset_spec.get("type")
    if not dataset_type:
        raise ValueError(
            f"{scope}: Dataset '{dataset_name}' is missing a 'type' declaration; "
            "see configs/dataset.yml for the schema."
        )

    def _collect_incompatible(
        artifacts: Iterable[Tuple[str, Dict[str, object]]],
    ) -> list[str]:
        incompatible: list[str] = []
        for name, spec in artifacts:
            supported = spec.get("supported_data_types", [])
            if not supported:
                incompatible.append(name)
            elif dataset_type not in supported:
                incompatible.append(name)
        return incompatible

    bad_models = _collect_incompatible(models)
    if bad_models:
        raise ValueError(
            f"{scope}: Dataset '{dataset_name}' is type '{dataset_type}' "
            f"but the following models do not declare support for it: {', '.join(bad_models)}"
        )

    bad_explainers = _collect_incompatible(explainers)
    if bad_explainers:
        raise ValueError(
            f"{scope}: Dataset '{dataset_name}' is type '{dataset_type}' "
            f"but the following explainers do not declare support for it: {', '.join(bad_explainers)}"
        )

    return str(dataset_type)
