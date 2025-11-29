"""
Lightweight registries over config YAML to expose dataset/model/explainer metadata.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


class BaseRegistry:
    """Common helper for loading YAML config entries with defensive copying."""

    config_name: str

    def __init__(self, config_name: str | None = None):
        name = config_name or getattr(self, "config_name", None)
        if not name:
            raise ValueError("config_name must be provided for registry instances")
        self.config_name = name
        self._raw_config = self._load_yaml(name)
        self._entries = self._build_index()

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        path = CONFIG_DIR / filename
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    def _source_section(self) -> Dict[str, Any]:
        return dict(self._raw_config)

    def _normalize_entry(self, name: str, spec: Dict[str, Any]) -> Dict[str, Any]:
        return dict(spec)

    def _is_template_key(self, key: str) -> bool:
        return key.startswith("_") or key == "templates"

    def _build_index(self) -> Dict[str, Dict[str, Any]]:
        section = self._source_section()
        entries: Dict[str, Dict[str, Any]] = {}
        for name, spec in section.items():
            if not isinstance(spec, dict):
                continue
            if self._is_template_key(name):
                continue
            entries[name] = self._normalize_entry(name, spec)
        return entries

    def names(self) -> Iterable[str]:
        return self._entries.keys()

    def get(self, name: str) -> Dict[str, Any]:
        try:
            spec = self._entries[name]
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise KeyError(f"Unknown registry entry '{name}' in {self.config_name}") from exc
        return copy.deepcopy(spec)


class DatasetRegistry(BaseRegistry):
    """Registry that exposes dataset specs grouped by data type."""

    config_name = "dataset.yml"

    def _source_section(self) -> Dict[str, Any]:
        datasets = self._raw_config.get("datasets")
        if isinstance(datasets, dict):
            return datasets
        return super()._source_section()

    def list_by_type(self, data_type: str) -> Dict[str, Dict[str, Any]]:
        return {
            name: copy.deepcopy(spec)
            for name, spec in self._entries.items()
            if spec.get("type") == data_type
        }


class _SupportsDataTypeRegistry(BaseRegistry):
    """Shared mixin for registries that store supported_data_types lists."""

    def _normalize_entry(self, name: str, spec: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(spec)
        supported = normalized.get("supported_data_types")
        if not supported:
            normalized["supported_data_types"] = ["tabular"]
        else:
            normalized["supported_data_types"] = list(supported)
        return normalized

    def list_by_type(self, data_type: str) -> Dict[str, Dict[str, Any]]:
        return {
            name: copy.deepcopy(spec)
            for name, spec in self._entries.items()
            if data_type in spec.get("supported_data_types", [])
        }


class ModelRegistry(_SupportsDataTypeRegistry):
    """Registry wrapper for configs/models.yml."""

    config_name = "models.yml"


class ExplainerRegistry(_SupportsDataTypeRegistry):
    """Registry wrapper for configs/explainers.yml."""

    config_name = "explainers.yml"
