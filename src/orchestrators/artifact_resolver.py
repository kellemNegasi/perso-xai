"""Helpers for resolving dataset/model/explainer references from configs."""

from __future__ import annotations

from typing import Any, Iterable, List, Optional


def resolve_artifact_key(
    entry: Any,
    kind: str,
    *,
    required: bool = True,
) -> Optional[str]:
    """Extract the canonical key/name for a dataset/model/explainer entry."""
    if entry is None:
        if required:
            raise ValueError(f"Experiment is missing a '{kind}' reference.")
        return None
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        key = entry.get("key") or entry.get("name")
        if key:
            return str(key)
    if required:
        raise ValueError(
            f"Experiment {kind} entries must be strings or mappings with a 'key'. Got: {entry!r}"
        )
    return None


def resolve_artifact_list(entries: Optional[Iterable[Any]], kind: str) -> List[str]:
    """Return all artifact keys for a list of config entries, skipping blanks."""
    if not entries:
        return []
    names: List[str] = []
    for entry in entries:
        key = resolve_artifact_key(entry, kind)
        if key:
            names.append(key)
    return names


__all__ = ["resolve_artifact_key", "resolve_artifact_list"]
