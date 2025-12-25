from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence


@dataclass(frozen=True)
class ObjectiveTerm:
    name: str
    metric_key: str
    direction: str  # "max" or "min"
    weight: float = 1.0

    def apply_direction(self, value: float) -> float:
        if self.direction == "max":
            return value
        if self.direction == "min":
            return -value
        raise ValueError(f"Unknown direction: {self.direction!r} (expected 'max' or 'min').")


@dataclass(frozen=True)
class CandidateScore:
    dataset_index: int
    method_variant: str
    method: str
    raw_terms: Mapping[str, float]
    scaled_terms: Mapping[str, float]
    aggregated_score: float


@dataclass(frozen=True)
class HPOTrial:
    method: str
    method_variant: str
    mean_score: float


@dataclass(frozen=True)
class HPOResult:
    method: str
    mode: str
    seed: int
    epochs: int
    default_variant: Optional[str]
    default_mean_score: Optional[float]
    trials: Sequence[HPOTrial]
    best_variant: str
    best_mean_score: float

