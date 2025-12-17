"""Configuration helpers for preference-learning experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class ExperimentConfig:
    """High-level knobs for preference-learning experiments."""

    test_size: float = 0.2
    random_state: int = 42
    top_k: Sequence[int] = field(default_factory=lambda: (1, 3, 5))

