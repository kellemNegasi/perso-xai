"""Configuration helpers for preference-learning experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class ExperimentConfig:
    """High-level knobs for preference-learning experiments."""

    test_size: float = 0.2
    random_state: int = 42
    top_k: Sequence[int] = field(default_factory=lambda: (3, 5, 8))
    num_users: int = 10
    persona_seed: int = 13
    label_seed: int = 41
    tau: float | None = None
    concentration_c: float | None = None
    exclude_feature_groups: Sequence[str] = field(default_factory=tuple)
    autoxai_include_all_metrics: bool = False
    autoxai_enabled: bool = True
