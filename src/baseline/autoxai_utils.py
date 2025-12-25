from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


def parse_top_k(values: Sequence[str] | None) -> Tuple[int, ...]:
    if not values:
        return (3, 5)
    parsed: List[int] = []
    for raw in values:
        try:
            parsed.append(int(raw))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid integer for --top-k: {raw}") from exc
    return tuple(parsed)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def standard_scale(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    std = math.sqrt(var)
    if std == 0.0:
        return [0.0 for _ in values]
    return [(v - mean) / std for v in values]


def minmax_scale(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi == lo:
        return [0.0 for _ in values]
    span = hi - lo
    return [(v - lo) / span for v in values]


def scalarize_terms(
    term_values: Mapping[str, float],
    *,
    weights: Mapping[str, float],
) -> float:
    if not term_values:
        return float("-inf")
    denom = len(term_values)
    return sum(weights.get(name, 1.0) * value for name, value in term_values.items()) / denom


def tie_break(seed: int, dataset_index: object, pair_1: str, pair_2: str) -> int:
    token = f"{seed}:{dataset_index}:{pair_1}:{pair_2}".encode("utf-8")
    digest = hashlib.sha256(token).digest()
    return digest[0] % 2

