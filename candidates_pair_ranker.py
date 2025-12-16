#!/usr/bin/env python3
"""
Generate pairwise ranking labels for Pareto-front explanation candidates.

For every Pareto JSON, the ranker enumerates all explanation pairs within each
instance and assigns a persona-specific label by comparing ordered metric
preferences (e.g., layperson = compactness -> contrastivity -> stability).
Each pair receives label 0 when the first candidate wins and 1 when the second
does. Ties cascade through the priority list and fall back to a deterministic
random choice based on the dataset index and method variants.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Sequence

import pandas as pd

DEFAULT_RESULTS_ROOT = Path("results") / "full_run_dec8"
DEFAULT_PARETO_DIR = DEFAULT_RESULTS_ROOT / "pareto_fronts"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_ROOT / "candidate_pair_rankings"

COMPACTNESS_METRICS = (
    "compactness_effective_features",
    "compactness_sparsity",
    "compactness_top10_coverage",
    "compactness_top5_coverage",
)

def _coerce_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _mean_metric(metrics: Mapping[str, object], keys: Sequence[str]) -> float | None:
    collected: List[float] = []
    for key in keys:
        value = _coerce_float(metrics.get(key))
        if value is not None:
            collected.append(value)
    if not collected:
        return None
    return sum(collected) / len(collected)


MetricFetcher = Callable[[Mapping[str, object]], float | None]

METRIC_FETCHERS: Dict[str, MetricFetcher] = {
    "compactness": lambda metrics: _mean_metric(metrics, COMPACTNESS_METRICS),
    "contrastivity": lambda metrics: _coerce_float(metrics.get("contrastivity")),
    "stability": lambda metrics: _coerce_float(metrics.get("relative_input_stability")),
    "faithfulness": lambda metrics: _coerce_float(metrics.get("correctness")),
    "completeness": lambda metrics: _coerce_float(metrics.get("completeness_score")),
    "consistency": lambda metrics: _coerce_float(metrics.get("consistency")),
}

PERSONA_PRIORITIES: Dict[str, Sequence[str]] = {
    "layperson": ("compactness", "contrastivity", "stability"),
    "regulator": ("faithfulness", "completeness", "consistency", "compactness"),
}

AVAILABLE_METRICS_TEXT = ", ".join(sorted(METRIC_FETCHERS))
PERSONA_CHOICES = tuple(sorted(PERSONA_PRIORITIES))


@dataclass
class CandidateScores:
    """Compact container for a method variant and its priority metrics."""

    method_variant: str
    values: MutableMapping[str, float | None]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build pairwise ranking labels from Pareto-front metrics.",
    )
    parser.add_argument(
        "--pareto-dir",
        type=Path,
        default=DEFAULT_PARETO_DIR,
        help=f"Directory containing Pareto JSON files (default: {DEFAULT_PARETO_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory where pair label Parquet files will be stored (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--epsilon-mode",
        choices=("fixed", "relative"),
        default="fixed",
        help="Comparison tolerance strategy. 'fixed' uses --epsilon for all metrics, "
        "whereas 'relative' sets epsilon to (max-min)*--relative-factor per instance/metric.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.05,
        help="Absolute epsilon used when --epsilon-mode=fixed (default: 0.05).",
    )
    parser.add_argument(
        "--relative-factor",
        type=float,
        default=0.1,
        help="Fraction of the metric range used when --epsilon-mode=relative (default: 0.1).",
    )
    parser.add_argument(
        "--tie-breaker-seed",
        type=int,
        default=13,
        help="Seed that controls deterministic tie-breaking.",
    )
    parser.add_argument(
        "--persona",
        choices=PERSONA_CHOICES,
        default="layperson",
        help="Named priority template to use (default: layperson).",
    )
    parser.add_argument(
        "--priority-order",
        type=str,
        help="Comma-separated list of metric keys overriding --persona. "
        f"Available metrics: {AVAILABLE_METRICS_TEXT}.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pareto_files = sorted(args.pareto_dir.glob("*_pareto.json"))
    if not pareto_files:
        raise FileNotFoundError(f"No Pareto files found under {args.pareto_dir}")

    priority_levels = _resolve_priority_levels(args.persona, args.priority_order)

    ranker = CandidatePairRanker(
        epsilon_mode=args.epsilon_mode,
        fixed_epsilon=args.epsilon,
        relative_factor=args.relative_factor,
        tie_breaker_seed=args.tie_breaker_seed,
        priority_levels=priority_levels,
    )

    for path in pareto_files:
        df = ranker.rank_file(path)
        output_path = args.output_dir / f"{path.stem}_pair_labels.parquet"
        df.to_parquet(output_path, index=False)
        print(f"Wrote {len(df)} pair labels for {path.name} to {output_path}")


class CandidatePairRanker:
    """Ranks candidate explanations by emitting pairwise preference labels."""

    def __init__(
        self,
        *,
        epsilon_mode: str = "fixed",
        fixed_epsilon: float = 0.05,
        relative_factor: float = 0.1,
        tie_breaker_seed: int = 13,
        priority_levels: Sequence[str] | None = None,
        metric_fetchers: Mapping[str, MetricFetcher] | None = None,
    ) -> None:
        self.epsilon_mode = epsilon_mode
        self.fixed_epsilon = fixed_epsilon
        self.relative_factor = relative_factor
        self.tie_breaker_seed = tie_breaker_seed
        if not priority_levels:
            raise ValueError("At least one priority metric is required.")
        self.priority_levels = list(priority_levels)
        self.metric_fetchers = metric_fetchers or METRIC_FETCHERS

    def rank_file(self, path: Path) -> pd.DataFrame:
        payload = _load_json(path)
        instances: Iterable[Mapping[str, object]] = payload.get("instances") or []
        rows: List[Dict[str, object]] = []
        for instance in instances:
            rows.extend(self._rank_instance(instance))
        return pd.DataFrame(rows, columns=("dataset_index", "pair_1", "pair_2", "label"))

    def _rank_instance(self, instance: Mapping[str, object]) -> List[Dict[str, object]]:
        pareto_entries = instance.get("pareto_front") or []
        candidates: List[CandidateScores] = []
        for entry in pareto_entries:
            variant = _safe_str(entry.get("method_variant"))
            metrics = entry.get("metrics") or {}
            if not variant or not isinstance(metrics, Mapping):
                continue
            metric_values = {
                metric: self.metric_fetchers[metric](metrics)
                for metric in self.priority_levels
            }
            candidates.append(CandidateScores(method_variant=variant, values=metric_values))

        if len(candidates) < 2:
            return []

        epsilon_map = self._compute_epsilon(candidates)
        dataset_index = instance.get("dataset_index")
        rows: List[Dict[str, object]] = []
        for cand_a, cand_b in combinations(candidates, 2):
            label = self._label_pair(cand_a, cand_b, epsilon_map, dataset_index)
            rows.append(
                {
                    "dataset_index": dataset_index,
                    "pair_1": cand_a.method_variant,
                    "pair_2": cand_b.method_variant,
                    "label": label,
                }
            )
        return rows

    def _compute_epsilon(self, candidates: Sequence[CandidateScores]) -> Dict[str, float]:
        epsilons: Dict[str, float] = {}
        for metric in self.priority_levels:
            values = [
                value
                for candidate in candidates
                if (value := candidate.values.get(metric)) is not None
            ]
            if self.epsilon_mode == "fixed":
                epsilons[metric] = self.fixed_epsilon
            else:
                epsilons[metric] = (
                    self.relative_factor * (max(values) - min(values))
                    if len(values) >= 2
                    else 0.0
                )
        return epsilons

    def _label_pair(
        self,
        cand_a: CandidateScores,
        cand_b: CandidateScores,
        epsilon_map: Mapping[str, float],
        dataset_index: object,
    ) -> int:
        for metric in self.priority_levels:
            value_a = cand_a.values.get(metric)
            value_b = cand_b.values.get(metric)
            if value_a is None or value_b is None:
                continue
            epsilon = epsilon_map.get(metric, 0.0)
            if abs(value_a - value_b) > epsilon:
                return 0 if value_a > value_b else 1
        return self._tie_break(dataset_index, cand_a.method_variant, cand_b.method_variant)

    def _tie_break(self, dataset_index: object, pair_1: str, pair_2: str) -> int:
        token = f"{self.tie_breaker_seed}:{dataset_index}:{pair_1}:{pair_2}".encode("utf-8")
        digest = hashlib.sha256(token).digest()
        return digest[0] % 2


def _safe_str(value: object) -> str:
    return value if isinstance(value, str) else ""


def _resolve_priority_levels(persona: str, override: str | None) -> List[str]:
    if override:
        tokens = [token.strip() for token in override.split(",")]
        priority = [token for token in tokens if token]
    else:
        if persona not in PERSONA_PRIORITIES:
            raise ValueError(f"Unknown persona '{persona}'.")
        priority = list(PERSONA_PRIORITIES[persona])
    if not priority:
        raise ValueError("Priority order cannot be empty.")
    unknown = [metric for metric in priority if metric not in METRIC_FETCHERS]
    if unknown:
        raise ValueError(
            f"Unknown metric keys in priority order: {', '.join(unknown)}. "
            f"Valid options: {AVAILABLE_METRICS_TEXT}."
        )
    return priority


def _load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    main()
