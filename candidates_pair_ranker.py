#!/usr/bin/env python3
"""
Generate pairwise ranking labels for Pareto-front explanation candidates.

For every Pareto JSON, the ranker enumerates all explanation pairs within each
instance and assigns a persona-specific label by sampling preferences from a
utility model:

  U(e) = sum_j w_j * z_j(e)
  P(e_i ≻ e_j) = sigmoid((U(e_i) - U(e_j)) / tau)

The weights w are sampled once per run from a hierarchical Dirichlet persona
configuration under `src/preference_learning/configs/`, then reused to label
every pair in the dataset.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

from src.preference_learning.persona import (
    HierarchicalDirichletUser,
    load_persona_config,
    z_normalize_matrix,
)

DEFAULT_RESULTS_ROOT = Path("results") / "full_run_dec8"
DEFAULT_PARETO_DIR = DEFAULT_RESULTS_ROOT / "pareto_fronts"

DEFAULT_PERSONA_CONFIG_DIR = Path("src") / "preference_learning" / "configs"
PERSONA_CONFIGS: Mapping[str, Path] = {
    "layperson": DEFAULT_PERSONA_CONFIG_DIR / "lay-person.json",
    "regulator": DEFAULT_PERSONA_CONFIG_DIR / "regulator.json",
    "clinician": DEFAULT_PERSONA_CONFIG_DIR / "clinician.json",
}

PERSONA_CHOICES = tuple(sorted(PERSONA_CONFIGS))


@dataclass
class CandidateScores:
    """Compact container for a method variant and its z-normalised metrics."""

    method_variant: str
    z_values: np.ndarray


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
        help=(
            "Directory where pair label Parquet files will be stored "
            f"(default: {DEFAULT_RESULTS_ROOT}/candidate_pair_rankings_<persona>)."
        ),
    )
    parser.add_argument(
        "--persona",
        choices=PERSONA_CHOICES,
        default="layperson",
        help="Named persona configuration to use (default: layperson).",
    )
    parser.add_argument(
        "--persona-config",
        type=Path,
        help="Optional path to a persona JSON file. Overrides --persona when provided.",
    )
    parser.add_argument(
        "--persona-seed",
        type=int,
        default=13,
        help="Seed used to sample the persona's Dirichlet weights (default: 13).",
    )
    parser.add_argument(
        "--label-seed",
        type=int,
        default=41,
        help="Seed used to sample pairwise preference labels (default: 41).",
    )
    parser.add_argument(
        "--tau",
        type=float,
        help="Temperature for the logistic preference model (overrides persona config).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir: Path = args.output_dir or (DEFAULT_RESULTS_ROOT / f"candidate_pair_rankings_{args.persona}")
    output_dir.mkdir(parents=True, exist_ok=True)
    pareto_files = sorted(args.pareto_dir.glob("*_pareto.json"))
    if not pareto_files:
        raise FileNotFoundError(f"No Pareto files found under {args.pareto_dir}")

    config_path = args.persona_config or PERSONA_CONFIGS[args.persona]
    persona_config = load_persona_config(config_path)
    user = HierarchicalDirichletUser(
        persona_config,
        seed=args.persona_seed,
        tau=args.tau,
        # Pareto-front metrics are already oriented as "higher is better" (min objectives are negated).
        metrics_already_oriented=True,
    )
    label_rng = np.random.default_rng(args.label_seed)

    ranker = CandidatePairRanker(
        user=user,
        label_rng=label_rng,
    )

    for path in pareto_files:
        df = ranker.rank_file(path)
        output_path = output_dir / f"{path.stem}_pair_labels.parquet"
        df.to_parquet(output_path, index=False)
        print(f"Wrote {len(df)} pair labels for {path.name} to {output_path}")


class CandidatePairRanker:
    """Ranks candidate explanations by emitting pairwise preference labels."""

    def __init__(
        self,
        *,
        user: HierarchicalDirichletUser,
        label_rng: np.random.Generator | None = None,
    ) -> None:
        self.user = user
        self.label_rng = label_rng or np.random.default_rng()

    def rank_file(self, path: Path) -> pd.DataFrame:
        payload = _load_json(path)
        instances: Iterable[Mapping[str, object]] = payload.get("instances") or []
        rows: List[Dict[str, object]] = []
        for instance in instances:
            rows.extend(self._rank_instance(instance))
        return pd.DataFrame(rows, columns=("dataset_index", "pair_1", "pair_2", "label"))

    def _rank_instance(self, instance: Mapping[str, object]) -> List[Dict[str, object]]:
        pareto_entries = instance.get("pareto_front") or []
        metric_vectors: List[np.ndarray] = []
        method_variants: List[str] = []
        for entry in pareto_entries:
            variant = _safe_str(entry.get("method_variant"))
            metrics = entry.get("metrics") or {}
            if not variant or not isinstance(metrics, Mapping):
                continue
            method_variants.append(variant)
            metric_vectors.append(self.user.vectorize_metrics(metrics))

        if len(method_variants) < 2:
            return []

        dataset_index = instance.get("dataset_index")
        z_matrix = z_normalize_matrix(np.vstack(metric_vectors))
        candidates = [
            CandidateScores(method_variant=variant, z_values=z_values)
            for variant, z_values in zip(method_variants, z_matrix)
        ]
        rows: List[Dict[str, object]] = []
        for cand_a, cand_b in combinations(candidates, 2):
            label = self._label_pair(cand_a, cand_b)
            rows.append(
                {
                    "dataset_index": dataset_index,
                    "pair_1": cand_a.method_variant,
                    "pair_2": cand_b.method_variant,
                    "label": label,
                }
            )
        return rows

    def _label_pair(
        self,
        cand_a: CandidateScores,
        cand_b: CandidateScores,
    ) -> int:
        p = self.user.preference_probability(cand_a.z_values, cand_b.z_values)
        return 0 if self.label_rng.random() < p else 1


def _safe_str(value: object) -> str:
    return value if isinstance(value, str) else ""


def _load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    main()
