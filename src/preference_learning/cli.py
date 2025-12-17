"""CLI entry point for running preference-learning experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .config import ExperimentConfig
from .models import LinearSVCConfig
from .pipeline import DEFAULT_PROCESSED_DIR, DEFAULT_RESULTS_ROOT, run_linear_svc_experiment


def parse_top_k(values: Sequence[str] | None) -> Sequence[int]:
    if not values:
        return (1, 3, 5)
    parsed = []
    for value in values:
        try:
            parsed.append(int(value))
        except ValueError as exc:  # pragma: no cover - argparse will surface error
            raise argparse.ArgumentTypeError(f"Invalid integer for --top-k: {value}") from exc
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run preference-learning experiments on encoded Pareto features.",
    )
    parser.add_argument(
        "--encoded-path",
        type=Path,
        required=True,
        help="Path to the encoded Pareto-front parquet file.",
    )
    parser.add_argument(
        "--persona",
        choices=("layperson", "regulator"),
        default="layperson",
        help="Persona whose pair labels should be used (default: layperson).",
    )
    parser.add_argument(
        "--pair-labels-dir",
        type=Path,
        help="Optional explicit directory containing *_pair_labels.parquet files. "
        "Defaults to results/full_run_dec8/candidate_pair_rankings_<persona>.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help=f"Directory where processed data/predictions will be stored (default: {DEFAULT_PROCESSED_DIR}).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of instances to use for the test split (default: 0.2).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for splitting and model training (default: 42).",
    )
    parser.add_argument(
        "--top-k",
        nargs="+",
        help="Space-separated list of k values for evaluation (default: 1 3 5).",
    )
    parser.add_argument(
        "--svc-C",
        type=float,
        default=1.0,
        help="Regularization strength for LinearSVC (default: 1.0).",
    )
    parser.add_argument(
        "--svc-max-iter",
        type=int,
        default=5000,
        help="Maximum optimization iterations for LinearSVC (default: 5000).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    encoded_path = args.encoded_path
    if not encoded_path.exists():
        raise FileNotFoundError(f"Encoded path does not exist: {encoded_path}")
    pair_labels_dir = args.pair_labels_dir
    if pair_labels_dir is None:
        pair_labels_dir = DEFAULT_RESULTS_ROOT / f"candidate_pair_rankings_{args.persona}"
    if not pair_labels_dir.exists():
        raise FileNotFoundError(f"Pair labels directory does not exist: {pair_labels_dir}")
    top_k = parse_top_k(args.top_k)
    experiment_config = ExperimentConfig(
        test_size=args.test_size,
        random_state=args.random_state,
        top_k=tuple(top_k),
    )
    model_config = LinearSVCConfig(
        C=args.svc_C,
        max_iter=args.svc_max_iter,
        random_state=args.random_state,
    )
    metrics = run_linear_svc_experiment(
        encoded_path=encoded_path,
        pair_labels_dir=pair_labels_dir,
        persona=args.persona,
        output_dir=args.output_dir,
        experiment_config=experiment_config,
        model_config=model_config,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI hook
    main()
