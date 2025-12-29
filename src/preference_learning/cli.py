"""CLI entry point for running preference-learning experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .data import FEATURE_GROUPS
from .config import ExperimentConfig
from .models import LinearSVCConfig
from .pipeline import (
    DEFAULT_PROCESSED_DIR,
    run_linear_svc_experiment,
    run_persona_linear_svc_simulation,
)

DEFAULT_PERSONA_CONFIG_DIR = Path("src") / "preference_learning" / "configs"
PERSONA_CONFIGS = {
    "layperson": DEFAULT_PERSONA_CONFIG_DIR / "lay-person.json",
    "regulator": DEFAULT_PERSONA_CONFIG_DIR / "regulator.json",
    "clinician": DEFAULT_PERSONA_CONFIG_DIR / "clinician.json",
}


def parse_top_k(values: Sequence[str] | None) -> Sequence[int]:
    if not values:
        return (3, 5, 8)
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
        choices=tuple(sorted(PERSONA_CONFIGS)),
        default="layperson",
        help="Persona whose pair labels should be used (default: layperson).",
    )
    parser.add_argument(
        "--persona-config",
        type=Path,
        help="Optional path to a persona JSON file. When set, runs end-to-end simulation (no pair-label files).",
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
        default=None,
        help=(
            "Optional output directory. In simulation mode, writes a summary JSON when provided. "
            f"In legacy mode, defaults to {DEFAULT_PROCESSED_DIR}."
        ),
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
        "--exclude-feature-groups",
        nargs="+",
        choices=tuple(sorted(FEATURE_GROUPS)),
        default=(),
        help="Feature groups to drop from model features (default: none).",
    )
    parser.add_argument(
        "--top-k",
        nargs="+",
        help="Space-separated list of k values for evaluation (default: 1 3 5).",
    )
    parser.add_argument(
        "--num-users",
        type=int,
        default=10,
        help="Number of sampled users for simulation mode (default: 10).",
    )
    parser.add_argument(
        "--persona-seed",
        type=int,
        default=13,
        help="Base seed for Dirichlet persona sampling in simulation mode (default: 13).",
    )
    parser.add_argument(
        "--label-seed",
        type=int,
        default=41,
        help="Base seed for sampling pairwise labels in simulation mode (default: 41).",
    )
    parser.add_argument(
        "--svc-C",
        type=float,
        default=1.0,
        help="Regularization strength for LinearSVC (default: 1.0).",
    )
    parser.add_argument(
        "--tune-svc",
        action="store_true",
        help="Tune LinearSVC C via 5-fold cross-validation on the training set (default: off).",
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
    top_k = parse_top_k(args.top_k)
    experiment_config = ExperimentConfig(
        test_size=args.test_size,
        random_state=args.random_state,
        top_k=tuple(top_k),
        num_users=int(args.num_users),
        persona_seed=int(args.persona_seed),
        label_seed=int(args.label_seed),
        exclude_feature_groups=tuple(args.exclude_feature_groups or ()),
    )
    model_config = LinearSVCConfig(
        C=args.svc_C,
        max_iter=args.svc_max_iter,
        random_state=args.random_state,
        tune=bool(args.tune_svc),
    )
    if args.persona_config is not None or args.pair_labels_dir is None:
        persona_config_path = args.persona_config or PERSONA_CONFIGS[args.persona]
        result = run_persona_linear_svc_simulation(
            encoded_path=encoded_path,
            persona_config_path=persona_config_path,
            output_dir=args.output_dir,
            experiment_config=experiment_config,
            model_config=model_config,
        )
        print(json.dumps(result, indent=2))
        return

    pair_labels_dir = args.pair_labels_dir
    if not pair_labels_dir.exists():
        raise FileNotFoundError(f"Pair labels directory does not exist: {pair_labels_dir}")
    metrics = run_linear_svc_experiment(
        encoded_path=encoded_path,
        pair_labels_dir=pair_labels_dir,
        persona=args.persona,
        output_dir=args.output_dir or DEFAULT_PROCESSED_DIR,
        experiment_config=experiment_config,
        model_config=model_config,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI hook
    main()
