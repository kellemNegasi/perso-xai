"""Utility script to run preference-learning experiments for every encoded file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

from .config import ExperimentConfig
from .models import LinearSVCConfig
from .pipeline import run_persona_linear_svc_simulation

DEFAULT_RESULTS_ROOT = Path("results") / "full_run_dec8"
DEFAULT_ENCODED_DIR = DEFAULT_RESULTS_ROOT / "encoded_pareto_fronts" / "features_full_lm_stats"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_ROOT / "preference_learning"
DEFAULT_PERSONA_CONFIG_DIR = Path("src") / "preference_learning" / "configs"
PERSONA_CONFIGS = {
    "layperson": DEFAULT_PERSONA_CONFIG_DIR / "lay-person.json",
    "regulator": DEFAULT_PERSONA_CONFIG_DIR / "regulator.json",
    "clinician": DEFAULT_PERSONA_CONFIG_DIR / "clinician.json",
}
PERSONAS = tuple(sorted(PERSONA_CONFIGS))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run preference-learning experiments for every encoded Pareto file.",
    )
    parser.add_argument(
        "--encoded-dir",
        type=Path,
        default=DEFAULT_ENCODED_DIR,
        help=f"Directory containing encoded Pareto parquet files (default: {DEFAULT_ENCODED_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory where experiment artifacts will be stored (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--personas",
        nargs="+",
        choices=PERSONAS,
        default=list(PERSONAS),
        help="Personas to evaluate (default: layperson regulator).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of instances used for test split (default: 0.2).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for splitting and model training (default: 42).",
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
        help="Number of sampled users per persona (default: 10).",
    )
    parser.add_argument(
        "--persona-seed",
        type=int,
        default=13,
        help="Base seed for Dirichlet persona sampling (default: 13).",
    )
    parser.add_argument(
        "--label-seed",
        type=int,
        default=41,
        help="Base seed for sampling pairwise labels (default: 41).",
    )
    parser.add_argument(
        "--svc-C",
        type=float,
        default=1.0,
        help="LinearSVC regularization strength (default: 1.0).",
    )
    parser.add_argument(
        "--svc-max-iter",
        type=int,
        default=5000,
        help="Maximum LinearSVC iterations (default: 5000).",
    )
    return parser.parse_args(argv)


def parse_top_k(values: Iterable[str] | None) -> Sequence[int]:
    if not values:
        return (3, 5, 8)
    parsed: list[int] = []
    for value in values:
        try:
            parsed.append(int(value))
        except ValueError as exc:  # pragma: no cover - argument parsing
            raise argparse.ArgumentTypeError(f"Invalid integer for --top-k: {value}") from exc
    return parsed


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    encoded_dir = args.encoded_dir
    if not encoded_dir.exists():
        raise FileNotFoundError(f"Encoded directory does not exist: {encoded_dir}")
    encoded_files = sorted(encoded_dir.glob("*_encoded.parquet"))
    if not encoded_files:
        raise FileNotFoundError(f"No encoded parquet files found under {encoded_dir}")
    top_k = parse_top_k(args.top_k)

    experiment_config = ExperimentConfig(
        test_size=args.test_size,
        random_state=args.random_state,
        top_k=tuple(top_k),
        num_users=int(args.num_users),
        persona_seed=int(args.persona_seed),
        label_seed=int(args.label_seed),
    )
    model_config = LinearSVCConfig(
        C=args.svc_C,
        max_iter=args.svc_max_iter,
        random_state=args.random_state,
    )

    for persona in args.personas:
        persona_config_path = PERSONA_CONFIGS[persona]
        for encoded_path in encoded_files:
            print(f"Running {persona} experiment for {encoded_path.name}")
            result = run_persona_linear_svc_simulation(
                encoded_path=encoded_path,
                persona_config_path=persona_config_path,
                output_dir=args.output_dir,
                experiment_config=experiment_config,
                model_config=model_config,
            )
            print(json.dumps(result, indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI hook
    main()
