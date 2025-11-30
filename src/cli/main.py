"""Command-line interface for running configured experiments."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, List, Sequence
import sys
from src.orchestrators.metrics_runner import run_experiment, run_experiments


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not a valid integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run HC-XAI experiments without launching the notebook UI.",
    )
    parser.add_argument(
        "experiments",
        nargs="+",
        help="One or more experiment names defined in src/configs/experiments.yml.",
    )
    parser.add_argument(
        "--max-instances",
        type=_positive_int,
        default=None,
        help="Optional cap on evaluation instances passed to the orchestrator.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where JSON results should be written (one file per experiment).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model override when running a single experiment.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level for orchestrator progress output (default: INFO).",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print a compact JSON summary of experiment descriptors to stdout.",
    )
    parser.add_argument(
        "--tune-models",
        action="store_true",
        help="Run hyperparameter tuning for every dataset/model pair before training.",
    )
    parser.add_argument(
        "--use-tuned-params",
        action="store_true",
        help="Reuse tuned hyperparameters from saved artifacts (if present).",
    )
    parser.add_argument(
        "--reuse-trained-models",
        action="store_true",
        help="Load persisted trained models when available and save new ones after training.",
    )
    parser.add_argument(
        "--tuning-output-dir",
        type=Path,
        default=None,
        help="Directory for hyperparameter tuning artifacts (defaults to saved_models/tuning_results).",
    )
    parser.add_argument(
        "--model-store-dir",
        type=Path,
        default=None,
        help="Directory for serialized trained models (defaults to saved_models).",
    )
    parser.add_argument(
        "--stop-after-training",
        action="store_true",
        help="Stop after hyperparameter tuning/model training (skip explanations + metrics).",
    )
    parser.add_argument(
        "--stop-after-explanations",
        action="store_true",
        help="Run explainers but skip all metric evaluations.",
    )
    return parser


def _configure_logging(level: str) -> None:
    try:
        numeric_level = getattr(logging, level.upper())
    except AttributeError:
        raise ValueError(f"Unknown log level: {level}")
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _ensure_output_dir(path: Path | None) -> Path | None:
    if path is None:
        return None
    path.mkdir(parents=True, exist_ok=True)
    return path


def _run_with_model_override(
    experiments: Sequence[str],
    *,
    model_name: str,
    max_instances: int | None,
    output_dir: Path | None,
    tune_models: bool,
    use_tuned_params: bool,
    reuse_trained_models: bool,
    tuning_output_dir: Path | None,
    model_store_dir: Path | None,
    stop_after_training: bool,
    stop_after_explanations: bool,
) -> List[dict]:
    results: List[dict] = []
    for name in experiments:
        file_path = None
        if output_dir is not None:
            suffix = f"{name}__{model_name}"
            file_path = output_dir / f"{suffix}.json"
        results.append(
            run_experiment(
                name,
                max_instances=max_instances,
                output_path=file_path,
                model_override=model_name,
                tune_models=tune_models,
                use_tuned_params=use_tuned_params,
                reuse_trained_models=reuse_trained_models,
                tuning_output_dir=tuning_output_dir,
                model_store_dir=model_store_dir,
                stop_after_training=stop_after_training,
                stop_after_explanations=stop_after_explanations,
            )
        )
    return results


def _summarize(results: Iterable[dict]) -> List[dict]:
    summary: List[dict] = []
    for exp in results:
        summary.append(
            {
                "experiment": exp.get("experiment"),
                "dataset": exp.get("dataset"),
                "model": exp.get("model"),
                "instances": len(exp.get("instances", [])),
            }
        )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.model and len(args.experiments) > 1:
        parser.error("--model can only be used when running a single experiment at a time")
    if args.stop_after_training and args.stop_after_explanations:
        parser.error("--stop-after-training and --stop-after-explanations are mutually exclusive")
    # TODO log the list command line arguments 
    
    _configure_logging(args.log_level)
    logger = logging.getLogger(__name__)
    cli_args = list(argv) if argv is not None else sys.argv[1:]
    logger.debug("CLI arguments: %s", cli_args)
    output_dir = _ensure_output_dir(args.output_dir)
    logger.info(
        "Requested experiments=%s max_instances=%s model_override=%s tune=%s use_tuned=%s reuse_models=%s stop_after_training=%s stop_after_explanations=%s",
        args.experiments,
        args.max_instances,
        args.model,
        args.tune_models,
        args.use_tuned_params,
        args.reuse_trained_models,
        args.stop_after_training,
        args.stop_after_explanations,
    )

    if args.model:
        results = _run_with_model_override(
            args.experiments,
            model_name=args.model,
            max_instances=args.max_instances,
            output_dir=output_dir,
            tune_models=args.tune_models,
            use_tuned_params=args.use_tuned_params,
            reuse_trained_models=args.reuse_trained_models,
            tuning_output_dir=args.tuning_output_dir,
            model_store_dir=args.model_store_dir,
            stop_after_training=args.stop_after_training,
            stop_after_explanations=args.stop_after_explanations,
        )
    else:
        results = run_experiments(
            args.experiments,
            max_instances=args.max_instances,
            output_dir=output_dir,
            tune_models=args.tune_models,
            use_tuned_params=args.use_tuned_params,
            reuse_trained_models=args.reuse_trained_models,
            tuning_output_dir=args.tuning_output_dir,
            model_store_dir=args.model_store_dir,
            stop_after_training=args.stop_after_training,
            stop_after_explanations=args.stop_after_explanations,
        )

    logger.info(
        "Completed %d experiment run(s). Results directory: %s",
        len(results),
        output_dir if output_dir else "<not written>",
    )

    if args.print_summary:
        print(json.dumps(_summarize(results), indent=2))

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
