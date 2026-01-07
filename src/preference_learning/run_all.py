"""Utility script to run preference-learning experiments for every encoded file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

from .data import FEATURE_GROUPS
from .config import ExperimentConfig
from .models import LinearSVCConfig
from .pipeline import run_persona_linear_svc_simulation

DEFAULT_RESULTS_ROOT = Path("results") / "full_run_dec8"
DEFAULT_ENCODED_DIR = DEFAULT_RESULTS_ROOT / "encoded_pareto_fronts" / "features_full_lm_stats"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_ROOT / "preference_learning"
DEFAULT_TAU_RESULTS_DIR = Path("preference-learning-simulation") / "basic-features" / "tau-tunning-results"
DEFAULT_CONCENTRATION_RESULTS_DIR = (
    Path("preference-learning-simulation") / "basic-features" / "concentration-c-sweep-results"
)
DEFAULT_NUM_USERS_RESULTS_DIR = (
    Path("preference-learning-simulation") / "basic-features" / "num-users-sweep-results"
)
DEFAULT_PERSONA_CONFIG_DIR = Path("src") / "preference_learning" / "configs"
PERSONA_CONFIGS = {
    "layperson": DEFAULT_PERSONA_CONFIG_DIR / "lay.yaml",
    "regulator": DEFAULT_PERSONA_CONFIG_DIR / "regulator.yaml",
    "clinician": DEFAULT_PERSONA_CONFIG_DIR / "clinician.yaml",
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
        "--exclude-feature-groups",
        nargs="+",
        choices=tuple(sorted(FEATURE_GROUPS)),
        default=(),
        help="Feature groups to drop from model features (default: none).",
    )
    parser.add_argument(
        "--autoxai-include-all-metrics",
        action="store_true",
        help=(
            "When set, the pipeline's AutoXAI baseline scoring uses all available Pareto-front metrics "
            "(in addition to robustness/correctness/compactness terms)."
        ),
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
        "--num-users-values",
        nargs="+",
        default=None,
        help=(
            "Optional space-separated list of num_users values to sweep. When set, runs every experiment for each "
            "value and writes aggregated Spearman correlations to --num-users-results-dir."
        ),
    )
    parser.add_argument(
        "--num-users-results-dir",
        type=Path,
        default=DEFAULT_NUM_USERS_RESULTS_DIR,
        help=f"Output directory for num-users sweep summaries (default: {DEFAULT_NUM_USERS_RESULTS_DIR}).",
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
        "--tau",
        type=float,
        default=None,
        help="Optional override for preference-sampling temperature tau (default: persona config).",
    )
    parser.add_argument(
        "--concentration-c",
        type=float,
        default=None,
        help="Optional override for preference-model concentration c (default: preference_model.yml fixed value).",
    )
    parser.add_argument(
        "--concentration-c-values",
        nargs="+",
        default=None,
        help=(
            "Optional space-separated list of concentration c values to sweep. When set, runs every experiment for "
            "each value and writes aggregated Spearman/precision summaries to --concentration-c-results-dir."
        ),
    )
    parser.add_argument(
        "--concentration-c-results-dir",
        type=Path,
        default=DEFAULT_CONCENTRATION_RESULTS_DIR,
        help=(
            "Output directory for concentration-c sweep summaries "
            f"(default: {DEFAULT_CONCENTRATION_RESULTS_DIR})."
        ),
    )
    parser.add_argument(
        "--tau-values",
        nargs="+",
        default=None,
        help=(
            "Optional space-separated list of tau values to sweep. When set, runs every experiment for each tau and "
            "writes aggregated Spearman correlations to --tau-results-dir."
        ),
    )
    parser.add_argument(
        "--tau-results-dir",
        type=Path,
        default=DEFAULT_TAU_RESULTS_DIR,
        help=f"Output directory for tau sweep summaries (default: {DEFAULT_TAU_RESULTS_DIR}).",
    )
    parser.add_argument(
        "--svc-C",
        type=float,
        default=1.0,
        help="LinearSVC regularization strength (default: 1.0).",
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


def _parse_tau_values(values: Iterable[str] | None) -> Sequence[float]:
    if not values:
        return ()
    parsed: list[float] = []
    for value in values:
        try:
            parsed_value = float(value)
        except ValueError as exc:  # pragma: no cover - argument parsing
            raise argparse.ArgumentTypeError(f"Invalid float for --tau-values: {value}") from exc
        if parsed_value <= 0:
            raise argparse.ArgumentTypeError("--tau-values must all be > 0.")
        parsed.append(parsed_value)
    # De-duplicate while preserving order
    seen: set[float] = set()
    ordered: list[float] = []
    for tau in parsed:
        if tau in seen:
            continue
        ordered.append(tau)
        seen.add(tau)
    return tuple(ordered)


def _parse_num_users_values(values: Iterable[str] | None) -> Sequence[int]:
    if not values:
        return ()
    parsed: list[int] = []
    for value in values:
        try:
            parsed_value = int(value)
        except ValueError as exc:  # pragma: no cover - argument parsing
            raise argparse.ArgumentTypeError(f"Invalid integer for --num-users-values: {value}") from exc
        if parsed_value < 1:
            raise argparse.ArgumentTypeError("--num-users-values must all be >= 1.")
        parsed.append(parsed_value)
    # De-duplicate while preserving order
    seen: set[int] = set()
    ordered: list[int] = []
    for num_users in parsed:
        if num_users in seen:
            continue
        ordered.append(num_users)
        seen.add(num_users)
    return tuple(ordered)


def _parse_concentration_values(values: Iterable[str] | None) -> Sequence[float]:
    if not values:
        return ()
    parsed: list[float] = []
    for value in values:
        try:
            parsed_value = float(value)
        except ValueError as exc:  # pragma: no cover - argument parsing
            raise argparse.ArgumentTypeError(
                f"Invalid float for --concentration-c-values: {value}"
            ) from exc
        if parsed_value <= 0:
            raise argparse.ArgumentTypeError("--concentration-c-values must all be > 0.")
        parsed.append(parsed_value)
    # De-duplicate while preserving order
    seen: set[float] = set()
    ordered: list[float] = []
    for c in parsed:
        if c in seen:
            continue
        ordered.append(c)
        seen.add(c)
    return tuple(ordered)


def _extract_aggregate_metric(
    simulation_result: dict,
    *,
    k_values: Sequence[int],
    metric_key: str,
    source_key: str = "aggregate_top_k_mean",
) -> list[float]:
    top_k_mean = simulation_result.get(source_key) or {}
    if not isinstance(top_k_mean, dict):
        return []
    keys = metric_key.split(".")
    values: list[float] = []
    for k in k_values:
        metrics = top_k_mean.get(str(k)) or {}
        if not isinstance(metrics, dict):
            continue
        value: object | None
        if metric_key in metrics:
            value = metrics.get(metric_key)
        else:
            value = metrics
            for part in keys:
                if not isinstance(value, dict):
                    value = None
                    break
                value = value.get(part)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _extract_aggregate_spearman(
    simulation_result: dict,
    *,
    k_values: Sequence[int],
) -> list[float]:
    return _extract_aggregate_metric(
        simulation_result,
        k_values=k_values,
        metric_key="rank_correlation.spearman",
        source_key="aggregate_top_k_mean",
    )


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    encoded_dir = args.encoded_dir
    if not encoded_dir.exists():
        raise FileNotFoundError(f"Encoded directory does not exist: {encoded_dir}")
    encoded_files = sorted(encoded_dir.glob("*_encoded.parquet"))
    if not encoded_files:
        raise FileNotFoundError(f"No encoded parquet files found under {encoded_dir}")
    top_k = parse_top_k(args.top_k)
    tau_values = _parse_tau_values(args.tau_values)
    num_users_values = _parse_num_users_values(args.num_users_values)
    concentration_c_values = _parse_concentration_values(args.concentration_c_values)
    sweep_count = sum(bool(values) for values in (tau_values, num_users_values, concentration_c_values))
    if sweep_count > 1:
        raise argparse.ArgumentTypeError(
            "Set at most one of --tau-values, --num-users-values, or --concentration-c-values."
        )

    experiment_config = ExperimentConfig(
        test_size=args.test_size,
        random_state=args.random_state,
        top_k=tuple(top_k),
        num_users=int(args.num_users),
        persona_seed=int(args.persona_seed),
        label_seed=int(args.label_seed),
        tau=float(args.tau) if args.tau is not None else None,
        concentration_c=float(args.concentration_c) if args.concentration_c is not None else None,
        exclude_feature_groups=tuple(args.exclude_feature_groups or ()),
        autoxai_include_all_metrics=bool(args.autoxai_include_all_metrics),
    )
    model_config = LinearSVCConfig(
        C=args.svc_C,
        max_iter=args.svc_max_iter,
        random_state=args.random_state,
        tune=bool(args.tune_svc),
    )

    if tau_values:
        results_per_tau: list[dict] = []
        for tau in tau_values:
            sweep_config = ExperimentConfig(
                test_size=experiment_config.test_size,
                random_state=experiment_config.random_state,
                top_k=experiment_config.top_k,
                num_users=experiment_config.num_users,
                persona_seed=experiment_config.persona_seed,
                label_seed=experiment_config.label_seed,
                tau=float(tau),
                concentration_c=experiment_config.concentration_c,
                exclude_feature_groups=experiment_config.exclude_feature_groups,
                autoxai_include_all_metrics=experiment_config.autoxai_include_all_metrics,
            )
            spearman_values: list[float] = []
            for persona in args.personas:
                persona_config_path = PERSONA_CONFIGS[persona]
                for encoded_path in encoded_files:
                    print(f"Running tau={tau} {persona} experiment for {encoded_path.name}")
                    result = run_persona_linear_svc_simulation(
                        encoded_path=encoded_path,
                        persona_config_path=persona_config_path,
                        output_dir=None,
                        experiment_config=sweep_config,
                        model_config=model_config,
                    )
                    spearman_values.extend(_extract_aggregate_spearman(result, k_values=sweep_config.top_k))

            results_per_tau.append(
                {
                    "tau": float(tau),
                    "aggregate_spearman": _mean(spearman_values),
                    "n_values": int(len(spearman_values)),
                }
            )

        best = max(results_per_tau, key=lambda row: row["aggregate_spearman"]) if results_per_tau else None
        output_dir = Path(args.tau_results_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "encoded_dir": str(encoded_dir),
            "personas": list(args.personas),
            "top_k": list(top_k),
            "num_users": int(experiment_config.num_users),
            "persona_seed": int(experiment_config.persona_seed),
            "label_seed": int(experiment_config.label_seed),
            "exclude_feature_groups": list(experiment_config.exclude_feature_groups),
            "svc_config": {
                "C": float(model_config.C),
                "max_iter": int(model_config.max_iter),
                "random_state": int(model_config.random_state),
                "tune": bool(model_config.tune),
            },
            "tau_values": [float(tau) for tau in tau_values],
            "results": results_per_tau,
            "best_tau": best["tau"] if best is not None else None,
            "best_aggregate_spearman": best["aggregate_spearman"] if best is not None else None,
            "aggregation": "Mean of aggregate_top_k_mean[k]['rank_correlation.spearman'] across all runs.",
        }
        (output_dir / "tau_tunning_summary.json").write_text(json.dumps(summary, indent=2))
        csv_lines = ["tau,aggregate_spearman,n_values"]
        for row in sorted(results_per_tau, key=lambda r: r["tau"]):
            csv_lines.append(f"{row['tau']},{row['aggregate_spearman']},{row['n_values']}")
        (output_dir / "tau_tunning_summary.csv").write_text("\n".join(csv_lines) + "\n")
        print(json.dumps(summary, indent=2))
        return

    if num_users_values:
        results_per_num_users: list[dict] = []
        for num_users in num_users_values:
            sweep_config = ExperimentConfig(
                test_size=experiment_config.test_size,
                random_state=experiment_config.random_state,
                top_k=experiment_config.top_k,
                num_users=int(num_users),
                persona_seed=experiment_config.persona_seed,
                label_seed=experiment_config.label_seed,
                tau=experiment_config.tau,
                concentration_c=experiment_config.concentration_c,
                exclude_feature_groups=experiment_config.exclude_feature_groups,
                autoxai_include_all_metrics=experiment_config.autoxai_include_all_metrics,
            )
            spearman_values: list[float] = []
            for persona in args.personas:
                persona_config_path = PERSONA_CONFIGS[persona]
                for encoded_path in encoded_files:
                    print(f"Running num_users={num_users} {persona} experiment for {encoded_path.name}")
                    result = run_persona_linear_svc_simulation(
                        encoded_path=encoded_path,
                        persona_config_path=persona_config_path,
                        output_dir=None,
                        experiment_config=sweep_config,
                        model_config=model_config,
                    )
                    spearman_values.extend(_extract_aggregate_spearman(result, k_values=sweep_config.top_k))

            results_per_num_users.append(
                {
                    "num_users": int(num_users),
                    "aggregate_spearman": _mean(spearman_values),
                    "n_values": int(len(spearman_values)),
                }
            )

        best = (
            max(results_per_num_users, key=lambda row: row["aggregate_spearman"])
            if results_per_num_users
            else None
        )
        output_dir = Path(args.num_users_results_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "encoded_dir": str(encoded_dir),
            "personas": list(args.personas),
            "top_k": list(top_k),
            "persona_seed": int(experiment_config.persona_seed),
            "label_seed": int(experiment_config.label_seed),
            "tau": float(experiment_config.tau) if experiment_config.tau is not None else None,
            "exclude_feature_groups": list(experiment_config.exclude_feature_groups),
            "svc_config": {
                "C": float(model_config.C),
                "max_iter": int(model_config.max_iter),
                "random_state": int(model_config.random_state),
                "tune": bool(model_config.tune),
            },
            "num_users_values": [int(v) for v in num_users_values],
            "results": results_per_num_users,
            "best_num_users": best["num_users"] if best is not None else None,
            "best_aggregate_spearman": best["aggregate_spearman"] if best is not None else None,
            "aggregation": "Mean of aggregate_top_k_mean[k]['rank_correlation.spearman'] across all runs.",
        }
        (output_dir / "num_users_sweep_summary.json").write_text(json.dumps(summary, indent=2))
        csv_lines = ["num_users,aggregate_spearman,n_values"]
        for row in sorted(results_per_num_users, key=lambda r: r["num_users"]):
            csv_lines.append(f"{row['num_users']},{row['aggregate_spearman']},{row['n_values']}")
        (output_dir / "num_users_sweep_summary.csv").write_text("\n".join(csv_lines) + "\n")
        print(json.dumps(summary, indent=2))
        return

    if concentration_c_values:
        results_per_c: list[dict] = []
        for concentration_c in concentration_c_values:
            sweep_config = ExperimentConfig(
                test_size=experiment_config.test_size,
                random_state=experiment_config.random_state,
                top_k=experiment_config.top_k,
                num_users=experiment_config.num_users,
                persona_seed=experiment_config.persona_seed,
                label_seed=experiment_config.label_seed,
                tau=experiment_config.tau,
                concentration_c=float(concentration_c),
                exclude_feature_groups=experiment_config.exclude_feature_groups,
                autoxai_include_all_metrics=experiment_config.autoxai_include_all_metrics,
            )
            svc_spearman_values: list[float] = []
            svc_precision_values: list[float] = []
            autoxai_spearman_values: list[float] = []
            autoxai_precision_values: list[float] = []
            for persona in args.personas:
                persona_config_path = PERSONA_CONFIGS[persona]
                for encoded_path in encoded_files:
                    print(
                        f"Running concentration_c={concentration_c} {persona} experiment for {encoded_path.name}"
                    )
                    result = run_persona_linear_svc_simulation(
                        encoded_path=encoded_path,
                        persona_config_path=persona_config_path,
                        output_dir=None,
                        experiment_config=sweep_config,
                        model_config=model_config,
                    )
                    svc_spearman_values.extend(
                        _extract_aggregate_metric(
                            result,
                            k_values=sweep_config.top_k,
                            metric_key="rank_correlation.spearman",
                        )
                    )
                    svc_precision_values.extend(
                        _extract_aggregate_metric(
                            result,
                            k_values=sweep_config.top_k,
                            metric_key="precision",
                        )
                    )
                    autoxai_spearman_values.extend(
                        _extract_aggregate_metric(
                            result,
                            k_values=sweep_config.top_k,
                            metric_key="rank_correlation.spearman",
                            source_key="aggregate_autoxai_top_k_mean",
                        )
                    )
                    autoxai_precision_values.extend(
                        _extract_aggregate_metric(
                            result,
                            k_values=sweep_config.top_k,
                            metric_key="precision",
                            source_key="aggregate_autoxai_top_k_mean",
                        )
                    )

            results_per_c.append(
                {
                    "concentration_c": float(concentration_c),
                    "svc_aggregate_spearman": _mean(svc_spearman_values),
                    "svc_aggregate_precision": _mean(svc_precision_values),
                    "autoxai_aggregate_spearman": _mean(autoxai_spearman_values),
                    "autoxai_aggregate_precision": _mean(autoxai_precision_values),
                    "svc_n_values": int(len(svc_spearman_values)),
                    "autoxai_n_values": int(len(autoxai_spearman_values)),
                }
            )

        best_svc = (
            max(results_per_c, key=lambda row: row["svc_aggregate_spearman"]) if results_per_c else None
        )
        best_autoxai = (
            max(results_per_c, key=lambda row: row["autoxai_aggregate_spearman"])
            if results_per_c
            else None
        )
        output_dir = Path(args.concentration_c_results_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "encoded_dir": str(encoded_dir),
            "personas": list(args.personas),
            "top_k": list(top_k),
            "num_users": int(experiment_config.num_users),
            "persona_seed": int(experiment_config.persona_seed),
            "label_seed": int(experiment_config.label_seed),
            "tau": float(experiment_config.tau) if experiment_config.tau is not None else None,
            "exclude_feature_groups": list(experiment_config.exclude_feature_groups),
            "svc_config": {
                "C": float(model_config.C),
                "max_iter": int(model_config.max_iter),
                "random_state": int(model_config.random_state),
                "tune": bool(model_config.tune),
            },
            "concentration_c_values": [float(c) for c in concentration_c_values],
            "results": results_per_c,
            "best_concentration_c_svc": best_svc["concentration_c"] if best_svc is not None else None,
            "best_svc_aggregate_spearman": (
                best_svc["svc_aggregate_spearman"] if best_svc is not None else None
            ),
            "best_concentration_c_autoxai": (
                best_autoxai["concentration_c"] if best_autoxai is not None else None
            ),
            "best_autoxai_aggregate_spearman": (
                best_autoxai["autoxai_aggregate_spearman"] if best_autoxai is not None else None
            ),
            "aggregation": (
                "Mean of aggregate_top_k_mean[k]['rank_correlation.spearman'/'precision'] "
                "and aggregate_autoxai_top_k_mean equivalents across all runs and k values."
            ),
        }
        (output_dir / "concentration_c_sweep_summary.json").write_text(json.dumps(summary, indent=2))
        csv_lines = [
            "concentration_c,svc_aggregate_spearman,svc_aggregate_precision,"
            "autoxai_aggregate_spearman,autoxai_aggregate_precision,svc_n_values,autoxai_n_values"
        ]
        for row in sorted(results_per_c, key=lambda r: r["concentration_c"]):
            csv_lines.append(
                f"{row['concentration_c']},{row['svc_aggregate_spearman']},{row['svc_aggregate_precision']},"
                f"{row['autoxai_aggregate_spearman']},{row['autoxai_aggregate_precision']},"
                f"{row['svc_n_values']},{row['autoxai_n_values']}"
            )
        (output_dir / "concentration_c_sweep_summary.csv").write_text("\n".join(csv_lines) + "\n")
        print(json.dumps(summary, indent=2))
        return

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
