from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from .autoxai_objectives import parse_objective_terms, persona_objective_terms
from .autoxai_runner import run
from .autoxai_types import CandidateScore, HPOResult, HPOTrial, ObjectiveTerm
from .autoxai_utils import parse_top_k

__all__ = [
    "CandidateScore",
    "HPOResult",
    "HPOTrial",
    "ObjectiveTerm",
    "build_arg_parser",
    "main",
    "parse_objective_terms",
    "persona_objective_terms",
    "run",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AutoXAI-style baseline ranking over cached HC-XAI metric artifacts.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="HC-XAI run directory (contains metrics_results/).",
    )
    parser.add_argument("--dataset", required=True, help="Dataset key (e.g. open_compas).")
    parser.add_argument("--model", required=True, help="Model key (e.g. mlp_classifier).")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["lime", "shap"],
        help="Which methods to include (default: lime shap).",
    )
    parser.add_argument(
        "--objective",
        nargs="*",
        default=None,
        help="Objective terms: name:direction:metric_key[:weight]. If omitted, uses a robust default.",
    )
    parser.add_argument(
        "--persona",
        choices=["autoxai", "layperson", "regulator"],
        default="autoxai",
        help="Select a preset objective aligned with an HC-XAI persona (default: autoxai).",
    )
    parser.add_argument(
        "--scaling",
        choices=["Std", "MinMax"],
        default="Std",
        help="Scaling used before scalarization (default: Std).",
    )
    parser.add_argument(
        "--scaling-scope",
        choices=["trial", "global", "instance"],
        default="trial",
        help=(
            "Scaling population for objective terms (default: trial). "
            "trial matches AutoXAI by scaling over per-variant means; "
            "global scales over all (instance,variant) candidates; "
            "instance scales within each instance across variants."
        ),
    )
    parser.add_argument(
        "--grid-config",
        type=Path,
        default=Path("src/configs/explainer_hyperparameters.yml"),
        help="Filter to variants defined in this hyperparameter grid (default: src/configs/explainer_hyperparameters.yml).",
    )
    parser.add_argument(
        "--explainer-config",
        type=Path,
        default=Path("src/configs/explainers.yml"),
        help="Explainer defaults file used to report the default variant score (default: src/configs/explainers.yml).",
    )
    parser.add_argument(
        "--hpo",
        choices=["grid", "random", "gp"],
        default="grid",
        help="Hyperparameter optimization mode over cached variants (default: grid).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of HPO trials when --hpo is random/gp (default: 20).",
    )
    parser.add_argument(
        "--hpo-seed",
        type=int,
        default=0,
        help="Random seed for HPO (default: 0).",
    )
    parser.add_argument(
        "--pair-labels",
        type=Path,
        default=None,
        help="Optional parquet file with HC-XAI pair labels to score against.",
    )
    parser.add_argument(
        "--hc-xai-split-json",
        type=Path,
        default=None,
        help="Optional path to HC-XAI preference-learning split file (processed/splits.json). "
        "If provided, scoring is restricted to the requested split set.",
    )
    parser.add_argument(
        "--split-set",
        choices=["train", "test", "all"],
        default="test",
        help="Which HC-XAI split to score/evaluate when --hc-xai-split-json is provided (default: test).",
    )
    parser.add_argument(
        "--top-k",
        nargs="+",
        default=None,
        help="Space-separated list of k values for top-k evaluation (default: 3 5).",
    )
    parser.add_argument(
        "--tie-breaker-seed",
        type=int,
        default=13,
        help="Tie breaker seed used for pairwise comparisons (default: 13).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional path to write a JSON report. "
            "When omitted, writes to <results-root>/autoxai_baseline__<dataset>__<model>__<persona>.json."
        ),
    )
    parser.add_argument(
        "--require-write",
        action="store_true",
        help="Fail if the JSON report cannot be written (default: print JSON to stdout and continue).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    objective = parse_objective_terms(args.objective) if args.objective else persona_objective_terms(args.persona)
    grid_path = args.grid_config if args.grid_config and args.grid_config.exists() else None
    explainer_path = args.explainer_config if args.explainer_config and args.explainer_config.exists() else None
    top_k = parse_top_k(args.top_k)

    report = run(
        results_root=args.results_root,
        dataset=args.dataset,
        model=args.model,
        methods=args.methods,
        objective=objective,
        scaling=args.scaling,
        scaling_scope=args.scaling_scope,
        grid_config_path=grid_path,
        explainers_config_path=explainer_path,
        hpo_mode=args.hpo,
        hpo_epochs=args.epochs,
        hpo_seed=args.hpo_seed,
        pair_labels_path=args.pair_labels,
        tie_breaker_seed=args.tie_breaker_seed,
        hc_xai_split_json=args.hc_xai_split_json,
        split_set=args.split_set,
        top_k=top_k,
    )

    payload = json.dumps(report, indent=2, sort_keys=True)
    output_path = args.output
    if output_path is None:
        output_path = args.results_root / f"autoxai_baseline__{args.dataset}__{args.model}__{args.persona}.json"
    wrote_report = False
    write_error: Optional[Exception] = None
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
        wrote_report = True
    except Exception as exc:  # pragma: no cover
        write_error = exc
        if args.require_write:
            raise

    print(payload)
    if wrote_report:
        print(f"[autoxai-baseline] wrote report to {output_path}", file=sys.stderr)
    elif write_error is not None:
        print(
            f"[autoxai-baseline] WARNING: failed to write report to {output_path}: {write_error}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

