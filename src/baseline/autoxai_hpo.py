from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .autoxai_scoring import _trial_objective_value
from .autoxai_types import HPOResult, HPOTrial, ObjectiveTerm


def run_hpo(
    *,
    method: str,
    variant_means: Mapping[str, Tuple[str, float]],
    variants: Sequence[str],
    mode: str,
    epochs: int,
    seed: int,
    default_variant: Optional[str],
) -> HPOResult:
    """
    Hyperparameter selection over precomputed variants.

    mode:
      - "grid": evaluate all variants (deterministic; effectively exhaustive search)
      - "random": sample `epochs` variants uniformly at random (with replacement)
      - "gp": optional Bayesian optimization using scikit-optimize over a categorical space
    """
    if mode not in {"grid", "random", "gp"}:
        raise ValueError("hpo mode must be one of: grid, random, gp.")
    if epochs <= 0:
        raise ValueError("epochs must be positive.")

    method_variants = [v for v in variants if variant_means.get(v, ("", 0.0))[0] == method]
    if not method_variants:
        raise ValueError(f"No variants found for method {method!r}.")

    default_mean: Optional[float] = None
    if default_variant is not None:
        record = variant_means.get(default_variant)
        if record is not None and record[0] == method:
            default_mean = record[1]

    trials: List[HPOTrial] = []
    best_variant = method_variants[0]
    best_mean = variant_means.get(best_variant, (method, float("-inf")))[1]

    if mode == "grid":
        for variant in method_variants:
            mean_score = variant_means[variant][1]
            trials.append(HPOTrial(method=method, method_variant=variant, mean_score=mean_score))
            if mean_score > best_mean:
                best_variant = variant
                best_mean = mean_score

    elif mode == "random":
        import random

        rng = random.Random(seed)
        for _ in range(epochs):
            variant = rng.choice(method_variants)
            mean_score = variant_means[variant][1]
            trials.append(HPOTrial(method=method, method_variant=variant, mean_score=mean_score))
            if mean_score > best_mean:
                best_variant = variant
                best_mean = mean_score

    else:  # mode == "gp"
        try:
            from skopt import gp_minimize  # type: ignore
            from skopt.space import Categorical  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "hpo=gp requires scikit-optimize. Install it with `pip install scikit-optimize`."
            ) from exc

        def objective_fn(params: List[str]) -> float:
            variant = params[0]
            mean_score = variant_means[variant][1]
            trials.append(HPOTrial(method=method, method_variant=variant, mean_score=mean_score))
            return -mean_score

        space = [Categorical(method_variants, name="variant")]
        result = gp_minimize(
            objective_fn,
            space,
            n_calls=epochs,
            random_state=seed,
            n_initial_points=min(5, epochs),
        )
        best_variant = str(result.x[0])
        best_mean = variant_means[best_variant][1]

    return HPOResult(
        method=method,
        mode=mode,
        seed=seed,
        epochs=epochs,
        default_variant=default_variant,
        default_mean_score=default_mean,
        trials=trials,
        best_variant=best_variant,
        best_mean_score=best_mean,
    )


def schedule_autoxai_trials(
    *,
    method: str,
    method_variants: Sequence[str],
    mode: str,
    epochs: int,
    seed: int,
    default_variant: Optional[str],
    variant_term_means: Mapping[str, Mapping[str, float]],
    objective: Sequence[ObjectiveTerm],
    scaling: str,
    global_history: List[str],
) -> List[str]:
    """
    Create an AutoXAI-like trial schedule for one method while updating a shared global history.

    For random/gp, trials are selected sequentially and the objective value for a candidate is
    computed using scalers fit on the score history observed so far (global_history).
    """
    if mode not in {"grid", "random", "gp"}:
        raise ValueError("hpo mode must be one of: grid, random, gp.")
    if mode in {"random", "gp"} and epochs <= 0:
        raise ValueError("epochs must be positive.")

    available = [v for v in method_variants if v in variant_term_means]
    if not available:
        raise ValueError(f"No variants with term means found for method {method!r}.")

    trials: List[str] = []
    seen: set[str] = set()

    def add_trial(variant: str) -> None:
        trials.append(variant)
        global_history.append(variant)
        seen.add(variant)

    if default_variant is not None and default_variant in available:
        add_trial(default_variant)
        if mode in {"random", "gp"} and len(trials) >= epochs:
            return trials

    if mode == "grid":
        for variant in sorted(available):
            if variant == default_variant:
                continue
            add_trial(variant)
        return trials

    if mode == "random":
        import random

        rng = random.Random(seed)
        while len(trials) < epochs:
            remaining = [v for v in available if v not in seen]
            if remaining:
                add_trial(rng.choice(remaining))
            else:
                add_trial(rng.choice(available))
        return trials

    try:
        from skopt import Optimizer  # type: ignore
        from skopt.space import Categorical  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "hpo=gp requires scikit-optimize. Install it with `pip install scikit-optimize`."
        ) from exc

    opt = Optimizer(
        [Categorical(list(available), name="variant")],
        random_state=seed,
        n_initial_points=min(5, max(1, epochs)),
    )

    if trials:
        variant = trials[-1]
        score = _trial_objective_value(
            history=global_history[:-1],
            candidate=variant,
            variant_term_means=variant_term_means,
            objective=objective,
            scaling=scaling,
        )
        opt.tell([variant], -(score if score is not None else float("-inf")))

    while len(trials) < epochs:
        variant = None
        for _ in range(len(available) + 5):
            proposed = opt.ask()[0]
            proposed = str(proposed)
            if proposed not in seen:
                variant = proposed
                break
        if variant is None:
            variant = str(opt.ask()[0])

        score = _trial_objective_value(
            history=global_history,
            candidate=variant,
            variant_term_means=variant_term_means,
            objective=objective,
            scaling=scaling,
        )
        opt.tell([variant], -(score if score is not None else float("-inf")))
        add_trial(variant)

    return trials

