from __future__ import annotations

from src.baseline.autoxai_objectives import default_objective_terms
from src.baseline.autoxai_scoring import compute_scores


def test_autoxai_baseline_compute_scores_instance_scope() -> None:
    candidate_metrics = {
        0: {
            "a": {
                "relative_input_stability": 0.1,  # lower better
                "infidelity": 0.01,  # lower better
                "compactness_effective_features": 0.9,  # higher better
            },
            "b": {
                "relative_input_stability": 0.5,
                "infidelity": 0.2,
                "compactness_effective_features": 0.2,
            },
        }
    }
    variant_to_method = {"a": "lime", "b": "shap"}
    scores = compute_scores(
        candidate_metrics=candidate_metrics,
        variant_to_method=variant_to_method,
        objective=default_objective_terms(),
        scaling="Std",
        scaling_scope="instance",
    )
    assert len(scores) == 2
    by_variant = {score.method_variant: score.aggregated_score for score in scores}
    assert by_variant["a"] > by_variant["b"]

