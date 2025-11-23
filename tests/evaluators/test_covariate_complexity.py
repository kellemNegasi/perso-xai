import numpy as np
import pytest

from src.evaluators.covariate_complexity import CovariateComplexityEvaluator


def _normalized_entropy(importance: np.ndarray) -> float:
    abs_imp = np.abs(importance)
    prob = abs_imp / np.sum(abs_imp)
    safe_prob = np.clip(prob, 1e-12, 1.0)
    entropy = -np.sum(safe_prob * np.log2(safe_prob))
    max_entropy = np.log2(len(prob)) if len(prob) > 1 else 0.0
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def test_covariate_complexity_matches_entropy_expectation():
    explanation_results = {
        "method": "shap",
        "explanations": [
            {"attributions": [0.5, 0.25, 0.25]},
            {"attributions": [1.0, 0.0, 0.0]},
        ],
    }

    evaluator = CovariateComplexityEvaluator()
    scores = evaluator.evaluate(model=None, explanation_results=explanation_results, dataset=None, explainer=None)

    norm1 = _normalized_entropy(np.array([0.5, 0.25, 0.25], dtype=float))
    norm2 = _normalized_entropy(np.array([1.0, 0.0, 0.0], dtype=float))
    expected_complexity = np.mean([norm1, norm2])
    expected_regularity = np.mean([1.0 - norm1, 1.0 - norm2])

    # Average normalized entropy must equal mean of the per-instance values.
    assert scores["covariate_complexity"] == pytest.approx(expected_complexity, abs=1e-9)
    # Regularity is defined as 1 - complexity on a per-instance basis, so averages match too.
    assert scores["covariate_regularity"] == pytest.approx(expected_regularity, abs=1e-9)
