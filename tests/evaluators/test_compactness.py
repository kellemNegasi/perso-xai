import pytest

from src.evaluators.compactness import CompactnessEvaluator


def test_compactness_scores_match_expected_distribution():
    evaluator = CompactnessEvaluator(zero_tolerance=1e-9)
    explanation_results = {
        "method": "shap",
        "explanations": [
            {
                "attributions": [0.5, 0.0, -0.5, 0.0],
                "instance": [1.0, 2.0, 3.0, 4.0],
            }
        ],
    }

    scores = evaluator.evaluate(
        model=None,
        explanation_results=explanation_results,
        dataset=None,
        explainer=None,
    )

    # Two of four features are non-zero -> sparsity 0.5.
    assert scores["compactness_sparsity"] == pytest.approx(0.5)
    # Total attribution mass captured by top 5 features equals 1.0.
    assert scores["compactness_top5_coverage"] == pytest.approx(1.0)
    # Total attribution mass captured by top 10 features also equals 1.0.
    assert scores["compactness_top10_coverage"] == pytest.approx(1.0)
    # Effective feature count normalization yields 2/3 for this distribution.
    assert scores["compactness_effective_features"] == pytest.approx(2.0 / 3.0, abs=1e-9)
