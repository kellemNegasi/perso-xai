import pytest

from src.evaluators.contrastivity import ContrastivityEvaluator


def test_contrastivity_returns_expected_score_for_two_labels():
    evaluator = ContrastivityEvaluator(pairs_per_instance=1, random_state=0)

    base_explanations = [
        {"attributions": [1.0, 0.0, 0.0], "prediction": 0},
        {"attributions": [0.0, 1.0, 0.0], "prediction": 1},
    ]

    explanation_results = {"method": "shap", "explanations": base_explanations}

    scores = evaluator.evaluate(
        model=None,
        explanation_results=explanation_results,
        dataset=None,
        explainer=None,
    )

    # Two cross-label comparisons (ordered pairs) are evaluated.
    assert scores["contrastivity_pairs"] == pytest.approx(2.0, abs=1e-9)
    # Orthogonal importance vectors should yield max contrastivity (1 - SSIM ≈ 1).
    assert scores["contrastivity"] == pytest.approx(1.0, abs=1e-9)


def test_contrastivity_per_instance_anchors_on_target_explanation():
    evaluator = ContrastivityEvaluator(pairs_per_instance=2, random_state=0)

    explanations = [
        {"attributions": [1.0, 0.0, 0.0], "prediction": 0},
        {"attributions": [0.0, 1.0, 0.0], "prediction": 1},
        {"attributions": [0.0, 0.0, 1.0], "prediction": 1},
    ]
    explanation_results = {"method": "shap", "explanations": explanations}
    payload = dict(explanation_results)
    payload["current_index"] = 0

    scores = evaluator.evaluate(
        model=None,
        explanation_results=payload,
        dataset=None,
        explainer=None,
    )

    assert scores["contrastivity_pairs"] == pytest.approx(2.0, abs=1e-9)
    assert scores["contrastivity"] == pytest.approx(1.0, abs=1e-9)
