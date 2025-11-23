import numpy as np
import pytest

from src.evaluators.completeness import CompletenessEvaluator


class LinearProbModel:
    def __init__(self, weights, bias=0.0):
        self.weights = np.asarray(weights, dtype=float)
        self.bias = float(bias)

    def predict_proba(self, X):
        logits = X @ self.weights + self.bias
        probs = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1.0 - probs, probs])


def test_completeness_drop_matches_manual_computation():
    model = LinearProbModel(weights=[1.0, -0.5, 0.25, 0.0])
    instance = np.array([2.0, 1.0, -1.0, 0.5], dtype=float)
    proba = model.predict_proba(instance.reshape(1, -1))[0]

    explanation = {
        "attributions": [0.9, 0.2, 0.02, 0.0],
        "instance": instance.tolist(),
        "metadata": {"baseline_instance": [0.0, 0.0, 0.0, 0.0]},
        "prediction_proba": proba.tolist(),
    }
    explanation_results = {"method": "shap", "explanations": [explanation]}

    evaluator = CompletenessEvaluator(
        magnitude_threshold=0.05,
        min_features=1,
        random_trials=0,
        default_baseline=0.0,
    )
    scores = evaluator.evaluate(model, explanation_results, dataset=None, explainer=None)

    mask_indices = np.array([0, 1], dtype=int)
    orig_pred = proba[1]
    perturbed = instance.copy()
    perturbed[mask_indices] = 0.0
    new_pred = model.predict_proba(perturbed.reshape(1, -1))[0, 1]
    expected_drop = abs(orig_pred - new_pred) / (abs(orig_pred) + 1e-8)

    # Deterministic drop should match direct model computation with masked features.
    assert scores["completeness_drop"] == pytest.approx(expected_drop, abs=1e-9)
    # Random baselines disabled => random drop must be exactly zero.
    assert scores["completeness_random_drop"] == pytest.approx(0.0, abs=1e-12)
    # Score equals target drop minus random baseline (zero here).
    assert scores["completeness_score"] == pytest.approx(expected_drop, abs=1e-9)
