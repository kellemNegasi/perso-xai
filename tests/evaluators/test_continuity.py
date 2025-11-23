import numpy as np
import pytest

from src.evaluators.correctness import CorrectnessEvaluator


class LinearProbModel:
    def __init__(self, weights, bias=0.0):
        self.weights = np.asarray(weights, dtype=float)
        self.bias = float(bias)

    def predict_proba(self, X):
        logits = X @ self.weights + self.bias
        probs = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1.0 - probs, probs])


def test_correctness_feature_removal_matches_expected_drop():
    model = LinearProbModel(weights=[0.6, -0.4, 0.2])
    instance = np.array([1.0, -2.0, 0.5], dtype=float)
    proba = model.predict_proba(instance.reshape(1, -1))[0]

    explanation = {
        "attributions": [0.9, 0.2, 0.6],
        "instance": instance.tolist(),
        "metadata": {"baseline_instance": [0.0, 0.0, 0.0]},
        "prediction_proba": proba.tolist(),
    }
    explanation_results = {"method": "shap", "explanations": [explanation]}

    evaluator = CorrectnessEvaluator(removal_fraction=0.5, default_baseline=0.0, min_features=1)
    scores = evaluator.evaluate(model, explanation_results, dataset=None, explainer=None)

    mask_indices = np.array([0, 2], dtype=int)
    orig_pred = proba[1]
    perturbed = instance.copy()
    perturbed[mask_indices] = 0.0
    new_pred = model.predict_proba(perturbed.reshape(1, -1))[0, 1]
    expected_drop = abs(orig_pred - new_pred) / (abs(orig_pred) + 1e-8)

    assert scores["correctness"] == pytest.approx(expected_drop, abs=1e-9)
