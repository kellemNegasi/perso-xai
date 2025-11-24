import numpy as np
import pytest

from src.evaluators.infidelity import InfidelityEvaluator


class LinearModel:
    def __init__(self, weights, bias=0.0):
        self.weights = np.asarray(weights, dtype=float)
        self.bias = float(bias)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.weights + self.bias


def _explanation(instance, attributions, model):
    instance = np.asarray(instance, dtype=float)
    pred = model.predict(instance.reshape(1, -1))[0]
    return {
        "instance": instance.tolist(),
        "attributions": attributions,
        "prediction": pred,
        "metadata": {"baseline_instance": [0.0] * instance.size},
    }


def test_infidelity_zero_for_linear_model_with_gradient_attributions():
    model = LinearModel(weights=[0.6, -0.2, 0.3])
    instance = np.array([1.0, -2.0, 0.5], dtype=float)
    explanation = _explanation(instance, model.weights.tolist(), model)
    explanation_results = {"method": "shap", "explanations": [explanation]}

    evaluator = InfidelityEvaluator(
        n_perturb_samples=64,
        features_per_sample=2,
        default_baseline=0.0,
        random_state=0,
    )
    score = evaluator.evaluate(model, explanation_results)["infidelity"]

    assert score == pytest.approx(0.0, abs=1e-9)


def test_infidelity_penalizes_incorrect_attributions():
    model = LinearModel(weights=[0.6, -0.2, 0.3])
    instance = np.array([1.0, -2.0, 0.5], dtype=float)
    wrong_attributions = (model.weights * np.array([1.0, -3.0, 2.0])).tolist()
    explanation = _explanation(instance, wrong_attributions, model)
    explanation_results = {"method": "shap", "explanations": [explanation]}

    evaluator = InfidelityEvaluator(
        n_perturb_samples=32,
        features_per_sample=2,
        default_baseline=0.0,
        random_state=1,
    )
    score = evaluator.evaluate(model, explanation_results)["infidelity"]

    assert score > 0.01
