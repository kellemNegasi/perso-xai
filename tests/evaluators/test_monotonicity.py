import numpy as np
import pytest

from src.evaluators.monotonicity import MonotonicityEvaluator


class LinearProbModel:
    def __init__(self, weights, bias=0.0):
        self.weights = np.asarray(weights, dtype=float)
        self.bias = float(bias)

    def predict_proba(self, X):
        logits = X @ self.weights + self.bias
        probs = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1.0 - probs, probs])


def _explanation(instance, attributions, model):
    instance = np.asarray(instance, dtype=float)
    proba = model.predict_proba(instance.reshape(1, -1))[0]
    return {
        "instance": instance.tolist(),
        "attributions": attributions,
        "metadata": {"baseline_instance": [0.0] * instance.size},
        "prediction_proba": proba.tolist(),
    }


def _squared_effects(model, instance):
    baseline = np.zeros_like(instance)
    proba = model.predict_proba(instance.reshape(1, -1))[0, 1]
    effects = []
    for idx in range(instance.size):
        perturbed = instance.copy()
        perturbed[idx] = baseline[idx]
        new_pred = model.predict_proba(perturbed.reshape(1, -1))[0, 1]
        effects.append((proba - new_pred) ** 2)
    return effects


def test_monotonicity_high_when_effects_follow_attribution_order():
    model = LinearProbModel(weights=[0.9, 0.5, 0.1])
    instance = np.array([1.0, 0.3, -0.4], dtype=float)
    effects = _squared_effects(model, instance)
    explanation = _explanation(instance, effects, model)
    explanation_results = {"method": "shap", "explanations": [explanation]}
    # Original probability ≈ 0.733; masking feature 0 drops it to ≈ 0.527
    #   ⇒ squared delta ≈ 0.0424, feature 1 ≈ 9.6e-4, feature 2 ≈ 6.4e-5.
    # We pass those exact numbers as attributions, so the metric should see perfect alignment.

    evaluator = MonotonicityEvaluator(
        nr_samples=1,
        features_in_step=1,
        default_baseline=0.0,
        noise_scale=0.0,
        random_state=0,
    )
    score = evaluator.evaluate(model, explanation_results)["monotonicity"]

    assert score == pytest.approx(1.0, abs=1e-6)


def test_monotonicity_penalizes_inverted_rankings():
    model = LinearProbModel(weights=[0.9, 0.5, 0.1])
    instance = np.array([1.0, 0.3, -0.4], dtype=float)
    effects = _squared_effects(model, instance)
    inverted = list(reversed(effects))
    explanation = _explanation(instance, inverted, model)
    explanation_results = {"method": "shap", "explanations": [explanation]}
    # Same predictions as above, but now we deliberately mis-order the attribution
    # magnitudes (largest effect reported last). The monotonicity score should turn negative.

    evaluator = MonotonicityEvaluator(
        nr_samples=1,
        features_in_step=1,
        default_baseline=0.0,
        noise_scale=0.0,
        random_state=0,
    )
    score = evaluator.evaluate(model, explanation_results)["monotonicity"]

    assert score < -0.8
