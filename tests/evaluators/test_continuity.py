import numpy as np
import pytest

from src.evaluators.continuity import ContinuityEvaluator


class DummyDataset:
    def __init__(self):
        self.X_train = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=float,
        )


class EchoExplainer:
    def __init__(self, importance):
        self.importance = np.asarray(importance, dtype=float)

    def explain_instance(self, instance):
        return {
            "attributions": self.importance.tolist(),
            "instance": np.asarray(instance, dtype=float).tolist(),
        }


def _base_results():
    base_importance = [0.3, -0.2, 0.1]
    explanations = [
        {"instance": [1.0, 2.0, 3.0], "attributions": base_importance},
        {"instance": [0.5, 1.5, 2.5], "attributions": base_importance},
    ]
    return {"method": "shap", "explanations": explanations}, base_importance


def test_continuity_batch_zero_noise_yields_one():
    explanation_results, base_importance = _base_results()
    evaluator = ContinuityEvaluator(max_instances=2, noise_scale=0.0)
    dataset = DummyDataset()
    explainer = EchoExplainer(base_importance)

    scores = evaluator.evaluate(
        model=None,
        explanation_results=explanation_results,
        dataset=dataset,
        explainer=explainer,
    )
    assert scores["continuity_stability"] == pytest.approx(1.0, abs=1e-9)


def test_continuity_per_instance_uses_current_index():
    explanation_results, base_importance = _base_results()
    evaluator = ContinuityEvaluator(max_instances=2, noise_scale=0.0)
    dataset = DummyDataset()
    explainer = EchoExplainer(base_importance)

    payload = dict(explanation_results)
    payload["current_index"] = 0
    scores = evaluator.evaluate(
        model=None,
        explanation_results=payload,
        dataset=dataset,
        explainer=explainer,
    )
    assert scores["continuity_stability"] == pytest.approx(1.0, abs=1e-9)
