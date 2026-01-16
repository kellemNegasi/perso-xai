from __future__ import annotations

from src.preference_learning.evaluation import evaluate_topk


def test_evaluate_topk_breaks_score_ties_by_variant_name() -> None:
    # Dict preserves insertion order: "b" is inserted before "a" even though their scores tie.
    predicted_scores = {"b": 1.0, "a": 1.0, "c": 0.0}
    ground_truth = ["a", "b", "c"]
    metrics = evaluate_topk(predicted_scores, ground_truth, k_values=(1,))
    assert metrics["1"]["hits"] == 1.0

