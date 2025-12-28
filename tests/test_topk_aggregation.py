from __future__ import annotations

from src.preference_learning.pipeline import _average_topk


def test_average_topk_flattens_nested_metrics() -> None:
    items = [
        {
            "1": {
                "precision": 1.0,
                "rank_correlation": {"spearman": 0.5, "kendall": 0.25},
            }
        },
        {
            "1": {
                "precision": 0.0,
                "rank_correlation": {"spearman": 1.0, "kendall": 0.75},
            }
        },
    ]
    avg = _average_topk(items)
    assert avg["1"]["precision"] == 0.5
    assert avg["1"]["rank_correlation.spearman"] == 0.75
    assert avg["1"]["rank_correlation.kendall"] == 0.5

