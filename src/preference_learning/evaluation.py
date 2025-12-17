"""Evaluation helpers for preference-learning experiments."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def compute_pairwise_scores(pair_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate wins/losses per method variant from pairwise labels."""
    wins = Counter()
    losses = Counter()
    for _, row in pair_df.iterrows():
        label = row.get("label")
        method_a = row.get("pair_1")
        method_b = row.get("pair_2")
        if label not in (0, 1):
            continue
        if label == 0:
            winner, loser = method_a, method_b
        else:
            winner, loser = method_b, method_a
        wins[winner] += 1
        losses[loser] += 1
    variants = sorted(set(list(wins) + list(losses)))
    records = []
    for variant in variants:
        win_count = wins.get(variant, 0)
        loss_count = losses.get(variant, 0)
        records.append(
            {
                "method_variant": variant,
                "wins": int(win_count),
                "losses": int(loss_count),
                "score": int(win_count - loss_count),
            }
        )
    return pd.DataFrame(records)


def build_ground_truth_order(pair_df: pd.DataFrame) -> List[str]:
    """Derive a deterministic total order using Copeland-style scores."""
    scores_df = compute_pairwise_scores(pair_df)
    if scores_df.empty:
        return []
    tie_groups = (
        scores_df.groupby(["score", "wins"])
        .filter(lambda frame: len(frame) > 1)
        .sort_values(by=["score", "wins"], ascending=[False, False])
    )
    if not tie_groups.empty:
        LOGGER.info(
            "Detected %d tied variants when constructing Copeland ranking. "
            "Ties will be broken by method_variant name for determinism. Examples: %s",
            len(tie_groups),
            tie_groups["method_variant"].tolist()[:5],
        )
    # Break ties lexicographically on method_variant so we emit a stable order even when
    # the pairwise graph contains cycles or perfectly balanced head-to-head records.
    ranked = scores_df.sort_values(
        by=["score", "wins", "method_variant"],
        ascending=[False, False, True],
    )
    return ranked["method_variant"].tolist()


def evaluate_topk(
    predicted_scores: Mapping[str, float],
    ground_truth_order: Sequence[str],
    k_values: Iterable[int] = (1, 3, 5),
) -> Dict[str, Dict[str, float]]:
    """Evaluate overlap between predicted and ground-truth top-k rankings."""
    if not ground_truth_order:
        return {}
    sorted_pred = sorted(
        predicted_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    pred_order = [variant for variant, _ in sorted_pred]
    metrics: Dict[str, Dict[str, float]] = {}
    rank_map_pred = {variant: idx for idx, variant in enumerate(pred_order)}
    rank_map_gt = {variant: idx for idx, variant in enumerate(ground_truth_order)}
    for k in k_values:
        k = max(1, int(k))
        gt_top = set(ground_truth_order[:k])
        pred_top = set(pred_order[:k])
        hits = len(gt_top & pred_top)
        union_items = sorted(gt_top | pred_top)
        rank_corr = compute_rank_correlations(
            union_items,
            rank_map_pred,
            rank_map_gt,
        )
        jaccard = hits / len(union_items) if union_items else 0.0
        metrics[str(k)] = {
            "hits": float(hits),
            "precision": hits / k,
            "recall": hits / len(gt_top) if gt_top else 0.0,
            "jaccard": jaccard,
            "rank_correlation": rank_corr,
        }
    return metrics


def compute_rank_correlations(
    union_items: Sequence[str],
    pred_rank_map: Mapping[str, int],
    gt_rank_map: Mapping[str, int],
) -> Dict[str, float]:
    """Compute Spearman and Kendall correlations on the union of top-k items."""
    pred_ranks: List[int] = []
    gt_ranks: List[int] = []
    for variant in union_items:
        if variant not in pred_rank_map or variant not in gt_rank_map:
            continue
        # use 1-based ranks for traditional formulas
        pred_ranks.append(pred_rank_map[variant] + 1)
        gt_ranks.append(gt_rank_map[variant] + 1)
    n = len(pred_ranks)
    if n < 2:
        return {"spearman": 0.0, "kendall": 0.0}

    spearman = _spearman_rank_correlation(pred_ranks, gt_ranks)
    kendall = _kendall_tau(pred_ranks, gt_ranks)
    return {"spearman": spearman, "kendall": kendall}


def _spearman_rank_correlation(r1: Sequence[int], r2: Sequence[int]) -> float:
    n = len(r1)
    if n < 2:
        return 0.0
    diffs = [(a - b) for a, b in zip(r1, r2)]
    numerator = 6 * sum(diff * diff for diff in diffs)
    denominator = n * (n * n - 1)
    return 1 - numerator / denominator if denominator else 0.0


def _kendall_tau(r1: Sequence[int], r2: Sequence[int]) -> float:
    n = len(r1)
    if n < 2:
        return 0.0
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            sign1 = r1[i] - r1[j]
            sign2 = r2[i] - r2[j]
            prod = sign1 * sign2
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1
            # ties (prod == 0) are ignored for simplicity
    total_pairs = concordant + discordant
    if total_pairs == 0:
        return 0.0
    return (concordant - discordant) / total_pairs
