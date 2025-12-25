from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .autoxai_types import CandidateScore
from .autoxai_utils import tie_break


def compare_against_pair_labels(
    *,
    pair_labels_path: Path,
    scores: Sequence[CandidateScore],
    tie_breaker_seed: int = 13,
) -> Dict[str, Any]:
    """
    Compare per-instance candidate scores against HC-XAI pairwise labels.

    Pair label convention matches `hc-xai/candidates_pair_ranker.py`:
      - label 0 means pair_1 is preferred
      - label 1 means pair_2 is preferred
    """
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "pandas is required to read the parquet pair label files. "
            "Install the hc-xai dependencies (or convert the parquet to CSV)."
        ) from exc

    df = pd.read_parquet(pair_labels_path)
    needed = {"dataset_index", "pair_1", "pair_2", "label"}
    if not needed.issubset(set(df.columns)):
        raise ValueError(f"Pair labels file missing required columns: {sorted(needed)}")

    score_lookup: Dict[Tuple[int, str], float] = {
        (score.dataset_index, score.method_variant): score.aggregated_score for score in scores
    }

    total = 0
    correct = 0
    skipped_missing = 0
    wins: Dict[str, int] = {}
    losses: Dict[str, int] = {}
    for row in df.itertuples(index=False):
        dataset_index = getattr(row, "dataset_index")
        pair_1 = getattr(row, "pair_1")
        pair_2 = getattr(row, "pair_2")
        label = getattr(row, "label")

        key_a = (int(dataset_index), str(pair_1))
        key_b = (int(dataset_index), str(pair_2))
        score_a = score_lookup.get(key_a)
        score_b = score_lookup.get(key_b)
        if score_a is None or score_b is None:
            skipped_missing += 1
            continue

        if int(label) == 0:
            winner = str(pair_1)
            loser = str(pair_2)
        else:
            winner = str(pair_2)
            loser = str(pair_1)
        wins[winner] = wins.get(winner, 0) + 1
        losses[loser] = losses.get(loser, 0) + 1

        if score_a == score_b:
            pred = tie_break(tie_breaker_seed, dataset_index, str(pair_1), str(pair_2))
        else:
            pred = 0 if score_a > score_b else 1

        total += 1
        if int(label) == int(pred):
            correct += 1

    accuracy = correct / total if total else 0.0
    variant_ranking = []
    for variant in sorted(set(wins) | set(losses)):
        w = wins.get(variant, 0)
        l = losses.get(variant, 0)
        n = w + l
        rate = (w / n) if n else 0.0
        variant_ranking.append(
            {"method_variant": variant, "wins": w, "losses": l, "games": n, "win_rate": rate}
        )
    variant_ranking.sort(key=lambda item: (item["win_rate"], item["games"]), reverse=True)
    return {
        "pairs_evaluated": total,
        "pairs_correct": correct,
        "pairwise_accuracy": accuracy,
        "pairs_skipped_missing_candidates": skipped_missing,
        "hc_xai_pair_label_ranking": variant_ranking,
    }


def load_hc_xai_splits(split_json_path: Path) -> Dict[str, Any]:
    payload = json.loads(split_json_path.read_text(encoding="utf-8"))
    train = payload.get("train_instances")
    test = payload.get("test_instances")
    if not isinstance(train, list) or not isinstance(test, list):
        raise ValueError(f"Invalid splits file (missing train/test instances): {split_json_path}")
    payload["train_instances"] = [int(x) for x in train]
    payload["test_instances"] = [int(x) for x in test]
    return payload


def evaluate_topk_against_pair_labels(
    *,
    pair_labels_path: Path,
    scores: Sequence[CandidateScore],
    k_values: Iterable[int] = (3, 5),
) -> Dict[str, Any]:
    """
    Produce HC-XAI-style per-instance top-k evaluation against pair-label ground truth.
    """
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "pandas is required to read the parquet pair label files. "
            "Install the hc-xai dependencies (or convert the parquet to CSV)."
        ) from exc

    from src.preference_learning.evaluation import build_ground_truth_order, evaluate_topk  # type: ignore

    df = pd.read_parquet(pair_labels_path)
    needed = {"dataset_index", "pair_1", "pair_2", "label"}
    if not needed.issubset(set(df.columns)):
        raise ValueError(f"Pair labels file missing required columns: {sorted(needed)}")

    scores_by_instance: Dict[int, Dict[str, float]] = {}
    for score in scores:
        scores_by_instance.setdefault(int(score.dataset_index), {})[str(score.method_variant)] = float(
            score.aggregated_score
        )

    metrics_by_instance: Dict[str, Any] = {}
    skipped_no_pairs = 0
    skipped_no_scores = 0
    for dataset_index, pair_df in df.groupby("dataset_index", sort=False):
        dataset_index = int(dataset_index)
        predicted_scores = scores_by_instance.get(dataset_index)
        if not predicted_scores:
            skipped_no_scores += 1
            continue
        if pair_df.empty:
            skipped_no_pairs += 1
            continue

        gt_variants = set(pair_df["pair_1"].astype(str)) | set(pair_df["pair_2"].astype(str))
        predicted_scores = {variant: value for variant, value in predicted_scores.items() if variant in gt_variants}
        if len(predicted_scores) < 2:
            skipped_no_scores += 1
            continue

        ground_truth = build_ground_truth_order(pair_df)
        topk_metrics = evaluate_topk(
            predicted_scores,
            ground_truth,
            k_values=k_values,
        )
        metrics_by_instance[str(dataset_index)] = {
            "ground_truth": ground_truth,
            "top_k": topk_metrics,
        }

    k_values_norm = [str(max(1, int(k))) for k in k_values]
    mean_topk: Dict[str, Any] = {}
    for k in k_values_norm:
        precision_vals: List[float] = []
        recall_vals: List[float] = []
        jaccard_vals: List[float] = []
        spearman_vals: List[float] = []
        kendall_vals: List[float] = []
        for item in metrics_by_instance.values():
            topk_blob = item.get("top_k", {}).get(k, {})
            if isinstance(topk_blob.get("precision"), (int, float)):
                precision_vals.append(float(topk_blob["precision"]))
            if isinstance(topk_blob.get("recall"), (int, float)):
                recall_vals.append(float(topk_blob["recall"]))
            if isinstance(topk_blob.get("jaccard"), (int, float)):
                jaccard_vals.append(float(topk_blob["jaccard"]))
            corr = topk_blob.get("rank_correlation")
            if isinstance(corr, dict):
                if isinstance(corr.get("spearman"), (int, float)):
                    spearman_vals.append(float(corr["spearman"]))
                if isinstance(corr.get("kendall"), (int, float)):
                    kendall_vals.append(float(corr["kendall"]))

        mean_topk[k] = {
            "precision": sum(precision_vals) / len(precision_vals) if precision_vals else 0.0,
            "recall": sum(recall_vals) / len(recall_vals) if recall_vals else 0.0,
            "jaccard": sum(jaccard_vals) / len(jaccard_vals) if jaccard_vals else 0.0,
            "rank_correlation": {
                "spearman": sum(spearman_vals) / len(spearman_vals) if spearman_vals else 0.0,
                "kendall": sum(kendall_vals) / len(kendall_vals) if kendall_vals else 0.0,
            },
        }

    return {
        "k_values": [int(k) for k in k_values_norm],
        "instances_evaluated": len(metrics_by_instance),
        "instances_skipped_missing_pairs": skipped_no_pairs,
        "instances_skipped_missing_scores": skipped_no_scores,
        "mean_top_k": mean_topk,
        "by_instance": metrics_by_instance,
    }

