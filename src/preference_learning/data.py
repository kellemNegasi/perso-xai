"""Utilities for building pairwise preference datasets from encoded features."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

LOGGER = logging.getLogger(__name__)

EXCLUDED_FEATURE_COLUMNS = {
    "dataset",
    "model",
    "instance_index",
    "method",
    "method_variant",
}

STATISTICAL_DATASET_FEATURES = {
    "dataset_log_dataset_size_z",
    "dataset_mean_of_means_z",
    "dataset_std_of_means_z",
    "dataset_mean_variance_z",
    "dataset_max_variance_z",
    "dataset_mean_skewness_z",
    "dataset_std_skewness_z",
    "dataset_max_kurtosis_z",
    "dataset_mean_std_z",
    "dataset_std_std_z",
    "dataset_max_std_z",
    "dataset_mean_range_z",
    "dataset_max_range_z",
    "dataset_mean_cardinality_z",
    "dataset_max_cardinality_z",
    "dataset_mean_cat_entropy_z",
    "dataset_std_cat_entropy_z",
    "dataset_mean_top_freq_z",
    "dataset_max_top_freq_z",
}

LANDMARKING_DATASET_FEATURES = {
    "dataset_landmark_acc_knn1_z",
    "dataset_landmark_acc_gaussian_nb_z",
    "dataset_landmark_acc_decision_stump_z",
    "dataset_landmark_acc_logreg_z",
}

FEATURE_GROUPS = {
    "statistical": STATISTICAL_DATASET_FEATURES,
    "landmarking": LANDMARKING_DATASET_FEATURES,
}


@dataclass
class InstanceData:
    """Container holding all data required to score a held-out instance."""

    instance_index: int
    candidates: pd.DataFrame
    pair_labels: pd.DataFrame


@dataclass
class PairwisePreferenceData:
    """Processed dataset ready for training/testing."""

    dataset_name: str
    model_name: str
    feature_columns: List[str]
    train_instances: List[int]
    test_instances: List[int]
    train_features: pd.DataFrame
    train_labels: pd.Series
    test_data: List[InstanceData]
    encoded_path: Path
    pair_labels_path: Path


class PreferenceDatasetBuilder:
    """Builds pairwise training data from encoded Pareto-front features."""

    def __init__(self, encoded_path: Path, pair_labels_dir: Path) -> None:
        self.encoded_path = encoded_path
        self.pair_labels_path = self._locate_pair_file(encoded_path, pair_labels_dir)

    @staticmethod
    def _locate_pair_file(encoded_path: Path, pair_labels_dir: Path) -> Path:
        base_name = encoded_path.stem.replace("_encoded", "")
        candidate = pair_labels_dir / f"{base_name}_pair_labels.parquet"
        if not candidate.exists():
            raise FileNotFoundError(
                f"Could not find pair labels for {encoded_path.name} under {pair_labels_dir}"
            )
        return candidate

    def build(
        self,
        *,
        test_size: float = 0.2,
        random_state: int = 42,
        excluded_feature_groups: Sequence[str] | None = None,
    ) -> PairwisePreferenceData:
        encoded_df = pd.read_parquet(self.encoded_path)
        if encoded_df.empty:
            raise ValueError(f"No rows found in encoded file {self.encoded_path}")
        pair_labels_df = pd.read_parquet(self.pair_labels_path)
        if pair_labels_df.empty:
            raise ValueError(f"No pair labels found at {self.pair_labels_path}")

        dataset_name = encoded_df["dataset"].iloc[0]
        model_name = encoded_df["model"].iloc[0]
        feature_columns = _infer_feature_columns(
            encoded_df,
            excluded_feature_groups=excluded_feature_groups,
        )

        instance_ids = encoded_df["instance_index"].unique()
        if len(instance_ids) < 2:
            raise ValueError("At least two instances are required to perform a split.")
        train_ids, test_ids = train_test_split(
            instance_ids,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )
        train_ids = sorted(int(idx) for idx in train_ids)
        test_ids = sorted(int(idx) for idx in test_ids)

        train_features, train_labels = _build_training_differences(
            encoded_df,
            pair_labels_df,
            train_ids,
            feature_columns,
        )
        if train_features.empty:
            raise ValueError("Training set is empty after generating difference vectors.")

        test_data: List[InstanceData] = []
        for instance_id in test_ids:
            instance_df = encoded_df.loc[encoded_df["instance_index"] == instance_id].copy()
            pair_df = pair_labels_df.loc[pair_labels_df["dataset_index"] == instance_id].copy()
            if instance_df.empty or pair_df.empty:
                LOGGER.warning(
                    "Skipping evaluation data for instance %s because features or pairs are empty.",
                    instance_id,
                )
                continue
            test_data.append(
                InstanceData(
                    instance_index=int(instance_id),
                    candidates=instance_df.reset_index(drop=True),
                    pair_labels=pair_df.reset_index(drop=True),
                )
            )

        return PairwisePreferenceData(
            dataset_name=dataset_name,
            model_name=model_name,
            feature_columns=feature_columns,
            train_instances=train_ids,
            test_instances=test_ids,
            train_features=train_features,
            train_labels=train_labels,
            test_data=test_data,
            encoded_path=self.encoded_path,
            pair_labels_path=self.pair_labels_path,
        )


def _infer_feature_columns(
    df: pd.DataFrame,
    *,
    excluded_feature_groups: Sequence[str] | None = None,
) -> List[str]:
    numeric_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    group_exclusions = set()
    for group in excluded_feature_groups or []:
        cols = FEATURE_GROUPS.get(group)
        if cols is None:
            LOGGER.warning("Unknown feature group '%s' requested for exclusion; ignoring.", group)
            continue
        group_exclusions.update(cols)
    feature_columns = [
        col
        for col in numeric_cols
        if col not in EXCLUDED_FEATURE_COLUMNS and col not in group_exclusions
    ]
    if not feature_columns:
        raise ValueError("No numeric feature columns were found in the encoded DataFrame.")
    return feature_columns


def _build_training_differences(
    encoded_df: pd.DataFrame,
    pair_labels_df: pd.DataFrame,
    train_ids: Sequence[int],
    feature_columns: Sequence[str],
) -> Tuple[pd.DataFrame, pd.Series]:
    features: List[np.ndarray] = []
    labels: List[int] = []
    for instance_id in train_ids:
        instance_df = encoded_df.loc[encoded_df["instance_index"] == instance_id]
        pair_df = pair_labels_df.loc[pair_labels_df["dataset_index"] == instance_id]
        if instance_df.empty or pair_df.empty:
            LOGGER.warning(
                "Skipping instance %s during training because features or pair labels are empty.",
                instance_id,
            )
            continue
        diff_matrix, diff_labels = _differences_for_instance(instance_df, pair_df, feature_columns)
        if diff_matrix.size == 0:
            LOGGER.warning("Instance %s produced no preference pairs.", instance_id)
            continue
        features.extend(diff_matrix)
        labels.extend(diff_labels)
    if len(features) != len(labels):
        raise ValueError(
            "Inconsistent training data lengths when building difference vectors: "
            f"features={len(features)} labels={len(labels)}. "
            "This can happen if encoded candidates contain duplicate method_variant rows."
        )
    if not features:
        return pd.DataFrame(columns=feature_columns), pd.Series(dtype="int64")
    X = pd.DataFrame(features, columns=feature_columns)
    y = pd.Series(labels, name="label")
    return X, y


def _differences_for_instance(
    instance_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    feature_columns: Sequence[str],
) -> Tuple[np.ndarray, List[int]]:
    instance_df = _dedupe_candidates_by_variant(instance_df)
    candidate_matrix = instance_df.set_index("method_variant")[list(feature_columns)].astype(float)
    if not candidate_matrix.index.is_unique:
        # Defensive fallback: `_dedupe_candidates_by_variant` should have ensured uniqueness.
        LOGGER.warning(
            "Duplicate method_variant values remain after de-duplication; averaging feature rows."
        )
        candidate_matrix = candidate_matrix.groupby(level=0).mean()
    rows: List[np.ndarray] = []
    labels: List[int] = []
    for _, row in pair_df.iterrows():
        method_a = row.get("pair_1")
        method_b = row.get("pair_2")
        label = row.get("label")
        if method_a == method_b:
            LOGGER.warning("Skipping degenerate pair (%s, %s) with identical variants.", method_a, method_b)
            continue
        if method_a not in candidate_matrix.index or method_b not in candidate_matrix.index:
            LOGGER.warning(
                "Pair (%s, %s) skipped because at least one variant is missing from features.",
                method_a,
                method_b,
            )
            continue
        if label not in (0, 1):
            LOGGER.warning("Unexpected label %s encountered. Skipping pair.", label)
            continue
        if label == 0:
            preferred, other = method_a, method_b
        else:
            preferred, other = method_b, method_a
        diff = candidate_matrix.loc[preferred].values - candidate_matrix.loc[other].values
        rows.append(diff)
        labels.append(1)
        rows.append(-diff)
        labels.append(-1)
    return np.vstack(rows) if rows else np.empty((0, len(feature_columns))), labels


def _dedupe_candidates_by_variant(candidates: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeated `method_variant` rows to ensure one candidate per variant.

    Some encoded Pareto artifacts may contain multiple rows per `method_variant` for a given
    `instance_index` (e.g., repeated runs). Downstream training expects a single feature
    vector per variant; duplicates can create inconsistent X/y lengths via broadcasting.
    """
    if candidates.empty or "method_variant" not in candidates.columns:
        return candidates
    method_variant = candidates["method_variant"].astype(str)
    if method_variant.is_unique:
        return candidates

    LOGGER.warning(
        "Found duplicate method_variant rows (n=%s -> %s unique); aggregating duplicates.",
        int(len(candidates)),
        int(method_variant.nunique(dropna=False)),
    )
    candidates = candidates.copy()
    candidates["method_variant"] = method_variant

    grouped = candidates.groupby("method_variant", sort=False)
    numeric_cols = candidates.select_dtypes(include=["number", "bool"]).columns.tolist()

    non_numeric_cols = [col for col in candidates.columns if col not in numeric_cols and col != "method_variant"]
    numeric_mean_cols = [col for col in numeric_cols if col != "instance_index"]

    parts: list[pd.DataFrame] = []
    if non_numeric_cols:
        parts.append(grouped[non_numeric_cols].first())
    if numeric_mean_cols:
        parts.append(grouped[numeric_mean_cols].mean())
    if "instance_index" in candidates.columns:
        parts.append(grouped[["instance_index"]].first())

    collapsed = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=grouped.size().index)
    collapsed.insert(0, "method_variant", collapsed.index.astype(str))
    collapsed = collapsed.reset_index(drop=True)

    desired_order = ["method_variant"] + [col for col in candidates.columns if col != "method_variant"]
    collapsed = collapsed[[col for col in desired_order if col in collapsed.columns]]

    if "instance_index" in collapsed.columns:
        collapsed["instance_index"] = pd.to_numeric(collapsed["instance_index"], errors="coerce")
        if collapsed["instance_index"].notna().all():
            collapsed["instance_index"] = collapsed["instance_index"].astype(int)

    return collapsed
