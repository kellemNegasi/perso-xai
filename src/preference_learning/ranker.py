"""In-memory ranker for sampling persona-based pairwise preferences.

This ranker operates on the encoded candidate DataFrame (the same one used for
training), and assumes metric columns are already oriented so that
"higher is better" (e.g., lower-is-better metrics have been negated upstream).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

from .persona import HierarchicalDirichletUser


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


@dataclass(frozen=True)
class PairwiseLabelConfig:
    """Configuration for sampling pairwise labels."""

    label: str = "label"
    pair_1: str = "pair_1"
    pair_2: str = "pair_2"
    dataset_index: str = "dataset_index"
    method_variant: str = "method_variant"


class PersonaPairwiseRanker:
    """Samples pairwise labels using a fixed persona (user) and RNG."""

    def __init__(
        self,
        *,
        user: HierarchicalDirichletUser,
        rng: np.random.Generator,
        metric_names: Sequence[str] | None = None,
        columns: PairwiseLabelConfig | None = None,
    ) -> None:
        self.user = user
        self.rng = rng
        self.metric_names = tuple(metric_names or user.metric_order)
        self.columns = columns or PairwiseLabelConfig()

    def _z_matrix(self, candidates: pd.DataFrame) -> np.ndarray:
        if not self.metric_names:
            raise ValueError("metric_names cannot be empty.")
        # Encoded candidates already store z-normalised metric columns; missing -> 0.0.
        z = np.zeros((len(candidates), len(self.metric_names)), dtype=float)
        for j, metric in enumerate(self.metric_names):
            if metric in candidates.columns:
                z[:, j] = pd.to_numeric(candidates[metric], errors="coerce").fillna(0.0).to_numpy()
        return z

    def label_instance(
        self,
        *,
        dataset_index: int,
        candidates: pd.DataFrame,
    ) -> pd.DataFrame:
        variants = candidates[self.columns.method_variant].astype(str).tolist()
        if len(variants) < 2:
            return pd.DataFrame(columns=(self.columns.dataset_index, self.columns.pair_1, self.columns.pair_2, self.columns.label))

        z = self._z_matrix(candidates)
        w = self.user.weight_vector
        if w.shape[0] != z.shape[1]:
            raise ValueError("User weights do not match metric_names.")

        index_pairs = list(combinations(range(len(variants)), 2))
        idx_a = np.fromiter((a for a, _ in index_pairs), dtype=int, count=len(index_pairs))
        idx_b = np.fromiter((b for _, b in index_pairs), dtype=int, count=len(index_pairs))

        delta = z[idx_a] - z[idx_b]
        logits = (delta @ w) / self.user.tau
        p = _sigmoid(logits)
        # Sample pairwise outcomes with P(pair_1 wins) = p using inverse-CDF sampling:
        # if u ~ Uniform(0,1), then P(u < p) = p, so `u < p` selects pair_1 with probability p.
        a_preferred = self.rng.random(size=p.shape[0]) < p
        labels = np.where(a_preferred, 0, 1).astype(int)

        rows: List[Dict[str, object]] = []
        for (a, b), label in zip(index_pairs, labels):
            rows.append(
                {
                    self.columns.dataset_index: dataset_index,
                    self.columns.pair_1: variants[a],
                    self.columns.pair_2: variants[b],
                    self.columns.label: int(label),
                }
            )
        return pd.DataFrame(rows, columns=(self.columns.dataset_index, self.columns.pair_1, self.columns.pair_2, self.columns.label))
