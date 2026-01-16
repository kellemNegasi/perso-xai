from __future__ import annotations

import numpy as np
import pandas as pd

from src.preference_learning.persona import HierarchicalDirichletUser, PersonaConfig
from src.preference_learning.ranker import PersonaPairwiseRanker


def test_ranker_uses_tau_and_weights() -> None:
    cfg = PersonaConfig.from_dict(
        {
            "persona": "unit_test",
            "type": "hierarchical_dirichlet",
            "tau": 0.05,
            "properties": {
                "g": {
                    "preference": 1,
                    "metrics": ["m1", "m2"],
                },
            },
        }
    )
    user = HierarchicalDirichletUser(cfg, seed=0, concentration_c=1.0)
    # Force deterministic weights: m1 dominates.
    user.metric_weights = {"m1": 1.0, "m2": 0.0}
    user._weight_vector = None

    candidates = pd.DataFrame(
        {
            "method_variant": ["a", "b"],
            "m1": [10.0, -10.0],
            "m2": [0.0, 0.0],
        }
    )
    ranker = PersonaPairwiseRanker(user=user, rng=np.random.default_rng(0))
    pair_df = ranker.label_instance(dataset_index=0, candidates=candidates)
    assert int(pair_df.iloc[0]["label"]) == 0
