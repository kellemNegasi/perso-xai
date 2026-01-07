from __future__ import annotations

from pathlib import Path

import numpy as np

from src.preference_learning.persona import (
    HierarchicalDirichletUser,
    DEFAULT_TAU,
    load_persona_config,
    z_normalize_matrix,
)


def test_load_persona_config_extracts_metric_order() -> None:
    config = load_persona_config(Path("src/preference_learning/configs/lay.yaml"))
    metrics = list(config.metric_names())
    assert metrics, "Expected at least one metric in the persona config."
    assert len(metrics) == len(set(metrics)), "Metric order should not contain duplicates."


def test_dirichlet_sampling_produces_valid_weights() -> None:
    config = load_persona_config(Path("src/preference_learning/configs/regulator.yaml"))
    user = HierarchicalDirichletUser(config, seed=7, tau=1.0)
    assert user.metric_weights, "Expected sampled metric weights."
    assert all(w >= 0 for w in user.metric_weights.values())
    assert abs(sum(user.metric_weights.values()) - 1.0) < 1e-9
    assert set(user.metric_order).issuperset(user.metric_weights.keys())


def test_user_defaults_tau_from_config_or_module_default() -> None:
    config = load_persona_config(Path("src/preference_learning/configs/lay.yaml"))
    user = HierarchicalDirichletUser(config, seed=0)
    assert user.tau == config.tau

    config_no_tau = config.__class__(
        persona=config.persona,
        type=config.type,
        description=config.description,
        tau=None,
        properties=config.properties,
    )
    user2 = HierarchicalDirichletUser(config_no_tau, seed=0)
    assert user2.tau == DEFAULT_TAU


def test_preference_probability_is_antisymmetric() -> None:
    config = load_persona_config(Path("src/preference_learning/configs/clinician.yaml"))
    user = HierarchicalDirichletUser(config, seed=11, tau=1.0)
    n = len(user.metric_order)
    z_i = np.linspace(-1.0, 1.0, n)
    z_j = -z_i
    p_ij = user.preference_probability(z_i, z_j)
    p_ji = user.preference_probability(z_j, z_i)
    assert abs((p_ij + p_ji) - 1.0) < 1e-12


def test_z_normalize_handles_nans_and_constant_columns() -> None:
    values = np.array(
        [
            [1.0, np.nan, 5.0],
            [1.0, 2.0, 5.0],
            [1.0, 3.0, 5.0],
        ],
        dtype=float,
    )
    z = z_normalize_matrix(values)
    assert z.shape == values.shape
    # First + third columns are constant -> all zeros after scaling.
    assert np.allclose(z[:, 0], 0.0)
    assert np.allclose(z[:, 2], 0.0)
    # NaN is converted to 0 after scaling.
    assert z[0, 1] == 0.0
