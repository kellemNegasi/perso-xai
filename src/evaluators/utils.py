"""
Shared helper functions for evaluator modules (similarities, normalisation, etc.).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["structural_similarity"]


def structural_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> Optional[float]:
    """
    Lightweight SSIM variant for 1-D attribution vectors.

    Parameters
    ----------
    vec_a, vec_b : np.ndarray
        Flattenable attribution vectors with identical shapes.

    Returns
    -------
    float | None
        Similarity score in [-1, 1]; None if the inputs are incompatible.
    """
    if vec_a.size == 0 or vec_b.size == 0 or vec_a.size != vec_b.size:
        return None

    a = vec_a.astype(float).ravel()
    b = vec_b.astype(float).ravel()
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return None

    mu_a = float(np.mean(a))
    mu_b = float(np.mean(b))
    diff_a = a - mu_a
    diff_b = b - mu_b

    var_a = float(np.mean(diff_a * diff_a))
    var_b = float(np.mean(diff_b * diff_b))
    cov_ab = float(np.mean(diff_a * diff_b))

    data_range = max(np.max(a) - np.min(a), np.max(b) - np.min(b))
    if data_range < 1e-12:
        data_range = 1.0

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    denom_mean = mu_a ** 2 + mu_b ** 2 + c1
    denom_var = var_a + var_b + c2
    if denom_mean <= 0.0 or denom_var <= 0.0:
        if np.allclose(a, b):
            return 1.0
        return 0.0

    numerator = (2 * mu_a * mu_b + c1) * (2 * cov_ab + c2)
    denominator = denom_mean * denom_var
    if denominator == 0.0:
        return 0.0

    ssim_value = numerator / denominator
    return float(np.clip(ssim_value, -1.0, 1.0))
