from __future__ import annotations

import argparse

import pytest

from src.preference_learning.run_all import _extract_aggregate_spearman, _parse_tau_values


def test_parse_tau_values_deduplicates_and_validates() -> None:
    assert _parse_tau_values(["0.1", "0.2", "0.1"]) == (0.1, 0.2)
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_tau_values(["0"])
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_tau_values(["-1.0"])
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_tau_values(["not-a-float"])


def test_extract_aggregate_spearman_reads_expected_key() -> None:
    dummy = {
        "aggregate_top_k_mean": {
            "3": {"rank_correlation.spearman": 0.25},
            "5": {"rank_correlation.spearman": 0.75},
            "8": {"rank_correlation.kendall": 0.1},
        }
    }
    assert _extract_aggregate_spearman(dummy, k_values=(3, 5, 8)) == [0.25, 0.75]

