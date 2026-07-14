"""Tests for calibration metric definitions."""

from __future__ import annotations

import pytest

from src.metrics import compute_calibration


def test_calibration_uses_equal_width_bins_and_weighted_gaps() -> None:
    summary = compute_calibration(
        [0.05, 0.15, 0.95, 1.0],
        [True, False, True, False],
        num_bins=10,
    )

    assert summary.bins[0].count == 1
    assert summary.bins[1].count == 1
    assert summary.bins[9].count == 2
    assert summary.bins[9].mean_confidence == pytest.approx(0.975)
    assert summary.bins[9].empirical_accuracy == pytest.approx(0.5)
    assert summary.ece == pytest.approx(0.5125)


def test_calibration_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="same length"):
        compute_calibration([0.5], [True, False])
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        compute_calibration([1.1], [True])
    with pytest.raises(ValueError, match="must be boolean"):
        compute_calibration([0.5], [1])  # type: ignore[list-item]
