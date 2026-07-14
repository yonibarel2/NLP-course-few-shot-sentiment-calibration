"""Tests for the training-derived majority-class baseline."""

from __future__ import annotations

import pytest

from src.baseline import (
    compute_majority_class_baseline,
    load_majority_baseline,
    save_majority_baseline,
)


def _records(labels: list[int]) -> list[dict[str, int]]:
    return [{"idx": index, "label": label} for index, label in enumerate(labels)]


def _baseline(
    train_labels: list[int], validation_labels: list[int]
) -> dict[str, object]:
    return compute_majority_class_baseline(
        _records(train_labels),
        _records(validation_labels),
        dataset_name="stanfordnlp/sst2",
        dataset_revision=None,
        train_fingerprint="train-fingerprint",
        validation_fingerprint="validation-fingerprint",
        train_split="train",
        validation_split="validation",
    )


def test_baseline_uses_training_majority_frequency_and_validation_accuracy() -> None:
    result = _baseline([0, 1, 1, 1], [0, 0, 1, 1])

    assert result["baseline"]["majority_label"] == 1
    assert result["training"]["majority_frequency"] == pytest.approx(0.75)
    assert result["validation"]["accuracy"] == pytest.approx(0.5)
    assert result["calibration"]["ece"] == pytest.approx(0.25)
    populated_bins = [
        bin_ for bin_ in result["calibration"]["reliability_bins"] if bin_["count"]
    ]
    assert len(populated_bins) == 1
    assert populated_bins[0]["index"] == 7
    assert populated_bins[0]["count"] == 4


def test_validation_distribution_cannot_change_training_prediction() -> None:
    first = _baseline([0, 1, 1], [0, 0, 0])
    second = _baseline([0, 1, 1], [1, 1, 1])

    assert first["baseline"]["majority_label"] == 1
    assert second["baseline"]["majority_label"] == 1
    assert first["training"]["majority_frequency"] == second["training"][
        "majority_frequency"
    ]


def test_training_tie_is_rejected() -> None:
    with pytest.raises(ValueError, match="no unique majority"):
        _baseline([0, 1], [0, 1])


def test_baseline_round_trip(tmp_path) -> None:
    result = _baseline([0, 1, 1], [0, 1])
    path = tmp_path / "majority_baseline.json"

    save_majority_baseline(result, path)

    assert load_majority_baseline(path) == result
