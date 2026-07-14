"""Training-derived majority-class baseline for SST-2."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from src.data import LABEL_NAMES
from src.metrics import CalibrationSummary, ReliabilityBin, compute_calibration


BASELINE_SCHEMA_VERSION = 1


def _labels(
    records: Iterable[Mapping[str, Any]], *, split_name: str
) -> tuple[int, ...]:
    labels: list[int] = []
    for position, record in enumerate(records):
        if "label" not in record:
            raise ValueError(f"{split_name} record {position} is missing label")
        label = record["label"]
        if (
            not isinstance(label, int)
            or isinstance(label, bool)
            or label not in LABEL_NAMES
        ):
            raise ValueError(
                f"{split_name} record {position} has label {label!r}; expected 0 or 1"
            )
        labels.append(label)
    if not labels:
        raise ValueError(f"{split_name} must not be empty")
    return tuple(labels)


def _bin_dict(bin_: ReliabilityBin) -> dict[str, Any]:
    return {
        "index": bin_.index,
        "lower_bound": bin_.lower_bound,
        "upper_bound": bin_.upper_bound,
        "count": bin_.count,
        "mean_confidence": bin_.mean_confidence,
        "empirical_accuracy": bin_.empirical_accuracy,
        "absolute_gap": bin_.absolute_gap,
        "weighted_gap": bin_.weighted_gap,
    }


def compute_majority_class_baseline(
    train_records: Iterable[Mapping[str, Any]],
    validation_records: Iterable[Mapping[str, Any]],
    *,
    dataset_name: str,
    dataset_revision: str | None,
    train_fingerprint: str | None,
    validation_fingerprint: str | None,
    train_split: str,
    validation_split: str,
    num_bins: int = 10,
) -> dict[str, Any]:
    """Compute the baseline without using validation labels for its prediction."""

    train_labels = _labels(train_records, split_name=train_split)
    validation_labels = _labels(validation_records, split_name=validation_split)
    train_counts = Counter(train_labels)
    validation_counts = Counter(validation_labels)

    largest_count = max(train_counts.values())
    majority_labels = [
        label for label in LABEL_NAMES if train_counts[label] == largest_count
    ]
    if len(majority_labels) != 1:
        raise ValueError("training split has no unique majority label")

    majority_label = majority_labels[0]
    training_frequency = largest_count / len(train_labels)
    correctness = tuple(label == majority_label for label in validation_labels)
    validation_accuracy = sum(correctness) / len(correctness)
    calibration: CalibrationSummary = compute_calibration(
        (training_frequency for _ in validation_labels),
        correctness,
        num_bins=num_bins,
    )

    result: dict[str, Any] = {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "dataset": {
            "name": dataset_name,
            "revision": dataset_revision,
            "fingerprints": {
                train_split: train_fingerprint,
                validation_split: validation_fingerprint,
            },
        },
        "source_splits": {
            "majority_selection": train_split,
            "evaluation": validation_split,
        },
        "baseline": {
            "type": "majority_class",
            "majority_label": majority_label,
            "majority_label_name": LABEL_NAMES[majority_label],
            "prediction_rule": "always predict the unique training-set majority label",
            "confidence_definition": "empirical training frequency of the majority label",
        },
        "training": {
            "size": len(train_labels),
            "label_counts": {
                str(label): train_counts[label] for label in LABEL_NAMES
            },
            "majority_frequency": training_frequency,
        },
        "validation": {
            "size": len(validation_labels),
            "label_counts": {
                str(label): validation_counts[label] for label in LABEL_NAMES
            },
            "accuracy": validation_accuracy,
        },
        "calibration": {
            "ece": calibration.ece,
            "num_bins": calibration.num_bins,
            "binning": "equal_width_left_inclusive_right_exclusive_last_bin_closed",
            "reliability_bins": [_bin_dict(bin_) for bin_ in calibration.bins],
        },
    }
    validate_majority_baseline(result)
    return result


def validate_majority_baseline(result: Mapping[str, Any]) -> None:
    """Validate internal consistency of a saved majority-baseline result."""

    if result.get("schema_version") != BASELINE_SCHEMA_VERSION:
        raise ValueError("unsupported majority baseline schema_version")
    baseline = result.get("baseline")
    training = result.get("training")
    validation = result.get("validation")
    calibration = result.get("calibration")
    if not all(
        isinstance(section, Mapping)
        for section in (baseline, training, validation, calibration)
    ):
        raise ValueError("majority baseline result is missing required sections")

    majority_label = baseline.get("majority_label")
    if majority_label not in LABEL_NAMES:
        raise ValueError("majority baseline result has an invalid majority label")
    train_size = training.get("size")
    train_counts = training.get("label_counts")
    frequency = training.get("majority_frequency")
    if not isinstance(train_size, int) or not isinstance(train_counts, Mapping):
        raise ValueError("majority baseline training counts are invalid")
    normalized_counts = {
        int(label): int(count) for label, count in train_counts.items()
    }
    if sum(normalized_counts.values()) != train_size:
        raise ValueError("majority baseline training counts do not sum to its size")
    expected_frequency = normalized_counts[majority_label] / train_size
    if frequency != expected_frequency:
        raise ValueError("majority baseline training frequency is inconsistent")

    validation_size = validation.get("size")
    validation_counts = validation.get("label_counts")
    accuracy = validation.get("accuracy")
    if not isinstance(validation_size, int) or not isinstance(
        validation_counts, Mapping
    ):
        raise ValueError("majority baseline validation counts are invalid")
    normalized_validation = {
        int(label): int(count) for label, count in validation_counts.items()
    }
    if sum(normalized_validation.values()) != validation_size:
        raise ValueError("majority baseline validation counts do not sum to its size")
    expected_accuracy = normalized_validation[majority_label] / validation_size
    if accuracy != expected_accuracy:
        raise ValueError("majority baseline validation accuracy is inconsistent")

    bins = calibration.get("reliability_bins")
    num_bins = calibration.get("num_bins")
    if not isinstance(bins, list) or len(bins) != num_bins:
        raise ValueError("majority baseline reliability bins are inconsistent")
    if sum(int(bin_["count"]) for bin_ in bins) != validation_size:
        raise ValueError("majority baseline reliability counts are inconsistent")
    expected_ece = sum(float(bin_["weighted_gap"]) for bin_ in bins)
    if calibration.get("ece") != expected_ece:
        raise ValueError("majority baseline ECE is inconsistent")


def save_majority_baseline(result: Mapping[str, Any], path: str | Path) -> None:
    """Save a validated result with deterministic formatting and replacement."""

    validate_majority_baseline(result)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_path.replace(output_path)


def load_majority_baseline(path: str | Path) -> dict[str, Any]:
    """Load and validate a saved majority-class baseline result."""

    with Path(path).open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise ValueError("majority baseline result root must be an object")
    validate_majority_baseline(result)
    return result
