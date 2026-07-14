"""Calibration metrics shared by baselines and model evaluation."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class ReliabilityBin:
    """Statistics for one equal-width confidence bin."""

    index: int
    lower_bound: float
    upper_bound: float
    count: int
    mean_confidence: float | None
    empirical_accuracy: float | None
    absolute_gap: float | None
    weighted_gap: float


@dataclass(frozen=True)
class CalibrationSummary:
    """Expected Calibration Error and its component bins."""

    num_examples: int
    num_bins: int
    ece: float
    bins: tuple[ReliabilityBin, ...]


def compute_calibration(
    confidences: Iterable[float],
    correctness: Iterable[bool],
    *,
    num_bins: int = 10,
) -> CalibrationSummary:
    """Compute ECE with equal-width bins over confidence in ``[0, 1]``.

    Bins are left-inclusive and right-exclusive, except that the last bin
    includes confidence exactly equal to one.
    """

    if not isinstance(num_bins, int) or isinstance(num_bins, bool) or num_bins <= 0:
        raise ValueError("num_bins must be a positive integer")

    confidence_values = tuple(confidences)
    correctness_values = tuple(correctness)
    if not confidence_values:
        raise ValueError("calibration inputs must not be empty")
    if len(confidence_values) != len(correctness_values):
        raise ValueError("confidences and correctness must have the same length")

    normalized_confidences: list[float] = []
    for position, value in enumerate(confidence_values):
        if isinstance(value, bool):
            raise ValueError(f"confidence {position} must be numeric, not boolean")
        try:
            confidence = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"confidence {position} is not numeric") from error
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence {position} must be finite and in [0, 1]")
        normalized_confidences.append(confidence)

    for position, value in enumerate(correctness_values):
        if not isinstance(value, bool):
            raise ValueError(f"correctness {position} must be boolean")

    bin_confidences: list[list[float]] = [[] for _ in range(num_bins)]
    bin_correctness: list[list[bool]] = [[] for _ in range(num_bins)]
    for confidence, is_correct in zip(
        normalized_confidences, correctness_values, strict=True
    ):
        bin_index = min(int(confidence * num_bins), num_bins - 1)
        bin_confidences[bin_index].append(confidence)
        bin_correctness[bin_index].append(is_correct)

    total = len(normalized_confidences)
    reliability_bins: list[ReliabilityBin] = []
    for index in range(num_bins):
        values = bin_confidences[index]
        outcomes = bin_correctness[index]
        lower_bound = index / num_bins
        upper_bound = (index + 1) / num_bins
        if values:
            mean_confidence = sum(values) / len(values)
            empirical_accuracy = sum(outcomes) / len(outcomes)
            absolute_gap = abs(mean_confidence - empirical_accuracy)
            weighted_gap = len(values) / total * absolute_gap
        else:
            mean_confidence = None
            empirical_accuracy = None
            absolute_gap = None
            weighted_gap = 0.0
        reliability_bins.append(
            ReliabilityBin(
                index=index,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                count=len(values),
                mean_confidence=mean_confidence,
                empirical_accuracy=empirical_accuracy,
                absolute_gap=absolute_gap,
                weighted_gap=weighted_gap,
            )
        )

    return CalibrationSummary(
        num_examples=total,
        num_bins=num_bins,
        ece=sum(bin_.weighted_gap for bin_ in reliability_bins),
        bins=tuple(reliability_bins),
    )
