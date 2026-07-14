"""Tests for full-condition aggregation and paired quantization gaps."""

from __future__ import annotations

import math

from src.reporting import (
    DEMONSTRATION_SEEDS,
    FULL_SHOT_COUNTS,
    PRECISION_CONDITIONS,
    aggregate_condition_metrics,
    compute_condition_metrics,
    compute_paired_gaps,
    validate_full_condition_coverage,
)


def _records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for precision in PRECISION_CONDITIONS:
        for shot_count in FULL_SHOT_COUNTS:
            seeds = (None,) if shot_count == 0 else DEMONSTRATION_SEEDS
            for seed in seeds:
                for example_id, gold_label in enumerate((0, 1)):
                    quantized_error = precision == "4bit_nf4" and example_id == 1
                    predicted_label = 0 if quantized_error else gold_label
                    confidence = 0.75
                    records.append(
                        {
                            "precision_condition": precision,
                            "shot_count": shot_count,
                            "demonstration_seed": seed,
                            "evaluation_example_identifier": example_id,
                            "gold_label": gold_label,
                            "predicted_label": predicted_label,
                            "selected_label_confidence": confidence,
                            "correctness": predicted_label == gold_label,
                        }
                    )
    return records


def test_full_coverage_aggregation_and_gap_direction() -> None:
    metrics = compute_condition_metrics(_records())
    validate_full_condition_coverage(metrics, expected_examples=2)

    aggregates = aggregate_condition_metrics(metrics)
    paired, summaries = compute_paired_gaps(metrics)

    assert len(metrics) == 50
    assert len(aggregates) == 10
    assert len(paired) == 25
    assert len(summaries) == 5
    assert all(
        math.isclose(row["mean_accuracy_gap_4bit_minus_bf16"], -0.5)
        for row in summaries
    )
    assert all(
        row["std_accuracy_gap"] is None
        if row["shot_count"] == 0
        else math.isclose(row["std_accuracy_gap"], 0.0)
        for row in summaries
    )
