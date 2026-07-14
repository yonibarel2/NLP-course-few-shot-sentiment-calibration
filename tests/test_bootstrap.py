"""Tests for paired hierarchical bootstrap methodology."""

from __future__ import annotations

from src.bootstrap import paired_hierarchical_bootstrap
from src.reporting import DEMONSTRATION_SEEDS, FULL_SHOT_COUNTS, PRECISION_CONDITIONS


def _records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for precision in PRECISION_CONDITIONS:
        for shot_count in FULL_SHOT_COUNTS:
            seeds = (None,) if shot_count == 0 else DEMONSTRATION_SEEDS
            for seed in seeds:
                for example_id, gold_label in enumerate((0, 1, 0, 1)):
                    quantized_error = precision == "4bit_nf4" and example_id == 1
                    predicted_label = 1 - gold_label if quantized_error else gold_label
                    records.append(
                        {
                            "precision_condition": precision,
                            "shot_count": shot_count,
                            "demonstration_seed": seed,
                            "evaluation_example_identifier": example_id,
                            "selected_label_confidence": 0.75,
                            "correctness": predicted_label == gold_label,
                        }
                    )
    return records


def test_bootstrap_is_deterministic_and_preserves_gap_direction() -> None:
    first = paired_hierarchical_bootstrap(
        _records(), num_samples=100, random_seed=7
    )
    second = paired_hierarchical_bootstrap(
        _records(), num_samples=100, random_seed=7
    )

    assert first == second
    accuracy_gaps = [
        row for row in first["gap_intervals"] if row["metric"] == "accuracy"
    ]
    assert all(row["point_estimate"] == -0.25 for row in accuracy_gaps)
    accuracy_interactions = [
        row
        for row in first["shot_effect_interaction_intervals"]
        if row["metric"] == "accuracy"
    ]
    assert all(row["point_estimate"] == 0.0 for row in accuracy_interactions)
    assert len(first["gap_intervals"]) == 10
    assert len(first["shot_effect_interaction_intervals"]) == 8


def test_bootstrap_rejects_incomplete_condition_coverage() -> None:
    records = _records()
    records = [
        row
        for row in records
        if not (
            row["precision_condition"] == "bf16"
            and row["shot_count"] == 8
            and row["demonstration_seed"] == 5
        )
    ]

    try:
        paired_hierarchical_bootstrap(records, num_samples=10)
    except ValueError as error:
        assert "condition coverage mismatch" in str(error)
    else:
        raise AssertionError("incomplete coverage should be rejected")
