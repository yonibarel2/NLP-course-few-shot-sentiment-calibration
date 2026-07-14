"""Tests for deterministic and methodological SST-2 partition behavior."""

from __future__ import annotations

from collections import Counter

import pytest

from src.data import (
    build_split_manifest,
    load_split_manifest,
    partition_training_records,
    save_split_manifest,
)


def _records(per_label: int) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for label in (0, 1):
        for offset in range(per_label):
            idx = label * 10_000 + offset
            records.append(
                {"idx": idx, "label": label, "sentence": f"sentence {idx}"}
            )
    return records


def _manifest(train_records: list[dict[str, object]]) -> dict[str, object]:
    validation_records = _records(4)
    return build_split_manifest(
        train_records,
        validation_records,
        dataset_name="stanfordnlp/sst2",
        dataset_revision=None,
        train_fingerprint="train-fingerprint",
        validation_fingerprint="validation-fingerprint",
        train_split="train",
        validation_split="validation",
        seed=42,
        examples_per_label=10,
        expected_train_size=len(train_records),
        expected_validation_size=len(validation_records),
        expected_validation_labels={0: 4, 1: 4},
    )


def test_partition_is_balanced_disjoint_and_complete() -> None:
    records = _records(30)
    partition = partition_training_records(
        records, seed=42, examples_per_label=10
    )

    prompt_ids = {example.idx for example in partition.prompt_development}
    pool_ids = {example.idx for example in partition.demonstration_pool}
    all_ids = {int(record["idx"]) for record in records}

    assert Counter(
        example.label for example in partition.prompt_development
    ) == Counter({0: 10, 1: 10})
    assert prompt_ids.isdisjoint(pool_ids)
    assert prompt_ids | pool_ids == all_ids


def test_partition_is_deterministic_and_input_order_independent() -> None:
    records = _records(30)

    first = partition_training_records(records, seed=42, examples_per_label=10)
    second = partition_training_records(
        reversed(records), seed=42, examples_per_label=10
    )

    assert first == second


def test_different_seed_changes_prompt_development_selection() -> None:
    records = _records(30)

    first = partition_training_records(records, seed=42, examples_per_label=10)
    second = partition_training_records(records, seed=43, examples_per_label=10)

    assert first.prompt_development != second.prompt_development


def test_duplicate_identifier_is_rejected() -> None:
    records = _records(30)
    records.append(dict(records[0]))

    with pytest.raises(ValueError, match="duplicate idx"):
        partition_training_records(records, seed=42, examples_per_label=10)


def test_unknown_label_is_rejected() -> None:
    records = _records(30)
    records[0]["label"] = 2

    with pytest.raises(ValueError, match="expected 0 or 1"):
        partition_training_records(records, seed=42, examples_per_label=10)


def test_manifest_is_stable_and_records_required_metadata() -> None:
    records = _records(30)

    first = _manifest(records)
    second = _manifest(list(reversed(records)))

    assert first == second
    assert first["prompt_development_selection"] == {
        "seed": 42,
        "sampling_method": "sorted_ids_then_seeded_sample_per_label",
        "examples_per_label": 10,
    }
    assert first["counts"]["demonstration_pool"] == 40
    assert len(first["prompt_development_examples"]) == 20


def test_manifest_round_trip(tmp_path) -> None:
    manifest = _manifest(_records(30))
    path = tmp_path / "split_manifest.json"

    save_split_manifest(manifest, path)

    assert load_split_manifest(path) == manifest


def test_unexpected_validation_counts_are_rejected() -> None:
    with pytest.raises(ValueError, match="label counts"):
        build_split_manifest(
            _records(30),
            _records(4),
            dataset_name="stanfordnlp/sst2",
            dataset_revision=None,
            train_fingerprint=None,
            validation_fingerprint=None,
            train_split="train",
            validation_split="validation",
            seed=42,
            examples_per_label=10,
            expected_train_size=60,
            expected_validation_size=8,
            expected_validation_labels={0: 3, 1: 5},
        )
