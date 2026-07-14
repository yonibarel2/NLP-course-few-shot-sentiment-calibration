"""Tests for deterministic, nested demonstration selection."""

from __future__ import annotations

from collections import Counter

import pytest

from src.prompts import (
    PromptExample,
    build_demonstration_manifest,
    build_prompt,
    demonstration_prefix,
    load_demonstration_manifest,
    materialize_demonstrations,
    save_demonstration_manifest,
    select_demonstration_sets,
)


def _records(per_label: int) -> list[dict[str, int]]:
    return [
        {"idx": label * 10_000 + offset, "label": label}
        for label in (0, 1)
        for offset in range(per_label)
    ]


def test_selections_are_deterministic_balanced_nested_and_disjoint() -> None:
    records = _records(20)
    excluded = {0, 10_000}
    seeds = [0, 1, 2, 3, 4, 5]

    first = select_demonstration_sets(
        records, excluded_ids=excluded, seeds=seeds, examples_per_label=4
    )
    second = select_demonstration_sets(
        reversed(records), excluded_ids=excluded, seeds=seeds, examples_per_label=4
    )

    assert first == second
    for seed, demonstrations in first.items():
        assert len(demonstrations) == 8
        assert not {item.idx for item in demonstrations} & excluded
        assert len({item.idx for item in demonstrations}) == 8
        assert Counter(item.label for item in demonstrations[:2]) == Counter({0: 1, 1: 1})
        assert Counter(item.label for item in demonstrations[:4]) == Counter({0: 2, 1: 2})
        assert Counter(item.label for item in demonstrations) == Counter({0: 4, 1: 4})
        assert demonstrations[0].label == seed % 2


def test_first_label_alternation_requires_balanced_seed_list() -> None:
    with pytest.raises(ValueError, match="evenly distributed"):
        select_demonstration_sets(
            _records(10), excluded_ids=[], seeds=[0, 1, 2], examples_per_label=4
        )


def test_prompt_development_id_must_exist_in_training() -> None:
    with pytest.raises(ValueError, match="missing from training"):
        select_demonstration_sets(
            _records(10), excluded_ids=[999_999], seeds=[0, 1], examples_per_label=4
        )


def test_demonstration_manifest_round_trip(tmp_path) -> None:
    manifest = build_demonstration_manifest(
        _records(20),
        excluded_ids=[0, 10_000],
        seeds=[0, 1, 2, 3, 4, 5],
        examples_per_label=4,
        dataset_name="stanfordnlp/sst2",
        dataset_revision=None,
        train_fingerprint="fingerprint",
        train_split="train",
        source_split_manifest_sha256="a" * 64,
    )
    path = tmp_path / "demonstrations.json"

    save_demonstration_manifest(manifest, path)

    assert load_demonstration_manifest(path) == manifest


def test_prompt_construction_matches_exact_specification() -> None:
    prompt = build_prompt(
        "The plot was dull and predictable.",
        [
            PromptExample(
                idx=7,
                sentence="The movie was funny and beautifully acted.",
                label=1,
            )
        ],
    )

    assert prompt == (
        "Classify the sentiment of each movie-review sentence as positive or negative.\n\n"
        "Review: The movie was funny and beautifully acted.\n"
        "Sentiment: positive\n\n"
        "Review: The plot was dull and predictable.\n"
        "Sentiment:"
    )


def test_zero_shot_prompt_omits_demonstration_blocks() -> None:
    assert build_prompt("A quiet but moving film.") == (
        "Classify the sentiment of each movie-review sentence as positive or negative.\n\n"
        "Review: A quiet but moving film.\n"
        "Sentiment:"
    )


def test_saved_prefixes_are_nested_and_materialize_in_order() -> None:
    manifest = build_demonstration_manifest(
        _records(20),
        excluded_ids=[0, 10_000],
        seeds=[0, 1, 2, 3, 4, 5],
        examples_per_label=4,
        dataset_name="stanfordnlp/sst2",
        dataset_revision=None,
        train_fingerprint="fingerprint",
        train_split="train",
        source_split_manifest_sha256="a" * 64,
    )
    two = demonstration_prefix(manifest, seed=0, shot_count=2)
    four = demonstration_prefix(manifest, seed=0, shot_count=4)
    assert four[:2] == two

    records = [
        {**record, "sentence": f"sentence {record['idx']}"}
        for record in _records(20)
    ]
    materialized = materialize_demonstrations(records, two)
    assert [example.idx for example in materialized] == [reference.idx for reference in two]
    assert [example.label for example in materialized] == [reference.label for reference in two]
