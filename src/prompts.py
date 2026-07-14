"""Deterministic demonstration selection and prompt construction utilities."""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.data import ExampleReference, LABEL_NAMES


DEMONSTRATION_SCHEMA_VERSION = 1
DEMONSTRATION_SAMPLING_METHOD = "sorted_pool_ids_then_seeded_sample_per_label"
SHOT_COUNTS = (0, 1, 2, 4, 8)
TASK_INSTRUCTION = (
    "Classify the sentiment of each movie-review sentence as positive or negative."
)


@dataclass(frozen=True)
class PromptExample:
    """A materialized labeled example used in a prompt."""

    idx: int
    sentence: str
    label: int


def demonstration_prefix(
    manifest: Mapping[str, Any], *, seed: int, shot_count: int
) -> tuple[ExampleReference, ...]:
    """Return the saved prefix for one seed and supported shot count."""

    validate_demonstration_manifest(manifest)
    if shot_count not in SHOT_COUNTS:
        raise ValueError(f"shot_count must be one of {SHOT_COUNTS}")
    if shot_count == 0:
        return ()
    sets = manifest["demonstration_sets"]
    matching = [set_ for set_ in sets if set_["seed"] == seed]
    if len(matching) != 1:
        raise ValueError(f"demonstration seed {seed} is not present exactly once")
    return tuple(
        ExampleReference(idx=int(item["idx"]), label=int(item["label"]))
        for item in matching[0]["demonstrations"][:shot_count]
    )


def materialize_demonstrations(
    train_records: Iterable[Mapping[str, Any]],
    references: Sequence[ExampleReference],
) -> tuple[PromptExample, ...]:
    """Resolve saved demonstration references against the matching train split."""

    requested_ids = {reference.idx for reference in references}
    records_by_id: dict[int, PromptExample] = {}
    for position, record in enumerate(train_records):
        if not requested_ids.intersection({record.get("idx")}):
            continue
        idx = record.get("idx")
        label = record.get("label")
        sentence = record.get("sentence")
        if not isinstance(idx, int) or isinstance(idx, bool):
            raise ValueError(f"train record {position} has an invalid idx")
        if (
            not isinstance(label, int)
            or isinstance(label, bool)
            or label not in LABEL_NAMES
        ):
            raise ValueError(f"train record {idx} has an invalid label")
        if not isinstance(sentence, str):
            raise ValueError(f"train record {idx} has an invalid sentence")
        if idx in records_by_id:
            raise ValueError(f"train contains duplicate requested idx {idx}")
        records_by_id[idx] = PromptExample(idx=idx, sentence=sentence, label=label)

    missing = requested_ids.difference(records_by_id)
    if missing:
        raise ValueError(f"demonstration IDs are missing from training: {sorted(missing)}")
    materialized: list[PromptExample] = []
    for reference in references:
        example = records_by_id[reference.idx]
        if example.label != reference.label:
            raise ValueError(
                f"demonstration {reference.idx} label does not match training"
            )
        materialized.append(example)
    return tuple(materialized)


def build_prompt(
    evaluation_sentence: str,
    demonstrations: Sequence[PromptExample] = (),
) -> str:
    """Construct the exact plain-text classification prompt."""

    if not isinstance(evaluation_sentence, str) or not evaluation_sentence.strip():
        raise ValueError("evaluation sentence must be a non-empty string")
    blocks = [TASK_INSTRUCTION]
    for demonstration in demonstrations:
        if (
            not isinstance(demonstration.label, int)
            or isinstance(demonstration.label, bool)
            or demonstration.label not in LABEL_NAMES
        ):
            raise ValueError(
                f"demonstration {demonstration.idx} has an invalid label"
            )
        if (
            not isinstance(demonstration.sentence, str)
            or not demonstration.sentence.strip()
        ):
            raise ValueError(
                f"demonstration {demonstration.idx} has an invalid sentence"
            )
        blocks.append(
            f"Review: {demonstration.sentence}\n"
            f"Sentiment: {LABEL_NAMES[demonstration.label]}"
        )
    blocks.append(f"Review: {evaluation_sentence}\nSentiment:")
    return "\n\n".join(blocks)


def select_demonstration_sets(
    train_records: Iterable[Mapping[str, Any]],
    *,
    excluded_ids: Iterable[int],
    seeds: Sequence[int],
    examples_per_label: int = 4,
) -> dict[int, tuple[ExampleReference, ...]]:
    """Select ordered, nested-ready demonstrations from the training pool."""

    if examples_per_label != 4:
        raise ValueError("the experiment requires four demonstrations per label")
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("demonstration seeds must be non-empty and unique")
    if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds):
        raise ValueError("demonstration seeds must be integers")

    excluded = set(excluded_ids)
    ids_by_label: dict[int, list[int]] = {label: [] for label in LABEL_NAMES}
    labels_by_id: dict[int, int] = {}
    for position, record in enumerate(train_records):
        if "idx" not in record or "label" not in record:
            raise ValueError(f"train record {position} is missing idx or label")
        idx = record["idx"]
        label = record["label"]
        if not isinstance(idx, int) or isinstance(idx, bool):
            raise ValueError(f"train record {position} has a non-integer idx")
        if idx in labels_by_id:
            raise ValueError(f"train contains duplicate idx {idx}")
        if (
            not isinstance(label, int)
            or isinstance(label, bool)
            or label not in LABEL_NAMES
        ):
            raise ValueError(f"train record {idx} has label {label!r}; expected 0 or 1")
        labels_by_id[idx] = label
        if idx not in excluded:
            ids_by_label[label].append(idx)

    missing_excluded = excluded.difference(labels_by_id)
    if missing_excluded:
        raise ValueError("excluded prompt-development IDs are missing from training")
    for label, candidates in ids_by_label.items():
        candidates.sort()
        if len(candidates) < examples_per_label:
            raise ValueError(
                f"demonstration pool label {label} has {len(candidates)} examples; "
                f"need at least {examples_per_label}"
            )

    selections: dict[int, tuple[ExampleReference, ...]] = {}
    for seed in seeds:
        sampled_by_label: dict[int, list[int]] = {}
        for label in sorted(LABEL_NAMES):
            rng = random.Random(f"sst2-demonstrations:{seed}:{label}")
            sampled_by_label[label] = rng.sample(
                ids_by_label[label], examples_per_label
            )

        first_label = seed % len(LABEL_NAMES)
        label_order = tuple(
            label
            for _ in range(examples_per_label)
            for label in (first_label, 1 - first_label)
        )
        offsets = {label: 0 for label in LABEL_NAMES}
        ordered: list[ExampleReference] = []
        for label in label_order:
            idx = sampled_by_label[label][offsets[label]]
            offsets[label] += 1
            ordered.append(ExampleReference(idx=idx, label=label))

        references = tuple(ordered)
        if any(reference.idx in excluded for reference in references):
            raise AssertionError("a demonstration overlaps prompt development")
        if len({reference.idx for reference in references}) != len(references):
            raise AssertionError("a demonstration set contains duplicate IDs")
        for prefix_size in (2, 4, examples_per_label * len(LABEL_NAMES)):
            prefix_counts = Counter(
                reference.label for reference in references[:prefix_size]
            )
            expected = prefix_size // len(LABEL_NAMES)
            if prefix_counts != Counter({label: expected for label in LABEL_NAMES}):
                raise AssertionError(
                    f"seed {seed} has an unbalanced {prefix_size}-shot prefix"
                )
        selections[seed] = references

    first_labels = [selections[seed][0].label for seed in seeds]
    if any(left == right for left, right in zip(first_labels, first_labels[1:])):
        raise ValueError("the first demonstration class does not alternate by seed")
    if Counter(first_labels) != Counter(
        {label: len(seeds) // len(LABEL_NAMES) for label in LABEL_NAMES}
    ):
        raise ValueError("first demonstration classes are not evenly distributed")
    return selections


def build_demonstration_manifest(
    train_records: Iterable[Mapping[str, Any]],
    *,
    excluded_ids: Iterable[int],
    seeds: Sequence[int],
    examples_per_label: int,
    dataset_name: str,
    dataset_revision: str | None,
    train_fingerprint: str | None,
    train_split: str,
    source_split_manifest_sha256: str,
) -> dict[str, Any]:
    """Build the saved representation of all ordered demonstration sets."""

    selections = select_demonstration_sets(
        train_records,
        excluded_ids=excluded_ids,
        seeds=seeds,
        examples_per_label=examples_per_label,
    )
    manifest: dict[str, Any] = {
        "schema_version": DEMONSTRATION_SCHEMA_VERSION,
        "dataset": {
            "name": dataset_name,
            "revision": dataset_revision,
            "train_fingerprint": train_fingerprint,
        },
        "source": {
            "split": train_split,
            "pool": "training split excluding prompt-development IDs",
            "split_manifest_sha256": source_split_manifest_sha256,
        },
        "selection": {
            "seeds": list(seeds),
            "sampling_method": DEMONSTRATION_SAMPLING_METHOD,
            "examples_per_label": examples_per_label,
            "shot_counts": list(SHOT_COUNTS),
            "first_label_rule": "seed modulo 2; 0=negative and 1=positive",
        },
        "demonstration_sets": [
            {
                "seed": seed,
                "demonstrations": [
                    {
                        "position": position,
                        "split": train_split,
                        "idx": reference.idx,
                        "label": reference.label,
                        "label_name": LABEL_NAMES[reference.label],
                    }
                    for position, reference in enumerate(selections[seed], start=1)
                ],
            }
            for seed in seeds
        ],
    }
    validate_demonstration_manifest(manifest)
    return manifest


def validate_demonstration_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate nesting, balance, uniqueness, and alternating first labels."""

    if manifest.get("schema_version") != DEMONSTRATION_SCHEMA_VERSION:
        raise ValueError("unsupported demonstration manifest schema_version")
    selection = manifest.get("selection")
    sets = manifest.get("demonstration_sets")
    if not isinstance(selection, Mapping) or not isinstance(sets, list):
        raise ValueError("demonstration manifest is missing required sections")
    seeds = selection.get("seeds")
    examples_per_label = selection.get("examples_per_label")
    if not isinstance(seeds, list) or not isinstance(examples_per_label, int):
        raise ValueError("demonstration selection metadata is invalid")
    if [set_.get("seed") for set_ in sets] != seeds:
        raise ValueError("demonstration sets do not match the declared seeds")

    expected_size = examples_per_label * len(LABEL_NAMES)
    first_labels: list[int] = []
    for set_ in sets:
        demonstrations = set_.get("demonstrations")
        if not isinstance(demonstrations, list) or len(demonstrations) != expected_size:
            raise ValueError("demonstration set has an invalid size")
        if [item.get("position") for item in demonstrations] != list(
            range(1, expected_size + 1)
        ):
            raise ValueError("demonstration positions are invalid")
        identifiers = [item.get("idx") for item in demonstrations]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("demonstration set contains duplicate IDs")
        for prefix_size in (2, 4, expected_size):
            counts = Counter(
                item.get("label") for item in demonstrations[:prefix_size]
            )
            per_label = prefix_size // len(LABEL_NAMES)
            if counts != Counter({label: per_label for label in LABEL_NAMES}):
                raise ValueError(
                    f"demonstration set has an unbalanced {prefix_size}-shot prefix"
                )
        first_labels.append(demonstrations[0].get("label"))

    if any(left == right for left, right in zip(first_labels, first_labels[1:])):
        raise ValueError("first demonstration labels do not alternate")
    expected_first = len(seeds) // len(LABEL_NAMES)
    if Counter(first_labels) != Counter(
        {label: expected_first for label in LABEL_NAMES}
    ):
        raise ValueError("first demonstration labels are not class-balanced")


def save_demonstration_manifest(
    manifest: Mapping[str, Any], path: str | Path
) -> None:
    """Save a validated demonstration manifest with deterministic formatting."""

    validate_demonstration_manifest(manifest)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_path.replace(output_path)


def load_demonstration_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate a saved demonstration manifest."""

    with Path(path).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("demonstration manifest root must be an object")
    validate_demonstration_manifest(manifest)
    return manifest
