"""Deterministic SST-2 partitioning and reproducibility manifests."""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LABEL_NAMES: dict[int, str] = {0: "negative", 1: "positive"}
MANIFEST_SCHEMA_VERSION = 1
SAMPLING_METHOD = "sorted_ids_then_seeded_sample_per_label"


@dataclass(frozen=True, order=True)
class ExampleReference:
    """The stable identifier and label needed to reproduce a partition."""

    idx: int
    label: int


@dataclass(frozen=True)
class TrainingPartition:
    """Prompt-development and demonstration-pool references."""

    prompt_development: tuple[ExampleReference, ...]
    demonstration_pool: tuple[ExampleReference, ...]


def _extract_references(
    records: Iterable[Mapping[str, Any]], *, split_name: str
) -> tuple[ExampleReference, ...]:
    references: list[ExampleReference] = []
    seen_ids: set[int] = set()

    for position, record in enumerate(records):
        missing = {"idx", "label", "sentence"}.difference(record)
        if missing:
            missing_fields = ", ".join(sorted(missing))
            raise ValueError(
                f"{split_name} record {position} is missing fields: {missing_fields}"
            )

        idx = record["idx"]
        label = record["label"]
        sentence = record["sentence"]

        if not isinstance(idx, int) or isinstance(idx, bool):
            raise ValueError(f"{split_name} record {position} has a non-integer idx")
        if idx in seen_ids:
            raise ValueError(f"{split_name} contains duplicate idx {idx}")
        if (
            not isinstance(label, int)
            or isinstance(label, bool)
            or label not in LABEL_NAMES
        ):
            raise ValueError(
                f"{split_name} record {idx} has label {label!r}; expected 0 or 1"
            )
        if not isinstance(sentence, str):
            raise ValueError(f"{split_name} record {idx} has a non-string sentence")

        seen_ids.add(idx)
        references.append(ExampleReference(idx=idx, label=label))

    if not references:
        raise ValueError(f"{split_name} must not be empty")

    return tuple(sorted(references))


def partition_training_records(
    records: Iterable[Mapping[str, Any]], *, seed: int, examples_per_label: int
) -> TrainingPartition:
    """Create a deterministic, class-balanced prompt-development partition.

    Candidate identifiers are sorted before sampling, making the result independent
    of input iteration order. Each class uses a seed derived from the declared split
    seed and label.
    """

    if examples_per_label <= 0:
        raise ValueError("examples_per_label must be positive")

    references = _extract_references(records, split_name="train")
    ids_by_label: dict[int, list[int]] = {label: [] for label in LABEL_NAMES}
    labels_by_id = {reference.idx: reference.label for reference in references}

    for reference in references:
        ids_by_label[reference.label].append(reference.idx)

    selected_ids: set[int] = set()
    for label in sorted(LABEL_NAMES):
        candidates = sorted(ids_by_label[label])
        if len(candidates) < examples_per_label:
            raise ValueError(
                f"train label {label} has {len(candidates)} examples; "
                f"need at least {examples_per_label}"
            )
        class_rng = random.Random(f"sst2-prompt-development:{seed}:{label}")
        selected_ids.update(class_rng.sample(candidates, examples_per_label))

    prompt_development = tuple(
        ExampleReference(idx=idx, label=labels_by_id[idx])
        for idx in sorted(selected_ids)
    )
    demonstration_pool = tuple(
        reference for reference in references if reference.idx not in selected_ids
    )

    prompt_ids = {reference.idx for reference in prompt_development}
    pool_ids = {reference.idx for reference in demonstration_pool}
    all_ids = {reference.idx for reference in references}
    if prompt_ids & pool_ids:
        raise AssertionError("prompt-development and demonstration-pool IDs overlap")
    if prompt_ids | pool_ids != all_ids:
        raise AssertionError("training partition does not cover every training ID")

    prompt_counts = Counter(reference.label for reference in prompt_development)
    expected_counts = {label: examples_per_label for label in LABEL_NAMES}
    if dict(prompt_counts) != expected_counts:
        raise AssertionError(
            f"prompt-development class counts {dict(prompt_counts)} "
            f"do not match {expected_counts}"
        )

    return TrainingPartition(
        prompt_development=prompt_development,
        demonstration_pool=demonstration_pool,
    )


def build_split_manifest(
    train_records: Iterable[Mapping[str, Any]],
    validation_records: Iterable[Mapping[str, Any]],
    *,
    dataset_name: str,
    dataset_revision: str | None,
    train_fingerprint: str | None,
    validation_fingerprint: str | None,
    train_split: str,
    validation_split: str,
    seed: int,
    examples_per_label: int,
    expected_train_size: int | None = None,
    expected_validation_size: int | None = None,
    expected_validation_labels: Mapping[int, int] | None = None,
) -> dict[str, Any]:
    """Build and validate the committed split-manifest representation."""

    train_references = _extract_references(train_records, split_name=train_split)
    validation_references = _extract_references(
        validation_records, split_name=validation_split
    )
    partition = partition_training_records(
        (
            {"idx": reference.idx, "label": reference.label, "sentence": "validated"}
            for reference in train_references
        ),
        seed=seed,
        examples_per_label=examples_per_label,
    )

    if expected_train_size is not None and len(train_references) != expected_train_size:
        raise ValueError(
            f"{train_split} contains {len(train_references)} examples; "
            f"expected {expected_train_size}"
        )
    if (
        expected_validation_size is not None
        and len(validation_references) != expected_validation_size
    ):
        raise ValueError(
            f"{validation_split} contains {len(validation_references)} examples; "
            f"expected {expected_validation_size}"
        )

    train_label_counts = Counter(reference.label for reference in train_references)
    validation_label_counts = Counter(
        reference.label for reference in validation_references
    )
    if expected_validation_labels is not None:
        normalized_expected = {
            int(label): int(count)
            for label, count in expected_validation_labels.items()
        }
        if dict(validation_label_counts) != normalized_expected:
            raise ValueError(
                f"{validation_split} label counts {dict(validation_label_counts)} "
                f"do not match {normalized_expected}"
            )

    prompt_counts = Counter(
        reference.label for reference in partition.prompt_development
    )
    pool_counts = Counter(reference.label for reference in partition.demonstration_pool)

    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset": {
            "name": dataset_name,
            "revision": dataset_revision,
            "fingerprints": {
                train_split: train_fingerprint,
                validation_split: validation_fingerprint,
            },
        },
        "label_mapping": {str(label): name for label, name in LABEL_NAMES.items()},
        "source_splits": {
            "train": train_split,
            "validation": validation_split,
        },
        "prompt_development_selection": {
            "seed": seed,
            "sampling_method": SAMPLING_METHOD,
            "examples_per_label": examples_per_label,
        },
        "counts": {
            "train": len(train_references),
            "train_by_label": {
                str(label): train_label_counts[label] for label in LABEL_NAMES
            },
            "prompt_development": len(partition.prompt_development),
            "prompt_development_by_label": {
                str(label): prompt_counts[label] for label in LABEL_NAMES
            },
            "demonstration_pool": len(partition.demonstration_pool),
            "demonstration_pool_by_label": {
                str(label): pool_counts[label] for label in LABEL_NAMES
            },
            "validation": len(validation_references),
            "validation_by_label": {
                str(label): validation_label_counts[label] for label in LABEL_NAMES
            },
        },
        "prompt_development_examples": [
            {
                "split": train_split,
                "idx": reference.idx,
                "label": reference.label,
                "label_name": LABEL_NAMES[reference.label],
            }
            for reference in partition.prompt_development
        ],
        "demonstration_pool": {
            "split": train_split,
            "definition": "all training examples excluding prompt_development_examples",
        },
        "evaluation": {
            "split": validation_split,
            "definition": "complete labeled validation split",
        },
    }
    validate_split_manifest(manifest)
    return manifest


def validate_split_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate methodological invariants represented in a split manifest."""

    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported split manifest schema_version")

    selection = manifest.get("prompt_development_selection")
    counts = manifest.get("counts")
    examples = manifest.get("prompt_development_examples")
    if not isinstance(selection, Mapping):
        raise ValueError("manifest is missing prompt_development_selection")
    if not isinstance(counts, Mapping):
        raise ValueError("manifest is missing counts")
    if not isinstance(examples, list):
        raise ValueError("manifest is missing prompt_development_examples")

    per_label = selection.get("examples_per_label")
    if not isinstance(per_label, int) or per_label <= 0:
        raise ValueError("manifest examples_per_label must be a positive integer")
    expected_total = per_label * len(LABEL_NAMES)
    if counts.get("prompt_development") != expected_total:
        raise ValueError("manifest prompt-development total is inconsistent")
    if len(examples) != expected_total:
        raise ValueError("manifest prompt-development example list is inconsistent")

    example_ids = [example.get("idx") for example in examples]
    if len(example_ids) != len(set(example_ids)):
        raise ValueError("manifest prompt-development IDs are not unique")
    example_counts = Counter(example.get("label") for example in examples)
    if example_counts != Counter({label: per_label for label in LABEL_NAMES}):
        raise ValueError("manifest prompt-development examples are not class-balanced")

    train_total = counts.get("train")
    pool_total = counts.get("demonstration_pool")
    if not isinstance(train_total, int) or not isinstance(pool_total, int):
        raise ValueError("manifest training counts must be integers")
    if pool_total + expected_total != train_total:
        raise ValueError("manifest training partition totals are inconsistent")


def save_split_manifest(manifest: Mapping[str, Any], path: str | Path) -> None:
    """Validate and save a manifest with stable formatting."""

    validate_split_manifest(manifest)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_path.replace(output_path)


def load_split_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate a saved split manifest."""

    with Path(path).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("split manifest root must be an object")
    validate_split_manifest(manifest)
    return manifest


def load_huggingface_dataset(
    dataset_name: str, *, revision: str | None = None
) -> Any:
    """Load a Hugging Face dataset while keeping the dependency optional in tests."""

    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "The 'datasets' package is required. Install requirements.txt first."
        ) from error

    load_kwargs: dict[str, Any] = {}
    if revision is not None:
        load_kwargs["revision"] = revision
    return load_dataset(dataset_name, **load_kwargs)


def build_manifest_from_dataset(
    dataset: Mapping[str, Any],
    *,
    dataset_name: str,
    dataset_revision: str | None,
    train_split: str,
    validation_split: str,
    seed: int,
    examples_per_label: int,
    expected_train_size: int,
    expected_validation_size: int,
    expected_validation_labels: Mapping[int, int],
) -> dict[str, Any]:
    """Build a manifest from a loaded Hugging Face DatasetDict."""

    missing_splits = {train_split, validation_split}.difference(dataset)
    if missing_splits:
        missing = ", ".join(sorted(missing_splits))
        raise ValueError(f"dataset is missing required splits: {missing}")

    train_dataset = dataset[train_split]
    validation_dataset = dataset[validation_split]
    return build_split_manifest(
        train_dataset,
        validation_dataset,
        dataset_name=dataset_name,
        dataset_revision=dataset_revision,
        train_fingerprint=getattr(train_dataset, "_fingerprint", None),
        validation_fingerprint=getattr(validation_dataset, "_fingerprint", None),
        train_split=train_split,
        validation_split=validation_split,
        seed=seed,
        examples_per_label=examples_per_label,
        expected_train_size=expected_train_size,
        expected_validation_size=expected_validation_size,
        expected_validation_labels=expected_validation_labels,
    )
