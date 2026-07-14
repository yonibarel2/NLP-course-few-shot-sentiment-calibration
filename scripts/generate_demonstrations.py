"""Generate the six deterministic SST-2 demonstration selections."""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_huggingface_dataset, load_split_manifest  # noqa: E402
from src.prompts import (  # noqa: E402
    build_demonstration_manifest,
    save_demonstration_manifest,
)


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "data.yaml"
DEFAULT_SPLIT_MANIFEST = PROJECT_ROOT / "data" / "splits" / "sst2_split_manifest.json"


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"configuration section '{name}' must be a mapping")
    return value


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def generate_from_config(
    config_path: Path, *, split_manifest_path: Path, output_path: Path | None
) -> Path:
    """Load the matching training split and save all demonstration sets."""

    with config_path.open("r", encoding="utf-8") as handle:
        config = _mapping(yaml.safe_load(handle), name="root")
    dataset_config = _mapping(config.get("dataset"), name="dataset")
    demonstration_config = _mapping(
        config.get("demonstrations"), name="demonstrations"
    )
    output_config = _mapping(config.get("output"), name="output")
    split_manifest = load_split_manifest(split_manifest_path)

    dataset_name = str(dataset_config["name"])
    revision_value = dataset_config.get("revision")
    dataset_revision = None if revision_value is None else str(revision_value)
    train_split = str(dataset_config["train_split"])
    seeds = [int(seed) for seed in demonstration_config["seeds"]]
    examples_per_label = int(demonstration_config["examples_per_label"])
    if seeds != [0, 1, 2, 3, 4, 5] or examples_per_label != 4:
        raise ValueError(
            "experiment specification requires seeds 0-5 and four examples per label"
        )
    if split_manifest["dataset"]["name"] != dataset_name:
        raise ValueError("split manifest dataset does not match data configuration")
    if split_manifest["dataset"]["revision"] != dataset_revision:
        raise ValueError("split manifest revision does not match data configuration")

    dataset = load_huggingface_dataset(dataset_name, revision=dataset_revision)
    train_dataset = dataset[train_split]
    current_fingerprint = getattr(train_dataset, "_fingerprint", None)
    saved_fingerprint = split_manifest["dataset"]["fingerprints"].get(train_split)
    if saved_fingerprint is not None and current_fingerprint != saved_fingerprint:
        raise ValueError("training fingerprint does not match the split manifest")

    excluded_ids = [
        int(example["idx"])
        for example in split_manifest["prompt_development_examples"]
    ]
    source_hash = hashlib.sha256(split_manifest_path.read_bytes()).hexdigest()
    manifest = build_demonstration_manifest(
        train_dataset,
        excluded_ids=excluded_ids,
        seeds=seeds,
        examples_per_label=examples_per_label,
        dataset_name=dataset_name,
        dataset_revision=dataset_revision,
        train_fingerprint=current_fingerprint,
        train_split=train_split,
        source_split_manifest_sha256=source_hash,
    )
    resolved_output = output_path
    if resolved_output is None:
        resolved_output = _project_path(
            str(output_config["demonstration_manifest_path"])
        )
    save_demonstration_manifest(manifest, resolved_output)
    return resolved_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic nested SST-2 demonstrations."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = generate_from_config(
        args.config.resolve(),
        split_manifest_path=args.split_manifest.resolve(),
        output_path=args.output,
    )
    print(f"Saved validated demonstration manifest to {output_path}")


if __name__ == "__main__":
    main()
