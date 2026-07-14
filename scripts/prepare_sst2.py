"""Create the deterministic SST-2 split manifest."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (  # noqa: E402
    build_manifest_from_dataset,
    load_huggingface_dataset,
    save_split_manifest,
)


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "data.yaml"


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"configuration section '{name}' must be a mapping")
    return value


def load_config(path: Path) -> Mapping[str, Any]:
    """Load the data-preparation YAML configuration."""

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return _mapping(config, name="root")


def prepare_from_config(config: Mapping[str, Any], *, output: Path | None) -> Path:
    """Load SST-2, validate it, and save its reproducibility manifest."""

    dataset_config = _mapping(config.get("dataset"), name="dataset")
    prompt_config = _mapping(
        config.get("prompt_development"), name="prompt_development"
    )
    output_config = _mapping(config.get("output"), name="output")

    dataset_name = str(dataset_config["name"])
    revision_value = dataset_config.get("revision")
    dataset_revision = None if revision_value is None else str(revision_value)
    train_split = str(dataset_config["train_split"])
    validation_split = str(dataset_config["validation_split"])
    expected_validation_labels = _mapping(
        dataset_config["expected_validation_labels"],
        name="dataset.expected_validation_labels",
    )

    output_path = output
    if output_path is None:
        configured_path = Path(str(output_config["manifest_path"]))
        output_path = (
            configured_path
            if configured_path.is_absolute()
            else PROJECT_ROOT / configured_path
        )

    dataset = load_huggingface_dataset(dataset_name, revision=dataset_revision)
    manifest = build_manifest_from_dataset(
        dataset,
        dataset_name=dataset_name,
        dataset_revision=dataset_revision,
        train_split=train_split,
        validation_split=validation_split,
        seed=int(prompt_config["seed"]),
        examples_per_label=int(prompt_config["examples_per_label"]),
        expected_train_size=int(dataset_config["expected_train_size"]),
        expected_validation_size=int(dataset_config["expected_validation_size"]),
        expected_validation_labels={
            int(label): int(count)
            for label, count in expected_validation_labels.items()
        },
    )
    save_split_manifest(manifest, output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare deterministic SST-2 split metadata."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Data configuration YAML path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional manifest output override.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    output_path = prepare_from_config(config, output=args.output)
    print(f"Saved validated SST-2 split manifest to {output_path}")


if __name__ == "__main__":
    main()
