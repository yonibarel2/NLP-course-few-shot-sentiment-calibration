"""Compute the SST-2 training-set majority-class baseline."""

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

from src.baseline import (  # noqa: E402
    compute_majority_class_baseline,
    save_majority_baseline,
)
from src.data import load_huggingface_dataset, load_split_manifest  # noqa: E402


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "data.yaml"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "splits" / "sst2_split_manifest.json"


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"configuration section '{name}' must be a mapping")
    return value


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def compute_from_config(
    config_path: Path, *, manifest_path: Path, output_path: Path | None
) -> Path:
    """Load the matching SST-2 revision and save validated baseline statistics."""

    with config_path.open("r", encoding="utf-8") as handle:
        config = _mapping(yaml.safe_load(handle), name="root")
    dataset_config = _mapping(config.get("dataset"), name="dataset")
    output_config = _mapping(config.get("output"), name="output")
    manifest = load_split_manifest(manifest_path)

    dataset_name = str(dataset_config["name"])
    revision_value = dataset_config.get("revision")
    dataset_revision = None if revision_value is None else str(revision_value)
    train_split = str(dataset_config["train_split"])
    validation_split = str(dataset_config["validation_split"])
    if manifest["dataset"]["name"] != dataset_name:
        raise ValueError("split manifest dataset does not match data configuration")
    if manifest["dataset"]["revision"] != dataset_revision:
        raise ValueError("split manifest revision does not match data configuration")

    dataset = load_huggingface_dataset(dataset_name, revision=dataset_revision)
    for split in (train_split, validation_split):
        current_fingerprint = getattr(dataset[split], "_fingerprint", None)
        saved_fingerprint = manifest["dataset"]["fingerprints"].get(split)
        if saved_fingerprint is not None and current_fingerprint != saved_fingerprint:
            raise ValueError(
                f"{split} fingerprint {current_fingerprint!r} does not match "
                f"manifest fingerprint {saved_fingerprint!r}"
            )

    result = compute_majority_class_baseline(
        dataset[train_split],
        dataset[validation_split],
        dataset_name=dataset_name,
        dataset_revision=dataset_revision,
        train_fingerprint=getattr(dataset[train_split], "_fingerprint", None),
        validation_fingerprint=getattr(
            dataset[validation_split], "_fingerprint", None
        ),
        train_split=train_split,
        validation_split=validation_split,
        num_bins=10,
    )
    resolved_output = output_path
    if resolved_output is None:
        resolved_output = _project_path(str(output_config["baseline_path"]))
    save_majority_baseline(result, resolved_output)
    return resolved_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the SST-2 majority-class baseline."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = compute_from_config(
        args.config.resolve(),
        manifest_path=args.manifest.resolve(),
        output_path=args.output,
    )
    print(f"Saved validated majority-class baseline to {output_path}")


if __name__ == "__main__":
    main()
