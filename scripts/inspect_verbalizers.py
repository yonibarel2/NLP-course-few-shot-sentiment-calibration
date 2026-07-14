"""Inspect Qwen label verbalizers after the exact prompt and chat template."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from importlib.metadata import version
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_huggingface_dataset, load_split_manifest  # noqa: E402
from src.model import (  # noqa: E402
    TOKENIZATION_SCHEMA_VERSION,
    inspect_prompt_verbalizers,
    validate_tokenization_report,
)
from src.prompts import (  # noqa: E402
    build_prompt,
    demonstration_prefix,
    load_demonstration_manifest,
    materialize_demonstrations,
)


DEFAULT_DATA_CONFIG = PROJECT_ROOT / "configs" / "data.yaml"
DEFAULT_HIGH_PRECISION_CONFIG = PROJECT_ROOT / "configs" / "high_precision.yaml"
DEFAULT_QUANTIZED_CONFIG = PROJECT_ROOT / "configs" / "quantized_4bit.yaml"
DEFAULT_SPLIT_MANIFEST = PROJECT_ROOT / "data" / "splits" / "sst2_split_manifest.json"
DEFAULT_DEMONSTRATION_MANIFEST = (
    PROJECT_ROOT / "data" / "splits" / "sst2_demonstration_sets.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "tables" / "verbalizer_tokenization.json"


def _load_yaml(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, Mapping):
        raise ValueError(f"configuration {path} must contain a mapping")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save_report(report: Mapping[str, Any], path: Path) -> None:
    validate_tokenization_report(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_path.replace(path)


def inspect(
    *,
    data_config_path: Path,
    high_precision_config_path: Path,
    quantized_config_path: Path,
    split_manifest_path: Path,
    demonstration_manifest_path: Path,
    output_path: Path,
) -> Path:
    """Build representative 0/2-shot prompts and inspect both continuations."""

    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "The 'transformers' package is required for tokenizer inspection."
        ) from error

    data_config = _load_yaml(data_config_path)
    high_precision = _load_yaml(high_precision_config_path)
    quantized = _load_yaml(quantized_config_path)
    high_model = high_precision["model"]
    quantized_model = quantized["model"]
    if high_model != quantized_model:
        raise ValueError("precision conditions do not declare the same model checkpoint")
    model_name = str(high_model["name"])
    revision_value = high_model.get("revision")
    requested_revision = None if revision_value is None else str(revision_value)

    dataset_config = data_config["dataset"]
    dataset_name = str(dataset_config["name"])
    dataset_revision_value = dataset_config.get("revision")
    dataset_revision = (
        None if dataset_revision_value is None else str(dataset_revision_value)
    )
    train_split = str(dataset_config["train_split"])
    split_manifest = load_split_manifest(split_manifest_path)
    demonstration_manifest = load_demonstration_manifest(
        demonstration_manifest_path
    )
    dataset = load_huggingface_dataset(dataset_name, revision=dataset_revision)
    train_dataset = dataset[train_split]
    fingerprint = getattr(train_dataset, "_fingerprint", None)
    if fingerprint != split_manifest["dataset"]["fingerprints"][train_split]:
        raise ValueError("training fingerprint does not match the split manifest")

    development_id = int(split_manifest["prompt_development_examples"][0]["idx"])
    development_rows = [row for row in train_dataset if int(row["idx"]) == development_id]
    if len(development_rows) != 1:
        raise ValueError("could not resolve the development query exactly once")
    evaluation_sentence = str(development_rows[0]["sentence"])
    two_references = demonstration_prefix(
        demonstration_manifest, seed=0, shot_count=2
    )
    two_demonstrations = materialize_demonstrations(train_dataset, two_references)
    prompts = {
        "0-shot": build_prompt(evaluation_sentence),
        "2-shot-seed-0": build_prompt(evaluation_sentence, two_demonstrations),
    }

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=requested_revision,
        use_fast=True,
    )
    init_kwargs = getattr(tokenizer, "init_kwargs", {})
    resolved_revision = init_kwargs.get("_commit_hash") or getattr(
        tokenizer, "_commit_hash", None
    )
    if resolved_revision is None:
        from huggingface_hub import scan_cache_dir

        ref_name = requested_revision or "main"
        cached_revisions = [
            cached_revision.commit_hash
            for repository in scan_cache_dir().repos
            if repository.repo_id == model_name
            for cached_revision in repository.revisions
            if ref_name in cached_revision.refs
        ]
        if len(cached_revisions) != 1:
            raise ValueError(
                f"could not resolve exactly one cached tokenizer revision for {ref_name!r}"
            )
        resolved_revision = cached_revisions[0]
    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str) or not chat_template:
        raise ValueError("the Qwen tokenizer does not provide a chat template")

    contexts: list[dict[str, Any]] = []
    for context_name, prompt in prompts.items():
        details = inspect_prompt_verbalizers(tokenizer, prompt)
        details.update(
            {
                "name": context_name,
                "evaluation_source": {
                    "split": train_split,
                    "idx": development_id,
                    "role": "reserved prompt-development query",
                },
                "demonstration_seed": 0 if context_name.startswith("2-shot") else None,
                "shot_count": 2 if context_name.startswith("2-shot") else 0,
            }
        )
        contexts.append(details)

    report: dict[str, Any] = {
        "schema_version": TOKENIZATION_SCHEMA_VERSION,
        "model_name": model_name,
        "requested_revision": requested_revision,
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "name_or_path": tokenizer.name_or_path,
            "is_fast": bool(getattr(tokenizer, "is_fast", False)),
            "vocab_size": int(tokenizer.vocab_size),
            "resolved_revision": resolved_revision,
            "chat_template_sha256": hashlib.sha256(
                chat_template.encode("utf-8")
            ).hexdigest(),
        },
        "software": {"transformers": version("transformers")},
        "inputs": {
            "split_manifest_sha256": _sha256(split_manifest_path),
            "demonstration_manifest_sha256": _sha256(
                demonstration_manifest_path
            ),
        },
        "contexts": contexts,
        "requires_sequence_scoring": any(
            details["token_count"] != 1
            for context in contexts
            for details in context["verbalizers"].values()
        ),
    }
    _save_report(report, output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect Qwen sentiment-label verbalizer tokenization."
    )
    parser.add_argument("--data-config", type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument(
        "--high-precision-config", type=Path, default=DEFAULT_HIGH_PRECISION_CONFIG
    )
    parser.add_argument("--quantized-config", type=Path, default=DEFAULT_QUANTIZED_CONFIG)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument(
        "--demonstration-manifest",
        type=Path,
        default=DEFAULT_DEMONSTRATION_MANIFEST,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = inspect(
        data_config_path=args.data_config.resolve(),
        high_precision_config_path=args.high_precision_config.resolve(),
        quantized_config_path=args.quantized_config.resolve(),
        split_manifest_path=args.split_manifest.resolve(),
        demonstration_manifest_path=args.demonstration_manifest.resolve(),
        output_path=args.output.resolve(),
    )
    print(f"Saved verbalizer tokenization report to {output_path}")


if __name__ == "__main__":
    main()
