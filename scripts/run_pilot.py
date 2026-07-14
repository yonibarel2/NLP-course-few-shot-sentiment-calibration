"""Run and validate the approved BF16/4-bit SST-2 pilot experiment."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_huggingface_dataset, load_split_manifest  # noqa: E402
from src.inference import LabelProbabilities, score_prompts  # noqa: E402
from src.metrics import compute_calibration  # noqa: E402
from src.model import (  # noqa: E402
    load_causal_lm,
    load_tokenizer,
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
DEFAULT_TOKENIZATION_REPORT = (
    PROJECT_ROOT / "results" / "tables" / "verbalizer_tokenization.json"
)
DEFAULT_PREDICTIONS = PROJECT_ROOT / "results" / "raw" / "pilot_predictions.jsonl"
DEFAULT_SUMMARY = PROJECT_ROOT / "results" / "tables" / "pilot_summary.json"
PILOT_SHOTS = (0, 2)
PILOT_SEED = 0


def _load_yaml(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, Mapping):
        raise ValueError(f"configuration {path} must contain a mapping")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON document {path} must contain an object")
    return value


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package_versions() -> dict[str, str]:
    packages = (
        "torch",
        "transformers",
        "datasets",
        "huggingface-hub",
        "accelerate",
        "bitsandbytes",
        "numpy",
        "pyyaml",
    )
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _gpu_metadata() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("pilot requires a CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("selected GPU does not report native BF16 support")
    properties = torch.cuda.get_device_properties(0)
    metadata: dict[str, Any] = {
        "name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "total_memory_bytes": int(properties.total_memory),
        "cuda_runtime": torch.version.cuda,
        "bf16_supported": True,
    }
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        metadata["driver_version"] = result.stdout.strip().splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        metadata["driver_version"] = None
    return metadata


def _resolve_experiment_revision(
    *,
    model_name: str,
    requested_revision: str | None,
    tokenization_report: Mapping[str, Any],
) -> str:
    tokenizer_metadata = tokenization_report.get("tokenizer")
    if not isinstance(tokenizer_metadata, Mapping):
        raise ValueError("tokenization report has no tokenizer metadata")
    inspected_revision = tokenizer_metadata.get("resolved_revision")
    if not isinstance(inspected_revision, str) or not inspected_revision:
        raise ValueError("tokenization report has no resolved tokenizer revision")
    if requested_revision is not None and requested_revision != inspected_revision:
        raise ValueError(
            "configured revision differs from the inspected tokenizer revision"
        )

    from huggingface_hub import HfApi

    resolved = HfApi().model_info(
        model_name,
        revision=requested_revision or inspected_revision,
    ).sha
    if not isinstance(resolved, str) or not resolved:
        raise ValueError("Hugging Face did not return an immutable model revision")
    if resolved != inspected_revision:
        raise ValueError("model and inspected tokenizer revisions do not match")
    return resolved


def _prepare_cases(
    *,
    dataset: Mapping[str, Any],
    train_split: str,
    validation_split: str,
    demonstration_manifest: Mapping[str, Any],
    pilot_size: int,
) -> list[dict[str, Any]]:
    train_dataset = dataset[train_split]
    validation_dataset = dataset[validation_split]
    if pilot_size > len(validation_dataset):
        raise ValueError("pilot size exceeds the validation split")
    demonstration_references = demonstration_prefix(
        demonstration_manifest,
        seed=PILOT_SEED,
        shot_count=2,
    )
    demonstrations = materialize_demonstrations(
        train_dataset, demonstration_references
    )

    cases: list[dict[str, Any]] = []
    for shot_count in PILOT_SHOTS:
        prompt_demonstrations = () if shot_count == 0 else demonstrations
        for position in range(pilot_size):
            row = validation_dataset[position]
            prompt = build_prompt(str(row["sentence"]), prompt_demonstrations)
            cases.append(
                {
                    "shot_count": shot_count,
                    "demonstration_seed": None if shot_count == 0 else PILOT_SEED,
                    "evaluation_example_identifier": int(row["idx"]),
                    "gold_label": int(row["label"]),
                    "prompt": prompt,
                    "prompt_sha256": _sha256_text(prompt),
                }
            )
    return cases


def _prediction_record(
    *,
    model_name: str,
    revision: str,
    precision_condition: str,
    case: Mapping[str, Any],
    result: LabelProbabilities,
) -> dict[str, Any]:
    predicted_label = int(
        result.positive_probability > result.negative_probability
    )
    confidence = max(result.negative_probability, result.positive_probability)
    gold_label = int(case["gold_label"])
    record = {
        "model_name": model_name,
        "model_revision": revision,
        "tokenizer_revision": revision,
        "precision_condition": precision_condition,
        "shot_count": int(case["shot_count"]),
        "demonstration_seed": case["demonstration_seed"],
        "evaluation_example_identifier": int(
            case["evaluation_example_identifier"]
        ),
        "gold_label": gold_label,
        "predicted_label": predicted_label,
        "negative_probability": result.negative_probability,
        "positive_probability": result.positive_probability,
        "selected_label_confidence": confidence,
        "correctness": predicted_label == gold_label,
        "negative_log_score": result.negative_log_score,
        "positive_log_score": result.positive_log_score,
        "prompt_sha256": str(case["prompt_sha256"]),
    }
    _validate_prediction(record)
    return record


def _validate_prediction(record: Mapping[str, Any]) -> None:
    required = {
        "model_name",
        "model_revision",
        "tokenizer_revision",
        "precision_condition",
        "shot_count",
        "demonstration_seed",
        "evaluation_example_identifier",
        "gold_label",
        "predicted_label",
        "negative_probability",
        "positive_probability",
        "selected_label_confidence",
        "correctness",
        "prompt_sha256",
    }
    missing = required.difference(record)
    if missing:
        raise ValueError(f"prediction is missing fields: {sorted(missing)}")
    negative = float(record["negative_probability"])
    positive = float(record["positive_probability"])
    if abs(negative + positive - 1.0) > 1e-6:
        raise ValueError("prediction probabilities do not sum to one")
    expected_prediction = int(positive > negative)
    if record["predicted_label"] != expected_prediction:
        raise ValueError("predicted label is not the larger label probability")
    if abs(float(record["selected_label_confidence"]) - max(negative, positive)) > 1e-7:
        raise ValueError("selected-label confidence is inconsistent")
    expected_correctness = expected_prediction == int(record["gold_label"])
    if record["correctness"] is not expected_correctness:
        raise ValueError("prediction correctness is inconsistent")


def _aggregate(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(str(record["precision_condition"]), int(record["shot_count"]))].append(
            record
        )

    summaries: list[dict[str, Any]] = []
    for (precision, shot_count), group in sorted(groups.items()):
        correctness = [bool(record["correctness"]) for record in group]
        confidences = [
            float(record["selected_label_confidence"]) for record in group
        ]
        calibration = compute_calibration(confidences, correctness, num_bins=10)
        summaries.append(
            {
                "precision_condition": precision,
                "shot_count": shot_count,
                "demonstration_seed": None if shot_count == 0 else PILOT_SEED,
                "num_examples": len(group),
                "accuracy": sum(correctness) / len(correctness),
                "ece_10_bins": calibration.ece,
            }
        )
    return summaries


def _save_jsonl(records: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    temporary.replace(path)


def _save_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_pilot(
    *,
    data_config_path: Path,
    high_precision_config_path: Path,
    quantized_config_path: Path,
    split_manifest_path: Path,
    demonstration_manifest_path: Path,
    tokenization_report_path: Path,
    predictions_path: Path,
    summary_path: Path,
    pilot_size: int,
    batch_size: int,
) -> Path:
    """Execute both precision conditions over the fixed pilot cases."""

    if pilot_size != 20:
        raise ValueError("the approved pilot uses exactly 20 validation examples")
    data_config = _load_yaml(data_config_path)
    high_config = _load_yaml(high_precision_config_path)
    quantized_config = _load_yaml(quantized_config_path)
    if high_config.get("model") != quantized_config.get("model"):
        raise ValueError("precision conditions must declare the same model")
    expected_high_precision = {
        "condition_name": "bf16",
        "quantized": False,
        "torch_dtype": "bfloat16",
    }
    if dict(high_config.get("precision", {})) != expected_high_precision:
        raise ValueError("BF16 configuration differs from the approved protocol")
    quantization = quantized_config.get("precision", {})
    expected_quantization = {
        "condition_name": "4bit_nf4",
        "quantized": True,
        "load_in_4bit": True,
        "quantization_type": "nf4",
        "compute_dtype": "bfloat16",
        "use_double_quantization": False,
    }
    if dict(quantization) != expected_quantization:
        raise ValueError("4-bit configuration differs from the approved NF4 protocol")

    model_config = high_config["model"]
    model_name = str(model_config["name"])
    requested_value = model_config.get("revision")
    requested_revision = None if requested_value is None else str(requested_value)
    tokenization_report = _load_json(tokenization_report_path)
    validate_tokenization_report(tokenization_report)
    if tokenization_report.get("model_name") != model_name:
        raise ValueError("tokenization report model does not match configuration")
    revision = _resolve_experiment_revision(
        model_name=model_name,
        requested_revision=requested_revision,
        tokenization_report=tokenization_report,
    )

    data_section = data_config["dataset"]
    dataset_name = str(data_section["name"])
    dataset_revision_value = data_section.get("revision")
    dataset_revision = (
        None if dataset_revision_value is None else str(dataset_revision_value)
    )
    train_split = str(data_section["train_split"])
    validation_split = str(data_section["validation_split"])
    split_manifest = load_split_manifest(split_manifest_path)
    demonstration_manifest = load_demonstration_manifest(
        demonstration_manifest_path
    )
    dataset = load_huggingface_dataset(dataset_name, revision=dataset_revision)
    for split_name in (train_split, validation_split):
        current = getattr(dataset[split_name], "_fingerprint", None)
        expected = split_manifest["dataset"]["fingerprints"][split_name]
        if current != expected:
            raise ValueError(f"{split_name} fingerprint differs from split manifest")
    cases = _prepare_cases(
        dataset=dataset,
        train_split=train_split,
        validation_split=validation_split,
        demonstration_manifest=demonstration_manifest,
        pilot_size=pilot_size,
    )
    prompts = [str(case["prompt"]) for case in cases]

    tokenizer = load_tokenizer(model_name, revision=revision)
    import torch

    hardware = _gpu_metadata()
    all_records: list[dict[str, Any]] = []
    condition_metadata: list[dict[str, Any]] = []
    for precision_condition, quantized in (("bf16", False), ("4bit_nf4", True)):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        model = load_causal_lm(
            model_name,
            revision=revision,
            quantized=quantized,
        )
        results = score_prompts(
            model,
            tokenizer,
            prompts,
            batch_size=batch_size,
            device="cuda",
        )
        elapsed = time.perf_counter() - started
        condition_records = [
            _prediction_record(
                model_name=model_name,
                revision=revision,
                precision_condition=precision_condition,
                case=case,
                result=result,
            )
            for case, result in zip(cases, results, strict=True)
        ]
        all_records.extend(condition_records)
        condition_metadata.append(
            {
                "precision_condition": precision_condition,
                "quantized": quantized,
                "elapsed_seconds_including_model_load": elapsed,
                "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            }
        )
        del results, model
        gc.collect()
        torch.cuda.empty_cache()

    expected_records = len(cases) * 2
    if len(all_records) != expected_records:
        raise AssertionError("pilot produced an unexpected record count")
    prompt_hashes: dict[tuple[int, int | None, int], set[str]] = defaultdict(set)
    for record in all_records:
        key = (
            int(record["shot_count"]),
            record["demonstration_seed"],
            int(record["evaluation_example_identifier"]),
        )
        prompt_hashes[key].add(str(record["prompt_sha256"]))
    if any(len(hashes) != 1 for hashes in prompt_hashes.values()):
        raise AssertionError("precision conditions did not use identical prompts")

    _save_jsonl(all_records, predictions_path)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "passed",
        "purpose": "pipeline validation only; not final experimental results",
        "model": {
            "name": model_name,
            "revision": revision,
            "tokenizer_revision": revision,
        },
        "pilot": {
            "validation_examples": pilot_size,
            "validation_example_identifiers": [
                int(dataset[validation_split][position]["idx"])
                for position in range(pilot_size)
            ],
            "shot_counts": list(PILOT_SHOTS),
            "demonstration_seed": PILOT_SEED,
            "batch_size": batch_size,
            "total_predictions": len(all_records),
        },
        "validation_checks": {
            "probabilities_sum_to_one": True,
            "predictions_match_larger_probability": True,
            "identical_prompts_across_precision_conditions": True,
            "required_output_fields_present": True,
            "full_sequence_verbalizer_scoring": True,
        },
        "metrics_for_pipeline_check_only": _aggregate(all_records),
        "conditions": condition_metadata,
        "hardware": hardware,
        "software": {
            "python": platform.python_version(),
            "packages": _package_versions(),
        },
        "inputs": {
            "split_manifest_sha256": _sha256_file(split_manifest_path),
            "demonstration_manifest_sha256": _sha256_file(
                demonstration_manifest_path
            ),
            "tokenization_report_sha256": _sha256_file(
                tokenization_report_path
            ),
        },
        "outputs": {
            "predictions": str(predictions_path.relative_to(PROJECT_ROOT)),
            "prediction_sha256": _sha256_file(predictions_path),
        },
    }
    _save_json(summary, summary_path)
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the approved GPU pilot.")
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
    parser.add_argument(
        "--tokenization-report", type=Path, default=DEFAULT_TOKENIZATION_REPORT
    )
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--pilot-size", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = run_pilot(
        data_config_path=args.data_config.resolve(),
        high_precision_config_path=args.high_precision_config.resolve(),
        quantized_config_path=args.quantized_config.resolve(),
        split_manifest_path=args.split_manifest.resolve(),
        demonstration_manifest_path=args.demonstration_manifest.resolve(),
        tokenization_report_path=args.tokenization_report.resolve(),
        predictions_path=args.predictions.resolve(),
        summary_path=args.summary.resolve(),
        pilot_size=args.pilot_size,
        batch_size=args.batch_size,
    )
    print(f"Pilot passed; saved summary to {summary_path}")


if __name__ == "__main__":
    main()
