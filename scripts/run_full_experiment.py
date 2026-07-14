"""Run the complete paired BF16 versus 4-bit NF4 SST-2 experiment."""

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
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_pilot import (  # noqa: E402
    DEFAULT_DATA_CONFIG,
    DEFAULT_DEMONSTRATION_MANIFEST,
    DEFAULT_HIGH_PRECISION_CONFIG,
    DEFAULT_QUANTIZED_CONFIG,
    DEFAULT_SPLIT_MANIFEST,
    DEFAULT_TOKENIZATION_REPORT,
    _gpu_metadata,
    _load_json,
    _load_yaml,
    _package_versions,
    _prediction_record,
    _resolve_experiment_revision,
    _save_json,
    _save_jsonl,
    _sha256_file,
)
from src.data import load_huggingface_dataset, load_split_manifest  # noqa: E402
from src.inference import LabelProbabilities, score_prompts  # noqa: E402
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
from src.reporting import (  # noqa: E402
    DEMONSTRATION_SEEDS,
    FULL_SHOT_COUNTS,
    aggregate_condition_metrics,
    compute_condition_metrics,
    compute_paired_gaps,
    generate_figures,
    pooled_reliability,
    validate_full_condition_coverage,
    write_csv,
)


DEFAULT_PREDICTIONS = PROJECT_ROOT / "results" / "raw" / "full_predictions.jsonl"
DEFAULT_SUMMARY = PROJECT_ROOT / "results" / "tables" / "full_summary.json"
DEFAULT_CONDITION_METRICS = (
    PROJECT_ROOT / "results" / "tables" / "full_condition_metrics.json"
)
DEFAULT_FIGURE_DIRECTORY = PROJECT_ROOT / "results" / "figures"
EXPECTED_VALIDATION_SIZE = 872


def _prepare_full_cases(
    *,
    dataset: Mapping[str, Any],
    train_split: str,
    validation_split: str,
    demonstration_manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    train_dataset = dataset[train_split]
    validation_dataset = dataset[validation_split]
    if len(validation_dataset) != EXPECTED_VALIDATION_SIZE:
        raise ValueError("full experiment requires all 872 validation examples")

    cases: list[dict[str, Any]] = []
    for shot_count in FULL_SHOT_COUNTS:
        seeds: Sequence[int | None] = (
            (None,) if shot_count == 0 else DEMONSTRATION_SEEDS
        )
        for seed in seeds:
            if seed is None:
                demonstrations = ()
                demonstration_identifiers: list[int] = []
            else:
                references = demonstration_prefix(
                    demonstration_manifest,
                    seed=seed,
                    shot_count=shot_count,
                )
                demonstrations = materialize_demonstrations(train_dataset, references)
                demonstration_identifiers = [reference.idx for reference in references]
            for position in range(len(validation_dataset)):
                row = validation_dataset[position]
                prompt = build_prompt(str(row["sentence"]), demonstrations)
                cases.append(
                    {
                        "shot_count": shot_count,
                        "demonstration_seed": seed,
                        "demonstration_identifiers": demonstration_identifiers,
                        "evaluation_example_identifier": int(row["idx"]),
                        "gold_label": int(row["label"]),
                        "prompt": prompt,
                        "prompt_sha256": hashlib.sha256(
                            prompt.encode("utf-8")
                        ).hexdigest(),
                    }
                )
    expected_cases = EXPECTED_VALIDATION_SIZE * (1 + 4 * len(DEMONSTRATION_SEEDS))
    if len(cases) != expected_cases:
        raise AssertionError("full case construction produced an unexpected count")
    return cases


def _score_with_progress(
    *,
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    batch_size: int,
    precision_condition: str,
    progress_chunk_size: int = 512,
) -> list[LabelProbabilities]:
    results: list[LabelProbabilities] = []
    for start in range(0, len(prompts), progress_chunk_size):
        chunk = prompts[start : start + progress_chunk_size]
        results.extend(
            score_prompts(
                model,
                tokenizer,
                chunk,
                batch_size=batch_size,
                device="cuda",
            )
        )
        completed = min(start + len(chunk), len(prompts))
        print(
            f"{precision_condition}: scored {completed}/{len(prompts)} prompts",
            flush=True,
        )
    return results


def _write_condition_checkpoint(
    records: Sequence[Mapping[str, Any]],
    *,
    predictions_path: Path,
    precision_condition: str,
) -> Path:
    checkpoint = predictions_path.with_name(
        f"{predictions_path.stem}_{precision_condition}{predictions_path.suffix}"
    )
    _save_jsonl(records, checkpoint)
    return checkpoint


def _validate_prediction_coverage(records: Sequence[Mapping[str, Any]]) -> None:
    expected_per_precision = EXPECTED_VALIDATION_SIZE * (
        1 + 4 * len(DEMONSTRATION_SEEDS)
    )
    expected_total = expected_per_precision * 2
    if len(records) != expected_total:
        raise ValueError(
            f"expected {expected_total} predictions, found {len(records)}"
        )
    keys = [
        (
            str(record["precision_condition"]),
            int(record["shot_count"]),
            record["demonstration_seed"],
            int(record["evaluation_example_identifier"]),
        )
        for record in records
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("full predictions contain duplicate condition/example keys")

    prompt_hashes: dict[tuple[int, int | None, int], set[str]] = defaultdict(set)
    for record in records:
        prompt_hashes[
            (
                int(record["shot_count"]),
                record["demonstration_seed"],
                int(record["evaluation_example_identifier"]),
            )
        ].add(str(record["prompt_sha256"]))
    if any(len(hashes) != 1 for hashes in prompt_hashes.values()):
        raise ValueError("BF16 and 4-bit conditions did not use identical prompts")


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_full_experiment(
    *,
    data_config_path: Path,
    high_precision_config_path: Path,
    quantized_config_path: Path,
    split_manifest_path: Path,
    demonstration_manifest_path: Path,
    tokenization_report_path: Path,
    pilot_summary_path: Path,
    predictions_path: Path,
    summary_path: Path,
    condition_metrics_path: Path,
    figure_directory: Path,
    batch_size: int,
) -> Path:
    """Run all 43,600 paired predictions and generate final artifacts."""

    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    pilot_summary = _load_json(pilot_summary_path)
    if pilot_summary.get("status") != "passed":
        raise ValueError("full experiment is blocked until the pilot passes")

    data_config = _load_yaml(data_config_path)
    high_config = _load_yaml(high_precision_config_path)
    quantized_config = _load_yaml(quantized_config_path)
    if high_config.get("model") != quantized_config.get("model"):
        raise ValueError("precision conditions do not declare the same model")
    if dict(high_config.get("precision", {})) != {
        "condition_name": "bf16",
        "quantized": False,
        "torch_dtype": "bfloat16",
    }:
        raise ValueError("BF16 configuration differs from the approved protocol")
    if dict(quantized_config.get("precision", {})) != {
        "condition_name": "4bit_nf4",
        "quantized": True,
        "load_in_4bit": True,
        "quantization_type": "nf4",
        "compute_dtype": "bfloat16",
        "use_double_quantization": False,
    }:
        raise ValueError("4-bit configuration differs from the approved protocol")

    model_config = high_config["model"]
    model_name = str(model_config["name"])
    requested_revision = str(model_config["revision"])
    tokenization_report = _load_json(tokenization_report_path)
    validate_tokenization_report(tokenization_report)
    revision = _resolve_experiment_revision(
        model_name=model_name,
        requested_revision=requested_revision,
        tokenization_report=tokenization_report,
    )
    if revision != pilot_summary.get("model", {}).get("revision"):
        raise ValueError("full experiment revision differs from successful pilot")

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
        if (
            getattr(dataset[split_name], "_fingerprint", None)
            != split_manifest["dataset"]["fingerprints"][split_name]
        ):
            raise ValueError(f"{split_name} fingerprint differs from split manifest")
    cases = _prepare_full_cases(
        dataset=dataset,
        train_split=train_split,
        validation_split=validation_split,
        demonstration_manifest=demonstration_manifest,
    )
    prompts = [str(case["prompt"]) for case in cases]
    tokenizer = load_tokenizer(model_name, revision=revision)

    import torch

    hardware = _gpu_metadata()
    all_records: list[dict[str, Any]] = []
    condition_metadata: list[dict[str, Any]] = []
    checkpoint_paths: list[Path] = []
    for precision_condition, quantized in (("bf16", False), ("4bit_nf4", True)):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        model = load_causal_lm(
            model_name,
            revision=revision,
            quantized=quantized,
        )
        results = _score_with_progress(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            batch_size=batch_size,
            precision_condition=precision_condition,
        )
        condition_records: list[dict[str, Any]] = []
        for case, result in zip(cases, results, strict=True):
            record = _prediction_record(
                model_name=model_name,
                revision=revision,
                precision_condition=precision_condition,
                case=case,
                result=result,
            )
            record["demonstration_identifiers"] = list(
                case["demonstration_identifiers"]
            )
            condition_records.append(record)
        elapsed = time.perf_counter() - started
        checkpoint_paths.append(
            _write_condition_checkpoint(
                condition_records,
                predictions_path=predictions_path,
                precision_condition=precision_condition,
            )
        )
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

    _validate_prediction_coverage(all_records)
    _save_jsonl(all_records, predictions_path)
    condition_metrics = compute_condition_metrics(all_records)
    validate_full_condition_coverage(
        condition_metrics, expected_examples=EXPECTED_VALIDATION_SIZE
    )
    aggregates = aggregate_condition_metrics(condition_metrics)
    paired_rows, paired_gap_summary = compute_paired_gaps(condition_metrics)
    pooled_bins = pooled_reliability(all_records)

    _save_json({"conditions": condition_metrics}, condition_metrics_path)
    aggregate_csv = summary_path.with_name("full_aggregate_metrics.csv")
    condition_csv = summary_path.with_name("full_condition_metrics.csv")
    paired_csv = summary_path.with_name("paired_seed_gaps.csv")
    paired_summary_csv = summary_path.with_name("paired_gap_summary.csv")
    pooled_bins_path = summary_path.with_name("pooled_reliability_bins.json")
    write_csv(condition_metrics, condition_csv, excluded=("reliability_bins",))
    write_csv(aggregates, aggregate_csv)
    write_csv(paired_rows, paired_csv)
    write_csv(paired_gap_summary, paired_summary_csv)
    _save_json({"pooled_conditions": pooled_bins}, pooled_bins_path)
    figures = generate_figures(
        aggregates=aggregates,
        paired_gap_summary=paired_gap_summary,
        pooled_bins=pooled_bins,
        output_directory=figure_directory,
    )

    output_files = [
        predictions_path,
        condition_metrics_path,
        aggregate_csv,
        condition_csv,
        paired_csv,
        paired_summary_csv,
        pooled_bins_path,
        *figures,
    ]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "passed",
        "model": {
            "name": model_name,
            "revision": revision,
            "tokenizer_revision": revision,
        },
        "experiment": {
            "validation_examples": EXPECTED_VALIDATION_SIZE,
            "shot_counts": list(FULL_SHOT_COUNTS),
            "demonstration_seeds": list(DEMONSTRATION_SEEDS),
            "batch_size": batch_size,
            "total_predictions": len(all_records),
            "ece_bins": 10,
            "quantization_gap_definition": "4-bit metric minus BF16 metric",
        },
        "validation_checks": {
            "pilot_passed_before_full_run": True,
            "complete_condition_coverage": True,
            "complete_validation_set_per_condition": True,
            "probabilities_sum_to_one": True,
            "predictions_match_larger_probability": True,
            "identical_prompts_across_precision_conditions": True,
            "paired_metric_comparison": True,
        },
        "aggregate_metrics": aggregates,
        "paired_gap_summary": paired_gap_summary,
        "conditions": condition_metadata,
        "hardware": hardware,
        "software": {
            "python": platform.python_version(),
            "packages": _package_versions(),
        },
        "git_commit": _git_revision(),
        "inputs": {
            "pilot_summary_sha256": _sha256_file(pilot_summary_path),
            "split_manifest_sha256": _sha256_file(split_manifest_path),
            "demonstration_manifest_sha256": _sha256_file(
                demonstration_manifest_path
            ),
            "tokenization_report_sha256": _sha256_file(
                tokenization_report_path
            ),
        },
        "condition_checkpoints": [
            str(path.relative_to(PROJECT_ROOT)) for path in checkpoint_paths
        ],
        "outputs": {
            str(path.relative_to(PROJECT_ROOT)): _sha256_file(path)
            for path in output_files
        },
    }
    _save_json(summary, summary_path)
    print(f"Full experiment passed; saved summary to {summary_path}", flush=True)
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full paired experiment.")
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
    parser.add_argument(
        "--pilot-summary",
        type=Path,
        default=PROJECT_ROOT / "results" / "tables" / "pilot_summary.json",
    )
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--condition-metrics", type=Path, default=DEFAULT_CONDITION_METRICS
    )
    parser.add_argument(
        "--figure-directory", type=Path, default=DEFAULT_FIGURE_DIRECTORY
    )
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_full_experiment(
        data_config_path=args.data_config.resolve(),
        high_precision_config_path=args.high_precision_config.resolve(),
        quantized_config_path=args.quantized_config.resolve(),
        split_manifest_path=args.split_manifest.resolve(),
        demonstration_manifest_path=args.demonstration_manifest.resolve(),
        tokenization_report_path=args.tokenization_report.resolve(),
        pilot_summary_path=args.pilot_summary.resolve(),
        predictions_path=args.predictions.resolve(),
        summary_path=args.summary.resolve(),
        condition_metrics_path=args.condition_metrics.resolve(),
        figure_directory=args.figure_directory.resolve(),
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
