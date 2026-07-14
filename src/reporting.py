"""Aggregation, paired gaps, and figures for full experiment predictions."""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.metrics import CalibrationSummary, compute_calibration


PRECISION_CONDITIONS = ("bf16", "4bit_nf4")
FULL_SHOT_COUNTS = (0, 1, 2, 4, 8)
DEMONSTRATION_SEEDS = (0, 1, 2, 3, 4, 5)


def _reliability_bins(calibration: CalibrationSummary) -> list[dict[str, Any]]:
    return [
        {
            "index": bin_.index,
            "lower_bound": bin_.lower_bound,
            "upper_bound": bin_.upper_bound,
            "count": bin_.count,
            "mean_confidence": bin_.mean_confidence,
            "empirical_accuracy": bin_.empirical_accuracy,
            "absolute_gap": bin_.absolute_gap,
            "weighted_gap": bin_.weighted_gap,
        }
        for bin_ in calibration.bins
    ]


def compute_condition_metrics(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Compute accuracy, ECE, and bins for every precision/shot/seed run."""

    if not records:
        raise ValueError("prediction records must not be empty")
    groups: dict[tuple[str, int, int | None], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    for record in records:
        key = (
            str(record["precision_condition"]),
            int(record["shot_count"]),
            record["demonstration_seed"],
        )
        groups[key].append(record)

    results: list[dict[str, Any]] = []
    for (precision, shot_count, seed), group in sorted(
        groups.items(),
        key=lambda item: (
            PRECISION_CONDITIONS.index(item[0][0]),
            item[0][1],
            -1 if item[0][2] is None else int(item[0][2]),
        ),
    ):
        correctness = [bool(record["correctness"]) for record in group]
        confidences = [
            float(record["selected_label_confidence"]) for record in group
        ]
        calibration = compute_calibration(confidences, correctness, num_bins=10)
        results.append(
            {
                "precision_condition": precision,
                "shot_count": shot_count,
                "demonstration_seed": seed,
                "num_examples": len(group),
                "accuracy": sum(correctness) / len(correctness),
                "ece_10_bins": calibration.ece,
                "reliability_bins": _reliability_bins(calibration),
            }
        )
    return results


def validate_full_condition_coverage(
    metrics: Sequence[Mapping[str, Any]], *, expected_examples: int
) -> None:
    """Assert the exact 50 run-level metric rows required by the protocol."""

    expected = {
        (precision, shot_count, seed)
        for precision in PRECISION_CONDITIONS
        for shot_count in FULL_SHOT_COUNTS
        for seed in ((None,) if shot_count == 0 else DEMONSTRATION_SEEDS)
    }
    actual = {
        (
            str(metric["precision_condition"]),
            int(metric["shot_count"]),
            metric["demonstration_seed"],
        )
        for metric in metrics
    }
    if actual != expected:
        missing = sorted(expected.difference(actual), key=str)
        extra = sorted(actual.difference(expected), key=str)
        raise ValueError(f"condition coverage mismatch; missing={missing}, extra={extra}")
    if any(int(metric["num_examples"]) != expected_examples for metric in metrics):
        raise ValueError("a condition does not contain the complete evaluation set")


def aggregate_condition_metrics(
    metrics: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate nonzero shots across seeds using sample standard deviation."""

    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for metric in metrics:
        grouped[(str(metric["precision_condition"]), int(metric["shot_count"]))].append(
            metric
        )
    aggregates: list[dict[str, Any]] = []
    for precision in PRECISION_CONDITIONS:
        for shot_count in FULL_SHOT_COUNTS:
            group = grouped[(precision, shot_count)]
            expected_count = 1 if shot_count == 0 else len(DEMONSTRATION_SEEDS)
            if len(group) != expected_count:
                raise ValueError("metric group has an unexpected number of seeds")
            accuracies = [float(metric["accuracy"]) for metric in group]
            eces = [float(metric["ece_10_bins"]) for metric in group]
            aggregates.append(
                {
                    "precision_condition": precision,
                    "shot_count": shot_count,
                    "num_demonstration_selections": expected_count,
                    "mean_accuracy": statistics.fmean(accuracies),
                    "std_accuracy": (
                        None if shot_count == 0 else statistics.stdev(accuracies)
                    ),
                    "mean_ece_10_bins": statistics.fmean(eces),
                    "std_ece_10_bins": (
                        None if shot_count == 0 else statistics.stdev(eces)
                    ),
                    "standard_deviation_definition": (
                        None if shot_count == 0 else "sample standard deviation (n-1)"
                    ),
                }
            )
    return aggregates


def compute_paired_gaps(
    metrics: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compute 4-bit minus BF16 gaps for matched shot/seed conditions."""

    by_key = {
        (
            str(metric["precision_condition"]),
            int(metric["shot_count"]),
            metric["demonstration_seed"],
        ): metric
        for metric in metrics
    }
    paired_rows: list[dict[str, Any]] = []
    for shot_count in FULL_SHOT_COUNTS:
        seeds = (None,) if shot_count == 0 else DEMONSTRATION_SEEDS
        for seed in seeds:
            bf16 = by_key[("bf16", shot_count, seed)]
            quantized = by_key[("4bit_nf4", shot_count, seed)]
            paired_rows.append(
                {
                    "shot_count": shot_count,
                    "demonstration_seed": seed,
                    "accuracy_gap_4bit_minus_bf16": float(quantized["accuracy"])
                    - float(bf16["accuracy"]),
                    "ece_gap_4bit_minus_bf16": float(quantized["ece_10_bins"])
                    - float(bf16["ece_10_bins"]),
                }
            )

    summaries: list[dict[str, Any]] = []
    for shot_count in FULL_SHOT_COUNTS:
        rows = [row for row in paired_rows if row["shot_count"] == shot_count]
        accuracy_gaps = [float(row["accuracy_gap_4bit_minus_bf16"]) for row in rows]
        ece_gaps = [float(row["ece_gap_4bit_minus_bf16"]) for row in rows]
        summaries.append(
            {
                "shot_count": shot_count,
                "num_pairs": len(rows),
                "mean_accuracy_gap_4bit_minus_bf16": statistics.fmean(
                    accuracy_gaps
                ),
                "std_accuracy_gap": (
                    None if shot_count == 0 else statistics.stdev(accuracy_gaps)
                ),
                "mean_ece_gap_4bit_minus_bf16": statistics.fmean(ece_gaps),
                "std_ece_gap": (
                    None if shot_count == 0 else statistics.stdev(ece_gaps)
                ),
            }
        )
    return paired_rows, summaries


def pooled_reliability(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Pool seeds only for visualization, retaining separate metric estimates."""

    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(str(record["precision_condition"]), int(record["shot_count"]))].append(
            record
        )
    pooled: list[dict[str, Any]] = []
    for precision in PRECISION_CONDITIONS:
        for shot_count in FULL_SHOT_COUNTS:
            group = groups[(precision, shot_count)]
            calibration = compute_calibration(
                [float(record["selected_label_confidence"]) for record in group],
                [bool(record["correctness"]) for record in group],
                num_bins=10,
            )
            pooled.append(
                {
                    "precision_condition": precision,
                    "shot_count": shot_count,
                    "num_predictions": len(group),
                    "ece_10_bins": calibration.ece,
                    "reliability_bins": _reliability_bins(calibration),
                    "note": "nonzero-shot rows pool six demonstration seeds for visualization only",
                }
            )
    return pooled


def write_csv(
    rows: Sequence[Mapping[str, Any]], path: Path, *, excluded: Sequence[str] = ()
) -> None:
    """Write a deterministic CSV table, excluding nested fields when requested."""

    if not rows:
        raise ValueError("CSV rows must not be empty")
    excluded_set = set(excluded)
    fieldnames = [key for key in rows[0] if key not in excluded_set]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def generate_figures(
    *,
    aggregates: Sequence[Mapping[str, Any]],
    paired_gap_summary: Sequence[Mapping[str, Any]],
    pooled_bins: Sequence[Mapping[str, Any]],
    output_directory: Path,
) -> list[Path]:
    """Generate final trend, gap, and reliability figures."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_directory.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    labels = {"bf16": "BF16", "4bit_nf4": "4-bit NF4"}
    colors = {"bf16": "#1f77b4", "4bit_nf4": "#d62728"}

    for metric_key, std_key, y_label, filename in (
        ("mean_accuracy", "std_accuracy", "Accuracy", "accuracy_by_shot.png"),
        ("mean_ece_10_bins", "std_ece_10_bins", "ECE (10 bins)", "ece_by_shot.png"),
    ):
        figure, axis = plt.subplots(figsize=(6.4, 4.2))
        for precision in PRECISION_CONDITIONS:
            rows = [row for row in aggregates if row["precision_condition"] == precision]
            errors = [0.0 if row[std_key] is None else float(row[std_key]) for row in rows]
            axis.errorbar(
                [int(row["shot_count"]) for row in rows],
                [float(row[metric_key]) for row in rows],
                yerr=errors,
                marker="o",
                capsize=3,
                label=labels[precision],
                color=colors[precision],
            )
        axis.set_xlabel("Shot count")
        axis.set_ylabel(y_label)
        axis.set_xticks(FULL_SHOT_COUNTS)
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        path = output_directory / filename
        figure.savefig(path, dpi=200)
        plt.close(figure)
        generated.append(path)

    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    for axis, key, label in (
        (axes[0], "mean_accuracy_gap_4bit_minus_bf16", "Accuracy gap"),
        (axes[1], "mean_ece_gap_4bit_minus_bf16", "ECE gap"),
    ):
        axis.axhline(0.0, color="black", linewidth=1)
        axis.plot(
            [int(row["shot_count"]) for row in paired_gap_summary],
            [float(row[key]) for row in paired_gap_summary],
            marker="o",
            color="#9467bd",
        )
        axis.set_xlabel("Shot count")
        axis.set_ylabel(f"{label} (4-bit - BF16)")
        axis.set_xticks(FULL_SHOT_COUNTS)
        axis.grid(alpha=0.25)
    figure.tight_layout()
    gap_path = output_directory / "paired_quantization_gaps.png"
    figure.savefig(gap_path, dpi=200)
    plt.close(figure)
    generated.append(gap_path)

    for row in pooled_bins:
        precision = str(row["precision_condition"])
        shot_count = int(row["shot_count"])
        nonempty = [bin_ for bin_ in row["reliability_bins"] if bin_["count"]]
        figure, axis = plt.subplots(figsize=(4.8, 4.8))
        axis.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="black")
        axis.plot(
            [float(bin_["mean_confidence"]) for bin_ in nonempty],
            [float(bin_["empirical_accuracy"]) for bin_ in nonempty],
            marker="o",
            color=colors[precision],
        )
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel("Mean confidence")
        axis.set_ylabel("Empirical accuracy")
        axis.set_title(f"{labels[precision]}, {shot_count}-shot")
        axis.grid(alpha=0.25)
        figure.tight_layout()
        path = output_directory / f"reliability_{precision}_{shot_count}shot.png"
        figure.savefig(path, dpi=200)
        plt.close(figure)
        generated.append(path)
    return generated
