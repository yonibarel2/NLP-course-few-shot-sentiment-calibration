"""Generate vector figures used by the ACL final report."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES = PROJECT_ROOT / "results" / "tables"
OUTPUT = PROJECT_ROOT / "report" / "figures"
SHOTS = (0, 1, 2, 4, 8)
PRECISIONS = ("bf16", "4bit_nf4")
LABELS = {"bf16": "BF16", "4bit_nf4": "4-bit NF4"}
COLORS = {"bf16": "#0072B2", "4bit_nf4": "#D55E00"}
MARKERS = {"bf16": "o", "4bit_nf4": "s"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _performance_figure(aggregates: list[dict[str, str]]) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(6.30, 2.45))
    settings = (
        ("mean_accuracy", "std_accuracy", "Accuracy", (0.80, 0.96)),
        ("mean_ece_10_bins", "std_ece_10_bins", "ECE (10 bins; lower is better)", (0.0, 0.15)),
    )
    for axis, (metric, spread, ylabel, ylim) in zip(axes, settings, strict=True):
        for precision in PRECISIONS:
            rows = [row for row in aggregates if row["precision_condition"] == precision]
            rows.sort(key=lambda row: int(row["shot_count"]))
            errors = [float(row[spread]) if row[spread] else 0.0 for row in rows]
            axis.errorbar(
                [int(row["shot_count"]) for row in rows],
                [float(row[metric]) for row in rows],
                yerr=errors,
                color=COLORS[precision],
                marker=MARKERS[precision],
                markersize=4,
                linewidth=1.4,
                capsize=2.5,
                label=LABELS[precision],
            )
        axis.set_xlabel("Number of demonstrations")
        axis.set_ylabel(ylabel)
        axis.set_xticks(SHOTS)
        axis.set_ylim(*ylim)
        axis.grid(alpha=0.22, linewidth=0.6)
    axes[0].legend(frameon=False, loc="lower right")
    figure.tight_layout(w_pad=1.8)
    path = OUTPUT / "performance_by_shot.pdf"
    figure.savefig(path)
    plt.close(figure)
    return path


def _interaction_figure(interactions: list[dict[str, str]]) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(6.30, 2.45))
    for axis, metric, ylabel in (
        (axes[0], "accuracy", "Accuracy interaction"),
        (axes[1], "ece_10_bins", "ECE interaction"),
    ):
        rows = [row for row in interactions if row["metric"] == metric]
        rows.sort(key=lambda row: int(row["shot_count"]))
        values = [float(row["point_estimate"]) for row in rows]
        lower = [value - float(row["ci_lower"]) for value, row in zip(values, rows, strict=True)]
        upper = [float(row["ci_upper"]) - value for value, row in zip(values, rows, strict=True)]
        axis.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        axis.errorbar(
            [int(row["shot_count"]) for row in rows],
            values,
            yerr=[lower, upper],
            fmt="o",
            markersize=4,
            capsize=2.5,
            color="#6A3D9A",
        )
        axis.set_xlabel("Number of demonstrations")
        axis.set_ylabel(ylabel)
        axis.set_xticks((1, 2, 4, 8))
        axis.grid(alpha=0.22, linewidth=0.6)
    figure.tight_layout(w_pad=1.8)
    path = OUTPUT / "bootstrap_interactions.pdf"
    figure.savefig(path)
    plt.close(figure)
    return path


def _reliability_panel(
    rows: list[dict[str, Any]], shots: tuple[int, ...], filename: str
) -> Path:
    figure, axes = plt.subplots(
        2,
        len(shots),
        figsize=(6.30, 4.15 if len(shots) == 3 else 3.45),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    by_key = {
        (str(row["precision_condition"]), int(row["shot_count"])): row
        for row in rows
    }
    for row_index, precision in enumerate(PRECISIONS):
        for column_index, shot in enumerate(shots):
            axis = axes[row_index][column_index]
            condition = by_key[(precision, shot)]
            bins = [item for item in condition["reliability_bins"] if item["count"]]
            axis.plot([0.5, 1.0], [0.5, 1.0], "--", color="#555555", linewidth=0.8)
            axis.plot(
                [float(item["mean_confidence"]) for item in bins],
                [float(item["empirical_accuracy"]) for item in bins],
                color=COLORS[precision],
                marker=MARKERS[precision],
                markersize=3.2,
                linewidth=1.1,
            )
            axis.set_xlim(0.5, 1.0)
            axis.set_ylim(0.0, 1.0)
            axis.grid(alpha=0.18, linewidth=0.5)
            axis.set_title(f"{LABELS[precision]}, {shot}-shot")
            if row_index == 1:
                axis.set_xlabel("Confidence")
            if column_index == 0:
                axis.set_ylabel("Empirical accuracy")
    figure.tight_layout(w_pad=0.8, h_pad=1.1)
    path = OUTPUT / filename
    figure.savefig(path)
    plt.close(figure)
    return path


def main() -> None:
    _configure_style()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    aggregates = _read_csv(TABLES / "full_aggregate_metrics.csv")
    interactions = _read_csv(TABLES / "bootstrap_interaction_intervals.csv")
    reliability = _read_json(TABLES / "pooled_reliability_bins.json")[
        "pooled_conditions"
    ]
    paths = [
        _performance_figure(aggregates),
        _interaction_figure(interactions),
        _reliability_panel(reliability, (0, 2, 8), "reliability_selected.pdf"),
        _reliability_panel(reliability, SHOTS, "reliability_all.pdf"),
    ]
    for path in paths:
        print(path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
