"""Run the approved paired hierarchical bootstrap on completed predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bootstrap import paired_hierarchical_bootstrap  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "analysis.yaml"
DEFAULT_PREDICTIONS = PROJECT_ROOT / "results" / "raw" / "full_predictions.jsonl"
DEFAULT_FULL_SUMMARY = PROJECT_ROOT / "results" / "tables" / "full_summary.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "tables" / "bootstrap_summary.json"
DEFAULT_FIGURE = PROJECT_ROOT / "results" / "figures" / "bootstrap_interactions.png"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            records.append(value)
    return records


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _generate_figure(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    matplotlib_cache = PROJECT_ROOT / ".cache" / "matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    for axis, metric, label in (
        (axes[0], "accuracy", "Accuracy interaction"),
        (axes[1], "ece_10_bins", "ECE interaction"),
    ):
        selected = [row for row in rows if row["metric"] == metric]
        points = [float(row["point_estimate"]) for row in selected]
        lower = [
            point - float(row["ci_lower"])
            for point, row in zip(points, selected, strict=True)
        ]
        upper = [
            float(row["ci_upper"]) - point
            for point, row in zip(points, selected, strict=True)
        ]
        axis.axhline(0.0, color="black", linewidth=1)
        axis.errorbar(
            [int(row["shot_count"]) for row in selected],
            points,
            yerr=[lower, upper],
            marker="o",
            capsize=4,
            color="#9467bd",
        )
        axis.set_xlabel("Shot count (reference: 0-shot)")
        axis.set_ylabel(f"{label}\n4-bit minus BF16")
        axis.set_xticks([1, 2, 4, 8])
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def run_analysis(
    *,
    config_path: Path,
    predictions_path: Path,
    full_summary_path: Path,
    output_path: Path,
    figure_path: Path,
) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("bootstrap"), dict):
        raise ValueError("analysis config must contain a bootstrap mapping")
    bootstrap_config = config["bootstrap"]
    if bootstrap_config.get("interval") != "percentile":
        raise ValueError("approved analysis requires percentile intervals")
    if (
        bootstrap_config.get("resampling")
        != "paired_crossed_examples_and_demonstration_seeds"
    ):
        raise ValueError("analysis resampling differs from the approved method")

    full_summary = _load_json(full_summary_path)
    if full_summary.get("status") != "passed":
        raise ValueError("bootstrap is blocked until the full experiment passes")
    relative_predictions = predictions_path.relative_to(PROJECT_ROOT).as_posix()
    expected_hash = full_summary.get("outputs", {}).get(relative_predictions)
    actual_hash = _sha256(predictions_path)
    if expected_hash != actual_hash:
        raise ValueError("prediction file hash differs from the validated full run")

    records = _load_jsonl(predictions_path)
    result = paired_hierarchical_bootstrap(
        records,
        num_samples=int(bootstrap_config["samples"]),
        random_seed=int(bootstrap_config["seed"]),
        confidence_level=float(bootstrap_config["confidence_level"]),
    )
    gap_csv = output_path.with_name("bootstrap_gap_intervals.csv")
    interaction_csv = output_path.with_name("bootstrap_interaction_intervals.csv")
    _write_csv(result["gap_intervals"], gap_csv)
    _write_csv(result["shot_effect_interaction_intervals"], interaction_csv)
    _generate_figure(result["shot_effect_interaction_intervals"], figure_path)

    summary = {
        "schema_version": 1,
        "status": "passed",
        "method": {
            "name": "paired hierarchical bootstrap with crossed shared resamples",
            "bootstrap_samples": result["num_bootstrap_samples"],
            "bootstrap_seed": result["random_seed"],
            "confidence_level": result["confidence_level"],
            "interval": "percentile",
            "example_resampling": "one shared with-replacement draw across all conditions",
            "demonstration_seed_resampling": "one shared with-replacement draw across all nonzero shots and both precision conditions",
            "gap_definition": "4-bit metric minus BF16 metric",
            "interaction_definition": "gap at the stated shot count minus the 0-shot gap",
        },
        "validation": {
            "full_experiment_status": "passed",
            "prediction_sha256_matches_full_summary": True,
            "evaluation_examples": result["num_evaluation_examples"],
            "demonstration_seeds": result["num_demonstration_seeds"],
        },
        "gap_intervals": result["gap_intervals"],
        "shot_effect_interaction_intervals": result[
            "shot_effect_interaction_intervals"
        ],
        "inputs": {
            relative_predictions: actual_hash,
            full_summary_path.relative_to(PROJECT_ROOT).as_posix(): _sha256(full_summary_path),
            config_path.relative_to(PROJECT_ROOT).as_posix(): _sha256(config_path),
        },
        "outputs": {
            gap_csv.relative_to(PROJECT_ROOT).as_posix(): _sha256(gap_csv),
            interaction_csv.relative_to(PROJECT_ROOT).as_posix(): _sha256(interaction_csv),
            figure_path.relative_to(PROJECT_ROOT).as_posix(): _sha256(figure_path),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    print(f"Bootstrap analysis passed; saved summary to {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paired bootstrap analysis.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--full-summary", type=Path, default=DEFAULT_FULL_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_analysis(
        config_path=args.config.resolve(),
        predictions_path=args.predictions.resolve(),
        full_summary_path=args.full_summary.resolve(),
        output_path=args.output.resolve(),
        figure_path=args.figure.resolve(),
    )


if __name__ == "__main__":
    main()
