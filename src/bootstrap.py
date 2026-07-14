"""Paired hierarchical bootstrap for the final quantization comparison."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from src.reporting import DEMONSTRATION_SEEDS, FULL_SHOT_COUNTS, PRECISION_CONDITIONS


METRICS = ("accuracy", "ece_10_bins")


def _ece(confidences: np.ndarray, correctness: np.ndarray) -> float:
    """Compute 10-bin equal-width ECE for one resampled condition."""

    bin_indices = np.minimum((confidences * 10).astype(np.int64), 9)
    total = confidences.size
    ece = 0.0
    for bin_index in range(10):
        mask = bin_indices == bin_index
        count = int(mask.sum())
        if count:
            ece += count / total * abs(
                float(confidences[mask].mean()) - float(correctness[mask].mean())
            )
    return ece


def _prepare_arrays(
    records: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[tuple[str, int, int | None], tuple[np.ndarray, np.ndarray]],
    tuple[int, ...],
]:
    """Validate records and index confidence/correctness arrays by condition."""

    if not records:
        raise ValueError("prediction records must not be empty")
    grouped: dict[
        tuple[str, int, int | None], list[Mapping[str, Any]]
    ] = defaultdict(list)
    for record in records:
        grouped[
            (
                str(record["precision_condition"]),
                int(record["shot_count"]),
                record["demonstration_seed"],
            )
        ].append(record)

    expected_keys = {
        (precision, shot_count, seed)
        for precision in PRECISION_CONDITIONS
        for shot_count in FULL_SHOT_COUNTS
        for seed in ((None,) if shot_count == 0 else DEMONSTRATION_SEEDS)
    }
    if set(grouped) != expected_keys:
        missing = sorted(expected_keys.difference(grouped), key=str)
        extra = sorted(set(grouped).difference(expected_keys), key=str)
        raise ValueError(f"condition coverage mismatch; missing={missing}, extra={extra}")

    arrays: dict[
        tuple[str, int, int | None], tuple[np.ndarray, np.ndarray]
    ] = {}
    reference_identifiers: tuple[int, ...] | None = None
    for key, group in grouped.items():
        ordered = sorted(group, key=lambda row: int(row["evaluation_example_identifier"]))
        identifiers = tuple(int(row["evaluation_example_identifier"]) for row in ordered)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError(f"condition {key} contains duplicate evaluation identifiers")
        if reference_identifiers is None:
            reference_identifiers = identifiers
        elif identifiers != reference_identifiers:
            raise ValueError("evaluation identifiers are not identical across conditions")
        confidences = np.asarray(
            [float(row["selected_label_confidence"]) for row in ordered],
            dtype=np.float64,
        )
        correctness = np.asarray(
            [bool(row["correctness"]) for row in ordered], dtype=np.float64
        )
        if not np.all(np.isfinite(confidences)) or np.any(
            (confidences < 0.0) | (confidences > 1.0)
        ):
            raise ValueError(f"condition {key} contains invalid confidences")
        arrays[key] = (confidences, correctness)

    if reference_identifiers is None:
        raise AssertionError("validated records unexpectedly produced no identifiers")
    return arrays, reference_identifiers


def _condition_metrics(
    arrays: Mapping[tuple[str, int, int | None], tuple[np.ndarray, np.ndarray]],
    *,
    precision: str,
    shot_count: int,
    example_indices: np.ndarray,
    seed_indices: np.ndarray,
) -> tuple[float, float]:
    seeds: Sequence[int | None]
    if shot_count == 0:
        seeds = (None,)
    else:
        seeds = tuple(DEMONSTRATION_SEEDS[int(index)] for index in seed_indices)

    accuracies: list[float] = []
    eces: list[float] = []
    for seed in seeds:
        confidences, correctness = arrays[(precision, shot_count, seed)]
        sampled_confidences = confidences[example_indices]
        sampled_correctness = correctness[example_indices]
        accuracies.append(float(sampled_correctness.mean()))
        eces.append(_ece(sampled_confidences, sampled_correctness))
    return float(np.mean(accuracies)), float(np.mean(eces))


def _paired_gaps(
    arrays: Mapping[tuple[str, int, int | None], tuple[np.ndarray, np.ndarray]],
    *,
    example_indices: np.ndarray,
    seed_indices: np.ndarray,
) -> np.ndarray:
    """Return shot-by-metric gaps, always defined as 4-bit minus BF16."""

    gaps = np.empty((len(FULL_SHOT_COUNTS), len(METRICS)), dtype=np.float64)
    for shot_position, shot_count in enumerate(FULL_SHOT_COUNTS):
        bf16 = _condition_metrics(
            arrays,
            precision="bf16",
            shot_count=shot_count,
            example_indices=example_indices,
            seed_indices=seed_indices,
        )
        quantized = _condition_metrics(
            arrays,
            precision="4bit_nf4",
            shot_count=shot_count,
            example_indices=example_indices,
            seed_indices=seed_indices,
        )
        gaps[shot_position] = np.asarray(quantized) - np.asarray(bf16)
    return gaps


def _interval_row(
    values: np.ndarray,
    *,
    point_estimate: float,
    confidence_level: float,
) -> dict[str, Any]:
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "point_estimate": point_estimate,
        "bootstrap_standard_error": float(values.std(ddof=1)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "ci_excludes_zero": bool(lower > 0.0 or upper < 0.0),
    }


def paired_hierarchical_bootstrap(
    records: Sequence[Mapping[str, Any]],
    *,
    num_samples: int = 1_000,
    random_seed: int = 42,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Estimate paired gap and shot-effect uncertainty with shared resamples.

    Each replicate resamples validation examples once and demonstration seeds once.
    Those same draws are applied across every precision and shot condition, preserving
    the experiment's crossed pairing and nested demonstration-set identities.
    """

    if not isinstance(num_samples, int) or isinstance(num_samples, bool) or num_samples < 2:
        raise ValueError("num_samples must be an integer of at least two")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")

    arrays, example_identifiers = _prepare_arrays(records)
    rng = np.random.default_rng(random_seed)
    all_examples = np.arange(len(example_identifiers), dtype=np.int64)
    all_seeds = np.arange(len(DEMONSTRATION_SEEDS), dtype=np.int64)
    observed_gaps = _paired_gaps(
        arrays, example_indices=all_examples, seed_indices=all_seeds
    )
    bootstrap_gaps = np.empty(
        (num_samples, len(FULL_SHOT_COUNTS), len(METRICS)), dtype=np.float64
    )
    for sample_index in range(num_samples):
        example_indices = rng.integers(
            0, len(example_identifiers), size=len(example_identifiers)
        )
        seed_indices = rng.integers(
            0, len(DEMONSTRATION_SEEDS), size=len(DEMONSTRATION_SEEDS)
        )
        bootstrap_gaps[sample_index] = _paired_gaps(
            arrays,
            example_indices=example_indices,
            seed_indices=seed_indices,
        )

    gap_intervals: list[dict[str, Any]] = []
    interaction_intervals: list[dict[str, Any]] = []
    for shot_position, shot_count in enumerate(FULL_SHOT_COUNTS):
        for metric_position, metric in enumerate(METRICS):
            gap_intervals.append(
                {
                    "shot_count": shot_count,
                    "metric": metric,
                    "estimand": "4bit_minus_bf16",
                    **_interval_row(
                        bootstrap_gaps[:, shot_position, metric_position],
                        point_estimate=float(observed_gaps[shot_position, metric_position]),
                        confidence_level=confidence_level,
                    ),
                }
            )
            if shot_count != 0:
                interactions = (
                    bootstrap_gaps[:, shot_position, metric_position]
                    - bootstrap_gaps[:, 0, metric_position]
                )
                point_interaction = (
                    observed_gaps[shot_position, metric_position]
                    - observed_gaps[0, metric_position]
                )
                interaction_intervals.append(
                    {
                        "shot_count": shot_count,
                        "reference_shot_count": 0,
                        "metric": metric,
                        "estimand": "(4bit_minus_bf16)_shot_minus_(4bit_minus_bf16)_0shot",
                        **_interval_row(
                            interactions,
                            point_estimate=float(point_interaction),
                            confidence_level=confidence_level,
                        ),
                    }
                )

    return {
        "num_bootstrap_samples": num_samples,
        "random_seed": random_seed,
        "confidence_level": confidence_level,
        "num_evaluation_examples": len(example_identifiers),
        "num_demonstration_seeds": len(DEMONSTRATION_SEEDS),
        "gap_intervals": gap_intervals,
        "shot_effect_interaction_intervals": interaction_intervals,
    }
