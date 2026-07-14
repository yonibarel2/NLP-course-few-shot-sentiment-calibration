"""Restricted-verbalizer scoring for causal language models."""

from __future__ import annotations

import math
import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from src.model import LABEL_VERBALIZERS, render_scoring_prefix


@dataclass(frozen=True)
class VerbalizerCandidate:
    """One prompt followed by one complete label verbalizer."""

    label: int
    verbalizer: str
    input_ids: tuple[int, ...]
    prefix_token_count: int
    continuation_ids: tuple[int, ...]


@dataclass(frozen=True)
class LabelProbabilities:
    """Restricted probabilities and underlying summed sequence scores."""

    negative_probability: float
    positive_probability: float
    negative_log_score: float
    positive_log_score: float


def _token_ids(tokenizer: Any, text: str) -> tuple[int, ...]:
    encoded = tokenizer(text, add_special_tokens=False)
    input_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
    if input_ids and isinstance(input_ids[0], list):
        if len(input_ids) != 1:
            raise ValueError("expected exactly one tokenized sequence")
        input_ids = input_ids[0]
    if not isinstance(input_ids, Sequence):
        raise ValueError("tokenizer did not return a token ID sequence")
    return tuple(int(token_id) for token_id in input_ids)


def prepare_verbalizer_candidates(
    tokenizer: Any, prompt: str
) -> tuple[VerbalizerCandidate, ...]:
    """Tokenize both labels after the exact chat-formatted assistant prefix."""

    rendered_prefix = render_scoring_prefix(tokenizer, prompt)
    prefix_ids = _token_ids(tokenizer, rendered_prefix)
    if not prefix_ids:
        raise ValueError("rendered scoring prefix contains no tokens")

    candidates: list[VerbalizerCandidate] = []
    for label in sorted(LABEL_VERBALIZERS):
        verbalizer = LABEL_VERBALIZERS[label]
        full_ids = _token_ids(tokenizer, rendered_prefix + verbalizer)
        if full_ids[: len(prefix_ids)] != prefix_ids:
            raise ValueError(
                f"tokenization of {verbalizer!r} changes the scoring prefix boundary"
            )
        continuation_ids = full_ids[len(prefix_ids) :]
        if not continuation_ids:
            raise ValueError(f"verbalizer {verbalizer!r} contains no continuation tokens")
        candidates.append(
            VerbalizerCandidate(
                label=label,
                verbalizer=verbalizer,
                input_ids=full_ids,
                prefix_token_count=len(prefix_ids),
                continuation_ids=continuation_ids,
            )
        )
    return tuple(candidates)


def summed_candidate_log_probabilities(
    logits: torch.Tensor, candidates: Sequence[VerbalizerCandidate]
) -> torch.Tensor:
    """Sum each candidate's conditional token log-probabilities."""

    if logits.ndim != 3:
        raise ValueError("logits must have shape [candidates, sequence, vocabulary]")
    if logits.shape[0] != len(candidates):
        raise ValueError("logit batch size does not match candidate count")

    scores: list[torch.Tensor] = []
    for row, candidate in enumerate(candidates):
        token_scores: list[torch.Tensor] = []
        for offset, token_id in enumerate(candidate.continuation_ids):
            logit_position = candidate.prefix_token_count + offset - 1
            if logit_position < 0 or logit_position >= logits.shape[1]:
                raise ValueError("candidate continuation position is outside logits")
            if token_id < 0 or token_id >= logits.shape[2]:
                raise ValueError("candidate continuation token is outside vocabulary")
            position_logits = logits[row, logit_position].float()
            token_scores.append(
                position_logits[token_id] - torch.logsumexp(position_logits, dim=-1)
            )
        scores.append(torch.stack(token_scores).sum())
    return torch.stack(scores)


def _pad_candidates(
    candidates: Sequence[VerbalizerCandidate], *, pad_token_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if not candidates:
        raise ValueError("candidate batch must not be empty")
    maximum_length = max(len(candidate.input_ids) for candidate in candidates)
    input_ids = torch.full(
        (len(candidates), maximum_length), pad_token_id, dtype=torch.long
    )
    attention_mask = torch.zeros_like(input_ids)
    for row, candidate in enumerate(candidates):
        length = len(candidate.input_ids)
        input_ids[row, :length] = torch.tensor(candidate.input_ids, dtype=torch.long)
        attention_mask[row, :length] = 1
    return input_ids, attention_mask


def _score_single_token_batch(
    model: Any,
    grouped_candidates: Sequence[Sequence[VerbalizerCandidate]],
    *,
    pad_token_id: int,
    device: torch.device | str,
) -> list[LabelProbabilities]:
    """Use one prefix pass per prompt when both labels are single tokens."""

    prefixes = [
        candidates[0].input_ids[: candidates[0].prefix_token_count]
        for candidates in grouped_candidates
    ]
    if any(
        candidate.input_ids[: candidate.prefix_token_count] != prefix
        for candidates, prefix in zip(grouped_candidates, prefixes, strict=True)
        for candidate in candidates
    ):
        raise ValueError("label candidates do not share an identical scoring prefix")
    maximum_length = max(len(prefix) for prefix in prefixes)
    input_ids = torch.full(
        (len(prefixes), maximum_length), pad_token_id, dtype=torch.long
    )
    attention_mask = torch.zeros_like(input_ids)
    for row, prefix in enumerate(prefixes):
        length = len(prefix)
        input_ids[row, maximum_length - length :] = torch.tensor(
            prefix, dtype=torch.long
        )
        attention_mask[row, maximum_length - length :] = 1

    forward_kwargs: dict[str, Any] = {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
    }
    if "logits_to_keep" in inspect.signature(model.forward).parameters:
        forward_kwargs["logits_to_keep"] = 1
    with torch.inference_mode():
        outputs = model(**forward_kwargs)
    final_logits = outputs.logits[:, -1, :].float()

    results: list[LabelProbabilities] = []
    for row, candidates in enumerate(grouped_candidates):
        if [candidate.label for candidate in candidates] != [0, 1]:
            raise AssertionError("verbalizer candidates must be ordered negative, positive")
        token_ids = [candidate.continuation_ids[0] for candidate in candidates]
        normalizer = torch.logsumexp(final_logits[row], dim=-1)
        pair_scores = torch.stack(
            [final_logits[row, token_id] - normalizer for token_id in token_ids]
        )
        probabilities = torch.softmax(pair_scores, dim=0).detach().cpu()
        pair_scores_cpu = pair_scores.detach().cpu()
        results.append(
            LabelProbabilities(
                negative_probability=float(probabilities[0]),
                positive_probability=float(probabilities[1]),
                negative_log_score=float(pair_scores_cpu[0]),
                positive_log_score=float(pair_scores_cpu[1]),
            )
        )
    return results


def score_prompts(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    batch_size: int,
    device: torch.device | str = "cuda",
) -> list[LabelProbabilities]:
    """Score complete label sequences and normalize only across the two labels."""

    if not prompts:
        raise ValueError("prompts must not be empty")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        raise ValueError("tokenizer must define pad_token_id")

    all_results: list[LabelProbabilities] = []
    for start in range(0, len(prompts), batch_size):
        prompt_batch = prompts[start : start + batch_size]
        grouped_candidates = [
            prepare_verbalizer_candidates(tokenizer, prompt) for prompt in prompt_batch
        ]
        if all(
            len(candidate.continuation_ids) == 1
            for candidates in grouped_candidates
            for candidate in candidates
        ):
            all_results.extend(
                _score_single_token_batch(
                    model,
                    grouped_candidates,
                    pad_token_id=int(pad_token_id),
                    device=device,
                )
            )
            continue
        flat_candidates = [
            candidate for candidates in grouped_candidates for candidate in candidates
        ]
        input_ids, attention_mask = _pad_candidates(
            flat_candidates, pad_token_id=int(pad_token_id)
        )
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        with torch.inference_mode():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        scores = summed_candidate_log_probabilities(outputs.logits, flat_candidates)

        for offset, candidates in enumerate(grouped_candidates):
            if [candidate.label for candidate in candidates] != [0, 1]:
                raise AssertionError("verbalizer candidates must be ordered negative, positive")
            pair_scores = scores[offset * 2 : offset * 2 + 2]
            probabilities = torch.softmax(pair_scores, dim=0).detach().cpu()
            pair_scores_cpu = pair_scores.detach().cpu()
            negative_probability = float(probabilities[0])
            positive_probability = float(probabilities[1])
            if not math.isclose(
                negative_probability + positive_probability,
                1.0,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise AssertionError("restricted label probabilities do not sum to one")
            all_results.append(
                LabelProbabilities(
                    negative_probability=negative_probability,
                    positive_probability=positive_probability,
                    negative_log_score=float(pair_scores_cpu[0]),
                    positive_log_score=float(pair_scores_cpu[1]),
                )
            )

    if len(all_results) != len(prompts):
        raise AssertionError("scoring returned an unexpected number of results")
    return all_results
