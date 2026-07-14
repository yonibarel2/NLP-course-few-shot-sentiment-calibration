"""Tests for full-sequence restricted verbalizer scoring."""

from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from src.inference import (
    VerbalizerCandidate,
    prepare_verbalizer_candidates,
    score_prompts,
    summed_candidate_log_probabilities,
)


class CharacterTokenizer:
    chat_template = "fake-template"
    pad_token_id = 0

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert not tokenize
        assert add_generation_prompt
        return f"<{messages[0]['content']}>:"

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        assert not add_special_tokens
        return {"input_ids": [ord(character) for character in text]}


def test_candidates_preserve_complete_multitoken_verbalizers() -> None:
    candidates = prepare_verbalizer_candidates(CharacterTokenizer(), "review")

    assert [candidate.label for candidate in candidates] == [0, 1]
    assert candidates[0].continuation_ids == tuple(map(ord, "negative"))
    assert candidates[1].continuation_ids == tuple(map(ord, "positive"))
    assert all(
        candidate.input_ids[-len(candidate.continuation_ids) :]
        == candidate.continuation_ids
        for candidate in candidates
    )


def test_sequence_score_sums_each_conditional_token_log_probability() -> None:
    candidates = (
        VerbalizerCandidate(
            label=0,
            verbalizer="negative",
            input_ids=(7, 8, 2, 3),
            prefix_token_count=2,
            continuation_ids=(2, 3),
        ),
        VerbalizerCandidate(
            label=1,
            verbalizer="positive",
            input_ids=(7, 8, 4),
            prefix_token_count=2,
            continuation_ids=(4,),
        ),
    )
    logits = torch.zeros((2, 4, 6), dtype=torch.float32)
    logits[0, 1, 2] = 4.0
    logits[0, 2, 3] = 3.0
    logits[1, 1, 4] = 2.0

    scores = summed_candidate_log_probabilities(logits, candidates)
    log_probs = torch.log_softmax(logits, dim=-1)

    assert torch.isclose(scores[0], log_probs[0, 1, 2] + log_probs[0, 2, 3])
    assert torch.isclose(scores[1], log_probs[1, 1, 4])


class SingleTokenTokenizer:
    chat_template = "fake-template"
    pad_token_id = 0

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert not tokenize
        assert add_generation_prompt
        return f"<{messages[0]['content']}>:"

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        assert not add_special_tokens
        if text.endswith("negative"):
            return {"input_ids": [7, 8, 2]}
        if text.endswith("positive"):
            return {"input_ids": [7, 8, 3]}
        return {"input_ids": [7, 8]}


class FinalLogitsModel:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
    ) -> object:
        assert logits_to_keep == 1
        assert input_ids.shape == attention_mask.shape
        self.batch_sizes.append(input_ids.shape[0])
        logits = torch.zeros((input_ids.shape[0], 1, 5), dtype=torch.float32)
        logits[:, 0, 2] = 1.0
        logits[:, 0, 3] = 3.0
        return type("Output", (), {"logits": logits})()

    __call__ = forward


def test_single_token_fast_path_uses_one_prefix_pass_per_prompt() -> None:
    model = FinalLogitsModel()

    results = score_prompts(
        model,
        SingleTokenTokenizer(),
        ["first", "second", "third"],
        batch_size=2,
        device="cpu",
    )

    assert model.batch_sizes == [2, 1]
    assert all(result.positive_probability > result.negative_probability for result in results)
    expected_positive = torch.softmax(torch.tensor([1.0, 3.0]), dim=0)[1]
    assert all(
        torch.isclose(torch.tensor(result.positive_probability), expected_positive)
        for result in results
    )
