"""Tests for prompt-context verbalizer tokenization."""

from __future__ import annotations

import pytest

from src.model import (
    TOKENIZATION_SCHEMA_VERSION,
    inspect_prompt_verbalizers,
    validate_tokenization_report,
)


class FakeTokenizer:
    chat_template = "fake-template"

    def apply_chat_template(
        self, messages: list[dict[str, str]], *, tokenize: bool, add_generation_prompt: bool
    ) -> str:
        assert not tokenize
        assert add_generation_prompt
        return f"<user>{messages[0]['content']}</user><assistant>\n"

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        assert not add_special_tokens
        return {"input_ids": [ord(character) for character in text]}

    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        return [chr(token_id) for token_id in token_ids]


def test_verbalizers_are_inspected_after_chat_generation_prefix() -> None:
    details = inspect_prompt_verbalizers(FakeTokenizer(), "Review: fine\nSentiment:")

    assert details["prefix_token_count"] > 0
    assert details["verbalizers"]["negative"]["token_count"] == 8
    assert details["verbalizers"]["positive"]["token_count"] == 8
    assert details["verbalizers"]["negative"]["tokens"] == list("negative")


def test_report_validation_requires_consistent_sequence_scoring_flag() -> None:
    context = inspect_prompt_verbalizers(FakeTokenizer(), "prompt")
    report = {
        "schema_version": TOKENIZATION_SCHEMA_VERSION,
        "tokenizer": {"class": "FakeTokenizer"},
        "contexts": [context],
        "requires_sequence_scoring": True,
    }
    validate_tokenization_report(report)

    report["requires_sequence_scoring"] = False
    with pytest.raises(ValueError, match="sequence-scoring flag"):
        validate_tokenization_report(report)
