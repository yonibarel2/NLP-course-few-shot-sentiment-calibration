"""Tokenizer and model-loading utilities for the fixed Qwen checkpoint."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any


LABEL_VERBALIZERS: dict[int, str] = {0: "negative", 1: "positive"}
TOKENIZATION_SCHEMA_VERSION = 1


def render_scoring_prefix(tokenizer: Any, prompt: str) -> str:
    """Apply the tokenizer's chat template through the assistant answer prefix."""

    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be a non-empty string")
    if getattr(tokenizer, "chat_template", None) is None:
        raise ValueError("the selected tokenizer does not define a chat template")
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(rendered, str) or not rendered:
        raise ValueError("chat template did not produce a non-empty string")
    return rendered


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    input_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
    if input_ids and isinstance(input_ids[0], list):
        if len(input_ids) != 1:
            raise ValueError("expected one tokenized sequence")
        input_ids = input_ids[0]
    if not isinstance(input_ids, Sequence):
        raise ValueError("tokenizer did not return a token ID sequence")
    return [int(token_id) for token_id in input_ids]


def inspect_prompt_verbalizers(tokenizer: Any, prompt: str) -> dict[str, Any]:
    """Inspect label continuations in the exact rendered prompt context."""

    rendered_prefix = render_scoring_prefix(tokenizer, prompt)
    prefix_ids = _token_ids(tokenizer, rendered_prefix)
    verbalizers: dict[str, Any] = {}
    for label in sorted(LABEL_VERBALIZERS):
        verbalizer = LABEL_VERBALIZERS[label]
        full_ids = _token_ids(tokenizer, rendered_prefix + verbalizer)
        if full_ids[: len(prefix_ids)] != prefix_ids:
            raise ValueError(
                f"tokenization of {verbalizer!r} changes the scoring prefix boundary"
            )
        continuation_ids = full_ids[len(prefix_ids) :]
        if not continuation_ids:
            raise ValueError(f"verbalizer {verbalizer!r} produced no continuation tokens")
        tokens = tokenizer.convert_ids_to_tokens(continuation_ids)
        verbalizers[verbalizer] = {
            "label": label,
            "text": verbalizer,
            "token_ids": continuation_ids,
            "tokens": list(tokens),
            "token_count": len(continuation_ids),
        }

    return {
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "rendered_prefix_sha256": hashlib.sha256(
            rendered_prefix.encode("utf-8")
        ).hexdigest(),
        "prefix_token_count": len(prefix_ids),
        "verbalizers": verbalizers,
    }


def validate_tokenization_report(report: Mapping[str, Any]) -> None:
    """Validate the saved verbalizer-tokenization inspection report."""

    if report.get("schema_version") != TOKENIZATION_SCHEMA_VERSION:
        raise ValueError("unsupported tokenization report schema_version")
    tokenizer_metadata = report.get("tokenizer")
    contexts = report.get("contexts")
    if not isinstance(tokenizer_metadata, Mapping) or not isinstance(contexts, list):
        raise ValueError("tokenization report is missing required sections")
    if not contexts:
        raise ValueError("tokenization report must contain at least one context")
    for context in contexts:
        if not isinstance(context, Mapping):
            raise ValueError("tokenization context must be a mapping")
        verbalizers = context.get("verbalizers")
        if not isinstance(verbalizers, Mapping) or set(verbalizers) != set(
            LABEL_VERBALIZERS.values()
        ):
            raise ValueError("tokenization context has incorrect verbalizers")
        for text, details in verbalizers.items():
            if not isinstance(details, Mapping):
                raise ValueError(f"verbalizer {text!r} metadata is invalid")
            token_ids = details.get("token_ids")
            tokens = details.get("tokens")
            if not isinstance(token_ids, list) or not token_ids:
                raise ValueError(f"verbalizer {text!r} has no token IDs")
            if not isinstance(tokens, list) or len(tokens) != len(token_ids):
                raise ValueError(f"verbalizer {text!r} token metadata is inconsistent")
            if details.get("token_count") != len(token_ids):
                raise ValueError(f"verbalizer {text!r} token count is inconsistent")

    expected_sequence_scoring = any(
        details["token_count"] != 1
        for context in contexts
        for details in context["verbalizers"].values()
    )
    if report.get("requires_sequence_scoring") != expected_sequence_scoring:
        raise ValueError("tokenization sequence-scoring flag is inconsistent")
