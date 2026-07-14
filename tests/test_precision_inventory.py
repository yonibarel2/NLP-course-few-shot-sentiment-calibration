from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.precision_inventory import inventory_model, validate_precision_pair


class Linear4bit(torch.nn.Linear):
    def __init__(self, input_features: int, output_features: int) -> None:
        super().__init__(input_features, output_features, dtype=torch.bfloat16)
        self.compute_dtype = torch.bfloat16
        self.weight.quant_type = "nf4"


Linear4bit.__module__ = "bitsandbytes.nn.modules"


class TinyModel(torch.nn.Module):
    def __init__(self, *, quantized: bool) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(8, 4, dtype=torch.bfloat16)
        self.projection = (
            Linear4bit(4, 4)
            if quantized
            else torch.nn.Linear(4, 4, dtype=torch.bfloat16)
        )
        self.norm = torch.nn.LayerNorm(4, dtype=torch.bfloat16)


def test_inventory_records_quantization_boundary_and_dtypes() -> None:
    bf16 = inventory_model(TinyModel(quantized=False))
    quantized = inventory_model(TinyModel(quantized=True))

    validate_precision_pair(bf16, quantized)

    assert bf16["quantized_weight_module_count"] == 0
    assert quantized["quantized_weight_module_count"] == 1
    record = quantized["quantized_weight_modules"][0]
    assert record["name"] == "projection"
    assert record["compute_dtype"] == "bfloat16"
    assert record["quant_type"] == "nf4"


def test_precision_pair_rejects_missing_quantized_modules() -> None:
    bf16 = inventory_model(TinyModel(quantized=False))
    with pytest.raises(AssertionError, match="no detected quantized"):
        validate_precision_pair(bf16, bf16)
