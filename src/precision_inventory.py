"""Inspect the numerical representation used by a loaded language model."""

from __future__ import annotations

from collections import Counter
from typing import Any


def _qualified_class_name(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _dtype_name(value: Any) -> str | None:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return None
    return str(dtype).removeprefix("torch.")


def _shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [int(dimension) for dimension in shape]


def _optional_text(value: Any) -> str | bool | int | float | None:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value).removeprefix("torch.")


def _weight_module_record(name: str, module: Any) -> dict[str, Any]:
    weight = getattr(module, "weight")
    bias = getattr(module, "bias", None)
    return {
        "name": name or "<root>",
        "module_class": _qualified_class_name(module),
        "weight_class": _qualified_class_name(weight),
        "weight_dtype": _dtype_name(weight),
        "weight_shape": _shape(weight),
        "bias_dtype": _dtype_name(bias),
        "compute_dtype": _optional_text(getattr(module, "compute_dtype", None)),
        "quant_type": _optional_text(
            getattr(weight, "quant_type", getattr(module, "quant_type", None))
        ),
    }


def inventory_model(model: Any) -> dict[str, Any]:
    """Return module-by-module and aggregate dtype/quantization information."""

    module_type_counts: Counter[str] = Counter()
    parameter_dtype_counts: Counter[str] = Counter()
    buffer_dtype_counts: Counter[str] = Counter()
    weight_modules: list[dict[str, Any]] = []
    quantized_modules: list[dict[str, Any]] = []

    for parameter in model.parameters():
        parameter_dtype_counts[_dtype_name(parameter) or "unknown"] += 1
    for buffer in model.buffers():
        buffer_dtype_counts[_dtype_name(buffer) or "unknown"] += 1

    for name, module in model.named_modules():
        class_name = _qualified_class_name(module)
        module_type_counts[class_name] += 1
        if hasattr(module, "weight") and getattr(module, "weight") is not None:
            record = _weight_module_record(name, module)
            weight_modules.append(record)
            lowered_class = class_name.lower()
            lowered_weight = str(record["weight_class"]).lower()
            if (
                "bitsandbytes" in lowered_class
                or "4bit" in lowered_class
                or "params4bit" in lowered_weight
            ):
                quantized_modules.append(record)

    quantized_module_type_counts = Counter(
        str(record["module_class"]) for record in quantized_modules
    )
    return {
        "model_class": _qualified_class_name(model),
        "module_count": sum(module_type_counts.values()),
        "module_type_counts": dict(sorted(module_type_counts.items())),
        "parameter_tensor_dtype_counts": dict(sorted(parameter_dtype_counts.items())),
        "buffer_tensor_dtype_counts": dict(sorted(buffer_dtype_counts.items())),
        "weight_module_count": len(weight_modules),
        "weight_modules": weight_modules,
        "quantized_weight_module_count": len(quantized_modules),
        "quantized_module_type_counts": dict(
            sorted(quantized_module_type_counts.items())
        ),
        "quantized_weight_modules": quantized_modules,
    }


def validate_precision_pair(
    bf16_inventory: dict[str, Any], quantized_inventory: dict[str, Any]
) -> None:
    """Validate the intended BF16-versus-4-bit module boundary."""

    if int(bf16_inventory.get("quantized_weight_module_count", -1)) != 0:
        raise AssertionError("BF16 model unexpectedly contains quantized weight modules")
    quantized_count = int(
        quantized_inventory.get("quantized_weight_module_count", 0)
    )
    if quantized_count <= 0:
        raise AssertionError("4-bit model contains no detected quantized weight modules")

    quantized_modules = quantized_inventory.get("quantized_weight_modules")
    if not isinstance(quantized_modules, list) or len(quantized_modules) != quantized_count:
        raise AssertionError("4-bit module inventory is internally inconsistent")
    for record in quantized_modules:
        if record.get("compute_dtype") != "bfloat16":
            raise AssertionError(
                f"quantized module {record.get('name')} does not use BF16 computation"
            )
        if record.get("quant_type") != "nf4":
            raise AssertionError(
                f"quantized module {record.get('name')} does not use NF4 weights"
            )
