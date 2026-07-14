"""Load both approved model conditions and save a module precision inventory."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
import sys
import time
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import load_causal_lm  # noqa: E402
from src.precision_inventory import (  # noqa: E402
    inventory_model,
    validate_precision_pair,
)


DEFAULT_HIGH_PRECISION_CONFIG = PROJECT_ROOT / "configs" / "high_precision.yaml"
DEFAULT_QUANTIZED_CONFIG = PROJECT_ROOT / "configs" / "quantized_4bit.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "tables" / "model_precision_inventory.json"


def _load_yaml(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, Mapping):
        raise ValueError(f"configuration {path} must contain a mapping")
    return value


def _validate_configs(
    high_config: Mapping[str, Any], quantized_config: Mapping[str, Any]
) -> tuple[str, str]:
    if high_config.get("model") != quantized_config.get("model"):
        raise ValueError("precision conditions must use the same model checkpoint")
    if dict(high_config.get("precision", {})) != {
        "condition_name": "bf16",
        "quantized": False,
        "torch_dtype": "bfloat16",
    }:
        raise ValueError("higher-precision configuration differs from the protocol")
    if dict(quantized_config.get("precision", {})) != {
        "condition_name": "4bit_nf4",
        "quantized": True,
        "load_in_4bit": True,
        "quantization_type": "nf4",
        "compute_dtype": "bfloat16",
        "use_double_quantization": False,
    }:
        raise ValueError("4-bit configuration differs from the protocol")
    model = high_config["model"]
    return str(model["name"]), str(model["revision"])


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("torch", "transformers", "accelerate", "bitsandbytes", "pyyaml"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _gpu_metadata() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("precision inspection requires a CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("selected GPU does not report native BF16 support")
    properties = torch.cuda.get_device_properties(0)
    metadata: dict[str, Any] = {
        "name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "total_memory_bytes": int(properties.total_memory),
        "cuda_runtime": torch.version.cuda,
        "bf16_supported": True,
    }
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
        metadata["driver_version"] = result.stdout.strip().splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        metadata["driver_version"] = None
    return metadata


def _model_metadata(model: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "device_map": getattr(model, "hf_device_map", None),
        "memory_footprint_bytes": int(model.get_memory_footprint()),
    }
    quantization_config = getattr(model.config, "quantization_config", None)
    if quantization_config is None:
        metadata["quantization_config"] = None
    elif hasattr(quantization_config, "to_dict"):
        metadata["quantization_config"] = quantization_config.to_dict()
    elif isinstance(quantization_config, Mapping):
        metadata["quantization_config"] = dict(quantization_config)
    else:
        metadata["quantization_config"] = str(quantization_config)
    return metadata


def run_inspection(
    *,
    high_precision_config_path: Path,
    quantized_config_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    import torch

    high_config = _load_yaml(high_precision_config_path)
    quantized_config = _load_yaml(quantized_config_path)
    model_name, revision = _validate_configs(high_config, quantized_config)

    conditions: dict[str, Any] = {}
    for condition, quantized in (("bf16", False), ("4bit_nf4", True)):
        print(f"Loading {condition} model for module inspection...", flush=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        model = load_causal_lm(
            model_name,
            revision=revision,
            quantized=quantized,
        )
        conditions[condition] = {
            "quantized": quantized,
            "load_and_inspection_seconds": time.monotonic() - started,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "model_metadata": _model_metadata(model),
            "inventory": inventory_model(model),
        }
        print(
            f"{condition}: detected "
            f"{conditions[condition]['inventory']['quantized_weight_module_count']} "
            "quantized weight modules",
            flush=True,
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()

    validate_precision_pair(
        conditions["bf16"]["inventory"],
        conditions["4bit_nf4"]["inventory"],
    )
    report = {
        "schema_version": 1,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_name": model_name,
        "model_revision": revision,
        "purpose": "post-experiment audit of the intended BF16 versus NF4 module boundary",
        "platform": platform.platform(),
        "gpu": _gpu_metadata(),
        "package_versions": _package_versions(),
        "conditions": conditions,
        "validation": {
            "same_checkpoint": True,
            "bf16_contains_no_detected_quantized_weight_modules": True,
            "4bit_contains_detected_nf4_weight_modules": True,
            "4bit_quantized_modules_use_bf16_compute": True,
            "double_quantization_disabled_by_validated_configuration": True,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_path.replace(output_path)
    print(f"Saved validated precision inventory to {output_path}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--high-precision-config", type=Path, default=DEFAULT_HIGH_PRECISION_CONFIG
    )
    parser.add_argument(
        "--quantized-config", type=Path, default=DEFAULT_QUANTIZED_CONFIG
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_inspection(
        high_precision_config_path=args.high_precision_config.resolve(),
        quantized_config_path=args.quantized_config.resolve(),
        output_path=args.output.resolve(),
    )


if __name__ == "__main__":
    main()
