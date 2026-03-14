from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class QuantizationSourcePolicy:
    source_format: str
    source_quantized: bool
    disable_runtime_quantization: bool
    reason: str


def _is_quantized_dtype(dtype_name: str) -> bool:
    norm = str(dtype_name or "").strip().lower()
    if not norm:
        return False
    if norm.startswith("q"):
        return True
    return norm in {
        "q2_k",
        "q3_k",
        "q4_0",
        "q4_1",
        "q4_k",
        "q5_0",
        "q5_1",
        "q5_k",
        "q6_k",
        "q8_0",
    }


def resolve_quantization_source_policy(weight_index: Optional[dict[str, Any]]) -> QuantizationSourcePolicy:
    if not weight_index:
        return QuantizationSourcePolicy(
            source_format="unknown",
            source_quantized=False,
            disable_runtime_quantization=False,
            reason="empty_weight_index",
        )

    source_formats: dict[str, int] = {}
    quantized_count = 0
    total_count = 0

    for info in weight_index.values():
        total_count += 1
        source_format = str(getattr(info, "source_format", "unknown") or "unknown").strip().lower()
        source_formats[source_format] = source_formats.get(source_format, 0) + 1
        dtype = str(getattr(info, "dtype", "") or "").strip().lower()
        if _is_quantized_dtype(dtype):
            quantized_count += 1

    source_format = max(source_formats.items(), key=lambda item: item[1])[0] if source_formats else "unknown"
    source_quantized = quantized_count > 0

    auto_detect = os.getenv("VSPEC_AUTO_USE_SOURCE_QUANT", "1").strip().lower() not in {"0", "false", "no", "off"}
    disable_runtime_quantization = bool(auto_detect and source_format == "gguf" and source_quantized)

    if disable_runtime_quantization:
        reason = "gguf_prequantized_passthrough"
    elif source_format == "gguf" and source_quantized:
        reason = "gguf_prequantized_forced_runtime_quant"
    elif source_quantized:
        reason = f"{source_format}_quantized"
    else:
        reason = f"{source_format}_float_or_unknown"

    return QuantizationSourcePolicy(
        source_format=source_format,
        source_quantized=source_quantized,
        disable_runtime_quantization=disable_runtime_quantization,
        reason=reason,
    )
