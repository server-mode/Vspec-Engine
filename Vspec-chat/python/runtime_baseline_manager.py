from __future__ import annotations

import os
from dataclasses import dataclass


def _truthy(name: str, default: str) -> bool:
    value = os.getenv(name, default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _clamp_bits(value: int) -> int:
    if value < 2:
        return 2
    if value > 3:
        return 3
    return value


@dataclass
class RuntimeBaselinePlan:
    fused_bits: int
    lowbit_enabled: bool
    compatible: bool
    reason: str


def resolve_runtime_baseline_plan(
    config: dict,
    use_native_cuda_norm: bool,
    int3_available: bool,
    int4_available: bool,
) -> RuntimeBaselinePlan:
    _ = int4_available

    if not use_native_cuda_norm:
        return RuntimeBaselinePlan(
            fused_bits=0,
            lowbit_enabled=False,
            compatible=True,
            reason="non_cuda_backend",
        )

    if not _truthy("VSPEC_PERFORMANCE_LOWBIT_DEFAULT", "1"):
        return RuntimeBaselinePlan(
            fused_bits=0,
            lowbit_enabled=False,
            compatible=True,
            reason="lowbit_default_disabled",
        )

    hidden = int(config.get("hidden_size", 0) or config.get("n_embd", 0) or 0)
    heads = int(config.get("num_attention_heads", 0) or config.get("n_head", 0) or 0)
    if hidden <= 0 or heads <= 0 or (hidden % heads) != 0:
        return RuntimeBaselinePlan(
            fused_bits=0,
            lowbit_enabled=False,
            compatible=False,
            reason="incompatible_shape",
        )

    target_bits = _clamp_bits(int(os.getenv("VSPEC_BASELINE_TARGET_BITS", "3") or "3"))
    force_lowbit = _truthy("VSPEC_BASELINE_FORCE_LOWBIT", "1")

    if target_bits == 3 and int3_available:
        return RuntimeBaselinePlan(
            fused_bits=3,
            lowbit_enabled=True,
            compatible=True,
            reason="int3_fused",
        )

    if force_lowbit:
        return RuntimeBaselinePlan(
            fused_bits=0,
            lowbit_enabled=False,
            compatible=False,
            reason="int3_unavailable",
        )

    return RuntimeBaselinePlan(
        fused_bits=0,
        lowbit_enabled=False,
        compatible=True,
        reason="fallback_full_precision",
    )
