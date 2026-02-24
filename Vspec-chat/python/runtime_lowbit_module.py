from __future__ import annotations

import os
from dataclasses import dataclass

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    from vspec_cuda_bridge import (
        fused_linear_int3,
        fused_linear_int3_available,
        fused_linear_int4,
        fused_linear_int4_available,
        gemm_f32,
        gemm_f32_available,
    )
except Exception:  # pragma: no cover
    fused_linear_int3 = None
    fused_linear_int3_available = lambda: False
    fused_linear_int4 = None
    fused_linear_int4_available = lambda: False
    gemm_f32 = None
    gemm_f32_available = lambda: False


def _clamp_sub4(bits: int) -> int:
    if bits < 2:
        return 2
    if bits > 3:
        return 3
    return bits


@dataclass
class LowbitModulePlan:
    enabled: bool
    bits: int
    compatible: bool
    reason: str
    profile: str


def build_lowbit_module_plan(config: dict, use_native_cuda_norm: bool, requested_bits: int) -> LowbitModulePlan:
    profile = os.getenv("VSPEC_LOWBIT_PROFILE", "aggressive").strip().lower()
    if profile not in {"aggressive", "baseline"}:
        profile = "aggressive"

    if np is None:
        return LowbitModulePlan(enabled=False, bits=0, compatible=False, reason="numpy_missing", profile=profile)

    if not use_native_cuda_norm:
        return LowbitModulePlan(enabled=False, bits=0, compatible=True, reason="non_cuda_backend", profile=profile)

    hidden = int(config.get("hidden_size", 0) or config.get("n_embd", 0) or 0)
    heads = int(config.get("num_attention_heads", 0) or config.get("n_head", 0) or 0)
    if hidden <= 0 or heads <= 0 or (hidden % heads) != 0:
        return LowbitModulePlan(enabled=False, bits=0, compatible=False, reason="incompatible_shape", profile=profile)

    bits = _clamp_sub4(int(requested_bits))
    if bits == 3 and fused_linear_int3_available():
        return LowbitModulePlan(enabled=True, bits=3, compatible=True, reason="int3_fused", profile=profile)

    return LowbitModulePlan(enabled=False, bits=0, compatible=True, reason="kernel_unavailable", profile=profile)


def lowbit_linear_project(
    vec,
    w,
    key: str,
    layer_idx: int,
    packed: dict,
    use_native_cuda_norm: bool,
    lowbit_plan: LowbitModulePlan,
):
    if np is None:
        return []

    if lowbit_plan.profile == "baseline" and key in {"w2"} and (layer_idx % 2 == 0):
        if use_native_cuda_norm and gemm_f32_available():
            return gemm_f32(vec, w)[0]
        return vec @ w.T

    if use_native_cuda_norm and lowbit_plan.enabled and lowbit_plan.bits in {3, 4} and key in packed:
        packed_w, scales, bits, out_n = packed[key]
        if bits == 4 and fused_linear_int4_available():
            return fused_linear_int4(vec, packed_w, scales, out_n)[0]
        if bits == 3 and fused_linear_int3_available():
            return fused_linear_int3(vec, packed_w, scales, out_n)[0]

    if use_native_cuda_norm and gemm_f32_available():
        return gemm_f32(vec, w)[0]

    return vec @ w.T
