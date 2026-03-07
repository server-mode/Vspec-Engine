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


def _clamp_lowbit(bits: int) -> int:
    if bits < 2:
        return 2
    if bits > 4:
        return 4
    return bits


@dataclass
class LowbitModulePlan:
    enabled: bool
    bits: int
    compatible: bool
    reason: str
    profile: str


_LOWBIT_PROJECTION_STATS = {
    "calls": 0,
    "lowbit_calls": 0,
    "fallback_gemm_calls": 0,
    "fallback_matmul_calls": 0,
}


def lowbit_projection_stats_reset() -> None:
    _LOWBIT_PROJECTION_STATS["calls"] = 0
    _LOWBIT_PROJECTION_STATS["lowbit_calls"] = 0
    _LOWBIT_PROJECTION_STATS["fallback_gemm_calls"] = 0
    _LOWBIT_PROJECTION_STATS["fallback_matmul_calls"] = 0


def lowbit_projection_stats_snapshot() -> dict[str, int]:
    return {
        "calls": int(_LOWBIT_PROJECTION_STATS["calls"]),
        "lowbit_calls": int(_LOWBIT_PROJECTION_STATS["lowbit_calls"]),
        "fallback_gemm_calls": int(_LOWBIT_PROJECTION_STATS["fallback_gemm_calls"]),
        "fallback_matmul_calls": int(_LOWBIT_PROJECTION_STATS["fallback_matmul_calls"]),
    }


def _unwrap_single_batch(output):
    try:
        if hasattr(output, "ndim") and int(output.ndim) == 2 and int(output.shape[0]) == 1:
            return output[0]
    except Exception:
        return output
    return output


_INT4_KERNEL_SELFTEST_DONE = False
_INT4_KERNEL_SELFTEST_OK = False


def _pack_signed_rowwise_int4(q: "np.ndarray") -> "np.ndarray":
    n, k = q.shape
    codes = (q.astype(np.int16, copy=False) & 0x0F).astype(np.uint8, copy=False)
    if (k & 1) != 0:
        pad = np.zeros((n, 1), dtype=np.uint8)
        codes = np.concatenate([codes, pad], axis=1)
    lo = codes[:, 0::2]
    hi = codes[:, 1::2] << 4
    return (lo | hi).astype(np.uint8, copy=False).reshape(-1)


def _quantize_rowwise_int4(weight: "np.ndarray") -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    w = weight.astype(np.float32, copy=False)
    max_abs = np.max(np.abs(w), axis=1)
    scales = np.where(max_abs > 0.0, max_abs / 7.0, 1.0).astype(np.float32, copy=False)
    q = np.clip(np.round(w / scales[:, None]), -8.0, 7.0).astype(np.int8, copy=False)
    packed = _pack_signed_rowwise_int4(q)
    dequant = (q.astype(np.float32, copy=False) * scales[:, None]).astype(np.float32, copy=False)
    return packed, scales, dequant


def _run_int4_kernel_selftest() -> bool:
    global _INT4_KERNEL_SELFTEST_DONE, _INT4_KERNEL_SELFTEST_OK
    if _INT4_KERNEL_SELFTEST_DONE:
        return _INT4_KERNEL_SELFTEST_OK
    _INT4_KERNEL_SELFTEST_DONE = True

    if np is None or not fused_linear_int4_available():
        _INT4_KERNEL_SELFTEST_OK = False
        return False

    try:
        rng = np.random.default_rng(20260307)
        n = 64
        k = 96
        vec = rng.standard_normal((1, k), dtype=np.float32)
        w = (rng.standard_normal((n, k), dtype=np.float32) * 0.35).astype(np.float32, copy=False)
        packed, scales, dequant = _quantize_rowwise_int4(w)

        ref = (vec @ dequant.T).astype(np.float32, copy=False)
        got = fused_linear_int4(vec, packed, scales, n).astype(np.float32, copy=False)

        denom = float(np.max(np.abs(ref)) + 1e-6)
        rel_max = float(np.max(np.abs(got - ref)) / denom)
        _INT4_KERNEL_SELFTEST_OK = rel_max <= 0.08
        return _INT4_KERNEL_SELFTEST_OK
    except Exception:
        _INT4_KERNEL_SELFTEST_OK = False
        return False


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

    requested = int(requested_bits)
    if requested >= 8:
        return LowbitModulePlan(enabled=False, bits=0, compatible=True, reason="high_precision_passthrough", profile=profile)

    bits = _clamp_lowbit(requested)
    if bits == 4 and fused_linear_int4_available():
        if not _run_int4_kernel_selftest():
            return LowbitModulePlan(enabled=False, bits=0, compatible=False, reason="int4_selftest_failed", profile=profile)
        return LowbitModulePlan(enabled=True, bits=4, compatible=True, reason="int4_fused", profile=profile)
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

    _LOWBIT_PROJECTION_STATS["calls"] += 1

    if lowbit_plan.profile == "baseline" and key in {"w2"} and (layer_idx % 2 == 0):
        if use_native_cuda_norm and gemm_f32_available():
            _LOWBIT_PROJECTION_STATS["fallback_gemm_calls"] += 1
            return _unwrap_single_batch(gemm_f32(vec, w))
        _LOWBIT_PROJECTION_STATS["fallback_matmul_calls"] += 1
        return vec @ w.T

    if use_native_cuda_norm and lowbit_plan.enabled and lowbit_plan.bits in {3, 4} and key in packed:
        packed_w, scales, bits, out_n = packed[key]
        if bits == 4 and fused_linear_int4_available():
            _LOWBIT_PROJECTION_STATS["lowbit_calls"] += 1
            return _unwrap_single_batch(fused_linear_int4(vec, packed_w, scales, out_n))
        if bits == 3 and fused_linear_int3_available():
            _LOWBIT_PROJECTION_STATS["lowbit_calls"] += 1
            return _unwrap_single_batch(fused_linear_int3(vec, packed_w, scales, out_n))

    if use_native_cuda_norm and gemm_f32_available():
        _LOWBIT_PROJECTION_STATS["fallback_gemm_calls"] += 1
        return _unwrap_single_batch(gemm_f32(vec, w))

    _LOWBIT_PROJECTION_STATS["fallback_matmul_calls"] += 1
    return vec @ w.T
