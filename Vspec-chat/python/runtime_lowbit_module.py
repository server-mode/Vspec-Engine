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
_INT4_DEBUG_COMPARE_SEEN: set[tuple[int, str]] = set()


def _pack_signed_rowwise_int4(q: "np.ndarray") -> "np.ndarray":
    n, k = q.shape
    codes = (q.astype(np.int16, copy=False) & 0x0F).astype(np.uint8, copy=False)
    if (k & 1) != 0:
        pad = np.zeros((n, 1), dtype=np.uint8)
        codes = np.concatenate([codes, pad], axis=1)
    lo = codes[:, 0::2]
    hi = codes[:, 1::2] << 4
    return (lo | hi).astype(np.uint8, copy=False).reshape(-1)


def _unpack_int4_rowwise(
    packed: "np.ndarray",
    scales: "np.ndarray",
    out_n: int,
    k: int,
    zero_points: "np.ndarray | None" = None,
) -> "np.ndarray":
    packed_flat = np.ascontiguousarray(packed.reshape(-1), dtype=np.uint8)
    scales_flat = np.ascontiguousarray(scales.reshape(-1), dtype=np.float32)
    packed_k = (k + 1) // 2
    expected = out_n * packed_k
    if packed_flat.size < expected:
        raise ValueError("invalid packed/scales shape for int4 unpack")

    # Row-wise layout: scales=[out_n], zero_points=[out_n]
    # Block-wise layout: scales=[out_n*n_blocks], zero_points=[out_n*n_blocks]
    n_blocks = 1
    if out_n > 0 and scales_flat.size >= out_n and (scales_flat.size % out_n) == 0:
        n_blocks = max(1, int(scales_flat.size // out_n))
    if scales_flat.size < out_n * n_blocks:
        raise ValueError("invalid scales shape for int4 unpack")

    block_size = max(1, (k + n_blocks - 1) // n_blocks)

    w = np.zeros((out_n, k), dtype=np.float32)
    for j in range(out_n):
        base = j * packed_k
        for t in range(k):
            blk = min(n_blocks - 1, t // block_size)
            s_idx = j * n_blocks + blk
            scale = float(scales_flat[s_idx])
            zp = 0.0
            if zero_points is not None and zero_points.size > s_idx:
                zp = float(zero_points[s_idx])
            byte = int(packed_flat[base + (t >> 1)])
            nibble = ((byte >> 4) & 0x0F) if (t & 1) else (byte & 0x0F)
            if nibble >= 8:
                nibble -= 16
            w[j, t] = (float(nibble) - zp) * scale
    return w


def _debug_compare_int4_kernel(
    vec,
    packed_w: "np.ndarray",
    scales: "np.ndarray",
    out_n: int,
    layer_idx: int,
    key: str,
    zero_points: "np.ndarray | None" = None,
) -> None:
    if np is None or not fused_linear_int4_available():
        return
    flag = os.getenv("VSPEC_INT4_DEBUG_COMPARE", "0").strip().lower()
    if flag not in {"1", "true", "yes", "on"}:
        return

    max_checks_env = os.getenv("VSPEC_INT4_DEBUG_MAX_CHECKS", "12").strip()
    try:
        max_checks = max(1, int(max_checks_env))
    except Exception:
        max_checks = 12

    marker = (int(layer_idx), str(key))
    if marker in _INT4_DEBUG_COMPARE_SEEN:
        return
    if len(_INT4_DEBUG_COMPARE_SEEN) >= max_checks:
        return

    _INT4_DEBUG_COMPARE_SEEN.add(marker)

    try:
        vec_np = np.ascontiguousarray(vec.astype(np.float32, copy=False))
        if vec_np.ndim == 1:
            vec_np = vec_np[None, :]
        k = int(vec_np.shape[-1])
        w_deq = _unpack_int4_rowwise(packed_w, scales, int(out_n), k, zero_points=zero_points)
        ref = np.ascontiguousarray(vec_np @ w_deq.T, dtype=np.float32)
        got = np.ascontiguousarray(
            fused_linear_int4(vec_np, packed_w, scales, int(out_n), zero_points=zero_points),
            dtype=np.float32,
        )
        err_abs = float(np.max(np.abs(ref - got)))
        denom = float(np.max(np.abs(ref)) + 1e-6)
        err_rel = float(err_abs / denom)
        print(f"[int4 debug] layer={layer_idx} key={key} max_abs_err={err_abs:.6f} max_rel_err={err_rel:.6f}")
    except Exception as exc:
        print(f"[int4 debug] layer={layer_idx} key={key} compare_failed={exc}")


def _quantize_rowwise_int4(weight: "np.ndarray") -> tuple["np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray"]:
    w = weight.astype(np.float32, copy=False)
    min_q = -8.0
    max_q = 7.0
    row_min = np.min(w, axis=1)
    row_max = np.max(w, axis=1)
    row_span = np.maximum(row_max - row_min, 1e-8)
    scales = (row_span / (max_q - min_q)).astype(np.float32, copy=False)
    zero_points = np.round(min_q - (row_min / scales)).astype(np.float32, copy=False)
    zero_points = np.clip(zero_points, min_q, max_q).astype(np.float32, copy=False)
    q = np.clip(np.round((w / scales[:, None]) + zero_points[:, None]), min_q, max_q).astype(np.int8, copy=False)
    packed = _pack_signed_rowwise_int4(q)
    dequant = ((q.astype(np.float32, copy=False) - zero_points[:, None]) * scales[:, None]).astype(np.float32, copy=False)
    return packed, scales, zero_points, dequant


def _apply_rowwise_zero_point_correction(output, vec, scales: "np.ndarray", zero_points: "np.ndarray | None"):
    if np is None or zero_points is None:
        return output
    try:
        zp = np.ascontiguousarray(zero_points.astype(np.float32, copy=False).reshape(-1))
        if zp.size == 0:
            return output
        s = np.ascontiguousarray(scales.astype(np.float32, copy=False).reshape(-1))
        if s.size != zp.size:
            return output
        v = np.ascontiguousarray(vec.astype(np.float32, copy=False))
        if v.ndim == 1:
            v = v[None, :]
        out = np.ascontiguousarray(output.astype(np.float32, copy=False))
        if out.ndim == 1:
            out = out[None, :]

        out_n = int(out.shape[-1]) if out.ndim >= 2 else int(out.shape[0])
        # Block-wise layout is corrected in-kernel (CUDA) or during CPU dequant ref path.
        # Keep this post-correction only for legacy row-wise layout where scales match output rows.
        if out_n <= 0 or s.size != out_n:
            return output

        vec_sum = np.sum(v, axis=-1, keepdims=True)
        corr = vec_sum * (s * zp)[None, :]
        if out.ndim == 1:
            return (out - corr)[0]
        return out - corr
    except Exception:
        return output


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
        packed, scales, zero_points, dequant = _quantize_rowwise_int4(w)

        ref = (vec @ dequant.T).astype(np.float32, copy=False)
        got = fused_linear_int4(vec, packed, scales, n, zero_points=zero_points).astype(np.float32, copy=False)

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
        entry = packed[key]
        if len(entry) >= 5:
            packed_w, scales, bits, out_n, zero_points = entry
        else:
            packed_w, scales, bits, out_n = entry
            zero_points = None
        if bits == 4 and fused_linear_int4_available():
            _debug_compare_int4_kernel(vec, packed_w, scales, out_n, layer_idx, key, zero_points=zero_points)
            _LOWBIT_PROJECTION_STATS["lowbit_calls"] += 1
            out = fused_linear_int4(vec, packed_w, scales, out_n, zero_points=zero_points)
            return _unwrap_single_batch(out)
        if bits == 3 and fused_linear_int3_available():
            _LOWBIT_PROJECTION_STATS["lowbit_calls"] += 1
            out = fused_linear_int3(vec, packed_w, scales, out_n)
            out = _apply_rowwise_zero_point_correction(out, vec, scales, zero_points)
            return _unwrap_single_batch(out)

    if use_native_cuda_norm and gemm_f32_available():
        _LOWBIT_PROJECTION_STATS["fallback_gemm_calls"] += 1
        return _unwrap_single_batch(gemm_f32(vec, w))

    _LOWBIT_PROJECTION_STATS["fallback_matmul_calls"] += 1
    return vec @ w.T
