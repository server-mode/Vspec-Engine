from __future__ import annotations

import os
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None

try:
    from safetensors import safe_open
except Exception:  # pragma: no cover - optional dependency
    safe_open = None

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None

try:
    from vspec_cuda_bridge import (
        attention_flash_single_f32,
        attention_flash_single_f32_available,
        attention_single_f32,
        attention_fused_single_f32,
        attention_fused_single_f32_available,
        attention_single_f32_available,
        fused_linear_int3,
        fused_linear_int3_available,
        fused_linear_int4,
        fused_linear_int4_available,
        get_lowbit_bridge_cache_caps,
        fused_linear_int4_register_weight,
        fused_linear_int4_registered_available,
        gemm_f32,
        gemm_f32_available,
        linear_f32,
        linear_f32_available,
        mul_f32,
        mul_f32_available,
        rmsnorm_f32,
        rmsnorm_f32_available,
        silu_f32,
        silu_f32_available,
    )
except Exception:  # pragma: no cover - optional dependency
    rmsnorm_f32 = None
    rmsnorm_f32_available = lambda: False
    linear_f32 = None
    linear_f32_available = lambda: False
    attention_flash_single_f32 = None
    attention_flash_single_f32_available = lambda: False
    attention_single_f32 = None
    attention_fused_single_f32 = None
    attention_fused_single_f32_available = lambda: False
    attention_single_f32_available = lambda: False
    fused_linear_int3 = None
    fused_linear_int3_available = lambda: False
    fused_linear_int4 = None
    fused_linear_int4_available = lambda: False
    get_lowbit_bridge_cache_caps = lambda: (256, 256)
    fused_linear_int4_register_weight = lambda *args, **kwargs: 0
    fused_linear_int4_registered_available = lambda: False
    gemm_f32 = None
    gemm_f32_available = lambda: False
    silu_f32 = None
    silu_f32_available = lambda: False
    mul_f32 = None
    mul_f32_available = lambda: False

from model_adapters import ModelAdapter
from model_loader import WeightInfo, get_torch_state_dict
from gguf_support import get_gguf_archive
from runtime_core_bridge import CorePagedKVCache
from runtime_baseline_manager import resolve_runtime_baseline_plan
from runtime_lowbit_module import LowbitModulePlan, build_lowbit_module_plan, lowbit_linear_project, lowbit_linear_project_many
from runtime_compat import (
    QuantizationSourcePolicy,
    resolve_model_stability_profile,
    resolve_qwen35_stability_profile,
    resolve_quantization_source_policy,
    resolve_runtime_target,
)


_NP_SAFEOPEN_CACHE: dict[str, object] = {}
_PT_SAFEOPEN_CACHE: dict[str, object] = {}


def _get_safe_open_handle(path: Path, framework: str):
    if safe_open is None:
        return None
    key = str(path)
    if framework == "np":
        handle = _NP_SAFEOPEN_CACHE.get(key)
        if handle is None:
            handle = safe_open(key, framework="np", device="cpu")
            _NP_SAFEOPEN_CACHE[key] = handle
        return handle
    handle = _PT_SAFEOPEN_CACHE.get(key)
    if handle is None:
        handle = safe_open(key, framework="pt", device="cpu")
        _PT_SAFEOPEN_CACHE[key] = handle
    return handle


def _clear_safe_open_caches() -> None:
    _NP_SAFEOPEN_CACHE.clear()
    _PT_SAFEOPEN_CACHE.clear()


def _softmax(x: "np.ndarray") -> "np.ndarray":
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def _attention_cpu_batched(
    q: "np.ndarray",
    keys: "np.ndarray",
    values: "np.ndarray",
    num_heads: int,
    num_kv_heads: int,
    kv_heads_equal: bool,
    kv_group_size: int,
    inv_sqrt_head_dim: float,
    attention_logit_clip: float,
) -> "np.ndarray":
    # Vectorized CPU attention fallback for equal-head layouts; grouped-KV keeps stable loop path.
    if num_heads <= 0:
        return np.empty((0, 0), dtype=np.float32)
    if kv_heads_equal:
        keys_h = np.transpose(keys, (1, 0, 2))
        values_h = np.transpose(values, (1, 0, 2))
        scores = np.einsum("hd,htd->ht", q.astype(np.float32, copy=False), keys_h.astype(np.float32, copy=False))
        scores = scores * float(inv_sqrt_head_dim)
        if attention_logit_clip > 0:
            np.clip(scores, -attention_logit_clip, attention_logit_clip, out=scores)
        probs = _softmax(scores)
        out = np.einsum("ht,htd->hd", probs.astype(np.float32, copy=False), values_h.astype(np.float32, copy=False))
        return out.astype(np.float32, copy=False)

    out = np.empty((num_heads, int(q.shape[-1])), dtype=np.float32)
    for h in range(num_heads):
        kv_h = min(num_kv_heads - 1, h // max(1, kv_group_size))
        kh = keys[:, kv_h, :]
        vh = values[:, kv_h, :]
        scores = (kh @ q[h]) * float(inv_sqrt_head_dim)
        if attention_logit_clip > 0:
            np.clip(scores, -attention_logit_clip, attention_logit_clip, out=scores)
        probs = _softmax(scores.reshape(1, -1))[0]
        out[h] = probs @ vh
    return out.astype(np.float32, copy=False)


def _dynamic_clamp_std_vec(x: "np.ndarray", alpha: float) -> "np.ndarray":
    if x.size == 0:
        return x
    if float(alpha) <= 0.0:
        return x
    mean = float(np.mean(x, axis=-1, keepdims=False))
    var = float(np.mean((x - mean) * (x - mean), axis=-1, keepdims=False))
    std = float(np.sqrt(max(var, 0.0)))
    th = max(1e-6, abs(float(alpha)) * std)
    return np.clip(x, -th, th)


def _infer_model_family(config: dict) -> str:
    model_type = str(config.get("model_type", "") or "").strip().lower()
    arch = str(config.get("architectures", "") or "").strip().lower()
    family = f"{model_type} {arch}".strip()
    if "qwen3.5" in family or "qwen35" in family:
        return "qwen35"
    if "qwen" in family:
        return "qwen"
    if "llama" in family or "mistral" in family:
        return "llama"
    if "gpt2" in family:
        return "gpt2"
    return "generic"


def _diag_enabled() -> bool:
    return os.getenv("VSPEC_RUNTIME_DIAG", "0").strip().lower() in {"1", "true", "yes", "on"}


def _diag_print(*parts) -> None:
    if _diag_enabled():
        print("[runtime diag]", *parts)


def _diag_logits_once(runtime_name: str, position: int, logits: "np.ndarray") -> None:
    if not _diag_enabled():
        return
    if int(position) != 1:
        return
    try:
        logits_np = np.asarray(logits, dtype=np.float32).reshape(-1)
        if logits_np.size == 0:
            _diag_print("logits_step1", "runtime=", runtime_name, "empty")
            return
        head = logits_np[:10]
        top_id = int(np.argmax(logits_np))
        top_val = float(logits_np[top_id])
        _diag_print(
            "logits_step1",
            "runtime=", runtime_name,
            "top_id=", top_id,
            "top_val=", f"{top_val:.5f}",
            "head10=", ",".join(f"{float(v):.5f}" for v in head),
        )
    except Exception:
        return


def _validate_config_or_warn(config: dict, runtime_name: str) -> list[str]:
    issues: list[str] = []

    def _as_int(name: str, fallback: int = 0) -> int:
        try:
            return int(config.get(name, fallback) or fallback)
        except Exception:
            return fallback

    hidden = _as_int("hidden_size")
    layers = _as_int("num_hidden_layers")
    heads = _as_int("num_attention_heads")
    kv_heads = _as_int("num_key_value_heads")

    if hidden <= 0:
        issues.append("invalid_hidden_size")
    if layers <= 0:
        issues.append("invalid_num_hidden_layers")
    if heads <= 0:
        issues.append("invalid_num_attention_heads")
    if runtime_name in {"generic", "qwen35"} and kv_heads <= 0:
        issues.append("invalid_num_key_value_heads")

    if runtime_name == "qwen35":
        for key in [
            "head_dim",
            "rope_dimension_count",
            "linear_num_key_heads",
            "linear_num_value_heads",
            "linear_key_head_dim",
            "linear_value_head_dim",
            "linear_conv_kernel_dim",
        ]:
            if _as_int(key) <= 0:
                issues.append(f"invalid_{key}")

    if issues:
        _diag_print(
            "config_invalid",
            "runtime=", runtime_name,
            "model_type=", str(config.get("model_type", "")),
            "issues=", ",".join(issues),
        )
    else:
        _diag_print(
            "config_ok",
            "runtime=", runtime_name,
            "hidden=", hidden,
            "layers=", layers,
            "heads=", heads,
            "kv_heads=", kv_heads,
        )

    return issues


def _stabilize_logits(logits: "np.ndarray", logit_clip: float, entropy_target: float, margin_floor: float, margin_gain: float) -> "np.ndarray":
    if logits.size == 0:
        return logits
    finite_cap = logit_clip if logit_clip > 0.0 else 80.0
    centered = np.nan_to_num(logits.astype(np.float32, copy=False), nan=0.0, posinf=finite_cap, neginf=-finite_cap)

    if logit_clip > 0.0:
        centered = np.clip(centered, -logit_clip, logit_clip)

    if entropy_target <= 0.0 and margin_floor <= 0.0:
        return centered.astype(np.float32, copy=False)

    centered = centered - np.mean(centered, dtype=np.float32)

    probs = _softmax(centered.reshape(1, -1))[0]
    entropy = -np.sum(probs * np.log(np.maximum(probs, 1e-12)), dtype=np.float64)
    if entropy_target > 0.0 and entropy > entropy_target:
        scale = min(1.45, 1.0 + 0.08 * float(entropy - entropy_target))
        centered = centered * np.float32(scale)

    if margin_floor > 0.0 and centered.size >= 2:
        top2 = np.argpartition(centered, -2)[-2:]
        a = int(top2[0])
        b = int(top2[1])
        if centered[a] < centered[b]:
            a, b = b, a
        margin = float(centered[a] - centered[b])
        if margin < margin_floor:
            centered[a] = centered[a] + np.float32(margin_gain * (margin_floor - margin))

    return centered.astype(np.float32, copy=False)


def _rms_norm(x: "np.ndarray", weight: "np.ndarray", eps: float, bias: Optional["np.ndarray"]) -> "np.ndarray":
    mean_sq = np.mean(x * x, axis=-1, keepdims=True)
    out = x / np.sqrt(mean_sq + eps)
    out = out * weight
    if bias is not None:
        out = out + bias
    return out


def _layer_norm(x: "np.ndarray", weight: "np.ndarray", eps: float, bias: Optional["np.ndarray"]) -> "np.ndarray":
    mean = np.mean(x, axis=-1, keepdims=True)
    centered = x - mean
    var = np.mean(centered * centered, axis=-1, keepdims=True)
    out = centered / np.sqrt(var + eps)
    out = out * weight
    if bias is not None:
        out = out + bias
    return out


def _silu(x: "np.ndarray") -> "np.ndarray":
    return x / (1.0 + np.exp(-x))


def _gelu_new(x: "np.ndarray") -> "np.ndarray":
    x32 = x.astype(np.float32, copy=False)
    return 0.5 * x32 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x32 + 0.044715 * (x32 ** 3))))


def _softplus(x: "np.ndarray") -> "np.ndarray":
    x32 = x.astype(np.float32, copy=False)
    return np.log1p(np.exp(-np.abs(x32))) + np.maximum(x32, 0.0)


def _apply_rotary(q: "np.ndarray", k: "np.ndarray", position: int, rope_theta: float) -> tuple["np.ndarray", "np.ndarray"]:
    if q.shape[-1] % 2 != 0:
        return q, k
    dim = q.shape[-1]
    half = dim // 2
    idx = np.arange(half, dtype=np.float32)
    inv_freq = 1.0 / (rope_theta ** (idx / half))
    angles = position * inv_freq
    cos = np.cos(angles)
    sin = np.sin(angles)

    q1, q2 = q[..., :half], q[..., half:]
    k1, k2 = k[..., :half], k[..., half:]
    q_rot = np.concatenate([q1 * cos - q2 * sin, q1 * sin + q2 * cos], axis=-1)
    k_rot = np.concatenate([k1 * cos - k2 * sin, k1 * sin + k2 * cos], axis=-1)
    return q_rot, k_rot


def _softmax_torch(x: "torch.Tensor") -> "torch.Tensor":
    return torch.softmax(x, dim=-1)


def _rms_norm_torch(x: "torch.Tensor", weight: "torch.Tensor", eps: float, bias: Optional["torch.Tensor"]) -> "torch.Tensor":
    mean_sq = torch.mean(x * x, dim=-1, keepdim=True)
    out = x / torch.sqrt(mean_sq + eps)
    out = out * weight
    if bias is not None:
        out = out + bias
    return out


def _silu_torch(x: "torch.Tensor") -> "torch.Tensor":
    return x / (1.0 + torch.exp(-x))


def _apply_rotary_torch(q: "torch.Tensor", k: "torch.Tensor", position: int, rope_theta: float) -> tuple["torch.Tensor", "torch.Tensor"]:
    if q.shape[-1] % 2 != 0:
        return q, k
    dim = q.shape[-1]
    half = dim // 2
    idx = torch.arange(half, device=q.device, dtype=torch.float32)
    inv_freq = 1.0 / (rope_theta ** (idx / half))
    angles = position * inv_freq
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    q1, q2 = q[..., :half], q[..., half:]
    k1, k2 = k[..., :half], k[..., half:]
    q_rot = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1).to(dtype=q.dtype)
    k_rot = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1).to(dtype=k.dtype)
    return q_rot, k_rot


def _matmul_with_weight_dtype(x: "torch.Tensor", w_t: "torch.Tensor") -> "torch.Tensor":
    if x.dtype != w_t.dtype:
        x = x.to(dtype=w_t.dtype)
    return torch.matmul(x, w_t)


def _pack_signed_rowwise(q: "np.ndarray", bits: int) -> "np.ndarray":
    n, k = q.shape
    row_bytes = (k * bits + 7) // 8
    mask = (1 << bits) - 1
    codes = (q.astype(np.int16, copy=False) & mask).astype(np.uint8, copy=False)

    if bits == 4:
        if (k & 1) != 0:
            pad = np.zeros((n, 1), dtype=np.uint8)
            codes = np.concatenate([codes, pad], axis=1)
        lo = codes[:, 0::2]
        hi = codes[:, 1::2] << 4
        packed = (lo | hi).astype(np.uint8, copy=False)
        return packed.reshape(-1)

    out = np.zeros((n, row_bytes), dtype=np.uint8)
    shifts = np.arange(bits, dtype=np.uint8)
    byte_weights = (1 << np.arange(8, dtype=np.uint16)).astype(np.uint16)

    for r in range(n):
        bit_stream = ((codes[r][:, None] >> shifts) & 1).reshape(-1)
        pad_bits = (-bit_stream.size) % 8
        if pad_bits:
            bit_stream = np.pad(bit_stream, (0, pad_bits), mode="constant")
        out[r] = (bit_stream.reshape(-1, 8) * byte_weights).sum(axis=1, dtype=np.uint16).astype(np.uint8)

    return out.reshape(-1)


def _quantize_weight_rowwise(weight: "np.ndarray", bits: int) -> tuple["np.ndarray", "np.ndarray", "np.ndarray | None"]:
    w = weight.astype(np.float32, copy=False)
    max_q = float((1 << (bits - 1)) - 1)
    min_q = float(-(1 << (bits - 1)))

    percentile_env = os.getenv("VSPEC_QUANT_ROW_PERCENTILE", "0.999").strip()
    try:
        percentile = float(percentile_env)
    except Exception:
        percentile = 0.995
    percentile = max(0.95, min(1.0, percentile))

    lo_q = 1.0 - percentile
    if percentile >= 0.9999:
        row_min = np.min(w, axis=1)
        row_max = np.max(w, axis=1)
    else:
        row_min = np.quantile(w, lo_q, axis=1)
        row_max = np.quantile(w, percentile, axis=1)

    levels = max(1.0, max_q - min_q)

    # Int4 default path: block-wise quantization (Q4_K-like granularity) to reduce outlier damage.
    # This materially improves factual stability vs coarse per-row scaling.
    blockwise_int4 = bits == 4 and os.getenv("VSPEC_INT4_BLOCKWISE_ENABLE", "1").strip().lower() in {"1", "true", "yes", "on"}
    if blockwise_int4:
        block_size_raw = os.getenv("VSPEC_INT4_BLOCK_SIZE", "32").strip()
        try:
            block_size = max(8, int(block_size_raw))
        except Exception:
            block_size = 32
        # Keep Q4_K-like granularity by normalizing to 32-multiple blocks.
        block_size = max(32, ((int(block_size) + 31) // 32) * 32)
        n, k = w.shape
        n_blocks = (k + block_size - 1) // block_size
        scales_2d = np.empty((n, n_blocks), dtype=np.float32)
        zero_points_2d = np.empty((n, n_blocks), dtype=np.float32)
        q = np.empty((n, k), dtype=np.int8)

        for b in range(n_blocks):
            s0 = b * block_size
            s1 = min(k, s0 + block_size)
            wb = w[:, s0:s1]
            if percentile >= 0.9999:
                b_min = np.min(wb, axis=1)
                b_max = np.max(wb, axis=1)
            else:
                b_min = np.quantile(wb, lo_q, axis=1)
                b_max = np.quantile(wb, percentile, axis=1)
            b_span = np.maximum(b_max - b_min, 1e-8)
            b_scale = (b_span / levels).astype(np.float32, copy=False)
            b_zp = np.round(min_q - (b_min / b_scale)).astype(np.float32, copy=False)
            b_zp = np.clip(b_zp, min_q, max_q).astype(np.float32, copy=False)

            qb = np.round((wb / b_scale[:, None]) + b_zp[:, None])
            qb = np.clip(qb, min_q, max_q).astype(np.int8, copy=False)

            q[:, s0:s1] = qb
            scales_2d[:, b] = b_scale
            zero_points_2d[:, b] = b_zp

        packed = _pack_signed_rowwise(q, bits)
        scales = scales_2d.reshape(-1)
        zero_points = zero_points_2d.reshape(-1)
    else:
        row_span = np.maximum(row_max - row_min, 1e-8)
        scales = (row_span / levels).astype(np.float32, copy=False)

        zero_points = np.round(min_q - (row_min / scales)).astype(np.float32, copy=False)
        zero_points = np.clip(zero_points, min_q, max_q).astype(np.float32, copy=False)

        q = np.round((w / scales[:, None]) + zero_points[:, None])
        q = np.clip(q, min_q, max_q).astype(np.int8, copy=False)
        packed = _pack_signed_rowwise(q, bits)

    if bits not in {3, 4}:
        zero_points = None
    return packed, scales, zero_points


def _combine_packed_entries(
    entries: list[tuple["np.ndarray", "np.ndarray", int, int, "np.ndarray | None"]],
) -> tuple["np.ndarray", "np.ndarray", int, int, "np.ndarray | None"] | None:
    if np is None or not entries:
        return None

    bits_ref: Optional[int] = None
    packed_row_bytes_ref: Optional[int] = None
    scales_per_row_ref: Optional[int] = None
    zps_per_row_ref: Optional[int] = None

    packed_rows: list["np.ndarray"] = []
    scale_rows: list["np.ndarray"] = []
    zp_rows: list["np.ndarray"] = []
    has_zero_points = True
    total_rows = 0

    for entry in entries:
        packed, scales, bits, out_n, zero_points = entry
        out_rows = int(out_n)
        if out_rows <= 0:
            return None
        packed_flat = np.ascontiguousarray(packed.reshape(-1), dtype=np.uint8)
        scales_flat = np.ascontiguousarray(scales.reshape(-1), dtype=np.float32)
        if packed_flat.size % out_rows != 0 or scales_flat.size % out_rows != 0:
            return None

        packed_row_bytes = int(packed_flat.size // out_rows)
        scales_per_row = int(scales_flat.size // out_rows)
        if packed_row_bytes <= 0 or scales_per_row <= 0:
            return None

        if zero_points is None:
            has_zero_points = False
            zps_flat = None
            zps_per_row = 0
        else:
            zps_flat = np.ascontiguousarray(zero_points.reshape(-1), dtype=np.float32)
            if zps_flat.size % out_rows != 0:
                return None
            zps_per_row = int(zps_flat.size // out_rows)

        if bits_ref is None:
            bits_ref = int(bits)
            packed_row_bytes_ref = packed_row_bytes
            scales_per_row_ref = scales_per_row
            zps_per_row_ref = zps_per_row
        else:
            if int(bits) != bits_ref:
                return None
            if packed_row_bytes != packed_row_bytes_ref:
                return None
            if scales_per_row != scales_per_row_ref:
                return None
            if zps_per_row != zps_per_row_ref:
                return None

        packed_rows.append(packed_flat.reshape(out_rows, packed_row_bytes))
        scale_rows.append(scales_flat.reshape(out_rows, scales_per_row))
        if zps_flat is not None:
            zp_rows.append(zps_flat.reshape(out_rows, zps_per_row))
        total_rows += out_rows

    if bits_ref is None or packed_row_bytes_ref is None or scales_per_row_ref is None:
        return None

    packed_out = np.ascontiguousarray(np.concatenate(packed_rows, axis=0).reshape(-1), dtype=np.uint8)
    scales_out = np.ascontiguousarray(np.concatenate(scale_rows, axis=0).reshape(-1), dtype=np.float32)
    zero_points_out: "np.ndarray | None" = None
    if has_zero_points and zp_rows and zps_per_row_ref is not None and zps_per_row_ref > 0:
        zero_points_out = np.ascontiguousarray(np.concatenate(zp_rows, axis=0).reshape(-1), dtype=np.float32)

    return packed_out, scales_out, int(bits_ref), int(total_rows), zero_points_out


def _maybe_add_packed_combo(
    packed_map: dict[str, tuple["np.ndarray", "np.ndarray", int, int, "np.ndarray | None"]],
    combo_key: str,
    keys: list[str],
) -> None:
    entries: list[tuple["np.ndarray", "np.ndarray", int, int, "np.ndarray | None"]] = []
    for key in keys:
        entry = packed_map.get(key)
        if entry is None:
            return
        entries.append(entry)

    combined = _combine_packed_entries(entries)
    if combined is None:
        return
    packed_map[combo_key] = combined


def _resolve_int4_precision_windows(total_layers: int) -> tuple[int, int, bool]:
    # Prefer a speed-balanced default while allowing explicit quality mode via env.
    mode = os.getenv("VSPEC_INT4_PRECISION_MODE", "balanced").strip().lower() or "balanced"
    if mode in {"quality", "safe"}:
        keep_first_default = 2
        keep_last_default = 2
        keep_sensitive_default = True
    elif mode in {"aggressive", "speed"}:
        keep_first_default = 0
        keep_last_default = 0
        keep_sensitive_default = False
    else:
        # balanced: keep only edge-most layers in FP and quantize sensitive matrices.
        keep_first_default = 1
        keep_last_default = 1
        keep_sensitive_default = False

    int4_keep_first = max(0, int(os.getenv("VSPEC_INT4_KEEP_FIRST_FP", str(keep_first_default)) or str(keep_first_default)))
    int4_keep_last = max(0, int(os.getenv("VSPEC_INT4_KEEP_LAST_FP", str(keep_last_default)) or str(keep_last_default)))
    int4_keep_sensitive = os.getenv("VSPEC_INT4_KEEP_SENSITIVE_FP", "1" if keep_sensitive_default else "0").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }

    if total_layers > 0:
        int4_keep_first = min(int4_keep_first, total_layers)
        int4_keep_last = min(int4_keep_last, total_layers)
    return int4_keep_first, int4_keep_last, int4_keep_sensitive


def _resolve_pack_workers(task_count: int) -> int:
    try:
        env_raw = os.getenv("VSPEC_PACK_WORKERS", "0").strip()
        env_workers = int(env_raw) if env_raw else 0
    except Exception:
        env_workers = 0
    cpu_count = max(1, int(os.cpu_count() or 1))
    default_workers = min(4, cpu_count)
    workers = env_workers if env_workers > 0 else default_workers
    workers = max(1, min(workers, max(1, int(task_count))))
    return workers


def _resolve_layer_load_workers(layer_count: int) -> int:
    try:
        env_raw = os.getenv("VSPEC_LAYER_LOAD_WORKERS", "2").strip()
        env_workers = int(env_raw) if env_raw else 2
    except Exception:
        env_workers = 2
    cpu_count = max(1, int(os.cpu_count() or 1))
    workers = max(1, min(env_workers, cpu_count, max(1, int(layer_count))))
    return workers


def _packed_cache_root() -> Path:
    custom = os.getenv("VSPEC_PACK_CACHE_DIR", "").strip()
    if custom:
        return Path(custom)
    return Path(__file__).resolve().parents[2] / "logs" / "pack_cache"


def _packed_cache_write_enabled() -> bool:
    # Default-on persistent pack cache dramatically reduces repeated startup quantization time.
    flag = os.getenv("VSPEC_PACK_CACHE_WRITE", "1").strip().lower()
    return flag in {"1", "true", "yes", "on"}


def _packed_cache_key(prefix: str, key: str, bits: int, w: "np.ndarray") -> str:
    flat = w.reshape(-1)
    head = np.ascontiguousarray(flat[:512], dtype=np.float32)
    tail = np.ascontiguousarray(flat[-512:], dtype=np.float32)
    digest = hashlib.sha1(head.tobytes() + tail.tobytes()).hexdigest()[:16]
    mode = "row"
    if bits == 4 and os.getenv("VSPEC_INT4_BLOCKWISE_ENABLE", "1").strip().lower() in {"1", "true", "yes", "on"}:
        block_size_raw = os.getenv("VSPEC_INT4_BLOCK_SIZE", "32").strip()
        try:
            block_size = max(8, int(block_size_raw))
        except Exception:
            block_size = 32
        block_size = max(32, ((int(block_size) + 31) // 32) * 32)
        mode = f"blk{block_size}"
    return f"{prefix}{key}.b{bits}.{mode}.{w.shape[0]}x{w.shape[1]}.{digest}"


def _load_packed_cache(cache_key: str) -> tuple["np.ndarray", "np.ndarray", "np.ndarray | None"] | None:
    cache_file = _packed_cache_root() / f"{cache_key}.npz"
    if not cache_file.exists():
        return None
    try:
        data = np.load(cache_file, allow_pickle=False)
        packed = data["packed"].astype(np.uint8, copy=False)
        scales = data["scales"].astype(np.float32, copy=False)
        zero_points = None
        if "zero_points" in data.files:
            zero_points = data["zero_points"].astype(np.float32, copy=False)
        return packed, scales, zero_points
    except Exception:
        return None


def _save_packed_cache(
    cache_key: str,
    packed: "np.ndarray",
    scales: "np.ndarray",
    zero_points: "np.ndarray | None" = None,
) -> None:
    if not _packed_cache_write_enabled():
        return
    try:
        root = _packed_cache_root()
        root.mkdir(parents=True, exist_ok=True)
        cache_file = root / f"{cache_key}.npz"
        if zero_points is not None:
            np.savez(cache_file, packed=packed, scales=scales, zero_points=zero_points)
        else:
            np.savez(cache_file, packed=packed, scales=scales)
    except Exception:
        return


@dataclass
class SimpleRuntime:
    embed: "np.ndarray"
    lm_head: "np.ndarray"
    eos_token_id: Optional[int]

    def forward_logits_np(self, token_ids: list[int]) -> "np.ndarray":
        if np is None:
            return []  # type: ignore[return-value]
        if not token_ids:
            token_ids = [0]
        embed_tokens = self.embed[token_ids]
        pooled = np.mean(embed_tokens, axis=0)
        logits = pooled @ self.lm_head
        return logits.astype(np.float32, copy=False)

    def forward_logits(self, token_ids: list[int]) -> list[float]:
        if np is None:
            return []
        logits = self.forward_logits_np(token_ids)
        if np is not None and hasattr(logits, "size") and logits.size > 0:
            _diag_logits_once("simple", 1, logits.astype(np.float32, copy=False))
        return logits.astype(float, copy=False).tolist()


@dataclass
class LayerWeights:
    wq: "np.ndarray"
    wk: "np.ndarray"
    wv: "np.ndarray"
    wo: "np.ndarray"
    w1: "np.ndarray"
    w2: "np.ndarray"
    w3: "np.ndarray"
    norm1: "np.ndarray"
    norm2: "np.ndarray"
    q_norm: Optional["np.ndarray"]
    k_norm: Optional["np.ndarray"]
    bq: Optional["np.ndarray"]
    bk: Optional["np.ndarray"]
    bv: Optional["np.ndarray"]
    bo: Optional["np.ndarray"]
    b1: Optional["np.ndarray"]
    b2: Optional["np.ndarray"]
    b3: Optional["np.ndarray"]
    norm1_bias: Optional["np.ndarray"]
    norm2_bias: Optional["np.ndarray"]
    packed: dict[str, tuple["np.ndarray", "np.ndarray", int, int, "np.ndarray | None"]] = field(default_factory=dict)


@dataclass
class GPT2LayerWeights:
    c_attn: "np.ndarray"
    c_attn_bias: Optional["np.ndarray"]
    c_proj: "np.ndarray"
    c_proj_bias: Optional["np.ndarray"]
    c_fc: "np.ndarray"
    c_fc_bias: Optional["np.ndarray"]
    mlp_proj: "np.ndarray"
    mlp_proj_bias: Optional["np.ndarray"]
    ln_1_weight: "np.ndarray"
    ln_1_bias: Optional["np.ndarray"]
    ln_2_weight: "np.ndarray"
    ln_2_bias: Optional["np.ndarray"]
    packed: dict[str, tuple["np.ndarray", "np.ndarray", int, int, "np.ndarray | None"]] = field(default_factory=dict)


@dataclass
class Qwen35LayerWeights:
    layer_type: str
    attn_norm: "np.ndarray"
    post_attention_norm: "np.ndarray"
    w1: "np.ndarray"
    w2: "np.ndarray"
    w3: "np.ndarray"
    wq: Optional["np.ndarray"] = None
    wk: Optional["np.ndarray"] = None
    wv: Optional["np.ndarray"] = None
    wo: Optional["np.ndarray"] = None
    q_norm: Optional["np.ndarray"] = None
    k_norm: Optional["np.ndarray"] = None
    wqkv: Optional["np.ndarray"] = None
    wgate: Optional["np.ndarray"] = None
    ssm_alpha: Optional["np.ndarray"] = None
    ssm_beta: Optional["np.ndarray"] = None
    ssm_a: Optional["np.ndarray"] = None
    ssm_conv1d: Optional["np.ndarray"] = None
    ssm_dt: Optional["np.ndarray"] = None
    ssm_norm: Optional["np.ndarray"] = None
    ssm_out: Optional["np.ndarray"] = None
    packed: dict[str, tuple["np.ndarray", "np.ndarray", int, int, "np.ndarray | None"]] = field(default_factory=dict)


def _apply_partial_rotary(q: "np.ndarray", k: "np.ndarray", position: int, rope_theta: float, rope_dim: int) -> tuple["np.ndarray", "np.ndarray"]:
    use_dim = max(0, min(int(rope_dim), int(q.shape[-1]), int(k.shape[-1])))
    if use_dim <= 0 or (use_dim % 2) != 0:
        return q, k
    q_rot, k_rot = _apply_rotary(q[..., :use_dim], k[..., :use_dim], position, rope_theta)
    if use_dim == q.shape[-1]:
        return q_rot, k_rot
    q_out = np.concatenate([q_rot, q[..., use_dim:]], axis=-1)
    k_out = np.concatenate([k_rot, k[..., use_dim:]], axis=-1)
    return q_out, k_out


def _split_qwen35_q_and_gate(q_proj: "np.ndarray", num_heads: int, head_dim: int) -> tuple["np.ndarray", "np.ndarray"]:
    proj = np.asarray(q_proj, dtype=np.float32)
    expected_q = max(1, int(num_heads) * int(head_dim))
    total = int(proj.size)

    if total == expected_q * 2:
        _diag_print("qwen35_split", "mode=dual", "num_heads=", int(num_heads), "head_dim=", int(head_dim), "total=", total)
        q_full = proj.reshape(int(num_heads), int(head_dim) * 2)
        return q_full[:, :head_dim], q_full[:, head_dim:]

    if total == expected_q:
        _diag_print("qwen35_split", "mode=query_only", "num_heads=", int(num_heads), "head_dim=", int(head_dim), "total=", total)
        q = proj.reshape(int(num_heads), int(head_dim))
        gate = np.ones_like(q, dtype=np.float32)
        return q, gate

    if int(num_heads) <= 0:
        q = proj.reshape(1, -1)
        gate = np.ones_like(q, dtype=np.float32)
        return q, gate

    per_head = max(1, total // int(num_heads))
    _diag_print("qwen35_split", "mode=fallback", "num_heads=", int(num_heads), "head_dim=", int(head_dim), "total=", total, "per_head=", per_head)
    trimmed = proj[: per_head * int(num_heads)].reshape(int(num_heads), per_head)
    q = np.zeros((int(num_heads), int(head_dim)), dtype=np.float32)
    copy_q = min(int(head_dim), per_head)
    if copy_q > 0:
        q[:, :copy_q] = trimmed[:, :copy_q]

    gate = np.ones((int(num_heads), int(head_dim)), dtype=np.float32)
    gate_start = int(head_dim)
    if per_head > gate_start:
        copy_gate = min(int(head_dim), per_head - gate_start)
        gate[:, :copy_gate] = trimmed[:, gate_start:gate_start + copy_gate]
    return q, gate


@dataclass
class GenericTransformerRuntime:
    embed: "np.ndarray"
    lm_head: "np.ndarray"
    lm_head_native: Optional["np.ndarray"]
    final_norm: "np.ndarray"
    layers: list[LayerWeights]
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_eps: float
    eos_token_id: Optional[int]
    rope_theta: float
    position: int
    cache_k: list["np.ndarray"]
    cache_v: list["np.ndarray"]
    cache_len: list[int]
    use_native_cuda_norm: bool
    fused_bits: int
    disable_fused_attention: bool
    flash_attention_min_tokens: int
    flash_attention_block_tokens: int
    inv_sqrt_head_dim: float
    lowbit_plan: LowbitModulePlan
    rope_inv_freq: "np.ndarray"
    rope_cos_cache: list["np.ndarray"]
    rope_sin_cache: list["np.ndarray"]
    attention_logit_clip: float
    attn_tmp_buffers: list["np.ndarray"]
    residual_error_buffers_attn: list["np.ndarray"]
    residual_error_buffers_ff: list["np.ndarray"]
    residual_feedback_gain: float
    residual_clamp_alpha: float
    logit_entropy_target: float
    logit_margin_floor: float
    logit_margin_gain: float
    phase3_flash_attn_calls: int
    phase3_fused_attn_calls: int
    phase3_scalar_attn_calls: int
    phase3_cpu_attn_calls: int

    def _get_rotary_cos_sin(self, position: int) -> tuple["np.ndarray", "np.ndarray"]:
        while len(self.rope_cos_cache) <= position:
            pos = len(self.rope_cos_cache)
            angles = float(pos) * self.rope_inv_freq
            self.rope_cos_cache.append(np.cos(angles).astype(np.float32, copy=False))
            self.rope_sin_cache.append(np.sin(angles).astype(np.float32, copy=False))
        return self.rope_cos_cache[position], self.rope_sin_cache[position]

    def _forward_token(self, token_id: int, return_logits: bool) -> Optional["np.ndarray"]:
        if np is None:
            return np.array([], dtype=np.float32) if return_logits else None

        x = self.embed[token_id].astype(np.float32)
        use_rotary_fast = (self.head_dim % 2) == 0
        if use_rotary_fast:
            cos, sin = self._get_rotary_cos_sin(self.position)
            half = self.head_dim // 2
        kv_heads_equal = self.num_kv_heads == self.num_heads
        kv_group_size = max(1, self.num_heads // max(1, self.num_kv_heads))

        for idx, layer in enumerate(self.layers):
            if self.use_native_cuda_norm and rmsnorm_f32_available():
                x_norm = rmsnorm_f32(x[None, :], layer.norm1, self.rms_eps)[0]
            else:
                x_norm = _rms_norm(x, layer.norm1, self.rms_eps, layer.norm1_bias)

            def _linear_native(vec: "np.ndarray", w: "np.ndarray", key: str) -> "np.ndarray":
                return lowbit_linear_project(
                    vec=vec,
                    w=w,
                    key=key,
                    layer_idx=idx,
                    packed=layer.packed,
                    use_native_cuda_norm=self.use_native_cuda_norm,
                    lowbit_plan=self.lowbit_plan,
                )

            def _linear_multi(vec: "np.ndarray", specs: list[tuple[str, "np.ndarray"]], combo_key: str) -> list["np.ndarray"]:
                many = lowbit_linear_project_many(
                    vec=vec,
                    specs=specs,
                    layer_idx=idx,
                    packed=layer.packed,
                    use_native_cuda_norm=self.use_native_cuda_norm,
                    lowbit_plan=self.lowbit_plan,
                )
                if many is not None and len(many) == len(specs):
                    return many

                use_combo = (
                    self.use_native_cuda_norm
                    and self.lowbit_plan.enabled
                    and self.lowbit_plan.bits in {3, 4}
                    and combo_key in layer.packed
                    and len(specs) >= 2
                )
                if use_combo:
                    anchor_w = specs[0][1]
                    merged = np.ascontiguousarray(_linear_native(vec, anchor_w, combo_key), dtype=np.float32).reshape(-1)
                    split_sizes = [int(w.shape[0]) for _, w in specs]
                    total = int(sum(split_sizes))
                    if merged.size == total:
                        cuts = np.cumsum(np.asarray(split_sizes[:-1], dtype=np.int64))
                        return [part.astype(np.float32, copy=False) for part in np.split(merged, cuts)]
                return [_linear_native(vec, w, key) for key, w in specs]

            q, k, v = _linear_multi(
                x_norm,
                [("wq", layer.wq), ("wk", layer.wk), ("wv", layer.wv)],
                "wq_wk_wv",
            )

            if layer.bq is not None:
                q = q + layer.bq
            if layer.bk is not None:
                k = k + layer.bk
            if layer.bv is not None:
                v = v + layer.bv

            q = q.reshape(self.num_heads, self.head_dim)
            k = k.reshape(self.num_kv_heads, self.head_dim)
            v = v.reshape(self.num_kv_heads, self.head_dim)

            if layer.q_norm is not None:
                q = _rms_norm(q, layer.q_norm, self.rms_eps, None)
            if layer.k_norm is not None:
                k = _rms_norm(k, layer.k_norm, self.rms_eps, None)

            if use_rotary_fast:
                q1, q2 = q[:, :half], q[:, half:]
                k1, k2 = k[:, :half], k[:, half:]
                q = np.concatenate([q1 * cos - q2 * sin, q1 * sin + q2 * cos], axis=-1)
                k = np.concatenate([k1 * cos - k2 * sin, k1 * sin + k2 * cos], axis=-1)
            else:
                q, k = _apply_rotary(q, k, self.position, self.rope_theta)

            mirrors = getattr(self, "kv_core_mirrors", None)
            mirror_layer = None
            if mirrors is not None and idx < len(mirrors):
                mirror_layer = mirrors[idx]

            mirror_only = bool(getattr(self, "kv_python_shadow_disabled", False) and (mirror_layer is not None))
            keys = None
            values = None
            used_len = 0

            if mirror_only:
                mirror_ok = False
                try:
                    mirror_ok = bool(mirror_layer.append(k, v))
                except Exception:
                    mirror_ok = False
                if mirror_ok:
                    try:
                        used_len = int(mirror_layer.session_tokens())
                        mirror_keys, mirror_values = mirror_layer.read_tokens(used_len)
                        if mirror_keys is not None and mirror_values is not None and int(mirror_keys.shape[0]) == used_len:
                            keys = mirror_keys
                            values = mirror_values
                        else:
                            mirror_only = False
                    except Exception:
                        mirror_only = False
                else:
                    mirror_only = False

            if not mirror_only:
                if len(self.cache_k) <= idx:
                    init_cap = 16
                    k_buf = np.empty((init_cap, self.num_kv_heads, self.head_dim), dtype=np.float32)
                    v_buf = np.empty((init_cap, self.num_kv_heads, self.head_dim), dtype=np.float32)
                    k_buf[0] = k
                    v_buf[0] = v
                    self.cache_k.append(k_buf)
                    self.cache_v.append(v_buf)
                    if len(self.cache_len) <= idx:
                        self.cache_len.append(1)
                    else:
                        self.cache_len[idx] = 1
                else:
                    used = self.cache_len[idx]
                    cap = self.cache_k[idx].shape[0]
                    if used >= cap:
                        new_cap = cap * 2
                        k_new = np.empty((new_cap, self.num_kv_heads, self.head_dim), dtype=np.float32)
                        v_new = np.empty((new_cap, self.num_kv_heads, self.head_dim), dtype=np.float32)
                        k_new[:used] = self.cache_k[idx][:used]
                        v_new[:used] = self.cache_v[idx][:used]
                        self.cache_k[idx] = k_new
                        self.cache_v[idx] = v_new
                    self.cache_k[idx][used] = k
                    self.cache_v[idx][used] = v
                    self.cache_len[idx] = used + 1

                if mirror_layer is not None:
                    try:
                        mirror_layer.append(k, v)
                    except Exception:
                        pass

                used_len = self.cache_len[idx]
                keys = self.cache_k[idx][:used_len]
                values = self.cache_v[idx][:used_len]
                if mirror_layer is not None:
                    try:
                        mirror_tokens = int(mirror_layer.session_tokens())
                        if mirror_tokens == used_len:
                            mirror_keys, mirror_values = mirror_layer.read_tokens(used_len)
                            if mirror_keys is not None and mirror_values is not None:
                                keys = mirror_keys
                                values = mirror_values
                    except Exception:
                        pass

            if keys is None or values is None or used_len <= 0:
                keys = np.ascontiguousarray(k[None, :, :], dtype=np.float32)
                values = np.ascontiguousarray(v[None, :, :], dtype=np.float32)
                used_len = int(keys.shape[0])

            if len(self.attn_tmp_buffers) <= idx:
                self.attn_tmp_buffers.append(np.empty((self.num_heads, self.head_dim), dtype=np.float32))
            attn = self.attn_tmp_buffers[idx]
            early_flash_turn1 = os.getenv("VSPEC_FLASH_ATTN_TURN1_FORCE", "1").strip().lower() in {"1", "true", "yes", "on"}
            flash_min_tokens = int(self.flash_attention_min_tokens)
            if early_flash_turn1 and int(self.position) <= 1:
                flash_min_tokens = 1

            use_native_attn = self.use_native_cuda_norm and (
                ((not self.disable_fused_attention) and attention_fused_single_f32_available())
                or attention_single_f32_available()
                or (((not self.disable_fused_attention) and attention_flash_single_f32_available() and used_len >= flash_min_tokens))
            )
            if use_native_attn:
                for h in range(self.num_heads):
                    qh = q[h]
                    kv_h = h if kv_heads_equal else min(self.num_kv_heads - 1, h // kv_group_size)
                    kh = keys[:, kv_h, :]
                    vh = values[:, kv_h, :]
                    if self.use_native_cuda_norm and (not self.disable_fused_attention) and attention_flash_single_f32_available() and used_len >= flash_min_tokens:
                        attn[h] = attention_flash_single_f32(qh, kh, vh, self.flash_attention_block_tokens)
                        self.phase3_flash_attn_calls += 1
                    elif self.use_native_cuda_norm and (not self.disable_fused_attention) and attention_fused_single_f32_available():
                        attn[h] = attention_fused_single_f32(qh, kh, vh)
                        self.phase3_fused_attn_calls += 1
                    else:
                        attn[h] = attention_single_f32(qh, kh, vh)
                        self.phase3_scalar_attn_calls += 1
            else:
                attn[:, :] = _attention_cpu_batched(
                    q=q,
                    keys=keys,
                    values=values,
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    kv_heads_equal=kv_heads_equal,
                    kv_group_size=kv_group_size,
                    inv_sqrt_head_dim=self.inv_sqrt_head_dim,
                    attention_logit_clip=self.attention_logit_clip,
                )
                self.phase3_cpu_attn_calls += int(self.num_heads)

            attn = attn.reshape(-1)
            attn = _linear_native(attn, layer.wo, "wo")
            if layer.bo is not None:
                attn = attn + layer.bo

            if len(self.residual_error_buffers_attn) <= idx:
                self.residual_error_buffers_attn.append(np.zeros_like(x, dtype=np.float32))
            residual_err = self.residual_error_buffers_attn[idx]
            attn_stable = _dynamic_clamp_std_vec(attn.astype(np.float32, copy=False), self.residual_clamp_alpha)
            attn_corrected = attn_stable + (self.residual_feedback_gain * residual_err)
            self.residual_error_buffers_attn[idx] = (attn.astype(np.float32, copy=False) - attn_stable).astype(np.float32, copy=False)
            x = x + attn_corrected

            if self.use_native_cuda_norm and rmsnorm_f32_available():
                x_norm = rmsnorm_f32(x[None, :], layer.norm2, self.rms_eps)[0]
            else:
                x_norm = _rms_norm(x, layer.norm2, self.rms_eps, layer.norm2_bias)
            gate, up = _linear_multi(
                x_norm,
                [("w1", layer.w1), ("w3", layer.w3)],
                "w1_w3",
            )
            if layer.b1 is not None:
                gate = gate + layer.b1
            if self.use_native_cuda_norm and silu_f32_available():
                gate = silu_f32(gate)
            else:
                gate = _silu(gate)
            if layer.b3 is not None:
                up = up + layer.b3

            if self.use_native_cuda_norm and mul_f32_available():
                fused = mul_f32(gate, up)
            else:
                fused = gate * up

            ff = _linear_native(fused, layer.w2, "w2")
            if layer.b2 is not None:
                ff = ff + layer.b2
            if len(self.residual_error_buffers_ff) <= idx:
                self.residual_error_buffers_ff.append(np.zeros_like(x, dtype=np.float32))
            ff_stable = _dynamic_clamp_std_vec(ff.astype(np.float32, copy=False), self.residual_clamp_alpha)
            ff_corrected = ff_stable + (self.residual_feedback_gain * self.residual_error_buffers_ff[idx])
            self.residual_error_buffers_ff[idx] = (ff.astype(np.float32, copy=False) - ff_stable).astype(np.float32, copy=False)
            x = x + ff_corrected

        self.position += 1

        if not return_logits:
            return None

        if self.use_native_cuda_norm and rmsnorm_f32_available():
            x_last = rmsnorm_f32(x[None, :], self.final_norm, self.rms_eps)[0]
        else:
            x_last = _rms_norm(x, self.final_norm, self.rms_eps, None)
        if self.use_native_cuda_norm and gemm_f32_available():
            native_lm_head = self.lm_head_native if self.lm_head_native is not None else np.ascontiguousarray(self.lm_head.T, dtype=np.float32)
            logits = gemm_f32(x_last, native_lm_head)[0]
        else:
            logits = x_last @ self.lm_head
        logits = _stabilize_logits(
            logits=logits.astype(np.float32, copy=False),
            logit_clip=self.attention_logit_clip,
            entropy_target=self.logit_entropy_target,
            margin_floor=self.logit_margin_floor,
            margin_gain=self.logit_margin_gain,
        )
        return logits.astype(np.float32, copy=False)

    def prefill_tokens(self, token_ids: list[int]) -> None:
        if np is None or not token_ids:
            return
        try:
            chunk_size = max(1, int(os.getenv("VSPEC_PREFILL_CHUNK_TOKENS", "64") or "64"))
        except Exception:
            chunk_size = 64
        forward = self._forward_token
        try:
            ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
            total = int(ids.size)
            for start in range(0, total, chunk_size):
                end = min(total, start + chunk_size)
                for token_id in ids[start:end]:
                    forward(int(token_id), return_logits=False)
            return
        except Exception:
            pass
        for token_id in token_ids:
            forward(int(token_id), return_logits=False)

    def reset_core_kv_mirrors(self) -> None:
        for mirror in list(getattr(self, "kv_core_mirrors", []) or []):
            try:
                mirror.reset()
            except Exception:
                pass

    def forward_logits(self, token_ids: list[int]) -> list[float]:
        if np is None or not token_ids:
            return []
        logits = self._forward_token(int(token_ids[-1]), return_logits=True)
        if logits is None or logits.size == 0:
            return []
        _diag_logits_once("generic", int(self.position), logits)
        return logits.astype(float, copy=False).tolist()

    def forward_logits_np(self, token_ids: list[int]) -> "np.ndarray":
        if np is None or not token_ids:
            return np.array([], dtype=np.float32)
        logits = self._forward_token(int(token_ids[-1]), return_logits=True)
        if logits is None:
            return np.array([], dtype=np.float32)
        return logits


@dataclass
class GPT2Runtime:
    embed: "np.ndarray"
    pos_embed: "np.ndarray"
    lm_head: "np.ndarray"
    final_norm_weight: "np.ndarray"
    final_norm_bias: Optional["np.ndarray"]
    layers: list[GPT2LayerWeights]
    num_heads: int
    num_kv_heads: int
    head_dim: int
    ln_eps: float
    eos_token_id: Optional[int]
    position: int
    cache_k: list["np.ndarray"]
    cache_v: list["np.ndarray"]
    cache_len: list[int]
    fused_bits: int
    lowbit_plan: LowbitModulePlan

    def _forward_token(self, token_id: int, return_logits: bool) -> Optional["np.ndarray"]:
        if np is None:
            return np.array([], dtype=np.float32) if return_logits else None

        pos = min(max(0, int(self.position)), max(0, int(self.pos_embed.shape[0]) - 1))
        x = self.embed[int(token_id)].astype(np.float32, copy=False) + self.pos_embed[pos].astype(np.float32, copy=False)

        for idx, layer in enumerate(self.layers):
            residual = x
            x_norm = _layer_norm(x, layer.ln_1_weight, self.ln_eps, layer.ln_1_bias)

            def _linear(vec: "np.ndarray", w: "np.ndarray", key: str) -> "np.ndarray":
                return lowbit_linear_project(
                    vec=vec,
                    w=w,
                    key=key,
                    layer_idx=idx,
                    packed=layer.packed,
                    use_native_cuda_norm=False,
                    lowbit_plan=self.lowbit_plan,
                )

            qkv = _linear(x_norm, layer.c_attn, "c_attn")
            if layer.c_attn_bias is not None:
                qkv = qkv + layer.c_attn_bias
            q, k, v = np.split(qkv, 3)
            q = q.reshape(self.num_heads, self.head_dim)
            k = k.reshape(self.num_heads, self.head_dim)
            v = v.reshape(self.num_heads, self.head_dim)

            if len(self.cache_k) <= idx:
                k_buf = np.empty((16, self.num_heads, self.head_dim), dtype=np.float32)
                v_buf = np.empty((16, self.num_heads, self.head_dim), dtype=np.float32)
                k_buf[0] = k
                v_buf[0] = v
                self.cache_k.append(k_buf)
                self.cache_v.append(v_buf)
                self.cache_len.append(1)
            else:
                used = self.cache_len[idx]
                cap = self.cache_k[idx].shape[0]
                if used >= cap:
                    new_cap = cap * 2
                    k_new = np.empty((new_cap, self.num_heads, self.head_dim), dtype=np.float32)
                    v_new = np.empty((new_cap, self.num_heads, self.head_dim), dtype=np.float32)
                    k_new[:used] = self.cache_k[idx][:used]
                    v_new[:used] = self.cache_v[idx][:used]
                    self.cache_k[idx] = k_new
                    self.cache_v[idx] = v_new
                self.cache_k[idx][used] = k
                self.cache_v[idx][used] = v
                self.cache_len[idx] = used + 1

            used_len = self.cache_len[idx]
            keys = self.cache_k[idx][:used_len]
            values = self.cache_v[idx][:used_len]
            attn_out = np.empty((self.num_heads, self.head_dim), dtype=np.float32)
            if self.use_native_cuda_norm and attention_fused_single_f32_available():
                for h in range(self.num_heads):
                    attn_out[h] = attention_fused_single_f32(q[h], keys[:, h, :], values[:, h, :])
            elif self.use_native_cuda_norm and attention_single_f32_available():
                for h in range(self.num_heads):
                    attn_out[h] = attention_single_f32(q[h], keys[:, h, :], values[:, h, :])
            else:
                attn_out[:, :] = _attention_cpu_batched(
                    q=q,
                    keys=keys,
                    values=values,
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_heads,
                    kv_heads_equal=True,
                    kv_group_size=1,
                    inv_sqrt_head_dim=(1.0 / np.sqrt(float(max(1, self.head_dim)))),
                    attention_logit_clip=0.0,
                )

            attn_flat = attn_out.reshape(-1)
            attn_proj = _linear(attn_flat, layer.c_proj, "c_proj")
            if layer.c_proj_bias is not None:
                attn_proj = attn_proj + layer.c_proj_bias
            x = residual + attn_proj.astype(np.float32, copy=False)

            residual = x
            x_norm = _layer_norm(x, layer.ln_2_weight, self.ln_eps, layer.ln_2_bias)
            ff = _linear(x_norm, layer.c_fc, "c_fc")
            if layer.c_fc_bias is not None:
                ff = ff + layer.c_fc_bias
            ff = _gelu_new(ff)
            ff = _linear(ff, layer.mlp_proj, "mlp_proj")
            if layer.mlp_proj_bias is not None:
                ff = ff + layer.mlp_proj_bias
            x = residual + ff.astype(np.float32, copy=False)

        self.position += 1

        if not return_logits:
            return None

        x_last = _layer_norm(x, self.final_norm_weight, self.ln_eps, self.final_norm_bias)
        logits = x_last @ self.lm_head
        return logits.astype(np.float32, copy=False)

    def prefill_tokens(self, token_ids: list[int]) -> None:
        if np is None or not token_ids:
            return
        try:
            chunk_size = max(1, int(os.getenv("VSPEC_PREFILL_CHUNK_TOKENS", "64") or "64"))
        except Exception:
            chunk_size = 64
        forward = self._forward_token
        try:
            ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
            total = int(ids.size)
            for start in range(0, total, chunk_size):
                end = min(total, start + chunk_size)
                for token_id in ids[start:end]:
                    forward(int(token_id), return_logits=False)
            return
        except Exception:
            pass
        for token_id in token_ids:
            forward(int(token_id), return_logits=False)

    def reset_core_kv_mirrors(self) -> None:
        self.conv_states = []
        self.ssm_states = []

    def forward_logits(self, token_ids: list[int]) -> list[float]:
        if np is None or not token_ids:
            return []
        logits = self._forward_token(int(token_ids[-1]), return_logits=True)
        if logits is None or logits.size == 0:
            return []
        _diag_logits_once("gpt2", int(self.position), logits)
        return logits.astype(float, copy=False).tolist()

    def forward_logits_np(self, token_ids: list[int]) -> "np.ndarray":
        if np is None or not token_ids:
            return np.array([], dtype=np.float32)
        logits = self._forward_token(int(token_ids[-1]), return_logits=True)
        if logits is None:
            return np.array([], dtype=np.float32)
        return logits.astype(np.float32, copy=False)


@dataclass
class Qwen35Runtime:
    embed: "np.ndarray"
    lm_head: "np.ndarray"
    lm_head_native: Optional["np.ndarray"]
    final_norm: "np.ndarray"
    layers: list[Qwen35LayerWeights]
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rope_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel: int
    rms_eps: float
    eos_token_id: Optional[int]
    rope_theta: float
    position: int
    cache_k: list[Optional["np.ndarray"]]
    cache_v: list[Optional["np.ndarray"]]
    cache_len: list[int]
    use_native_cuda_norm: bool
    fused_bits: int
    lowbit_plan: LowbitModulePlan
    conv_states: list[Optional["np.ndarray"]]
    ssm_states: list[Optional["np.ndarray"]]
    attention_logit_clip: float
    residual_error_buffers_attn: list["np.ndarray"]
    residual_error_buffers_ff: list["np.ndarray"]
    residual_feedback_gain: float
    residual_clamp_alpha: float
    logit_entropy_target: float
    logit_margin_floor: float
    logit_margin_gain: float
    ssm_warmup_tokens: int

    def _ensure_state_lists(self) -> None:
        layer_count = len(self.layers)
        while len(self.cache_k) < layer_count:
            self.cache_k.append(None)
        while len(self.cache_v) < layer_count:
            self.cache_v.append(None)
        while len(self.cache_len) < layer_count:
            self.cache_len.append(0)
        while len(self.conv_states) < layer_count:
            self.conv_states.append(None)
        while len(self.ssm_states) < layer_count:
            self.ssm_states.append(None)

    def _forward_token(self, token_id: int, return_logits: bool) -> Optional["np.ndarray"]:
        if np is None:
            return np.array([], dtype=np.float32) if return_logits else None

        self._ensure_state_lists()
        x = self.embed[int(token_id)].astype(np.float32, copy=False)
        kv_heads_equal = self.num_kv_heads == self.num_heads
        kv_group_size = max(1, self.num_heads // max(1, self.num_kv_heads))
        linear_scale = 1.0 / np.sqrt(float(max(1, self.linear_key_head_dim)))

        for idx, layer in enumerate(self.layers):
            if self.use_native_cuda_norm and rmsnorm_f32_available():
                x_norm = rmsnorm_f32(x[None, :], layer.attn_norm, self.rms_eps)[0]
            else:
                x_norm = _rms_norm(x, layer.attn_norm, self.rms_eps, None)

            def _linear(vec: "np.ndarray", w: "np.ndarray", key: str) -> "np.ndarray":
                return lowbit_linear_project(
                    vec=vec,
                    w=w,
                    key=key,
                    layer_idx=idx,
                    packed=layer.packed,
                    use_native_cuda_norm=self.use_native_cuda_norm,
                    lowbit_plan=self.lowbit_plan,
                )

            def _linear_multi(vec: "np.ndarray", specs: list[tuple[str, "np.ndarray"]], combo_key: str) -> list["np.ndarray"]:
                many = lowbit_linear_project_many(
                    vec=vec,
                    specs=specs,
                    layer_idx=idx,
                    packed=layer.packed,
                    use_native_cuda_norm=self.use_native_cuda_norm,
                    lowbit_plan=self.lowbit_plan,
                )
                if many is not None and len(many) == len(specs):
                    return many

                use_combo = (
                    self.use_native_cuda_norm
                    and self.lowbit_plan.enabled
                    and self.lowbit_plan.bits in {3, 4}
                    and combo_key in layer.packed
                    and len(specs) >= 2
                )
                if use_combo:
                    anchor_w = specs[0][1]
                    merged = np.ascontiguousarray(_linear(vec, anchor_w, combo_key), dtype=np.float32).reshape(-1)
                    split_sizes = [int(w.shape[0]) for _, w in specs]
                    total = int(sum(split_sizes))
                    if merged.size == total:
                        cuts = np.cumsum(np.asarray(split_sizes[:-1], dtype=np.int64))
                        return [part.astype(np.float32, copy=False) for part in np.split(merged, cuts)]
                return [_linear(vec, w, key) for key, w in specs]

            if layer.layer_type == "full_attention":
                assert layer.wq is not None and layer.wk is not None and layer.wv is not None and layer.wo is not None
                q_proj, k_proj, v_proj = _linear_multi(
                    x_norm,
                    [("wq", layer.wq), ("wk", layer.wk), ("wv", layer.wv)],
                    "wq_wk_wv",
                )
                q, gate = _split_qwen35_q_and_gate(q_proj, self.num_heads, self.head_dim)
                k = k_proj.reshape(self.num_kv_heads, self.head_dim)
                v = v_proj.reshape(self.num_kv_heads, self.head_dim)

                if layer.q_norm is not None:
                    q = _rms_norm(q, layer.q_norm, self.rms_eps, None)
                if layer.k_norm is not None:
                    k = _rms_norm(k, layer.k_norm, self.rms_eps, None)

                q, k = _apply_partial_rotary(q, k, self.position, self.rope_theta, self.rope_dim)

                if self.cache_k[idx] is None or self.cache_v[idx] is None:
                    init_cap = 16
                    k_buf = np.empty((init_cap, self.num_kv_heads, self.head_dim), dtype=np.float32)
                    v_buf = np.empty((init_cap, self.num_kv_heads, self.head_dim), dtype=np.float32)
                    k_buf[0] = k
                    v_buf[0] = v
                    self.cache_k[idx] = k_buf
                    self.cache_v[idx] = v_buf
                    self.cache_len[idx] = 1
                else:
                    used = int(self.cache_len[idx])
                    cap = int(self.cache_k[idx].shape[0])
                    if used >= cap:
                        new_cap = cap * 2
                        k_new = np.empty((new_cap, self.num_kv_heads, self.head_dim), dtype=np.float32)
                        v_new = np.empty((new_cap, self.num_kv_heads, self.head_dim), dtype=np.float32)
                        k_new[:used] = self.cache_k[idx][:used]
                        v_new[:used] = self.cache_v[idx][:used]
                        self.cache_k[idx] = k_new
                        self.cache_v[idx] = v_new
                    self.cache_k[idx][used] = k
                    self.cache_v[idx][used] = v
                    self.cache_len[idx] = used + 1

                used_len = int(self.cache_len[idx])
                keys = self.cache_k[idx][:used_len]
                values = self.cache_v[idx][:used_len]
                attn = np.empty((self.num_heads, self.head_dim), dtype=np.float32)
                use_native_attn = self.use_native_cuda_norm and (attention_fused_single_f32_available() or attention_single_f32_available())
                if use_native_attn:
                    for h in range(self.num_heads):
                        qh = q[h]
                        kv_h = h if kv_heads_equal else min(self.num_kv_heads - 1, h // kv_group_size)
                        kh = keys[:, kv_h, :]
                        vh = values[:, kv_h, :]
                        if self.use_native_cuda_norm and attention_fused_single_f32_available():
                            attn[h] = attention_fused_single_f32(qh, kh, vh)
                        else:
                            attn[h] = attention_single_f32(qh, kh, vh)
                else:
                    attn[:, :] = _attention_cpu_batched(
                        q=q,
                        keys=keys,
                        values=values,
                        num_heads=self.num_heads,
                        num_kv_heads=self.num_kv_heads,
                        kv_heads_equal=kv_heads_equal,
                        kv_group_size=kv_group_size,
                        inv_sqrt_head_dim=(1.0 / np.sqrt(float(max(1, self.head_dim)))),
                        attention_logit_clip=self.attention_logit_clip,
                    )

                attn = attn * (1.0 / (1.0 + np.exp(-gate.astype(np.float32, copy=False))))
                attn_out = _linear(attn.reshape(-1), layer.wo, "wo")
            else:
                assert layer.wqkv is not None and layer.wgate is not None and layer.ssm_alpha is not None and layer.ssm_beta is not None
                assert layer.ssm_a is not None and layer.ssm_conv1d is not None and layer.ssm_dt is not None and layer.ssm_norm is not None and layer.ssm_out is not None
                qkv = _linear(x_norm, layer.wqkv, "wqkv")
                z = _linear(x_norm, layer.wgate, "wgate").reshape(self.linear_num_value_heads, self.linear_value_head_dim)
                beta = 1.0 / (1.0 + np.exp(-_linear(x_norm, layer.ssm_beta, "ssm_beta").astype(np.float32, copy=False)))
                alpha = _softplus(_linear(x_norm, layer.ssm_alpha, "ssm_alpha").astype(np.float32, copy=False) + layer.ssm_dt.astype(np.float32, copy=False))
                decay_log = alpha * layer.ssm_a.astype(np.float32, copy=False)

                conv_dim = int(qkv.shape[0])
                kernel_size = int(self.linear_conv_kernel)
                prev_conv = self.conv_states[idx]
                if prev_conv is None or prev_conv.shape != (max(0, kernel_size - 1), conv_dim):
                    prev_conv = np.zeros((max(0, kernel_size - 1), conv_dim), dtype=np.float32)
                conv_input = np.concatenate([prev_conv, qkv.reshape(1, conv_dim).astype(np.float32, copy=False)], axis=0)
                conv_weights = layer.ssm_conv1d.astype(np.float32, copy=False)
                conv_raw = np.sum(conv_input * conv_weights, axis=0, dtype=np.float32)
                self.conv_states[idx] = conv_input[1:].astype(np.float32, copy=False)

                conv_activated = _silu(conv_raw)
                key_total = self.linear_num_key_heads * self.linear_key_head_dim
                value_total = self.linear_num_value_heads * self.linear_value_head_dim
                q_conv = conv_activated[:key_total].reshape(self.linear_num_key_heads, self.linear_key_head_dim)
                k_conv = conv_activated[key_total:(2 * key_total)].reshape(self.linear_num_key_heads, self.linear_key_head_dim)
                v_conv = conv_activated[(2 * key_total):(2 * key_total + value_total)].reshape(self.linear_num_value_heads, self.linear_value_head_dim)

                q_conv = q_conv / np.sqrt(np.sum(q_conv * q_conv, axis=-1, keepdims=True) + 1e-6)
                k_conv = k_conv / np.sqrt(np.sum(k_conv * k_conv, axis=-1, keepdims=True) + 1e-6)

                state = self.ssm_states[idx]
                expected_shape = (self.linear_num_value_heads, self.linear_value_head_dim, self.linear_key_head_dim)
                if state is None or state.shape != expected_shape:
                    state = np.zeros(expected_shape, dtype=np.float32)

                decay = np.exp(np.clip(decay_log.astype(np.float32, copy=False), -60.0, 20.0)).astype(np.float32, copy=False)
                state *= decay[:, None, None]
                projected = np.einsum("hvk,hk->hv", state, k_conv.astype(np.float32, copy=False), optimize=True)
                value_update = (v_conv.astype(np.float32, copy=False) - projected) * beta[:, None]
                state += np.einsum("hv,hk->hvk", value_update, k_conv.astype(np.float32, copy=False), optimize=True)
                self.ssm_states[idx] = state.astype(np.float32, copy=False)

                linear_out = np.einsum("hvk,hk->hv", state, q_conv.astype(np.float32, copy=False) * np.float32(linear_scale), optimize=True)
                linear_out = _rms_norm(linear_out, layer.ssm_norm.astype(np.float32, copy=False), self.rms_eps, None)
                linear_out = linear_out * _silu(z.astype(np.float32, copy=False))
                attn_out = _linear(linear_out.reshape(-1), layer.ssm_out, "ssm_out")
                if self.ssm_warmup_tokens > 0:
                    warmup_scale = min(1.0, float(self.position + 1) / float(max(1, self.ssm_warmup_tokens)))
                    attn_out = attn_out.astype(np.float32, copy=False) * np.float32(warmup_scale)
                self.cache_len[idx] = self.position + 1

            if len(self.residual_error_buffers_attn) <= idx:
                self.residual_error_buffers_attn.append(np.zeros_like(x, dtype=np.float32))
            attn_out_f32 = np.nan_to_num(attn_out.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
            attn_stable = _dynamic_clamp_std_vec(attn_out_f32, self.residual_clamp_alpha)
            attn_corrected = attn_stable + (self.residual_feedback_gain * self.residual_error_buffers_attn[idx])
            self.residual_error_buffers_attn[idx] = (attn_out_f32 - attn_stable).astype(np.float32, copy=False)
            x = x + attn_corrected

            if self.use_native_cuda_norm and rmsnorm_f32_available():
                x_norm = rmsnorm_f32(x[None, :], layer.post_attention_norm, self.rms_eps)[0]
            else:
                x_norm = _rms_norm(x, layer.post_attention_norm, self.rms_eps, None)

            gate, up = _linear_multi(
                x_norm,
                [("w1", layer.w1), ("w3", layer.w3)],
                "w1_w3",
            )
            if self.use_native_cuda_norm and silu_f32_available():
                gate = silu_f32(gate)
            else:
                gate = _silu(gate)
            if self.use_native_cuda_norm and mul_f32_available():
                ff = mul_f32(gate, up)
            else:
                ff = gate * up
            ff = _linear(ff, layer.w2, "w2")
            if len(self.residual_error_buffers_ff) <= idx:
                self.residual_error_buffers_ff.append(np.zeros_like(x, dtype=np.float32))
            ff_f32 = np.nan_to_num(ff.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
            ff_stable = _dynamic_clamp_std_vec(ff_f32, self.residual_clamp_alpha)
            ff_corrected = ff_stable + (self.residual_feedback_gain * self.residual_error_buffers_ff[idx])
            self.residual_error_buffers_ff[idx] = (ff_f32 - ff_stable).astype(np.float32, copy=False)
            x = x + ff_corrected
            if layer.layer_type == "full_attention":
                self.cache_len[idx] = self.position + 1

        self.position += 1

        if not return_logits:
            return None

        if self.use_native_cuda_norm and rmsnorm_f32_available():
            x_last = rmsnorm_f32(x[None, :], self.final_norm, self.rms_eps)[0]
        else:
            x_last = _rms_norm(x, self.final_norm, self.rms_eps, None)
        if self.use_native_cuda_norm and gemm_f32_available():
            native_lm_head = self.lm_head_native if self.lm_head_native is not None else np.ascontiguousarray(self.lm_head.T, dtype=np.float32)
            logits = gemm_f32(x_last, native_lm_head)[0]
        else:
            logits = x_last @ self.lm_head
        logits = _stabilize_logits(
            logits=logits.astype(np.float32, copy=False),
            logit_clip=self.attention_logit_clip,
            entropy_target=self.logit_entropy_target,
            margin_floor=self.logit_margin_floor,
            margin_gain=self.logit_margin_gain,
        )
        return logits.astype(np.float32, copy=False)

    def prefill_tokens(self, token_ids: list[int]) -> None:
        if np is None or not token_ids:
            return
        try:
            chunk_size = max(1, int(os.getenv("VSPEC_PREFILL_CHUNK_TOKENS", "64") or "64"))
        except Exception:
            chunk_size = 64
        forward = self._forward_token
        try:
            ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
            total = int(ids.size)
            for start in range(0, total, chunk_size):
                end = min(total, start + chunk_size)
                for token_id in ids[start:end]:
                    forward(int(token_id), return_logits=False)
            return
        except Exception:
            pass
        for token_id in token_ids:
            forward(int(token_id), return_logits=False)

    def reset_core_kv_mirrors(self) -> None:
        return

    def forward_logits(self, token_ids: list[int]) -> list[float]:
        if np is None or not token_ids:
            return []
        logits = self._forward_token(int(token_ids[-1]), return_logits=True)
        if logits is None or logits.size == 0:
            return []
        _diag_logits_once("qwen35", int(self.position), logits)
        return logits.astype(float, copy=False).tolist()

    def forward_logits_np(self, token_ids: list[int]) -> "np.ndarray":
        if np is None or not token_ids:
            return np.array([], dtype=np.float32)
        logits = self._forward_token(int(token_ids[-1]), return_logits=True)
        if logits is None:
            return np.array([], dtype=np.float32)
        return logits.astype(np.float32, copy=False)


@dataclass
class GenericTransformerRuntimeTorch:
    embed: "torch.Tensor"
    lm_head: "torch.Tensor"
    final_norm: "torch.Tensor"
    layers: list[LayerWeights]
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_eps: float
    eos_token_id: Optional[int]
    rope_theta: float
    position: int
    cache_k: list["torch.Tensor"]
    cache_v: list["torch.Tensor"]

    def _forward_token(self, token_id: int, return_logits: bool) -> Optional["torch.Tensor"]:
        if torch is None:
            return torch.empty(0, dtype=torch.float32) if return_logits else None

        x = self.embed[token_id]

        for idx, layer in enumerate(self.layers):
            x_norm = _rms_norm_torch(x, layer.norm1, self.rms_eps, layer.norm1_bias)

            q = _matmul_with_weight_dtype(x_norm, layer.wq.t())
            k = _matmul_with_weight_dtype(x_norm, layer.wk.t())
            v = _matmul_with_weight_dtype(x_norm, layer.wv.t())

            if layer.bq is not None:
                q = q + layer.bq
            if layer.bk is not None:
                k = k + layer.bk
            if layer.bv is not None:
                v = v + layer.bv

            q = q.reshape(self.num_heads, self.head_dim)
            k = k.reshape(self.num_kv_heads, self.head_dim)
            v = v.reshape(self.num_kv_heads, self.head_dim)

            if layer.q_norm is not None:
                q = _rms_norm_torch(q, layer.q_norm, self.rms_eps, None)
            if layer.k_norm is not None:
                k = _rms_norm_torch(k, layer.k_norm, self.rms_eps, None)

            q, k = _apply_rotary_torch(q, k, self.position, self.rope_theta)

            if self.num_kv_heads != self.num_heads:
                repeat = self.num_heads // self.num_kv_heads
                k = k.repeat_interleave(repeat, dim=0)
                v = v.repeat_interleave(repeat, dim=0)

            if len(self.cache_k) <= idx:
                self.cache_k.append(k.unsqueeze(0))
                self.cache_v.append(v.unsqueeze(0))
            else:
                self.cache_k[idx] = torch.cat([self.cache_k[idx], k.unsqueeze(0)], dim=0)
                self.cache_v[idx] = torch.cat([self.cache_v[idx], v.unsqueeze(0)], dim=0)

            keys = self.cache_k[idx]
            values = self.cache_v[idx]

            attn_out = []
            for h in range(self.num_heads):
                qh = q[h]
                kh = keys[:, h, :]
                vh = values[:, h, :]
                scores = torch.matmul(kh, qh) / torch.sqrt(torch.tensor(self.head_dim, device=kh.device, dtype=kh.dtype))
                probs = _softmax_torch(scores.view(1, -1))[0]
                attn_out.append(torch.matmul(probs, vh))

            attn = torch.cat(attn_out, dim=0)
            attn = _matmul_with_weight_dtype(attn, layer.wo.t())
            if layer.bo is not None:
                attn = attn + layer.bo
            x = x + attn

            x_norm = _rms_norm_torch(x, layer.norm2, self.rms_eps, layer.norm2_bias)
            gate = _matmul_with_weight_dtype(x_norm, layer.w1.t())
            if layer.b1 is not None:
                gate = gate + layer.b1
            gate = _silu_torch(gate)

            up = _matmul_with_weight_dtype(x_norm, layer.w3.t())
            if layer.b3 is not None:
                up = up + layer.b3

            ff = _matmul_with_weight_dtype(gate * up, layer.w2.t())
            if layer.b2 is not None:
                ff = ff + layer.b2
            x = x + ff

        self.position += 1

        if not return_logits:
            return None

        x_last = _rms_norm_torch(x, self.final_norm, self.rms_eps, None)
        logits = _matmul_with_weight_dtype(x_last, self.lm_head)
        return logits.float()

    def prefill_tokens(self, token_ids: list[int]) -> None:
        if torch is None or not token_ids:
            return
        with torch.no_grad():
            for token_id in token_ids:
                self._forward_token(int(token_id), return_logits=False)

    def forward_logits(self, token_ids: list[int]) -> list[float]:
        if torch is None or not token_ids:
            return []
        with torch.no_grad():
            logits = self._forward_token(int(token_ids[-1]), return_logits=True)
        if logits is None or logits.numel() == 0:
            return []
        if np is not None:
            _diag_logits_once("generic_torch", int(self.position), logits.detach().cpu().numpy().astype(np.float32, copy=False))
        return logits.cpu().tolist()

    def forward_logits_np(self, token_ids: list[int]) -> "np.ndarray":
        if torch is None or np is None or not token_ids:
            if np is None:
                return []  # type: ignore[return-value]
            return np.array([], dtype=np.float32)
        with torch.no_grad():
            logits = self._forward_token(int(token_ids[-1]), return_logits=True)
        if logits is None or logits.numel() == 0:
            return np.array([], dtype=np.float32)
        return logits.detach().cpu().numpy().astype(np.float32, copy=False)


def _load_tensor(info: WeightInfo) -> Optional["np.ndarray"]:
    if getattr(info, "source_format", "safetensors") == "gguf":
        return get_gguf_archive(info.path).load_tensor(info.name)
    if getattr(info, "source_format", "safetensors") == "pytorch_bin":
        if torch is None:
            return None
        state = get_torch_state_dict(info.path)
        tensor = state.get(info.name)
        if tensor is None:
            return None
        if info.name.endswith(".weight") and any(tag in info.name for tag in (".attn.c_attn", ".attn.c_proj", ".mlp.c_fc", ".mlp.c_proj")):
            tensor = tensor.t()
        return tensor.float().cpu().numpy()
    if np is None or safe_open is None:
        return None
    f = _get_safe_open_handle(info.path, "np")
    try:
        return f.get_tensor(info.name)
    except TypeError:
        if torch is None:
            return None
    except Exception:
        if torch is None:
            return None
    if torch is None:
        return None
    f_pt = _get_safe_open_handle(info.path, "pt")
    tensor = f_pt.get_tensor(info.name)
    return tensor.float().cpu().numpy()


def _load_tensor_torch(info: WeightInfo, device: str) -> Optional["torch.Tensor"]:
    if getattr(info, "source_format", "safetensors") == "pytorch_bin":
        if torch is None:
            return None
        state = get_torch_state_dict(info.path)
        tensor = state.get(info.name)
        if tensor is None:
            return None
        if info.name.endswith(".weight") and any(tag in info.name for tag in (".attn.c_attn", ".attn.c_proj", ".mlp.c_fc", ".mlp.c_proj")):
            tensor = tensor.t()
        return tensor.to(device=device)
    if torch is None or safe_open is None:
        return None
    f = _get_safe_open_handle(info.path, "pt")
    tensor = f.get_tensor(info.name)
    return tensor.to(device=device)


def _select_weight(weight_index: dict[str, WeightInfo], candidates: list[str]) -> Optional[WeightInfo]:
    for name in candidates:
        if name in weight_index:
            return weight_index[name]
    return None


def runtime_load_reason(weight_index: dict[str, WeightInfo], embed_names: list[str], lm_head_names: list[str]) -> str:
    if np is None:
        return "missing numpy or safetensors"

    embed_info = _select_weight(weight_index, embed_names)
    lm_head_info = _select_weight(weight_index, lm_head_names)
    if not embed_info or not lm_head_info:
        return "missing embed or lm_head weights"

    source_pair = {
        getattr(embed_info, "source_format", "safetensors"),
        getattr(lm_head_info, "source_format", "safetensors"),
    }
    if source_pair == {"gguf"} or source_pair == {"pytorch_bin"}:
        return "unknown"

    if safe_open is None:
        return "missing numpy or safetensors"

    needs_torch = False
    for info in (embed_info, lm_head_info):
        if str(info.dtype).lower() in {"bf16", "bfloat16"}:
            needs_torch = True

    if needs_torch and torch is None:
        return "missing torch for bfloat16 weights"

    return "unknown"


def _load_pair(weight_index: dict[str, WeightInfo], embed_names: list[str], lm_head_names: list[str]) -> Optional[SimpleRuntime]:
    embed_info = _select_weight(weight_index, embed_names)
    lm_head_info = _select_weight(weight_index, lm_head_names)
    if not embed_info:
        return None
    if lm_head_info is None:
        # Many modern checkpoints tie output projection to token embeddings.
        lm_head_info = embed_info

    embed = _load_tensor(embed_info)
    lm_head = _load_tensor(lm_head_info)
    if embed is None or lm_head is None:
        return None

    if embed.ndim != 2 or lm_head.ndim != 2:
        return None

    embed_vocab, embed_dim = embed.shape
    if lm_head.shape[0] == embed_dim:
        pass
    elif lm_head.shape[1] == embed_dim:
        lm_head = lm_head.T
    else:
        return None

    return SimpleRuntime(embed=embed, lm_head=lm_head, eos_token_id=None)


def _load_first_available(weight_index: dict[str, WeightInfo], name: str) -> Optional["np.ndarray"]:
    info = weight_index.get(name)
    if not info:
        return None
    return _load_tensor(info)


def _load_first_available_torch(weight_index: dict[str, WeightInfo], name: str, device: str) -> Optional["torch.Tensor"]:
    info = weight_index.get(name)
    if not info:
        return None
    return _load_tensor_torch(info, device)


def _load_layer(weight_index: dict[str, WeightInfo], layer_idx: int, fused_bits: int, total_layers: int) -> Optional[LayerWeights]:
    prefix = f"model.layers.{layer_idx}."
    blk_prefix = f"blk.{layer_idx}."

    q = _load_first_available(weight_index, prefix + "self_attn.q_proj.weight")
    k = _load_first_available(weight_index, prefix + "self_attn.k_proj.weight")
    v = _load_first_available(weight_index, prefix + "self_attn.v_proj.weight")

    if q is None:
        q = _load_first_available(weight_index, blk_prefix + "attn_q.weight")
    if k is None:
        k = _load_first_available(weight_index, blk_prefix + "attn_k.weight")
    if v is None:
        v = _load_first_available(weight_index, blk_prefix + "attn_v.weight")

    qkv = _load_first_available(weight_index, prefix + "self_attn.qkv_proj.weight")
    if qkv is not None and q is None and k is None and v is None:
        q, k, v = np.split(qkv, 3, axis=0)

    o = _load_first_available(weight_index, prefix + "self_attn.o_proj.weight")
    if o is None:
        o = _load_first_available(weight_index, prefix + "self_attn.out_proj.weight")
    if o is None:
        o = _load_first_available(weight_index, blk_prefix + "attn_output.weight")

    n1 = _load_first_available(weight_index, prefix + "input_layernorm.weight")
    if n1 is None:
        n1 = _load_first_available(weight_index, prefix + "attention_norm.weight")
    if n1 is None:
        n1 = _load_first_available(weight_index, blk_prefix + "attn_norm.weight")

    n2 = _load_first_available(weight_index, prefix + "post_attention_layernorm.weight")
    if n2 is None:
        n2 = _load_first_available(weight_index, prefix + "mlp_norm.weight")
    if n2 is None:
        n2 = _load_first_available(weight_index, blk_prefix + "post_attention_norm.weight")
    if n2 is None:
        n2 = _load_first_available(weight_index, blk_prefix + "ffn_norm.weight")

    q_norm = _load_first_available(weight_index, prefix + "self_attn.q_norm.weight")
    k_norm = _load_first_available(weight_index, prefix + "self_attn.k_norm.weight")
    if q_norm is None:
        q_norm = _load_first_available(weight_index, blk_prefix + "attn_q_norm.weight")
    if k_norm is None:
        k_norm = _load_first_available(weight_index, blk_prefix + "attn_k_norm.weight")

    w1 = _load_first_available(weight_index, prefix + "mlp.gate_proj.weight")
    w2 = _load_first_available(weight_index, prefix + "mlp.down_proj.weight")
    w3 = _load_first_available(weight_index, prefix + "mlp.up_proj.weight")

    if w1 is None:
        w1 = _load_first_available(weight_index, prefix + "mlp.w1.weight")
    if w2 is None:
        w2 = _load_first_available(weight_index, prefix + "mlp.w2.weight")
    if w3 is None:
        w3 = _load_first_available(weight_index, prefix + "mlp.w3.weight")
    if w1 is None:
        w1 = _load_first_available(weight_index, blk_prefix + "ffn_gate.weight")
    if w2 is None:
        w2 = _load_first_available(weight_index, blk_prefix + "ffn_down.weight")
    if w3 is None:
        w3 = _load_first_available(weight_index, blk_prefix + "ffn_up.weight")

    if q is not None and k is not None:
        try:
            if q.ndim == 2 and k.ndim == 2 and int(q.shape[0]) == int(k.shape[0]) * 2:
                q = q[: int(k.shape[0]), :]
        except Exception:
            pass

    if any(x is None for x in (q, k, v, o, w1, w2, w3, n1, n2)):
        return None

    bq = _load_first_available(weight_index, prefix + "self_attn.q_proj.bias")
    bk = _load_first_available(weight_index, prefix + "self_attn.k_proj.bias")
    bv = _load_first_available(weight_index, prefix + "self_attn.v_proj.bias")
    bo = _load_first_available(weight_index, prefix + "self_attn.o_proj.bias")
    b1 = _load_first_available(weight_index, prefix + "mlp.gate_proj.bias")
    b2 = _load_first_available(weight_index, prefix + "mlp.down_proj.bias")
    b3 = _load_first_available(weight_index, prefix + "mlp.up_proj.bias")
    n1_bias = _load_first_available(weight_index, prefix + "input_layernorm.bias")
    n2_bias = _load_first_available(weight_index, prefix + "post_attention_layernorm.bias")

    layer = LayerWeights(
        wq=q,
        wk=k,
        wv=v,
        wo=o,
        w1=w1,
        w2=w2,
        w3=w3,
        norm1=n1,
        norm2=n2,
        q_norm=q_norm,
        k_norm=k_norm,
        bq=bq,
        bk=bk,
        bv=bv,
        bo=bo,
        b1=b1,
        b2=b2,
        b3=b3,
        norm1_bias=n1_bias,
        norm2_bias=n2_bias,
    )

    if fused_bits in {3, 4}:
        keep_last_at_4 = max(0, int(os.getenv("VSPEC_THREEBIT_KEEP_LAST4", "4") or "4"))
        sensitive_threebit_keys = {"wq", "wk", "wo", "norm1", "norm2"}
        int4_keys_raw = os.getenv("VSPEC_INT4_MATRIX_KEYS", "wq,wk,wv,wo,w1,w2,w3")
        int4_allowed_keys = {k.strip().lower() for k in int4_keys_raw.split(",") if k.strip()}
        if not int4_allowed_keys:
            int4_allowed_keys = {"wq", "wk", "wv", "wo", "w1", "w2", "w3"}
        int4_keep_first, int4_keep_last, int4_keep_sensitive = _resolve_int4_precision_windows(total_layers)
        sensitive_int4_keys = {"wq", "wk", "wo"}
        pack_jobs: list[tuple[str, "np.ndarray", int]] = []
        for key, w in {
            "wq": layer.wq,
            "wk": layer.wk,
            "wv": layer.wv,
            "wo": layer.wo,
            "w1": layer.w1,
            "w2": layer.w2,
            "w3": layer.w3,
        }.items():
            matrix_bits = fused_bits
            if fused_bits == 3:
                if key in sensitive_threebit_keys:
                    matrix_bits = 4
                if keep_last_at_4 > 0 and layer_idx >= max(0, total_layers - keep_last_at_4):
                    matrix_bits = 4
            elif fused_bits == 4:
                in_first_window = (int4_keep_first > 0 and layer_idx < int4_keep_first)
                in_last_window = (int4_keep_last > 0 and layer_idx >= max(0, total_layers - int4_keep_last))
                if key.lower() not in int4_allowed_keys:
                    matrix_bits = 0
                elif in_first_window or in_last_window:
                    matrix_bits = 0
                elif int4_keep_sensitive and (key in sensitive_int4_keys):
                    matrix_bits = 0

            if matrix_bits <= 0:
                continue

            pack_jobs.append((key, w, int(matrix_bits)))

        def _pack_one(entry: tuple[str, "np.ndarray", int]) -> tuple[str, tuple["np.ndarray", "np.ndarray", int, int, "np.ndarray | None"]]:
            key, w, matrix_bits = entry
            cache_key = _packed_cache_key(prefix, key, matrix_bits, w)
            cached = _load_packed_cache(cache_key)
            if cached is not None:
                packed, scales, zero_points = cached
            else:
                packed, scales, zero_points = _quantize_weight_rowwise(w, matrix_bits)
                _save_packed_cache(cache_key, packed, scales, zero_points)
            return key, (packed, scales, matrix_bits, int(w.shape[0]), zero_points)

        workers = _resolve_pack_workers(len(pack_jobs))
        if workers <= 1 or len(pack_jobs) <= 1:
            for job in pack_jobs:
                key, packed_entry = _pack_one(job)
                layer.packed[key] = packed_entry
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_pack_one, job) for job in pack_jobs]
                for fut in as_completed(futures):
                    key, packed_entry = fut.result()
                    layer.packed[key] = packed_entry

        # Reduce per-token kernel launches by combining compatible projections.
        _maybe_add_packed_combo(layer.packed, "wq_wk_wv", ["wq", "wk", "wv"])
        _maybe_add_packed_combo(layer.packed, "w1_w3", ["w1", "w3"])

    return layer


def _load_layer_torch(weight_index: dict[str, WeightInfo], layer_idx: int, device: str) -> Optional[LayerWeights]:
    prefix = f"model.layers.{layer_idx}."

    q = _load_first_available_torch(weight_index, prefix + "self_attn.q_proj.weight", device)
    k = _load_first_available_torch(weight_index, prefix + "self_attn.k_proj.weight", device)
    v = _load_first_available_torch(weight_index, prefix + "self_attn.v_proj.weight", device)

    qkv = _load_first_available_torch(weight_index, prefix + "self_attn.qkv_proj.weight", device)
    if qkv is not None and q is None and k is None and v is None:
        q, k, v = torch.split(qkv, qkv.shape[0] // 3, dim=0)

    o = _load_first_available_torch(weight_index, prefix + "self_attn.o_proj.weight", device)
    if o is None:
        o = _load_first_available_torch(weight_index, prefix + "self_attn.out_proj.weight", device)

    n1 = _load_first_available_torch(weight_index, prefix + "input_layernorm.weight", device)
    if n1 is None:
        n1 = _load_first_available_torch(weight_index, prefix + "attention_norm.weight", device)

    n2 = _load_first_available_torch(weight_index, prefix + "post_attention_layernorm.weight", device)
    if n2 is None:
        n2 = _load_first_available_torch(weight_index, prefix + "mlp_norm.weight", device)

    q_norm = _load_first_available_torch(weight_index, prefix + "self_attn.q_norm.weight", device)
    k_norm = _load_first_available_torch(weight_index, prefix + "self_attn.k_norm.weight", device)

    w1 = _load_first_available_torch(weight_index, prefix + "mlp.gate_proj.weight", device)
    w2 = _load_first_available_torch(weight_index, prefix + "mlp.down_proj.weight", device)
    w3 = _load_first_available_torch(weight_index, prefix + "mlp.up_proj.weight", device)

    if w1 is None:
        w1 = _load_first_available_torch(weight_index, prefix + "mlp.w1.weight", device)
    if w2 is None:
        w2 = _load_first_available_torch(weight_index, prefix + "mlp.w2.weight", device)
    if w3 is None:
        w3 = _load_first_available_torch(weight_index, prefix + "mlp.w3.weight", device)

    if any(x is None for x in (q, k, v, o, w1, w2, w3, n1, n2)):
        return None

    bq = _load_first_available_torch(weight_index, prefix + "self_attn.q_proj.bias", device)
    bk = _load_first_available_torch(weight_index, prefix + "self_attn.k_proj.bias", device)
    bv = _load_first_available_torch(weight_index, prefix + "self_attn.v_proj.bias", device)
    bo = _load_first_available_torch(weight_index, prefix + "self_attn.o_proj.bias", device)
    b1 = _load_first_available_torch(weight_index, prefix + "mlp.gate_proj.bias", device)
    b2 = _load_first_available_torch(weight_index, prefix + "mlp.down_proj.bias", device)
    b3 = _load_first_available_torch(weight_index, prefix + "mlp.up_proj.bias", device)
    n1_bias = _load_first_available_torch(weight_index, prefix + "input_layernorm.bias", device)
    n2_bias = _load_first_available_torch(weight_index, prefix + "post_attention_layernorm.bias", device)

    return LayerWeights(
        wq=q,
        wk=k,
        wv=v,
        wo=o,
        w1=w1,
        w2=w2,
        w3=w3,
        norm1=n1,
        norm2=n2,
        q_norm=q_norm,
        k_norm=k_norm,
        bq=bq,
        bk=bk,
        bv=bv,
        bo=bo,
        b1=b1,
        b2=b2,
        b3=b3,
        norm1_bias=n1_bias,
        norm2_bias=n2_bias,
    )


def _load_qwen35_layer(weight_index: dict[str, WeightInfo], layer_idx: int, fused_bits: int) -> Optional[Qwen35LayerWeights]:
    prefix = f"blk.{layer_idx}."
    attn_norm = _load_first_available(weight_index, prefix + "attn_norm.weight")
    post_attention_norm = _load_first_available(weight_index, prefix + "post_attention_norm.weight")
    w1 = _load_first_available(weight_index, prefix + "ffn_gate.weight")
    w2 = _load_first_available(weight_index, prefix + "ffn_down.weight")
    w3 = _load_first_available(weight_index, prefix + "ffn_up.weight")
    if any(x is None for x in (attn_norm, post_attention_norm, w1, w2, w3)):
        return None

    if prefix + "attn_q.weight" in weight_index:
        wq = _load_first_available(weight_index, prefix + "attn_q.weight")
        wk = _load_first_available(weight_index, prefix + "attn_k.weight")
        wv = _load_first_available(weight_index, prefix + "attn_v.weight")
        wo = _load_first_available(weight_index, prefix + "attn_output.weight")
        q_norm = _load_first_available(weight_index, prefix + "attn_q_norm.weight")
        k_norm = _load_first_available(weight_index, prefix + "attn_k_norm.weight")
        if any(x is None for x in (wq, wk, wv, wo, q_norm, k_norm)):
            return None
        layer = Qwen35LayerWeights(
            layer_type="full_attention",
            attn_norm=attn_norm,
            post_attention_norm=post_attention_norm,
            w1=w1,
            w2=w2,
            w3=w3,
            wq=wq,
            wk=wk,
            wv=wv,
            wo=wo,
            q_norm=q_norm,
            k_norm=k_norm,
        )
        pack_targets = {
            "wq": layer.wq,
            "wk": layer.wk,
            "wv": layer.wv,
            "wo": layer.wo,
            "w1": layer.w1,
            "w2": layer.w2,
            "w3": layer.w3,
        }
    elif prefix + "attn_qkv.weight" in weight_index:
        wqkv = _load_first_available(weight_index, prefix + "attn_qkv.weight")
        wgate = _load_first_available(weight_index, prefix + "attn_gate.weight")
        ssm_alpha = _load_first_available(weight_index, prefix + "ssm_alpha.weight")
        ssm_beta = _load_first_available(weight_index, prefix + "ssm_beta.weight")
        ssm_a = _load_first_available(weight_index, prefix + "ssm_a")
        ssm_conv1d = _load_first_available(weight_index, prefix + "ssm_conv1d.weight")
        ssm_dt = _load_first_available(weight_index, prefix + "ssm_dt.bias")
        ssm_norm = _load_first_available(weight_index, prefix + "ssm_norm.weight")
        ssm_out = _load_first_available(weight_index, prefix + "ssm_out.weight")
        if any(x is None for x in (wqkv, wgate, ssm_alpha, ssm_beta, ssm_a, ssm_conv1d, ssm_dt, ssm_norm, ssm_out)):
            return None
        if ssm_conv1d.ndim == 2 and ssm_conv1d.shape[0] > ssm_conv1d.shape[1]:
            ssm_conv1d = ssm_conv1d.T
        layer = Qwen35LayerWeights(
            layer_type="linear_attention",
            attn_norm=attn_norm,
            post_attention_norm=post_attention_norm,
            w1=w1,
            w2=w2,
            w3=w3,
            wqkv=wqkv,
            wgate=wgate,
            ssm_alpha=ssm_alpha,
            ssm_beta=ssm_beta,
            ssm_a=ssm_a,
            ssm_conv1d=ssm_conv1d,
            ssm_dt=ssm_dt,
            ssm_norm=ssm_norm,
            ssm_out=ssm_out,
        )
        pack_targets = {
            "wqkv": layer.wqkv,
            "wgate": layer.wgate,
            "ssm_alpha": layer.ssm_alpha,
            "ssm_beta": layer.ssm_beta,
            "ssm_out": layer.ssm_out,
            "w1": layer.w1,
            "w2": layer.w2,
            "w3": layer.w3,
        }
    else:
        return None

    if fused_bits in {3, 4}:
        total_layers_guess = 0
        try:
            qwen_layer_ids = {
                int(name.split(".")[1])
                for name in weight_index.keys()
                if name.startswith("blk.") and len(name.split(".")) > 2 and name.split(".")[1].isdigit()
            }
            if qwen_layer_ids:
                total_layers_guess = max(qwen_layer_ids) + 1
        except Exception:
            total_layers_guess = 0

        int4_keep_first, int4_keep_last, int4_keep_sensitive = _resolve_int4_precision_windows(total_layers_guess)
        int4_keys_raw = os.getenv("VSPEC_INT4_MATRIX_KEYS", "wq,wk,wv,wo,w1,w2,w3,wqkv,wgate,ssm_alpha,ssm_beta,ssm_out")
        int4_allowed_keys = {k.strip().lower() for k in int4_keys_raw.split(",") if k.strip()}
        if not int4_allowed_keys:
            int4_allowed_keys = {"wq", "wk", "wv", "wo", "w1", "w2", "w3", "wqkv", "wgate", "ssm_alpha", "ssm_beta", "ssm_out"}
        sensitive_int4_keys = {"wq", "wk", "wo", "wqkv", "ssm_out"}

        pack_jobs: list[tuple[str, "np.ndarray", int]] = []
        for key, w in pack_targets.items():
            if w is None:
                continue
            matrix_bits = fused_bits
            if fused_bits == 4:
                in_first_window = int4_keep_first > 0 and layer_idx < int4_keep_first
                in_last_window = int4_keep_last > 0 and total_layers_guess > 0 and layer_idx >= max(0, total_layers_guess - int4_keep_last)
                if key.lower() not in int4_allowed_keys:
                    matrix_bits = 0
                elif in_first_window or in_last_window:
                    matrix_bits = 0
                elif int4_keep_sensitive and key in sensitive_int4_keys:
                    matrix_bits = 0

            if matrix_bits <= 0:
                continue

            pack_jobs.append((key, w, int(matrix_bits)))

        def _pack_one(entry: tuple[str, "np.ndarray", int]) -> tuple[str, tuple["np.ndarray", "np.ndarray", int, int, "np.ndarray | None"]]:
            key, w, matrix_bits = entry
            cache_key = _packed_cache_key(f"qwen35.layers.{layer_idx}.", key, matrix_bits, w)
            cached = _load_packed_cache(cache_key)
            if cached is not None:
                packed, scales, zero_points = cached
            else:
                packed, scales, zero_points = _quantize_weight_rowwise(w, matrix_bits)
                _save_packed_cache(cache_key, packed, scales, zero_points)
            return key, (packed, scales, matrix_bits, int(w.shape[0]), zero_points)

        workers = _resolve_pack_workers(len(pack_jobs))
        if workers <= 1 or len(pack_jobs) <= 1:
            for job in pack_jobs:
                key, packed_entry = _pack_one(job)
                layer.packed[key] = packed_entry
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_pack_one, job) for job in pack_jobs]
                for fut in as_completed(futures):
                    key, packed_entry = fut.result()
                    layer.packed[key] = packed_entry

        if layer.layer_type == "full_attention":
            _maybe_add_packed_combo(layer.packed, "wq_wk_wv", ["wq", "wk", "wv"])
        _maybe_add_packed_combo(layer.packed, "w1_w3", ["w1", "w3"])

    return layer


def _resolve_runtime_lowbit_plan(
    config: dict,
    weight_index: dict[str, WeightInfo],
    use_native_cuda_norm: bool,
    requested_fused_bits: int,
) -> tuple[LowbitModulePlan, int, QuantizationSourcePolicy]:
    quant_policy = resolve_quantization_source_policy(weight_index)
    if quant_policy.disable_runtime_quantization:
        profile = os.getenv("VSPEC_LOWBIT_PROFILE", "aggressive").strip().lower() or "aggressive"
        return (
            LowbitModulePlan(
                enabled=False,
                bits=0,
                compatible=True,
                reason=quant_policy.reason,
                profile=profile,
            ),
            0,
            quant_policy,
        )

    plan = build_lowbit_module_plan(config, use_native_cuda_norm, requested_fused_bits)
    effective_bits = plan.bits if plan.enabled else 0
    return plan, effective_bits, quant_policy


def _load_qwen35_runtime(
    config: dict,
    weight_index: dict[str, WeightInfo],
    max_layers: Optional[int],
    use_native_cuda_norm: bool,
    fused_bits_override: Optional[int] = None,
    progress_cb: Optional[Callable[[str, int, int], None]] = None,
) -> Optional[Qwen35Runtime]:
    try:
        if _validate_config_or_warn(config, "qwen35"):
            return None
        embed = _load_pair(
            weight_index,
            ["model.embed_tokens.weight", "token_embd.weight"],
            ["lm_head.weight", "output.weight"],
        )
        if embed is None:
            return None

        final_norm = _load_first_available(weight_index, "model.norm.weight")
        if final_norm is None:
            final_norm = _load_first_available(weight_index, "output_norm.weight")
        if final_norm is None:
            return None

        num_layers = int(config.get("num_hidden_layers", 0) or 0)
        if num_layers <= 0:
            return None
        if max_layers:
            num_layers = min(num_layers, max_layers)

        if fused_bits_override is not None:
            try:
                override_bits = int(fused_bits_override)
            except Exception:
                override_bits = 0
            fused_bits = override_bits if override_bits in {0, 3, 4} else 0
        else:
            env_value = os.getenv("VSPEC_FUSED_BITS")
            if env_value is not None:
                try:
                    env_override = int(env_value or "0")
                except Exception:
                    env_override = 0
                fused_bits = env_override if env_override in {0, 3, 4} else 0
            else:
                baseline_plan = resolve_runtime_baseline_plan(
                    config=config,
                    use_native_cuda_norm=use_native_cuda_norm,
                    int3_available=fused_linear_int3_available(),
                    int4_available=fused_linear_int4_available(),
                )
                fused_bits = baseline_plan.fused_bits

        lowbit_plan, fused_bits, quant_policy = _resolve_runtime_lowbit_plan(
            config,
            weight_index,
            use_native_cuda_norm,
            fused_bits,
        )

        layers: list[Qwen35LayerWeights] = []
        if progress_cb is not None:
            progress_cb("layer_load", 0, num_layers)
        layer_load_workers = _resolve_layer_load_workers(num_layers)
        if layer_load_workers <= 1 or num_layers <= 1:
            for idx in range(num_layers):
                layer = _load_qwen35_layer(weight_index, idx, fused_bits)
                if layer is None:
                    break
                layers.append(layer)
                if progress_cb is not None:
                    progress_cb("layer_load", idx + 1, num_layers)
        else:
            layer_map: dict[int, Qwen35LayerWeights] = {}
            try:
                with ThreadPoolExecutor(max_workers=layer_load_workers) as executor:
                    futures = {
                        executor.submit(_load_qwen35_layer, weight_index, idx, fused_bits): idx
                        for idx in range(num_layers)
                    }
                    completed = 0
                    for fut in as_completed(futures):
                        idx = futures[fut]
                        layer = fut.result()
                        if layer is not None:
                            layer_map[idx] = layer
                        completed += 1
                        if progress_cb is not None:
                            progress_cb("layer_load", min(completed, num_layers), num_layers)
                for idx in range(num_layers):
                    layer = layer_map.get(idx)
                    if layer is None:
                        break
                    layers.append(layer)
            except Exception:
                layers = []
                if progress_cb is not None:
                    progress_cb("layer_load", 0, num_layers)
                for idx in range(num_layers):
                    layer = _load_qwen35_layer(weight_index, idx, fused_bits)
                    if layer is None:
                        break
                    layers.append(layer)
                    if progress_cb is not None:
                        progress_cb("layer_load", idx + 1, num_layers)
        if not layers:
            return None

        if _diag_enabled():
            l0 = layers[0]
            _diag_print(
                "qwen35_layer0_shapes",
                "wq=", (None if l0.wq is None else tuple(int(v) for v in l0.wq.shape)),
                "wk=", (None if l0.wk is None else tuple(int(v) for v in l0.wk.shape)),
                "wv=", (None if l0.wv is None else tuple(int(v) for v in l0.wv.shape)),
                "wo=", (None if l0.wo is None else tuple(int(v) for v in l0.wo.shape)),
                "wqkv=", (None if l0.wqkv is None else tuple(int(v) for v in l0.wqkv.shape)),
            )

        lm_head = embed.lm_head
        hidden = int(embed.embed.shape[1])
        if lm_head.shape[0] != hidden and lm_head.shape[1] == hidden:
            lm_head = lm_head.T
        elif lm_head.shape[0] != hidden:
            return None

        stability = resolve_qwen35_stability_profile()
        q_residual_clamp_alpha = stability.residual_clamp_alpha
        if quant_policy.disable_runtime_quantization:
            q_residual_clamp_alpha = 0.0

        runtime = Qwen35Runtime(
            embed=embed.embed.astype(np.float32, copy=False),
            lm_head=lm_head.astype(np.float32, copy=False),
            lm_head_native=np.ascontiguousarray(lm_head.T, dtype=np.float32) if use_native_cuda_norm else None,
            final_norm=final_norm.astype(np.float32, copy=False),
            layers=layers,
            num_heads=int(config.get("num_attention_heads", 0) or 0),
            num_kv_heads=int(config.get("num_key_value_heads", 0) or 0),
            head_dim=int(config.get("head_dim", 0) or 0),
            rope_dim=int(config.get("rope_dimension_count", 0) or 0),
            linear_num_key_heads=int(config.get("linear_num_key_heads", 0) or 0),
            linear_num_value_heads=int(config.get("linear_num_value_heads", 0) or 0),
            linear_key_head_dim=int(config.get("linear_key_head_dim", 0) or 0),
            linear_value_head_dim=int(config.get("linear_value_head_dim", 0) or 0),
            linear_conv_kernel=int(config.get("linear_conv_kernel_dim", 0) or 0),
            rms_eps=float(config.get("rms_norm_eps", 1e-6) or 1e-6),
            eos_token_id=None,
            rope_theta=float(config.get("rope_theta", 10000.0) or 10000.0),
            position=0,
            cache_k=[],
            cache_v=[],
            cache_len=[],
            use_native_cuda_norm=use_native_cuda_norm,
            fused_bits=fused_bits,
            lowbit_plan=lowbit_plan,
            conv_states=[],
            ssm_states=[],
            attention_logit_clip=stability.attention_logit_clip,
            residual_error_buffers_attn=[],
            residual_error_buffers_ff=[],
            residual_feedback_gain=stability.residual_feedback_gain,
            residual_clamp_alpha=q_residual_clamp_alpha,
            logit_entropy_target=stability.logit_entropy_target,
            logit_margin_floor=stability.logit_margin_floor,
            logit_margin_gain=stability.logit_margin_gain,
            ssm_warmup_tokens=stability.ssm_warmup_tokens,
        )
        reg_ok, reg_fail = _warm_register_runtime_int4_handles(runtime)
        runtime.int4_pre_registered = int(reg_ok)
        runtime.int4_pre_register_failures = int(reg_fail)
        return runtime
    finally:
        _clear_safe_open_caches()


def _infer_head_dim(layers: list[LayerWeights], num_heads: int, num_kv_heads: int, hidden: int) -> int:
    if layers:
        first = layers[0]
        try:
            q_rows = int(first.wq.shape[0])
            if num_heads > 0 and q_rows > 0 and (q_rows % num_heads) == 0:
                return q_rows // num_heads
        except Exception:
            pass
        try:
            k_rows = int(first.wk.shape[0])
            if num_kv_heads > 0 and k_rows > 0 and (k_rows % num_kv_heads) == 0:
                return k_rows // num_kv_heads
        except Exception:
            pass
    return hidden // max(1, num_heads)


def _load_gpt2_layer(weight_index: dict[str, WeightInfo], layer_idx: int, fused_bits: int) -> Optional[GPT2LayerWeights]:
    prefix = f"transformer.h.{layer_idx}."
    c_attn = _load_first_available(weight_index, prefix + "attn.c_attn.weight")
    c_attn_bias = _load_first_available(weight_index, prefix + "attn.c_attn.bias")
    c_proj = _load_first_available(weight_index, prefix + "attn.c_proj.weight")
    c_proj_bias = _load_first_available(weight_index, prefix + "attn.c_proj.bias")
    c_fc = _load_first_available(weight_index, prefix + "mlp.c_fc.weight")
    c_fc_bias = _load_first_available(weight_index, prefix + "mlp.c_fc.bias")
    mlp_proj = _load_first_available(weight_index, prefix + "mlp.c_proj.weight")
    mlp_proj_bias = _load_first_available(weight_index, prefix + "mlp.c_proj.bias")
    ln_1_weight = _load_first_available(weight_index, prefix + "ln_1.weight")
    ln_1_bias = _load_first_available(weight_index, prefix + "ln_1.bias")
    ln_2_weight = _load_first_available(weight_index, prefix + "ln_2.weight")
    ln_2_bias = _load_first_available(weight_index, prefix + "ln_2.bias")

    if any(x is None for x in (c_attn, c_proj, c_fc, mlp_proj, ln_1_weight, ln_2_weight)):
        return None

    layer = GPT2LayerWeights(
        c_attn=c_attn,
        c_attn_bias=c_attn_bias,
        c_proj=c_proj,
        c_proj_bias=c_proj_bias,
        c_fc=c_fc,
        c_fc_bias=c_fc_bias,
        mlp_proj=mlp_proj,
        mlp_proj_bias=mlp_proj_bias,
        ln_1_weight=ln_1_weight,
        ln_1_bias=ln_1_bias,
        ln_2_weight=ln_2_weight,
        ln_2_bias=ln_2_bias,
    )

    if fused_bits in {3, 4}:
        for key, w in {
            "c_attn": layer.c_attn,
            "c_proj": layer.c_proj,
            "c_fc": layer.c_fc,
            "mlp_proj": layer.mlp_proj,
        }.items():
            cache_key = _packed_cache_key(f"gpt2.layers.{layer_idx}.", key, fused_bits, w)
            cached = _load_packed_cache(cache_key)
            if cached is not None:
                packed, scales, zero_points = cached
            else:
                packed, scales, zero_points = _quantize_weight_rowwise(w, fused_bits)
                _save_packed_cache(cache_key, packed, scales, zero_points)
            layer.packed[key] = (packed, scales, fused_bits, int(w.shape[0]), zero_points)

    return layer


def _load_gpt2_runtime(
    config: dict,
    weight_index: dict[str, WeightInfo],
    max_layers: Optional[int],
    use_native_cuda_norm: bool,
    fused_bits_override: Optional[int] = None,
    progress_cb: Optional[Callable[[str, int, int], None]] = None,
) -> Optional[GPT2Runtime]:
    try:
        if _validate_config_or_warn(config, "gpt2"):
            return None
        embed = _load_first_available(weight_index, "transformer.wte.weight")
        pos_embed = _load_first_available(weight_index, "transformer.wpe.weight")
        lm_head = _load_first_available(weight_index, "lm_head.weight")
        if lm_head is None:
            lm_head = embed
        final_norm_weight = _load_first_available(weight_index, "transformer.ln_f.weight")
        final_norm_bias = _load_first_available(weight_index, "transformer.ln_f.bias")
        if any(x is None for x in (embed, pos_embed, lm_head, final_norm_weight)):
            return None

        num_layers = int(config.get("num_hidden_layers", 0) or config.get("n_layer", 0) or 0)
        if num_layers <= 0:
            num_layers = 1
        if max_layers:
            num_layers = min(num_layers, max_layers)

        if fused_bits_override is not None:
            try:
                fused_bits = int(fused_bits_override)
            except Exception:
                fused_bits = 0
            fused_bits = fused_bits if fused_bits in {0, 3, 4} else 0
        else:
            env_value = os.getenv("VSPEC_FUSED_BITS")
            if env_value is not None:
                try:
                    env_override = int(env_value or "0")
                except Exception:
                    env_override = 0
                fused_bits = env_override if env_override in {0, 3, 4} else 0
            else:
                baseline_plan = resolve_runtime_baseline_plan(
                    config=config,
                    use_native_cuda_norm=use_native_cuda_norm,
                    int3_available=fused_linear_int3_available(),
                    int4_available=fused_linear_int4_available(),
                )
                fused_bits = baseline_plan.fused_bits

        lowbit_plan, fused_bits, _ = _resolve_runtime_lowbit_plan(
            config,
            weight_index,
            use_native_cuda_norm,
            fused_bits,
        )

        layers: list[GPT2LayerWeights] = []
        if progress_cb is not None:
            progress_cb("layer_load", 0, num_layers)
        for idx in range(num_layers):
            layer = _load_gpt2_layer(weight_index, idx, fused_bits)
            if layer is None:
                break
            layers.append(layer)
            if progress_cb is not None:
                progress_cb("layer_load", idx + 1, num_layers)
        if not layers:
            return None

        hidden = int(embed.shape[1])
        num_heads = int(config.get("num_attention_heads", 0) or config.get("n_head", 0) or 0)
        if num_heads <= 0:
            num_heads = 1
        head_dim = hidden // max(1, num_heads)
        ln_eps = float(config.get("layer_norm_epsilon", 1e-5) or 1e-5)

        if lm_head.shape[0] != hidden and lm_head.shape[1] == hidden:
            lm_head = lm_head.T
        elif lm_head.shape[0] != hidden:
            return None

        runtime = GPT2Runtime(
            embed=embed.astype(np.float32, copy=False),
            pos_embed=pos_embed.astype(np.float32, copy=False),
            lm_head=lm_head.astype(np.float32, copy=False),
            final_norm_weight=final_norm_weight.astype(np.float32, copy=False),
            final_norm_bias=None if final_norm_bias is None else final_norm_bias.astype(np.float32, copy=False),
            layers=layers,
            num_heads=num_heads,
            num_kv_heads=num_heads,
            head_dim=head_dim,
            ln_eps=ln_eps,
            eos_token_id=None,
            position=0,
            cache_k=[],
            cache_v=[],
            cache_len=[],
            fused_bits=fused_bits,
            lowbit_plan=lowbit_plan,
        )
        reg_ok, reg_fail = _warm_register_runtime_int4_handles(runtime)
        runtime.int4_pre_registered = int(reg_ok)
        runtime.int4_pre_register_failures = int(reg_fail)
        return runtime
    finally:
        _clear_safe_open_caches()


def _load_generic_transformer(
    config: dict,
    weight_index: dict[str, WeightInfo],
    max_layers: Optional[int],
    use_native_cuda_norm: bool,
    fused_bits_override: Optional[int] = None,
    progress_cb: Optional[Callable[[str, int, int], None]] = None,
) -> Optional[GenericTransformerRuntime]:
    try:
        if _validate_config_or_warn(config, "generic"):
            return None
        embed = _load_pair(
            weight_index,
            [
                "model.embed_tokens.weight",
                "tok_embeddings.weight",
                "transformer.wte.weight",
            ],
            [
                "lm_head.weight",
                "output.weight",
                "transformer.lm_head.weight",
            ],
        )
        if embed is None:
            return None

        embed_weight = embed.embed
        lm_head = embed.lm_head

        final_norm = _load_first_available(weight_index, "model.norm.weight")
        if final_norm is None:
            final_norm = _load_first_available(weight_index, "norm.weight")
        if final_norm is None:
            return None

        num_layers = int(config.get("num_hidden_layers", 0) or config.get("n_layer", 0) or 0)
        if num_layers == 0:
            num_layers = 1
        if max_layers:
            num_layers = min(num_layers, max_layers)

        if fused_bits_override is not None:
            try:
                override_bits = int(fused_bits_override)
            except Exception:
                override_bits = 0
            fused_bits = override_bits if override_bits in {0, 3, 4} else 0
        else:
            env_value = os.getenv("VSPEC_FUSED_BITS")
            if env_value is not None:
                try:
                    env_override = int(env_value or "0")
                except Exception:
                    env_override = 0
                fused_bits = env_override if env_override in {0, 3, 4} else 0
            else:
                baseline_plan = resolve_runtime_baseline_plan(
                    config=config,
                    use_native_cuda_norm=use_native_cuda_norm,
                    int3_available=fused_linear_int3_available(),
                    int4_available=fused_linear_int4_available(),
                )
                fused_bits = baseline_plan.fused_bits

        lowbit_plan, fused_bits, quant_policy = _resolve_runtime_lowbit_plan(
            config,
            weight_index,
            use_native_cuda_norm,
            fused_bits,
        )

        layers: list[LayerWeights] = []
        if progress_cb is not None:
            progress_cb("layer_load", 0, num_layers)
        layer_load_workers = _resolve_layer_load_workers(num_layers)
        if layer_load_workers <= 1 or num_layers <= 1:
            for idx in range(num_layers):
                layer = _load_layer(weight_index, idx, fused_bits, num_layers)
                if layer is None:
                    break
                layers.append(layer)
                if progress_cb is not None:
                    progress_cb("layer_load", idx + 1, num_layers)
        else:
            layer_map: dict[int, LayerWeights] = {}
            try:
                with ThreadPoolExecutor(max_workers=layer_load_workers) as executor:
                    futures = {
                        executor.submit(_load_layer, weight_index, idx, fused_bits, num_layers): idx
                        for idx in range(num_layers)
                    }
                    completed = 0
                    for fut in as_completed(futures):
                        idx = futures[fut]
                        layer = fut.result()
                        if layer is not None:
                            layer_map[idx] = layer
                        completed += 1
                        if progress_cb is not None:
                            progress_cb("layer_load", min(completed, num_layers), num_layers)
                for idx in range(num_layers):
                    layer = layer_map.get(idx)
                    if layer is None:
                        break
                    layers.append(layer)
            except Exception:
                layers = []
                if progress_cb is not None:
                    progress_cb("layer_load", 0, num_layers)
                for idx in range(num_layers):
                    layer = _load_layer(weight_index, idx, fused_bits, num_layers)
                    if layer is None:
                        break
                    layers.append(layer)
                    if progress_cb is not None:
                        progress_cb("layer_load", idx + 1, num_layers)

        if not layers:
            return None

        hidden = embed_weight.shape[1]
        num_heads = int(config.get("num_attention_heads", 0) or config.get("n_head", 0) or 0)
        num_kv_heads = int(config.get("num_key_value_heads", 0) or config.get("n_kv_head", 0) or num_heads)
        if num_heads <= 0:
            num_heads = 32
        if num_kv_heads <= 0:
            num_kv_heads = num_heads
        head_dim = _infer_head_dim(layers, num_heads, num_kv_heads, hidden)

        rms_eps = float(config.get("rms_norm_eps", 1e-6))
        rope_theta = float(config.get("rope_theta", 10000.0))
        stability = resolve_model_stability_profile(_infer_model_family(config))
        g_residual_clamp_alpha = stability.residual_clamp_alpha
        if quant_policy.disable_runtime_quantization:
            g_residual_clamp_alpha = 0.0
        flash_attention_min_tokens = int(os.getenv("VSPEC_FLASH_ATTN_MIN_TOKENS", "1") or "1")
        flash_attention_block_tokens = int(os.getenv("VSPEC_FLASH_ATTN_BLOCK_TOKENS", "128") or "128")

        runtime = GenericTransformerRuntime(
            embed=embed_weight,
            lm_head=lm_head,
            lm_head_native=np.ascontiguousarray(lm_head.T, dtype=np.float32) if use_native_cuda_norm else None,
            final_norm=final_norm,
            layers=layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rms_eps=rms_eps,
            eos_token_id=None,
            rope_theta=rope_theta,
            position=0,
            cache_k=[],
            cache_v=[],
            cache_len=[],
            use_native_cuda_norm=use_native_cuda_norm,
            fused_bits=fused_bits,
            disable_fused_attention=(os.getenv("VSPEC_DISABLE_FUSED_ATTN", "0").strip().lower() in {"1", "true", "yes", "on"}),
            flash_attention_min_tokens=max(1, flash_attention_min_tokens),
            flash_attention_block_tokens=max(1, flash_attention_block_tokens),
            inv_sqrt_head_dim=(1.0 / np.sqrt(float(head_dim))),
            lowbit_plan=lowbit_plan,
            rope_inv_freq=(1.0 / (rope_theta ** (np.arange(head_dim // 2, dtype=np.float32) / max(1, (head_dim // 2))))).astype(np.float32, copy=False),
            rope_cos_cache=[],
            rope_sin_cache=[],
            attention_logit_clip=stability.attention_logit_clip,
            attn_tmp_buffers=[],
            residual_error_buffers_attn=[],
            residual_error_buffers_ff=[],
            residual_feedback_gain=stability.residual_feedback_gain,
            residual_clamp_alpha=g_residual_clamp_alpha,
            logit_entropy_target=stability.logit_entropy_target,
            logit_margin_floor=stability.logit_margin_floor,
            logit_margin_gain=stability.logit_margin_gain,
            phase3_flash_attn_calls=0,
            phase3_fused_attn_calls=0,
            phase3_scalar_attn_calls=0,
            phase3_cpu_attn_calls=0,
        )
        reg_ok, reg_fail = _warm_register_runtime_int4_handles(runtime)
        runtime.int4_pre_registered = int(reg_ok)
        runtime.int4_pre_register_failures = int(reg_fail)
        return runtime
    finally:
        _clear_safe_open_caches()


def _load_generic_transformer_torch(
    config: dict,
    weight_index: dict[str, WeightInfo],
    max_layers: Optional[int],
    device: str,
    progress_cb: Optional[Callable[[str, int, int], None]] = None,
) -> Optional[GenericTransformerRuntimeTorch]:
    if torch is None:
        return None
    try:
        if _validate_config_or_warn(config, "generic"):
            return None
        embed_info = _select_weight(weight_index, [
            "model.embed_tokens.weight",
            "tok_embeddings.weight",
            "transformer.wte.weight",
        ])
        lm_info = _select_weight(weight_index, [
            "lm_head.weight",
            "output.weight",
            "transformer.lm_head.weight",
        ])
        if embed_info is None or lm_info is None:
            return None

        embed_weight = _load_tensor_torch(embed_info, device)
        lm_head = _load_tensor_torch(lm_info, device)
        if embed_weight is None or lm_head is None:
            return None

        embed_vocab, embed_dim = embed_weight.shape
        if lm_head.shape[0] == embed_dim:
            pass
        elif lm_head.shape[1] == embed_dim:
            lm_head = lm_head.t()
        else:
            return None

        final_norm = _load_first_available_torch(weight_index, "model.norm.weight", device)
        if final_norm is None:
            final_norm = _load_first_available_torch(weight_index, "norm.weight", device)
        if final_norm is None:
            return None

        num_layers = int(config.get("num_hidden_layers", 0) or config.get("n_layer", 0) or 0)
        if num_layers == 0:
            num_layers = 1
        if max_layers:
            num_layers = min(num_layers, max_layers)

        layers: list[LayerWeights] = []
        if progress_cb is not None:
            progress_cb("layer_load", 0, num_layers)
        for idx in range(num_layers):
            layer = _load_layer_torch(weight_index, idx, device)
            if layer is None:
                break
            layers.append(layer)
            if progress_cb is not None:
                progress_cb("layer_load", idx + 1, num_layers)

        if not layers:
            return None

        hidden = embed_weight.shape[1]
        num_heads = int(config.get("num_attention_heads", 0) or config.get("n_head", 0) or 0)
        num_kv_heads = int(config.get("num_key_value_heads", 0) or config.get("n_kv_head", 0) or num_heads)
        if num_heads <= 0:
            num_heads = 32
        if num_kv_heads <= 0:
            num_kv_heads = num_heads
        head_dim = _infer_head_dim(layers, num_heads, num_kv_heads, hidden)

        rms_eps = float(config.get("rms_norm_eps", 1e-6))
        rope_theta = float(config.get("rope_theta", 10000.0))

        return GenericTransformerRuntimeTorch(
            embed=embed_weight,
            lm_head=lm_head,
            final_norm=final_norm,
            layers=layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rms_eps=rms_eps,
            eos_token_id=None,
            rope_theta=rope_theta,
            position=0,
            cache_k=[],
            cache_v=[],
        )
    finally:
        _clear_safe_open_caches()


def build_runtime(adapter: ModelAdapter, weight_index: dict[str, WeightInfo]) -> Optional[SimpleRuntime]:
    return None


def runtime_matrix_bits_summary(layers: list[LayerWeights]) -> tuple[str, float, bool, float]:
    if not layers:
        return "none", 0.0, False, 0.0

    if hasattr(layers[0], "layer_type"):
        per_layer_values: list[float] = []
        has_lowbit = False
        total_matrices = 0
        lowbit_matrices = 0
        for layer in layers:
            if getattr(layer, "layer_type", "") == "linear_attention":
                matrix_order = ["wqkv", "wgate", "ssm_out", "w1", "w2", "w3"]
            else:
                matrix_order = ["wq", "wk", "wv", "wo", "w1", "w2", "w3"]
            vals: list[float] = []
            for key in matrix_order:
                total_matrices += 1
                packed_entry = layer.packed.get(key)
                if packed_entry is None:
                    vals.append(16.0)
                else:
                    bit_val = float(int(packed_entry[2]))
                    vals.append(bit_val)
                    if bit_val <= 4.0:
                        has_lowbit = True
                        lowbit_matrices += 1
            per_layer_values.append(sum(vals) / float(len(vals)))
        if len(per_layer_values) <= 8:
            summary = ",".join(f"{v:.1f}" for v in per_layer_values)
        else:
            head = ",".join(f"{v:.1f}" for v in per_layer_values[:4])
            tail = ",".join(f"{v:.1f}" for v in per_layer_values[-4:])
            summary = f"{head},...,{tail}"
        eff = float(sum(per_layer_values) / float(len(per_layer_values)))
        coverage = float(lowbit_matrices) / float(max(1, total_matrices)) if total_matrices > 0 else 0.0
        return summary, eff, has_lowbit, coverage
    if hasattr(layers[0], "wq"):
        matrix_order = ["wq", "wk", "wv", "wo", "w1", "w2", "w3"]
    else:
        matrix_order = ["c_attn", "c_proj", "c_fc", "mlp_proj"]
    per_layer_values: list[float] = []
    has_lowbit = False
    total_matrices = 0
    lowbit_matrices = 0

    for layer in layers:
        vals: list[float] = []
        for key in matrix_order:
            total_matrices += 1
            packed_entry = layer.packed.get(key)
            if packed_entry is None:
                vals.append(16.0)
            else:
                bit_val = float(int(packed_entry[2]))
                vals.append(bit_val)
                if bit_val <= 4.0:
                    has_lowbit = True
                    lowbit_matrices += 1
        per_layer_values.append(sum(vals) / float(len(vals)))

    if len(per_layer_values) <= 8:
        summary = ",".join(f"{v:.1f}" for v in per_layer_values)
    else:
        head = ",".join(f"{v:.1f}" for v in per_layer_values[:4])
        tail = ",".join(f"{v:.1f}" for v in per_layer_values[-4:])
        summary = f"{head},...,{tail}"

    eff = float(sum(per_layer_values) / float(len(per_layer_values)))
    coverage = float(lowbit_matrices) / float(max(1, total_matrices)) if total_matrices > 0 else 0.0
    return summary, eff, has_lowbit, coverage


def _warm_register_runtime_int4_handles(runtime_obj) -> tuple[int, int]:
    if np is None or runtime_obj is None:
        return 0, 0
    enabled = os.getenv("VSPEC_INT4_PRE_REGISTER", "1").strip().lower() in {"1", "true", "yes", "on"}
    if not enabled or not fused_linear_int4_registered_available():
        return 0, 0

    layers = list(getattr(runtime_obj, "layers", []) or [])
    if not layers:
        return 0, 0

    required_handles = 0
    for layer in layers:
        packed_map = getattr(layer, "packed", None)
        if not isinstance(packed_map, dict) or not packed_map:
            continue
        for entry in packed_map.values():
            if entry is None or len(entry) < 5:
                continue
            try:
                if int(entry[2]) == 4:
                    required_handles += 1
            except Exception:
                continue

    try:
        bridge_cap, _ = get_lowbit_bridge_cache_caps()
    except Exception:
        bridge_cap = 256
    if int(bridge_cap) < int(max(64, required_handles)):
        return 0, int(required_handles)

    registered = 0
    failures = 0
    for layer in layers:
        packed_map = getattr(layer, "packed", None)
        if not isinstance(packed_map, dict) or not packed_map:
            continue
        for key, entry in list(packed_map.items()):
            if entry is None or len(entry) < 5:
                continue
            if not hasattr(layer, key):
                continue
            packed_w, scales, bits, out_n, zero_points = entry[:5]
            if int(bits) != 4:
                continue
            handle = 0
            if len(entry) >= 6:
                try:
                    handle = int(entry[5] or 0)
                except Exception:
                    handle = 0
            if handle > 0:
                continue

            try:
                k_in = int(getattr(layer, key).shape[-1])
            except Exception:
                failures += 1
                continue

            try:
                handle = int(fused_linear_int4_register_weight(packed_w, scales, int(out_n), k_in, zero_points=zero_points) or 0)
            except Exception:
                handle = 0
            if handle > 0:
                packed_map[key] = (packed_w, scales, bits, out_n, zero_points, handle)
                registered += 1
            else:
                failures += 1

    return registered, failures


def build_generic_runtime(
    config: dict,
    weight_index: dict[str, WeightInfo],
    max_layers: Optional[int],
    device: str,
    fused_bits_override: Optional[int] = None,
    use_native_cuda_norm_override: Optional[bool] = None,
    progress_cb: Optional[Callable[[str, int, int], None]] = None,
) -> Optional[object]:
    runtime_target = resolve_runtime_target(config, list(weight_index.keys()), weight_index)
    quant_policy = resolve_quantization_source_policy(weight_index)
    runtime_config = runtime_target.config
    model_type = runtime_target.runtime_name
    _diag_print(
        "route",
        "runtime=", model_type,
        "reason=", runtime_target.reason,
        "warnings=", ";".join(runtime_target.warnings),
        "model_type=", str(runtime_config.get("model_type", "")),
        "hidden=", int(runtime_config.get("hidden_size", 0) or 0),
        "layers=", int(runtime_config.get("num_hidden_layers", 0) or 0),
        "heads=", int(runtime_config.get("num_attention_heads", 0) or 0),
        "head_dim=", int(runtime_config.get("head_dim", 0) or 0),
    )
    if model_type == "gpt2":
        return _load_gpt2_runtime(
            runtime_config,
            weight_index,
            max_layers,
            device in {"cuda", "cuda-native"},
            fused_bits_override=fused_bits_override,
            progress_cb=progress_cb,
        )
    if model_type == "qwen35":
        runtime = _load_qwen35_runtime(
            runtime_config,
            weight_index,
            max_layers,
            device in {"cuda", "cuda-native"},
            fused_bits_override=fused_bits_override,
            progress_cb=progress_cb,
        )
        if runtime is not None:
            runtime.compat_reason = runtime_target.reason
            runtime.compat_warnings = list(runtime_target.warnings)
            runtime.quant_source_format = quant_policy.source_format
            runtime.quant_source_quantized = bool(quant_policy.source_quantized)
            runtime.quant_runtime_disabled = bool(quant_policy.disable_runtime_quantization)
            runtime.quant_policy_reason = quant_policy.reason
        return runtime

    any_weight = next(iter(weight_index.values()), None)
    has_gguf = getattr(any_weight, "source_format", "safetensors") == "gguf"
    if device == "torch-cuda" and (not has_gguf) and torch is not None and torch.cuda.is_available():
        runtime = _load_generic_transformer_torch(runtime_config, weight_index, max_layers, "cuda", progress_cb)
        if runtime is not None:
            runtime.compat_reason = runtime_target.reason
            runtime.compat_warnings = list(runtime_target.warnings)
            runtime.quant_source_format = quant_policy.source_format
            runtime.quant_source_quantized = bool(quant_policy.source_quantized)
            runtime.quant_runtime_disabled = bool(quant_policy.disable_runtime_quantization)
            runtime.quant_policy_reason = quant_policy.reason
            return runtime
    use_native_cuda_norm = device in {"cuda", "cuda-native"}
    if use_native_cuda_norm_override is not None:
        use_native_cuda_norm = bool(use_native_cuda_norm_override)
    runtime = _load_generic_transformer(
        runtime_config,
        weight_index,
        max_layers,
        use_native_cuda_norm,
        fused_bits_override=fused_bits_override,
        progress_cb=progress_cb,
    )
    if runtime is not None and np is not None:
        runtime.compat_reason = runtime_target.reason
        runtime.compat_warnings = list(runtime_target.warnings)
        runtime.quant_source_format = quant_policy.source_format
        runtime.quant_source_quantized = bool(quant_policy.source_quantized)
        runtime.quant_runtime_disabled = bool(quant_policy.disable_runtime_quantization)
        runtime.quant_policy_reason = quant_policy.reason
        runtime.kv_core_mirror_enabled = False
        runtime.kv_core_mirror_count = 0
        runtime.kv_python_shadow_disabled = False
        try:
            mirror_disabled = os.getenv("VSPEC_DISABLE_CORE_KV_MIRROR", "0").strip().lower() in {"1", "true", "yes", "on"}
            shadow_disabled = os.getenv("VSPEC_DISABLE_PY_KV_SHADOW", "0").strip().lower() in {"1", "true", "yes", "on"}
            mirror_only_experimental = os.getenv("VSPEC_EXPERIMENTAL_MIRROR_ONLY", "0").strip().lower() in {"1", "true", "yes", "on"}
            if mirror_disabled:
                runtime.kv_core_mirrors = []
            else:
                page_tokens = max(16, int(os.getenv("VSPEC_CORE_KV_PAGE_TOKENS", "64") or "64"))
                max_tokens = max(128, int(os.getenv("VSPEC_CORE_KV_MAX_TOKENS", "4096") or "4096"))
                max_pages = max(4, (max_tokens + page_tokens - 1) // page_tokens)
                runtime.kv_core_mirrors = [
                    CorePagedKVCache(page_tokens, max_pages, runtime.num_kv_heads, runtime.head_dim, session_id=idx + 1)
                    for idx in range(len(getattr(runtime, "layers", []) or []))
                ]
                runtime.kv_core_mirror_count = len(runtime.kv_core_mirrors)
                runtime.kv_core_mirror_enabled = runtime.kv_core_mirror_count > 0
                runtime.kv_python_shadow_disabled = bool(runtime.kv_core_mirror_enabled and shadow_disabled and mirror_only_experimental)
        except Exception:
            runtime.kv_core_mirrors = []
            runtime.kv_core_mirror_enabled = False
            runtime.kv_core_mirror_count = 0
            runtime.kv_python_shadow_disabled = False
    return runtime
