from __future__ import annotations

import ctypes
import os
from ctypes import POINTER, c_char_p, c_float, c_int, c_size_t, c_uint32, c_uint64, c_uint8, create_string_buffer
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


def _load_lib() -> ctypes.CDLL | None:
    root = Path(__file__).resolve().parents[2]
    candidates = [
        root / "build" / "Release" / "vspec_engine_capi.dll",
        root / "build" / "vspec_engine_capi.dll",
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                return ctypes.CDLL(str(candidate))
            except Exception:
                pass
    return None


_lib = _load_lib()
_HAS_GRAPH_CACHE_STATS = False

if _lib is not None:
    _lib.vspec_py_weight_canonical_name.argtypes = [c_char_p, ctypes.c_char_p, c_size_t]
    _lib.vspec_py_weight_canonical_name.restype = c_int
    _lib.vspec_py_sample_candidate.argtypes = [POINTER(c_int), POINTER(c_float), c_size_t, c_int, c_uint64, POINTER(c_int)]
    _lib.vspec_py_sample_candidate.restype = c_int
    _lib.vspec_py_kv_cache_create.argtypes = [c_size_t, c_size_t, c_size_t, c_size_t]
    _lib.vspec_py_kv_cache_create.restype = c_int
    _lib.vspec_py_kv_cache_destroy.argtypes = [c_int]
    _lib.vspec_py_kv_cache_destroy.restype = None
    _lib.vspec_py_kv_cache_reset.argtypes = [c_int]
    _lib.vspec_py_kv_cache_reset.restype = c_int
    _lib.vspec_py_kv_cache_append.argtypes = [c_int, c_uint64, POINTER(c_float), POINTER(c_float)]
    _lib.vspec_py_kv_cache_append.restype = c_int
    _lib.vspec_py_kv_cache_session_tokens.argtypes = [c_int, c_uint64]
    _lib.vspec_py_kv_cache_session_tokens.restype = c_size_t
    _lib.vspec_py_kv_cache_read.argtypes = [c_int, c_uint64, POINTER(c_float), POINTER(c_float), c_size_t]
    _lib.vspec_py_kv_cache_read.restype = c_size_t
    _lib.vspec_py_decode_session_create.argtypes = [c_size_t, c_size_t, c_size_t, c_size_t]
    _lib.vspec_py_decode_session_create.restype = c_int
    _lib.vspec_py_decode_session_destroy.argtypes = [c_int]
    _lib.vspec_py_decode_session_destroy.restype = None
    _lib.vspec_py_decode_session_begin.argtypes = [c_int, c_size_t, c_size_t, c_size_t, ctypes.c_uint16]
    _lib.vspec_py_decode_session_begin.restype = c_int
    _lib.vspec_py_decode_session_next_quota.argtypes = [c_int]
    _lib.vspec_py_decode_session_next_quota.restype = c_size_t
    _lib.vspec_py_decode_session_commit.argtypes = [c_int, c_size_t, c_int]
    _lib.vspec_py_decode_session_commit.restype = c_int
    _lib.vspec_py_decode_session_cancel.argtypes = [c_int]
    _lib.vspec_py_decode_session_cancel.restype = c_int
    _lib.vspec_py_decode_session_is_active.argtypes = [c_int]
    _lib.vspec_py_decode_session_is_active.restype = c_int
    _lib.vspec_py_decode_session_remaining_tokens.argtypes = [c_int]
    _lib.vspec_py_decode_session_remaining_tokens.restype = c_size_t
    _lib.vspec_py_native_decode_loop_create.argtypes = [c_size_t, c_size_t, c_size_t, c_size_t]
    _lib.vspec_py_native_decode_loop_create.restype = c_int
    _lib.vspec_py_native_decode_loop_destroy.argtypes = [c_int]
    _lib.vspec_py_native_decode_loop_destroy.restype = None
    _lib.vspec_py_native_decode_loop_begin.argtypes = [c_int, c_size_t, c_size_t, c_size_t, ctypes.c_uint16, c_uint64]
    _lib.vspec_py_native_decode_loop_begin.restype = c_int
    _lib.vspec_py_native_decode_loop_next_quota.argtypes = [c_int]
    _lib.vspec_py_native_decode_loop_next_quota.restype = c_size_t
    _lib.vspec_py_native_decode_loop_commit.argtypes = [c_int, c_size_t, c_int]
    _lib.vspec_py_native_decode_loop_commit.restype = c_int
    _lib.vspec_py_native_decode_loop_cancel.argtypes = [c_int]
    _lib.vspec_py_native_decode_loop_cancel.restype = c_int
    _lib.vspec_py_native_decode_loop_stats.argtypes = [c_int, POINTER(c_uint64), POINTER(c_uint64), POINTER(c_uint64), POINTER(c_uint64)]
    _lib.vspec_py_native_decode_loop_stats.restype = c_int
    _HAS_GRAPH_CACHE_STATS = hasattr(_lib, "vspec_py_native_decode_loop_graph_cache_stats")
    if _HAS_GRAPH_CACHE_STATS:
        _lib.vspec_py_native_decode_loop_graph_cache_stats.argtypes = [c_int, POINTER(c_uint64), POINTER(c_uint64), POINTER(c_uint64), POINTER(c_int)]
        _lib.vspec_py_native_decode_loop_graph_cache_stats.restype = c_int
    else:
        _HAS_GRAPH_CACHE_STATS = False
    _lib.vspec_py_continuous_batch_create.argtypes = [c_size_t, c_size_t, c_size_t, c_size_t, c_size_t, c_size_t]
    _lib.vspec_py_continuous_batch_create.restype = c_int
    _lib.vspec_py_continuous_batch_destroy.argtypes = [c_int]
    _lib.vspec_py_continuous_batch_destroy.restype = None
    _lib.vspec_py_continuous_batch_submit.argtypes = [c_int, c_size_t, c_size_t, c_size_t, ctypes.c_uint16, POINTER(c_uint64)]
    _lib.vspec_py_continuous_batch_submit.restype = c_int
    _lib.vspec_py_continuous_batch_next.argtypes = [c_int, POINTER(c_uint64), POINTER(ctypes.c_uint32), POINTER(c_size_t), POINTER(c_size_t), c_size_t]
    _lib.vspec_py_continuous_batch_next.restype = c_size_t
    _lib.vspec_py_continuous_batch_commit_prefill.argtypes = [c_int, c_uint64, c_size_t]
    _lib.vspec_py_continuous_batch_commit_prefill.restype = c_int
    _lib.vspec_py_continuous_batch_commit_decode.argtypes = [c_int, c_uint64, c_size_t, c_int]
    _lib.vspec_py_continuous_batch_commit_decode.restype = c_int
    _lib.vspec_py_continuous_batch_cancel.argtypes = [c_int, c_uint64]
    _lib.vspec_py_continuous_batch_cancel.restype = c_int
    _lib.vspec_py_continuous_batch_stats.argtypes = [c_int, POINTER(c_uint64), POINTER(c_uint64), POINTER(c_uint64), POINTER(c_uint64), POINTER(c_size_t), POINTER(c_size_t), POINTER(c_size_t)]
    _lib.vspec_py_continuous_batch_stats.restype = c_int
    _lib.vspec_py_runtime_adaptive_step.argtypes = [
        c_char_p,
        c_float,
        c_float,
        c_float,
        c_float,
        c_float,
        c_uint32,
        POINTER(c_uint8),
        POINTER(c_uint8),
        POINTER(c_uint8),
        POINTER(c_uint8),
        POINTER(c_uint8),
        POINTER(c_uint32),
        POINTER(c_uint32),
        POINTER(c_float),
        POINTER(c_uint32),
    ]
    _lib.vspec_py_runtime_adaptive_step.restype = c_int
    _lib.vspec_py_runtime_output_guard_init.argtypes = [c_float]
    _lib.vspec_py_runtime_output_guard_init.restype = c_int
    _lib.vspec_py_runtime_output_guard_allow.argtypes = [c_char_p]
    _lib.vspec_py_runtime_output_guard_allow.restype = c_int
    _lib.vspec_py_runtime_output_guard_score_adjustment.argtypes = [c_char_p]
    _lib.vspec_py_runtime_output_guard_score_adjustment.restype = c_float
    _lib.vspec_py_runtime_output_guard_observe.argtypes = [c_char_p]
    _lib.vspec_py_runtime_output_guard_observe.restype = c_int
    _lib.vspec_py_native_forward_create.argtypes = [c_char_p, c_uint64]
    _lib.vspec_py_native_forward_create.restype = c_int
    _lib.vspec_py_native_forward_destroy.argtypes = [c_int]
    _lib.vspec_py_native_forward_destroy.restype = None
    _lib.vspec_py_native_forward_step.argtypes = [c_int, c_char_p, c_size_t, POINTER(c_int), POINTER(c_float), c_size_t, c_float, POINTER(c_float)]
    _lib.vspec_py_native_forward_step.restype = c_int
    _lib.vspec_py_plugin_load_dynamic.argtypes = [c_char_p, c_char_p, ctypes.c_char_p, c_size_t]
    _lib.vspec_py_plugin_load_dynamic.restype = c_int
    _lib.vspec_py_plugin_unload_dynamic.argtypes = [c_char_p, ctypes.c_char_p, c_size_t]
    _lib.vspec_py_plugin_unload_dynamic.restype = c_int


@dataclass
class AdaptiveStepDecision:
    target_bits: int
    skip_compute: bool
    reduce_attention_depth: bool
    enable_kv_compression: bool
    routed_bits: int
    attention_depth_hint: int
    token_tier: int
    token_importance: float
    kv_action: int


def adaptive_step(
    token_text: str,
    token_entropy: float,
    attention_entropy_collapse: float,
    latency_ms: float,
    vram_pressure: float,
    quality_drift: float,
    layer_type: int = 0,
) -> AdaptiveStepDecision | None:
    if _lib is None:
        return None
    out_target_bits = c_uint8(0)
    out_skip_compute = c_uint8(0)
    out_reduce_attention_depth = c_uint8(0)
    out_enable_kv_compression = c_uint8(0)
    out_routed_bits = c_uint8(0)
    out_attention_depth_hint = c_uint32(0)
    out_token_tier = c_uint32(0)
    out_token_importance = c_float(0.0)
    out_kv_action = c_uint32(0)
    ok = _lib.vspec_py_runtime_adaptive_step(
        (token_text or "").encode("utf-8", errors="ignore"),
        c_float(float(token_entropy)),
        c_float(float(attention_entropy_collapse)),
        c_float(float(latency_ms)),
        c_float(float(vram_pressure)),
        c_float(float(quality_drift)),
        c_uint32(max(0, int(layer_type))),
        ctypes.byref(out_target_bits),
        ctypes.byref(out_skip_compute),
        ctypes.byref(out_reduce_attention_depth),
        ctypes.byref(out_enable_kv_compression),
        ctypes.byref(out_routed_bits),
        ctypes.byref(out_attention_depth_hint),
        ctypes.byref(out_token_tier),
        ctypes.byref(out_token_importance),
        ctypes.byref(out_kv_action),
    )
    if not ok:
        return None
    return AdaptiveStepDecision(
        target_bits=int(out_target_bits.value),
        skip_compute=bool(out_skip_compute.value),
        reduce_attention_depth=bool(out_reduce_attention_depth.value),
        enable_kv_compression=bool(out_enable_kv_compression.value),
        routed_bits=int(out_routed_bits.value),
        attention_depth_hint=int(out_attention_depth_hint.value),
        token_tier=int(out_token_tier.value),
        token_importance=float(out_token_importance.value),
        kv_action=int(out_kv_action.value),
    )


def plugin_load_dynamic(path: str, symbol_name: str = "") -> tuple[bool, str]:
    if _lib is None:
        return False, "bridge_unavailable"
    msg = create_string_buffer(256)
    ok = _lib.vspec_py_plugin_load_dynamic(
        str(path or "").encode("utf-8", errors="ignore"),
        str(symbol_name or "").encode("utf-8", errors="ignore"),
        msg,
        c_size_t(len(msg)),
    )
    return bool(ok), msg.value.decode("utf-8", errors="ignore")


def plugin_unload_dynamic(name: str) -> tuple[bool, str]:
    if _lib is None:
        return False, "bridge_unavailable"
    msg = create_string_buffer(256)
    ok = _lib.vspec_py_plugin_unload_dynamic(
        str(name or "").encode("utf-8", errors="ignore"),
        msg,
        c_size_t(len(msg)),
    )
    return bool(ok), msg.value.decode("utf-8", errors="ignore")


def sample_candidate_available() -> bool:
    return _lib is not None and np is not None


def native_output_guard_available() -> bool:
    return _lib is not None


def native_output_guard_init(strictness: float) -> bool:
    if _lib is None:
        return False
    try:
        value = float(strictness)
    except Exception:
        value = 0.72
    value = max(0.0, min(1.0, value))
    return bool(_lib.vspec_py_runtime_output_guard_init(c_float(value)))


def native_output_guard_allow(text_fragment: str) -> bool:
    if _lib is None:
        return True
    return bool(_lib.vspec_py_runtime_output_guard_allow(str(text_fragment or "").encode("utf-8", errors="ignore")))


def native_output_guard_score_adjustment(text_fragment: str) -> float:
    if _lib is None:
        return 0.0
    return float(_lib.vspec_py_runtime_output_guard_score_adjustment(str(text_fragment or "").encode("utf-8", errors="ignore")))


def native_output_guard_observe(text_fragment: str) -> bool:
    if _lib is None:
        return False
    return bool(_lib.vspec_py_runtime_output_guard_observe(str(text_fragment or "").encode("utf-8", errors="ignore")))


def sample_candidate(token_ids: list[int], scores: list[float], greedy: bool, random_bits: int) -> int | None:
    if _lib is None or np is None or not token_ids or len(token_ids) != len(scores):
        return None
    token_arr = (c_int * len(token_ids))(*[int(v) for v in token_ids])
    score_np = np.ascontiguousarray(scores, dtype=np.float32)
    out_token = c_int(0)
    ok = _lib.vspec_py_sample_candidate(
        token_arr,
        score_np.ctypes.data_as(POINTER(c_float)),
        c_size_t(len(token_ids)),
        c_int(1 if greedy else 0),
        c_uint64(int(random_bits) & 0xFFFFFFFFFFFFFFFF),
        ctypes.byref(out_token),
    )
    if not ok:
        return None
    return int(out_token.value)


def canonical_weight_name(raw_name: str) -> str | None:
    if _lib is None or not raw_name:
        return None
    buf = create_string_buffer(256)
    ok = _lib.vspec_py_weight_canonical_name(raw_name.encode("utf-8"), buf, c_size_t(len(buf)))
    if not ok:
        return None
    value = buf.value.decode("utf-8", errors="ignore").strip()
    return value or None


class CorePagedKVCache:
    def __init__(self, page_tokens: int, max_pages: int, num_heads: int, head_dim: int, session_id: int = 1) -> None:
        self.handle = 0 if _lib is None else int(_lib.vspec_py_kv_cache_create(page_tokens, max_pages, num_heads, head_dim))
        self.session_id = int(session_id)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)

    @property
    def available(self) -> bool:
        return bool(_lib is not None and self.handle > 0 and np is not None)

    def reset(self) -> None:
        if self.available:
            _lib.vspec_py_kv_cache_reset(int(self.handle))

    def append(self, key_token, value_token) -> bool:
        if not self.available or np is None:
            return False
        key_np = np.ascontiguousarray(key_token, dtype=np.float32).reshape(-1)
        value_np = np.ascontiguousarray(value_token, dtype=np.float32).reshape(-1)
        ok = _lib.vspec_py_kv_cache_append(
            int(self.handle),
            c_uint64(int(self.session_id)),
            key_np.ctypes.data_as(POINTER(c_float)),
            value_np.ctypes.data_as(POINTER(c_float)),
        )
        return bool(ok)

    def session_tokens(self) -> int:
        if not self.available:
            return 0
        return int(_lib.vspec_py_kv_cache_session_tokens(int(self.handle), c_uint64(int(self.session_id))))

    def read_tokens(self, max_tokens: int | None = None):
        if not self.available or np is None:
            return None, None
        token_count = self.session_tokens()
        if max_tokens is not None:
            token_count = min(token_count, int(max_tokens))
        if token_count <= 0:
            empty = np.empty((0, self.num_heads, self.head_dim), dtype=np.float32)
            return empty, empty.copy()
        flat_size = token_count * self.num_heads * self.head_dim
        keys = np.empty((flat_size,), dtype=np.float32)
        values = np.empty((flat_size,), dtype=np.float32)
        copied = int(
            _lib.vspec_py_kv_cache_read(
                int(self.handle),
                c_uint64(int(self.session_id)),
                keys.ctypes.data_as(POINTER(c_float)),
                values.ctypes.data_as(POINTER(c_float)),
                c_size_t(token_count),
            )
        )
        if copied <= 0:
            return None, None
        return (
            keys[: copied * self.num_heads * self.head_dim].reshape(copied, self.num_heads, self.head_dim),
            values[: copied * self.num_heads * self.head_dim].reshape(copied, self.num_heads, self.head_dim),
        )

    def close(self) -> None:
        if _lib is not None and self.handle > 0:
            _lib.vspec_py_kv_cache_destroy(int(self.handle))
            self.handle = 0

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class CoreDecodeSession:
    def __init__(self, total_vram_bytes: int, max_active: int = 1, max_batch_tokens: int = 8, token_quantum: int = 1) -> None:
        self.handle = 0 if _lib is None else int(
            _lib.vspec_py_decode_session_create(
                c_size_t(max(1, int(total_vram_bytes))),
                c_size_t(max(1, int(max_active))),
                c_size_t(max(1, int(max_batch_tokens))),
                c_size_t(max(1, int(token_quantum))),
            )
        )
        self.reserve_bytes = 1

    @property
    def available(self) -> bool:
        return bool(_lib is not None and self.handle > 0)

    @staticmethod
    def estimate_reserve_bytes(runtime, max_new_tokens: int) -> int:
        layers = len(getattr(runtime, "layers", []) or []) if runtime is not None else 0
        num_kv_heads = int(getattr(runtime, "num_kv_heads", 0) or getattr(runtime, "num_heads", 0) or 1)
        head_dim = int(getattr(runtime, "head_dim", 0) or 1)
        hidden = 0
        embed = getattr(runtime, "embed", None)
        if embed is not None:
            try:
                hidden = int(embed.shape[1])
            except Exception:
                hidden = 0
        kv_bytes_per_token = max(1, layers) * max(1, num_kv_heads) * max(1, head_dim) * 2 * 4
        work_bytes = max(1, hidden) * 24
        return int(max(1, work_bytes + kv_bytes_per_token * max(1, int(max_new_tokens))))

    @classmethod
    def from_runtime(cls, runtime, max_new_tokens: int):
        total_vram = 8 * 1024 * 1024 * 1024
        try:
            total_vram = int(os.getenv("VSPEC_SCHED_TOTAL_BYTES", "0") or "0") or total_vram
        except Exception:
            pass
        max_active = int(os.getenv("VSPEC_SCHED_MAX_ACTIVE", "1") or "1")
        max_batch_tokens = int(os.getenv("VSPEC_SCHED_MAX_BATCH_TOKENS", "8") or "8")
        token_quantum = int(os.getenv("VSPEC_SCHED_TOKEN_QUANTUM", "1") or "1")
        session = cls(total_vram, max_active=max_active, max_batch_tokens=max_batch_tokens, token_quantum=token_quantum)
        session.reserve_bytes = cls.estimate_reserve_bytes(runtime, max_new_tokens)
        return session

    def begin(self, prompt_tokens: int, max_new_tokens: int, reserve_bytes: int | None = None, priority: int = 0) -> bool:
        if not self.available:
            return False
        budget = int(reserve_bytes if reserve_bytes is not None else self.reserve_bytes)
        return bool(
            _lib.vspec_py_decode_session_begin(
                int(self.handle),
                c_size_t(max(1, budget)),
                c_size_t(max(0, int(prompt_tokens))),
                c_size_t(max(1, int(max_new_tokens))),
                ctypes.c_uint16(max(0, min(65535, int(priority)))),
            )
        )

    def next_quota(self) -> int:
        if not self.available:
            return 0
        return int(_lib.vspec_py_decode_session_next_quota(int(self.handle)))

    def commit(self, generated_tokens: int = 1, reached_eos: bool = False) -> bool:
        if not self.available:
            return False
        return bool(_lib.vspec_py_decode_session_commit(int(self.handle), c_size_t(max(0, int(generated_tokens))), c_int(1 if reached_eos else 0)))

    def cancel(self) -> bool:
        if not self.available:
            return False
        return bool(_lib.vspec_py_decode_session_cancel(int(self.handle)))

    def is_active(self) -> bool:
        if not self.available:
            return False
        return bool(_lib.vspec_py_decode_session_is_active(int(self.handle)))

    def remaining_tokens(self) -> int:
        if not self.available:
            return 0
        return int(_lib.vspec_py_decode_session_remaining_tokens(int(self.handle)))

    def close(self) -> None:
        if _lib is not None and self.handle > 0:
            _lib.vspec_py_decode_session_destroy(int(self.handle))
            self.handle = 0

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class CoreNativeDecodeLoop:
    def __init__(self, total_vram_bytes: int, max_active: int = 1, max_batch_tokens: int = 8, token_quantum: int = 1) -> None:
        self.handle = 0 if _lib is None else int(
            _lib.vspec_py_native_decode_loop_create(
                c_size_t(max(1, int(total_vram_bytes))),
                c_size_t(max(1, int(max_active))),
                c_size_t(max(1, int(max_batch_tokens))),
                c_size_t(max(1, int(token_quantum))),
            )
        )
        self.reserve_bytes = 1

    @property
    def available(self) -> bool:
        return bool(_lib is not None and self.handle > 0)

    @classmethod
    def from_runtime(cls, runtime, max_new_tokens: int):
        total_vram = 8 * 1024 * 1024 * 1024
        try:
            total_vram = int(os.getenv("VSPEC_SCHED_TOTAL_BYTES", "0") or "0") or total_vram
        except Exception:
            pass
        max_active = int(os.getenv("VSPEC_SCHED_MAX_ACTIVE", "1") or "1")
        max_batch_tokens = int(os.getenv("VSPEC_SCHED_MAX_BATCH_TOKENS", "8") or "8")
        token_quantum = int(os.getenv("VSPEC_SCHED_TOKEN_QUANTUM", "1") or "1")
        session = cls(total_vram, max_active=max_active, max_batch_tokens=max_batch_tokens, token_quantum=token_quantum)
        session.reserve_bytes = CoreDecodeSession.estimate_reserve_bytes(runtime, max_new_tokens)
        return session

    def begin(self, prompt_tokens: int, max_new_tokens: int, graph_signature: int, reserve_bytes: int | None = None, priority: int = 0) -> bool:
        if not self.available:
            return False
        budget = int(reserve_bytes if reserve_bytes is not None else self.reserve_bytes)
        return bool(
            _lib.vspec_py_native_decode_loop_begin(
                int(self.handle),
                c_size_t(max(1, budget)),
                c_size_t(max(0, int(prompt_tokens))),
                c_size_t(max(1, int(max_new_tokens))),
                ctypes.c_uint16(max(0, min(65535, int(priority)))),
                c_uint64(int(graph_signature) & 0xFFFFFFFFFFFFFFFF),
            )
        )

    def next_quota(self) -> int:
        if not self.available:
            return 0
        return int(_lib.vspec_py_native_decode_loop_next_quota(int(self.handle)))

    def commit(self, generated_tokens: int = 1, reached_eos: bool = False) -> bool:
        if not self.available:
            return False
        return bool(_lib.vspec_py_native_decode_loop_commit(int(self.handle), c_size_t(max(0, int(generated_tokens))), c_int(1 if reached_eos else 0)))

    def cancel(self) -> bool:
        if not self.available:
            return False
        return bool(_lib.vspec_py_native_decode_loop_cancel(int(self.handle)))

    def stats(self) -> dict[str, int]:
        if not self.available:
            return {}
        out_sig = c_uint64(0)
        out_hits = c_uint64(0)
        out_misses = c_uint64(0)
        out_steps = c_uint64(0)
        out_captures = c_uint64(0)
        out_replays = c_uint64(0)
        out_cached = c_uint64(0)
        out_enabled = c_int(0)
        ok = _lib.vspec_py_native_decode_loop_stats(
            int(self.handle),
            ctypes.byref(out_sig),
            ctypes.byref(out_hits),
            ctypes.byref(out_misses),
            ctypes.byref(out_steps),
        )
        if not ok:
            return {}
        ok_cache = 0
        if _HAS_GRAPH_CACHE_STATS:
            ok_cache = _lib.vspec_py_native_decode_loop_graph_cache_stats(
                int(self.handle),
                ctypes.byref(out_captures),
                ctypes.byref(out_replays),
                ctypes.byref(out_cached),
                ctypes.byref(out_enabled),
            )
        return {
            "graph_signature": int(out_sig.value),
            "graph_reuse_hits": int(out_hits.value),
            "graph_reuse_misses": int(out_misses.value),
            "steps": int(out_steps.value),
            "graph_captures": int(out_captures.value) if ok_cache else 0,
            "graph_replays": int(out_replays.value) if ok_cache else 0,
            "graph_cached_signatures": int(out_cached.value) if ok_cache else 0,
            "graph_capture_enabled": int(out_enabled.value) if ok_cache else 0,
        }

    def close(self) -> None:
        if _lib is not None and self.handle > 0:
            _lib.vspec_py_native_decode_loop_destroy(int(self.handle))
            self.handle = 0

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class CoreContinuousBatcher:
    def __init__(
        self,
        total_vram_bytes: int,
        max_active: int = 8,
        max_batch_items: int = 32,
        max_batch_tokens: int = 64,
        prefill_quantum: int = 64,
        decode_quantum: int = 1,
    ) -> None:
        self.handle = 0 if _lib is None else int(
            _lib.vspec_py_continuous_batch_create(
                c_size_t(max(1, int(total_vram_bytes))),
                c_size_t(max(1, int(max_active))),
                c_size_t(max(1, int(max_batch_items))),
                c_size_t(max(1, int(max_batch_tokens))),
                c_size_t(max(1, int(prefill_quantum))),
                c_size_t(max(1, int(decode_quantum))),
            )
        )

    @property
    def available(self) -> bool:
        return bool(_lib is not None and self.handle > 0)

    @classmethod
    def from_runtime(cls, runtime, max_batch_items: int = 32, max_batch_tokens: int = 64):
        total_vram = 8 * 1024 * 1024 * 1024
        try:
            total_vram = int(os.getenv("VSPEC_SCHED_TOTAL_BYTES", "0") or "0") or total_vram
        except Exception:
            pass
        max_active = int(os.getenv("VSPEC_SCHED_MAX_ACTIVE", "8") or "8")
        prefill_quantum = int(os.getenv("VSPEC_PREFILL_QUANTUM", "64") or "64")
        decode_quantum = int(os.getenv("VSPEC_SCHED_TOKEN_QUANTUM", "1") or "1")
        if runtime is not None:
            max_batch_tokens = max(max_batch_tokens, decode_quantum * max_active)
        return cls(
            total_vram,
            max_active=max_active,
            max_batch_items=max_batch_items,
            max_batch_tokens=max_batch_tokens,
            prefill_quantum=prefill_quantum,
            decode_quantum=decode_quantum,
        )

    def submit(self, reserve_bytes: int, prompt_tokens: int, max_new_tokens: int, priority: int = 0) -> int:
        if not self.available:
            return 0
        out_request_id = c_uint64(0)
        ok = _lib.vspec_py_continuous_batch_submit(
            int(self.handle),
            c_size_t(max(1, int(reserve_bytes))),
            c_size_t(max(0, int(prompt_tokens))),
            c_size_t(max(1, int(max_new_tokens))),
            ctypes.c_uint16(max(0, min(65535, int(priority)))),
            ctypes.byref(out_request_id),
        )
        return int(out_request_id.value) if ok else 0

    def next_batch(self, cap: int = 32) -> list[dict[str, int]]:
        if not self.available:
            return []
        n = max(1, int(cap))
        req_ids = (c_uint64 * n)()
        phases = (ctypes.c_uint32 * n)()
        quotas = (c_size_t * n)()
        cursors = (c_size_t * n)()
        count = int(_lib.vspec_py_continuous_batch_next(int(self.handle), req_ids, phases, quotas, cursors, c_size_t(n)))
        items: list[dict[str, int]] = []
        for idx in range(count):
            items.append(
                {
                    "request_id": int(req_ids[idx]),
                    "phase": int(phases[idx]),
                    "token_quota": int(quotas[idx]),
                    "prompt_cursor": int(cursors[idx]),
                }
            )
        return items

    def commit_prefill(self, request_id: int, consumed_tokens: int) -> bool:
        if not self.available:
            return False
        return bool(_lib.vspec_py_continuous_batch_commit_prefill(int(self.handle), c_uint64(int(request_id)), c_size_t(max(0, int(consumed_tokens)))))

    def commit_decode(self, request_id: int, generated_tokens: int = 1, reached_eos: bool = False) -> bool:
        if not self.available:
            return False
        return bool(
            _lib.vspec_py_continuous_batch_commit_decode(
                int(self.handle),
                c_uint64(int(request_id)),
                c_size_t(max(0, int(generated_tokens))),
                c_int(1 if reached_eos else 0),
            )
        )

    def cancel(self, request_id: int) -> bool:
        if not self.available:
            return False
        return bool(_lib.vspec_py_continuous_batch_cancel(int(self.handle), c_uint64(int(request_id))))

    def stats(self) -> dict[str, int]:
        if not self.available:
            return {}
        prefill_steps = c_uint64(0)
        decode_steps = c_uint64(0)
        prefill_tokens = c_uint64(0)
        decode_tokens = c_uint64(0)
        active_prefill = c_size_t(0)
        active_decode = c_size_t(0)
        reserved_vram = c_size_t(0)
        ok = _lib.vspec_py_continuous_batch_stats(
            int(self.handle),
            ctypes.byref(prefill_steps),
            ctypes.byref(decode_steps),
            ctypes.byref(prefill_tokens),
            ctypes.byref(decode_tokens),
            ctypes.byref(active_prefill),
            ctypes.byref(active_decode),
            ctypes.byref(reserved_vram),
        )
        if not ok:
            return {}
        return {
            "prefill_steps": int(prefill_steps.value),
            "decode_steps": int(decode_steps.value),
            "prefill_tokens": int(prefill_tokens.value),
            "decode_tokens": int(decode_tokens.value),
            "active_prefill": int(active_prefill.value),
            "active_decode": int(active_decode.value),
            "reserved_vram": int(reserved_vram.value),
        }

    def close(self) -> None:
        if _lib is not None and self.handle > 0:
            _lib.vspec_py_continuous_batch_destroy(int(self.handle))
            self.handle = 0

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class CoreNativeForwardContext:
    def __init__(self, model_path: str, seed: int = 0) -> None:
        self.model_path = str(model_path or "")
        self.handle = 0
        if _lib is not None and self.model_path:
            self.handle = int(
                _lib.vspec_py_native_forward_create(
                    self.model_path.encode("utf-8", errors="ignore"),
                    c_uint64(int(seed) & 0xFFFFFFFFFFFFFFFF),
                )
            )

    @property
    def available(self) -> bool:
        return bool(_lib is not None and self.handle > 0 and np is not None)

    def blend_candidates(
        self,
        prompt: str,
        produced_tokens: int,
        candidate_ids: list[int],
        base_scores: list[float],
        blend: float,
    ) -> list[float] | None:
        if not self.available:
            return None
        if not candidate_ids:
            return None
        if len(candidate_ids) != len(base_scores):
            return None

        ids_arr = (c_int * len(candidate_ids))(*[int(v) for v in candidate_ids])
        base_np = np.ascontiguousarray(base_scores, dtype=np.float32)
        out_np = np.empty((len(candidate_ids),), dtype=np.float32)
        ok = _lib.vspec_py_native_forward_step(
            int(self.handle),
            str(prompt or "").encode("utf-8", errors="ignore"),
            c_size_t(max(0, int(produced_tokens))),
            ids_arr,
            base_np.ctypes.data_as(POINTER(c_float)),
            c_size_t(len(candidate_ids)),
            c_float(float(blend)),
            out_np.ctypes.data_as(POINTER(c_float)),
        )
        if not ok:
            return None
        return out_np.astype(float, copy=False).tolist()

    def close(self) -> None:
        if _lib is not None and self.handle > 0:
            _lib.vspec_py_native_forward_destroy(int(self.handle))
            self.handle = 0

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass