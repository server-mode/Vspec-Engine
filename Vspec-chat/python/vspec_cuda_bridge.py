import ctypes
import os
from ctypes import c_float, c_int, c_size_t, c_uint32, c_uint64, POINTER
from pathlib import Path

import numpy as np


def _load_lib() -> ctypes.CDLL | None:
    if os.name == "nt":
        cuda_path = os.environ.get("CUDA_PATH")
        candidates = []
        if cuda_path:
            candidates.append(Path(cuda_path) / "bin")
        candidates.append(Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin"))
        for p in candidates:
            try:
                if p.exists():
                    os.add_dll_directory(str(p))
            except Exception:
                pass

    root = Path(__file__).resolve().parents[2]
    candidates = [
        root / "build" / "Release" / "vspec_cuda_bridge.dll",
        root / "build" / "vspec_cuda_bridge.dll",
    ]
    for c in candidates:
        if c.exists():
            return ctypes.CDLL(str(c))
    return None


_lib = _load_lib()
_HAS_FUSED_INT4 = False
_HAS_FUSED_INT3 = False
_INT4_BRIDGE_ABI = "unknown"
_INT4_BRIDGE_SIGNATURE = "unknown"
_HAS_INT4_REGISTERED = False
_HAS_INT4_REGISTERED_MANY = False
_HAS_FUSED_HYBRID = False


def _ensure_c_array(arr: np.ndarray, dtype) -> np.ndarray:
    if arr.dtype != dtype:
        return np.ascontiguousarray(arr, dtype=dtype)
    if not arr.flags.c_contiguous:
        return np.ascontiguousarray(arr)
    return arr


def _set_int4_bridge_signature(signature: str) -> None:
    global _INT4_BRIDGE_SIGNATURE
    if _lib is None or not _HAS_FUSED_INT4:
        return
    if _INT4_BRIDGE_SIGNATURE == signature:
        return
    bridge = _lib.vspec_cuda_fused_linear_int4_bridge
    if signature == "legacy":
        bridge.argtypes = [
            POINTER(c_float),
            ctypes.POINTER(ctypes.c_ubyte),
            POINTER(c_float),
            c_size_t,
            c_size_t,
            c_size_t,
            POINTER(c_float),
        ]
        bridge.restype = c_int
        _INT4_BRIDGE_SIGNATURE = "legacy"
        return
    bridge.argtypes = [
        POINTER(c_float),
        ctypes.POINTER(ctypes.c_ubyte),
        POINTER(c_float),
        POINTER(c_float),
        c_size_t,
        c_size_t,
        c_size_t,
        c_size_t,
        POINTER(c_float),
    ]
    bridge.restype = c_int
    _INT4_BRIDGE_SIGNATURE = "new"

if _lib is not None:
    _lib.vspec_cuda_rmsnorm_f32_bridge.argtypes = [
        POINTER(c_float),
        POINTER(c_float),
        c_float,
        c_size_t,
        c_size_t,
        POINTER(c_float),
    ]
    _lib.vspec_cuda_rmsnorm_f32_bridge.restype = None

    _lib.vspec_cuda_linear_f32_bridge.argtypes = [
        POINTER(c_float),
        POINTER(c_float),
        c_size_t,
        c_size_t,
        c_size_t,
        POINTER(c_float),
    ]
    _lib.vspec_cuda_linear_f32_bridge.restype = None

    _lib.vspec_cuda_gemm_f32_bridge.argtypes = [
        POINTER(c_float),
        POINTER(c_float),
        c_size_t,
        c_size_t,
        c_size_t,
        POINTER(c_float),
    ]
    _lib.vspec_cuda_gemm_f32_bridge.restype = None

    _lib.vspec_cuda_attention_single_f32_bridge.argtypes = [
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_float),
        c_size_t,
        c_size_t,
        POINTER(c_float),
    ]
    _lib.vspec_cuda_attention_single_f32_bridge.restype = None

    _HAS_FUSED_ATTN = hasattr(_lib, "vspec_cuda_attention_fused_single_f32_bridge")
    if _HAS_FUSED_ATTN:
        _lib.vspec_cuda_attention_fused_single_f32_bridge.argtypes = [
            POINTER(c_float),
            POINTER(c_float),
            POINTER(c_float),
            c_size_t,
            c_size_t,
            POINTER(c_float),
        ]
        _lib.vspec_cuda_attention_fused_single_f32_bridge.restype = None
    _HAS_FLASH_ATTN = hasattr(_lib, "vspec_attention_flash_single_f32_bridge")
    if _HAS_FLASH_ATTN:
        _lib.vspec_attention_flash_single_f32_bridge.argtypes = [
            POINTER(c_float),
            POINTER(c_float),
            POINTER(c_float),
            c_size_t,
            c_size_t,
            c_size_t,
            POINTER(c_float),
        ]
        _lib.vspec_attention_flash_single_f32_bridge.restype = None
    _lib.vspec_cuda_silu_f32_bridge.argtypes = [
        POINTER(c_float),
        c_size_t,
    ]
    _lib.vspec_cuda_silu_f32_bridge.restype = None

    _lib.vspec_cuda_mul_f32_bridge.argtypes = [
        POINTER(c_float),
        POINTER(c_float),
        c_size_t,
        POINTER(c_float),
    ]
    _lib.vspec_cuda_mul_f32_bridge.restype = None

    _lib.vspec_cuda_mem_info_bridge.argtypes = [POINTER(c_size_t), POINTER(c_size_t)]
    _lib.vspec_cuda_mem_info_bridge.restype = c_int

    _HAS_DEVICE_CAP = hasattr(_lib, "vspec_cuda_device_capability_bridge")
    if _HAS_DEVICE_CAP:
        _lib.vspec_cuda_device_capability_bridge.argtypes = [POINTER(c_int), POINTER(c_int), POINTER(c_int)]
        _lib.vspec_cuda_device_capability_bridge.restype = c_int

    _HAS_INT4_TENSORCORE = hasattr(_lib, "vspec_cuda_int4_tensorcore_available_bridge")
    if _HAS_INT4_TENSORCORE:
        _lib.vspec_cuda_int4_tensorcore_available_bridge.argtypes = []
        _lib.vspec_cuda_int4_tensorcore_available_bridge.restype = c_int

    if hasattr(_lib, "vspec_cuda_fused_linear_int4_bridge"):
        _lib.vspec_cuda_fused_linear_int4_bridge.argtypes = [
            POINTER(c_float),
            ctypes.POINTER(ctypes.c_ubyte),
            POINTER(c_float),
            POINTER(c_float),
            c_size_t,
            c_size_t,
            c_size_t,
            c_size_t,
            POINTER(c_float),
        ]
        _lib.vspec_cuda_fused_linear_int4_bridge.restype = c_int
        _INT4_BRIDGE_SIGNATURE = "new"
        _HAS_FUSED_INT4 = True

    _HAS_INT4_REGISTERED = hasattr(_lib, "vspec_cuda_fused_linear_int4_register_weight_bridge") and hasattr(_lib, "vspec_cuda_fused_linear_int4_cached_bridge")
    if _HAS_INT4_REGISTERED:
        _lib.vspec_cuda_fused_linear_int4_register_weight_bridge.argtypes = [
            ctypes.POINTER(ctypes.c_ubyte),
            POINTER(c_float),
            POINTER(c_float),
            c_size_t,
            c_size_t,
            c_size_t,
        ]
        _lib.vspec_cuda_fused_linear_int4_register_weight_bridge.restype = c_int

        _lib.vspec_cuda_fused_linear_int4_cached_bridge.argtypes = [
            POINTER(c_float),
            c_size_t,
            c_size_t,
            c_int,
            c_size_t,
            POINTER(c_float),
        ]
        _lib.vspec_cuda_fused_linear_int4_cached_bridge.restype = c_int

        _HAS_INT4_REGISTERED_MANY = hasattr(_lib, "vspec_cuda_fused_linear_int4_cached_many_bridge")
        if _HAS_INT4_REGISTERED_MANY:
            _lib.vspec_cuda_fused_linear_int4_cached_many_bridge.argtypes = [
                POINTER(c_float),
                c_size_t,
                c_size_t,
                POINTER(c_int),
                POINTER(c_size_t),
                c_size_t,
                POINTER(c_float),
            ]
            _lib.vspec_cuda_fused_linear_int4_cached_many_bridge.restype = c_int
        else:
            _HAS_INT4_REGISTERED_MANY = False

    _HAS_INT4_CACHED_STATS = hasattr(_lib, "vspec_cuda_int4_cached_stats_bridge")
    if _HAS_INT4_CACHED_STATS:
        _lib.vspec_cuda_int4_cached_stats_bridge.argtypes = [
            POINTER(c_uint64),
            POINTER(c_uint64),
            POINTER(c_uint64),
            POINTER(c_uint64),
            POINTER(c_uint64),
            POINTER(c_uint64),
        ]
        _lib.vspec_cuda_int4_cached_stats_bridge.restype = c_int

    if hasattr(_lib, "vspec_cuda_fused_linear_int3_bridge"):
        _lib.vspec_cuda_fused_linear_int3_bridge.argtypes = [
            POINTER(c_float),
            ctypes.POINTER(ctypes.c_ubyte),
            POINTER(c_float),
            c_size_t,
            c_size_t,
            c_size_t,
            POINTER(c_float),
        ]
        _lib.vspec_cuda_fused_linear_int3_bridge.restype = c_int
        _HAS_FUSED_INT3 = True

    _HAS_FUSED_HYBRID = hasattr(_lib, "vspec_cuda_fused_linear_hybrid_bridge")
    if _HAS_FUSED_HYBRID:
        _lib.vspec_cuda_fused_linear_hybrid_bridge.argtypes = [
            POINTER(c_float),
            ctypes.POINTER(ctypes.c_ubyte),
            POINTER(c_float),
            ctypes.POINTER(ctypes.c_ubyte),
            POINTER(c_float),
            POINTER(c_float),
            c_size_t,
            c_size_t,
            c_size_t,
            POINTER(c_uint32),
            c_size_t,
            c_size_t,
            POINTER(c_float),
        ]
        _lib.vspec_cuda_fused_linear_hybrid_bridge.restype = c_int
else:
    _HAS_FUSED_ATTN = False
    _HAS_FLASH_ATTN = False
    _HAS_DEVICE_CAP = False
    _HAS_INT4_TENSORCORE = False
    _HAS_INT4_CACHED_STATS = False
    _HAS_INT4_REGISTERED_MANY = False
    _HAS_FUSED_HYBRID = False


def rmsnorm_f32_available() -> bool:
    return _lib is not None


def rmsnorm_f32(input_array: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    if _lib is None:
        raise RuntimeError("vspec_cuda_bridge.dll not found")
    if input_array.dtype != np.float32:
        input_array = input_array.astype(np.float32)
    if weight.dtype != np.float32:
        weight = weight.astype(np.float32)

    rows, dim = input_array.shape
    output = np.empty_like(input_array)

    _lib.vspec_cuda_rmsnorm_f32_bridge(
        input_array.ctypes.data_as(POINTER(c_float)),
        weight.ctypes.data_as(POINTER(c_float)),
        c_float(eps),
        c_size_t(rows),
        c_size_t(dim),
        output.ctypes.data_as(POINTER(c_float)),
    )

    return output


def linear_f32_available() -> bool:
    return _lib is not None


def gemm_f32_available() -> bool:
    return _lib is not None


def linear_f32(input_array: np.ndarray, weight: np.ndarray) -> np.ndarray:
    if _lib is None:
        raise RuntimeError("vspec_cuda_bridge.dll not found")
    if input_array.dtype != np.float32:
        input_array = input_array.astype(np.float32)
    if weight.dtype != np.float32:
        weight = weight.astype(np.float32)

    if input_array.ndim == 1:
        input_array = input_array[None, :]
    m, k = input_array.shape
    n = weight.shape[0]

    output = np.empty((m, n), dtype=np.float32)

    _lib.vspec_cuda_linear_f32_bridge(
        input_array.ctypes.data_as(POINTER(c_float)),
        weight.ctypes.data_as(POINTER(c_float)),
        c_size_t(m),
        c_size_t(k),
        c_size_t(n),
        output.ctypes.data_as(POINTER(c_float)),
    )

    return output


def gemm_f32(input_array: np.ndarray, weight: np.ndarray) -> np.ndarray:
    if _lib is None:
        raise RuntimeError("vspec_cuda_bridge.dll not found")
    if input_array.dtype != np.float32:
        input_array = input_array.astype(np.float32)
    if weight.dtype != np.float32:
        weight = weight.astype(np.float32)

    if input_array.ndim == 1:
        input_array = input_array[None, :]
    m, k = input_array.shape
    n = weight.shape[0]

    output = np.empty((m, n), dtype=np.float32)

    _lib.vspec_cuda_gemm_f32_bridge(
        input_array.ctypes.data_as(POINTER(c_float)),
        weight.ctypes.data_as(POINTER(c_float)),
        c_size_t(m),
        c_size_t(k),
        c_size_t(n),
        output.ctypes.data_as(POINTER(c_float)),
    )

    return output


def attention_single_f32_available() -> bool:
    return _lib is not None


def attention_fused_single_f32_available() -> bool:
    return _lib is not None and _HAS_FUSED_ATTN


def attention_flash_single_f32_available() -> bool:
    return _lib is not None and _HAS_FLASH_ATTN


def attention_single_f32(query: np.ndarray, keys: np.ndarray, values: np.ndarray) -> np.ndarray:
    if _lib is None:
        raise RuntimeError("vspec_cuda_bridge.dll not found")
    if query.dtype != np.float32:
        query = query.astype(np.float32)
    if keys.dtype != np.float32:
        keys = keys.astype(np.float32)
    if values.dtype != np.float32:
        values = values.astype(np.float32)

    query = np.ascontiguousarray(query)
    keys = np.ascontiguousarray(keys)
    values = np.ascontiguousarray(values)

    seq_len, head_dim = keys.shape
    output = np.empty((head_dim,), dtype=np.float32)

    _lib.vspec_cuda_attention_single_f32_bridge(
        query.ctypes.data_as(POINTER(c_float)),
        keys.ctypes.data_as(POINTER(c_float)),
        values.ctypes.data_as(POINTER(c_float)),
        c_size_t(seq_len),
        c_size_t(head_dim),
        output.ctypes.data_as(POINTER(c_float)),
    )

    return output


def attention_fused_single_f32(query: np.ndarray, keys: np.ndarray, values: np.ndarray) -> np.ndarray:
    if _lib is None or not _HAS_FUSED_ATTN:
        raise RuntimeError("fused attention bridge not available")
    if query.dtype != np.float32:
        query = query.astype(np.float32)
    if keys.dtype != np.float32:
        keys = keys.astype(np.float32)
    if values.dtype != np.float32:
        values = values.astype(np.float32)

    query = np.ascontiguousarray(query)
    keys = np.ascontiguousarray(keys)
    values = np.ascontiguousarray(values)

    seq_len, head_dim = keys.shape
    output = np.empty((head_dim,), dtype=np.float32)

    _lib.vspec_cuda_attention_fused_single_f32_bridge(
        query.ctypes.data_as(POINTER(c_float)),
        keys.ctypes.data_as(POINTER(c_float)),
        values.ctypes.data_as(POINTER(c_float)),
        c_size_t(seq_len),
        c_size_t(head_dim),
        output.ctypes.data_as(POINTER(c_float)),
    )

    return output


def attention_flash_single_f32(query: np.ndarray, keys: np.ndarray, values: np.ndarray, block_tokens: int = 128) -> np.ndarray:
    if _lib is None or not _HAS_FLASH_ATTN:
        raise RuntimeError("flash attention bridge not available")
    if query.dtype != np.float32:
        query = query.astype(np.float32)
    if keys.dtype != np.float32:
        keys = keys.astype(np.float32)
    if values.dtype != np.float32:
        values = values.astype(np.float32)

    query = np.ascontiguousarray(query)
    keys = np.ascontiguousarray(keys)
    values = np.ascontiguousarray(values)

    seq_len, head_dim = keys.shape
    output = np.empty((head_dim,), dtype=np.float32)

    _lib.vspec_attention_flash_single_f32_bridge(
        query.ctypes.data_as(POINTER(c_float)),
        keys.ctypes.data_as(POINTER(c_float)),
        values.ctypes.data_as(POINTER(c_float)),
        c_size_t(seq_len),
        c_size_t(head_dim),
        c_size_t(max(1, int(block_tokens))),
        output.ctypes.data_as(POINTER(c_float)),
    )

    return output


def silu_f32_available() -> bool:
    return _lib is not None


def silu_f32(data: np.ndarray) -> np.ndarray:
    if _lib is None:
        raise RuntimeError("vspec_cuda_bridge.dll not found")
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    data = np.ascontiguousarray(data)
    _lib.vspec_cuda_silu_f32_bridge(data.ctypes.data_as(POINTER(c_float)), c_size_t(data.size))
    return data


def mul_f32_available() -> bool:
    return _lib is not None


def mul_f32(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if _lib is None:
        raise RuntimeError("vspec_cuda_bridge.dll not found")
    if a.dtype != np.float32:
        a = a.astype(np.float32)
    if b.dtype != np.float32:
        b = b.astype(np.float32)

    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    output = np.empty_like(a)
    _lib.vspec_cuda_mul_f32_bridge(
        a.ctypes.data_as(POINTER(c_float)),
        b.ctypes.data_as(POINTER(c_float)),
        c_size_t(a.size),
        output.ctypes.data_as(POINTER(c_float)),
    )
    return output


def cuda_mem_info() -> tuple[int, int] | None:
    if _lib is None:
        return None
    free_b = c_size_t(0)
    total_b = c_size_t(0)
    ok = _lib.vspec_cuda_mem_info_bridge(ctypes.byref(free_b), ctypes.byref(total_b))
    if ok != 1:
        return None
    return int(free_b.value), int(total_b.value)


def cuda_device_capability() -> tuple[int, int, int] | None:
    if _lib is None or not _HAS_DEVICE_CAP:
        return None
    major = c_int(0)
    minor = c_int(0)
    multiprocessors = c_int(0)
    ok = _lib.vspec_cuda_device_capability_bridge(ctypes.byref(major), ctypes.byref(minor), ctypes.byref(multiprocessors))
    if ok != 1:
        return None
    return int(major.value), int(minor.value), int(multiprocessors.value)


def int4_tensorcore_available() -> bool:
    if _lib is None or not _HAS_INT4_TENSORCORE:
        return False
    return bool(_lib.vspec_cuda_int4_tensorcore_available_bridge())


def int4_compute_mode() -> str:
    return str(os.getenv("VSPEC_INT4_COMPUTE_MODE", "kernel")).strip().lower() or "kernel"


def fused_linear_int4_available() -> bool:
    return _lib is not None and _HAS_FUSED_INT4


def fused_linear_int3_available() -> bool:
    return _lib is not None and _HAS_FUSED_INT3


def fused_linear_hybrid_available() -> bool:
    return _lib is not None and _HAS_FUSED_HYBRID


def fused_linear_int4_registered_available() -> bool:
    return _lib is not None and _HAS_FUSED_INT4 and _HAS_INT4_REGISTERED


def fused_linear_int4_cached_many_available() -> bool:
    return _lib is not None and _HAS_FUSED_INT4 and _HAS_INT4_REGISTERED and _HAS_INT4_REGISTERED_MANY


def fused_linear_int4_register_weight(
    packed_weight: np.ndarray,
    scales: np.ndarray,
    n: int,
    k: int,
    zero_points: np.ndarray | None = None,
) -> int:
    if _lib is None or not fused_linear_int4_registered_available():
        return 0

    packed_weight = _ensure_c_array(packed_weight, np.uint8)
    scales = _ensure_c_array(scales, np.float32)
    zero_points_arr = _ensure_c_array(zero_points, np.float32) if zero_points is not None else None
    zp_ptr = zero_points_arr.ctypes.data_as(POINTER(c_float)) if zero_points_arr is not None else None

    n_blocks = 1
    if int(n) > 0 and int(scales.size) >= int(n) and (int(scales.size) % int(n)) == 0:
        n_blocks = max(1, int(scales.size // int(n)))

    return int(
        _lib.vspec_cuda_fused_linear_int4_register_weight_bridge(
            packed_weight.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            scales.ctypes.data_as(POINTER(c_float)),
            zp_ptr,
            c_size_t(int(k)),
            c_size_t(int(n)),
            c_size_t(int(n_blocks)),
        )
    )


def fused_linear_int4_cached(input_array: np.ndarray, weight_handle: int, n: int) -> np.ndarray:
    if _lib is None or not fused_linear_int4_registered_available():
        raise RuntimeError("int4 registered bridge unavailable")
    if int(weight_handle) <= 0:
        raise RuntimeError("invalid int4 weight handle")

    if input_array.ndim == 1:
        input_array = input_array[None, :]
    input_array = _ensure_c_array(input_array, np.float32)
    m, k = input_array.shape
    output = np.empty((m, int(n)), dtype=np.float32)
    ok = _lib.vspec_cuda_fused_linear_int4_cached_bridge(
        input_array.ctypes.data_as(POINTER(c_float)),
        c_size_t(int(m)),
        c_size_t(int(k)),
        c_int(int(weight_handle)),
        c_size_t(int(n)),
        output.ctypes.data_as(POINTER(c_float)),
    )
    if ok != 1:
        raise RuntimeError("vspec_cuda_fused_linear_int4_cached_bridge failed")
    return output


def fused_linear_int4_cached_many(input_array: np.ndarray, weight_handles: list[int], output_dims: list[int]) -> list[np.ndarray]:
    if _lib is None or not fused_linear_int4_cached_many_available():
        raise RuntimeError("int4 registered multi bridge unavailable")
    if not weight_handles or len(weight_handles) != len(output_dims):
        raise RuntimeError("invalid int4 multi dispatch arguments")

    if input_array.ndim == 1:
        input_array = input_array[None, :]
    input_array = _ensure_c_array(input_array, np.float32)
    m, k = input_array.shape

    handles_arr = (c_int * len(weight_handles))(*[int(h) for h in weight_handles])
    dims_arr = (c_size_t * len(output_dims))(*[max(1, int(n)) for n in output_dims])
    total_n = int(sum(max(1, int(n)) for n in output_dims))
    output = np.empty((m, total_n), dtype=np.float32)

    ok = _lib.vspec_cuda_fused_linear_int4_cached_many_bridge(
        input_array.ctypes.data_as(POINTER(c_float)),
        c_size_t(int(m)),
        c_size_t(int(k)),
        handles_arr,
        dims_arr,
        c_size_t(int(len(weight_handles))),
        output.ctypes.data_as(POINTER(c_float)),
    )
    if ok != 1:
        raise RuntimeError("vspec_cuda_fused_linear_int4_cached_many_bridge failed")

    results: list[np.ndarray] = []
    offset = 0
    for n in output_dims:
        nn = int(n)
        results.append(np.ascontiguousarray(output[:, offset:offset + nn], dtype=np.float32))
        offset += nn
    return results


def int4_cached_stats() -> dict[str, int]:
    if _lib is None or not _HAS_INT4_CACHED_STATS:
        return {}
    dispatch_calls = c_uint64(0)
    dispatch_hits = c_uint64(0)
    dispatch_misses = c_uint64(0)
    register_calls = c_uint64(0)
    register_reuse = c_uint64(0)
    register_evictions = c_uint64(0)
    ok = _lib.vspec_cuda_int4_cached_stats_bridge(
        ctypes.byref(dispatch_calls),
        ctypes.byref(dispatch_hits),
        ctypes.byref(dispatch_misses),
        ctypes.byref(register_calls),
        ctypes.byref(register_reuse),
        ctypes.byref(register_evictions),
    )
    if ok != 1:
        return {}
    return {
        "dispatch_calls": int(dispatch_calls.value),
        "dispatch_hits": int(dispatch_hits.value),
        "dispatch_misses": int(dispatch_misses.value),
        "register_calls": int(register_calls.value),
        "register_reuse": int(register_reuse.value),
        "register_evictions": int(register_evictions.value),
    }


def configure_lowbit_bridge_cache_caps(int4_cap: int | None = None, int3_cap: int | None = None) -> None:
    if int4_cap is not None:
        os.environ["VSPEC_INT4_BRIDGE_CACHE_CAP"] = str(max(1, int(int4_cap)))
    if int3_cap is not None:
        os.environ["VSPEC_INT3_BRIDGE_CACHE_CAP"] = str(max(1, int(int3_cap)))


def get_lowbit_bridge_cache_caps() -> tuple[int, int]:
    def _read(name: str, fallback: int) -> int:
        raw = os.getenv(name, str(fallback)).strip()
        try:
            val = int(raw)
            return max(1, val)
        except Exception:
            return fallback

    return _read("VSPEC_INT4_BRIDGE_CACHE_CAP", 256), _read("VSPEC_INT3_BRIDGE_CACHE_CAP", 256)


def fused_linear_int4(
    input_array: np.ndarray,
    packed_weight: np.ndarray,
    scales: np.ndarray,
    n: int,
    zero_points: np.ndarray | None = None,
) -> np.ndarray:
    global _INT4_BRIDGE_ABI
    if _lib is None or not _HAS_FUSED_INT4:
        raise RuntimeError("vspec_cuda_bridge.dll not found")
    has_zero_points = zero_points is not None

    if input_array.ndim == 1:
        input_array = input_array[None, :]

    input_array = _ensure_c_array(input_array, np.float32)
    packed_weight = _ensure_c_array(packed_weight, np.uint8)
    scales = _ensure_c_array(scales, np.float32)
    zero_points_arr = _ensure_c_array(zero_points, np.float32) if has_zero_points else None
    zp_ptr = zero_points_arr.ctypes.data_as(POINTER(c_float)) if zero_points_arr is not None else None

    m, k = input_array.shape
    n_blocks = 1
    if int(n) > 0 and int(scales.size) >= int(n) and (int(scales.size) % int(n)) == 0:
        n_blocks = max(1, int(scales.size // int(n)))
    output = np.empty((m, n), dtype=np.float32)

    bridge = _lib.vspec_cuda_fused_linear_int4_bridge

    def _call_new_abi() -> int:
        _set_int4_bridge_signature("new")
        return bridge(
            input_array.ctypes.data_as(POINTER(c_float)),
            packed_weight.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            scales.ctypes.data_as(POINTER(c_float)),
            zp_ptr,
            c_size_t(m),
            c_size_t(k),
            c_size_t(n),
            c_size_t(n_blocks),
            output.ctypes.data_as(POINTER(c_float)),
        )

    def _call_legacy_abi() -> int:
        _set_int4_bridge_signature("legacy")
        return bridge(
            input_array.ctypes.data_as(POINTER(c_float)),
            packed_weight.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            scales.ctypes.data_as(POINTER(c_float)),
            c_size_t(m),
            c_size_t(k),
            c_size_t(n),
            output.ctypes.data_as(POINTER(c_float)),
        )

    if _INT4_BRIDGE_ABI == "new":
        ok_new = _call_new_abi()
        if ok_new == 1:
            return output
        _INT4_BRIDGE_ABI = "unknown"

    if _INT4_BRIDGE_ABI == "unknown":
        ok_new = _call_new_abi()
        if ok_new == 1:
            _INT4_BRIDGE_ABI = "new"
            return output

    ok_legacy = _call_legacy_abi()
    if ok_legacy != 1:
        raise RuntimeError("vspec_cuda_fused_linear_int4_bridge failed")

    _INT4_BRIDGE_ABI = "legacy"
    # Legacy bridge applies row-wise scale only (no zero-point or n_blocks).
    # Keep runtime stable by forcing row-wise quantization for subsequent packing.
    os.environ["VSPEC_INT4_BLOCKWISE_ENABLE"] = "0"

    if n_blocks > 1:
        raise RuntimeError("legacy int4 bridge does not support block-wise scales")

    # Emulate asymmetric zero-point correction for legacy bridge output.
    if zero_points_arr is not None and zero_points_arr.size == int(n):
        vec_sum = np.sum(input_array.astype(np.float32, copy=False), axis=-1, keepdims=True)
        zp_scale = np.ascontiguousarray(zero_points_arr.astype(np.float32, copy=False) * scales.astype(np.float32, copy=False))
        output = output - (vec_sum * zp_scale[None, :])
    return output


def fused_linear_int3(input_array: np.ndarray, packed_weight: np.ndarray, scales: np.ndarray, n: int) -> np.ndarray:
    if _lib is None or not _HAS_FUSED_INT3:
        raise RuntimeError("vspec_cuda_bridge.dll not found")
    if input_array.ndim == 1:
        input_array = input_array[None, :]

    input_array = _ensure_c_array(input_array, np.float32)
    packed_weight = _ensure_c_array(packed_weight, np.uint8)
    scales = _ensure_c_array(scales, np.float32)

    m, k = input_array.shape
    output = np.empty((m, n), dtype=np.float32)
    ok = _lib.vspec_cuda_fused_linear_int3_bridge(
        input_array.ctypes.data_as(POINTER(c_float)),
        packed_weight.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
        scales.ctypes.data_as(POINTER(c_float)),
        c_size_t(m),
        c_size_t(k),
        c_size_t(n),
        output.ctypes.data_as(POINTER(c_float)),
    )
    if ok != 1:
        raise RuntimeError("vspec_cuda_fused_linear_int3_bridge failed")
    return output


def fused_linear_hybrid(
    input_array: np.ndarray,
    packed_weight_int2: np.ndarray,
    scales_int2: np.ndarray,
    n: int,
    hot_indices: np.ndarray | None = None,
    packed_weight_int4: np.ndarray | None = None,
    scales_int4: np.ndarray | None = None,
    zero_points_int4: np.ndarray | None = None,
    n_blocks_int4: int = 1,
) -> np.ndarray:
    if _lib is None or not _HAS_FUSED_HYBRID:
        raise RuntimeError("vspec_cuda_fused_linear_hybrid_bridge unavailable")

    if input_array.ndim == 1:
        input_array = input_array[None, :]

    input_array = _ensure_c_array(input_array, np.float32)
    packed_weight_int2 = _ensure_c_array(packed_weight_int2, np.uint8)
    scales_int2 = _ensure_c_array(scales_int2, np.float32)
    hot_arr = _ensure_c_array(hot_indices, np.uint32) if hot_indices is not None else None
    packed_weight_int4_arr = _ensure_c_array(packed_weight_int4, np.uint8) if packed_weight_int4 is not None else None
    scales_int4_arr = _ensure_c_array(scales_int4, np.float32) if scales_int4 is not None else None
    zero_points_int4_arr = _ensure_c_array(zero_points_int4, np.float32) if zero_points_int4 is not None else None

    m, k = input_array.shape
    output = np.empty((m, int(n)), dtype=np.float32)

    hot_ptr = hot_arr.ctypes.data_as(POINTER(c_uint32)) if hot_arr is not None and hot_arr.size > 0 else None
    hot_count = int(hot_arr.size) if hot_arr is not None else 0

    int4_w_ptr = packed_weight_int4_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)) if packed_weight_int4_arr is not None else None
    int4_s_ptr = scales_int4_arr.ctypes.data_as(POINTER(c_float)) if scales_int4_arr is not None else None
    int4_zp_ptr = zero_points_int4_arr.ctypes.data_as(POINTER(c_float)) if zero_points_int4_arr is not None else None

    ok = _lib.vspec_cuda_fused_linear_hybrid_bridge(
        input_array.ctypes.data_as(POINTER(c_float)),
        packed_weight_int2.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
        scales_int2.ctypes.data_as(POINTER(c_float)),
        int4_w_ptr,
        int4_s_ptr,
        int4_zp_ptr,
        c_size_t(int(m)),
        c_size_t(int(k)),
        c_size_t(int(n)),
        hot_ptr,
        c_size_t(hot_count),
        c_size_t(max(1, int(n_blocks_int4))),
        output.ctypes.data_as(POINTER(c_float)),
    )
    if ok != 1:
        raise RuntimeError("vspec_cuda_fused_linear_hybrid_bridge failed")
    return output
