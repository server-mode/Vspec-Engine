import ctypes
import os
from ctypes import c_float, c_int, c_size_t, POINTER
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

    if hasattr(_lib, "vspec_cuda_fused_linear_int4_bridge"):
        _lib.vspec_cuda_fused_linear_int4_bridge.argtypes = [
            POINTER(c_float),
            ctypes.POINTER(ctypes.c_ubyte),
            POINTER(c_float),
            c_size_t,
            c_size_t,
            c_size_t,
            POINTER(c_float),
        ]
        _lib.vspec_cuda_fused_linear_int4_bridge.restype = c_int
        _HAS_FUSED_INT4 = True

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
else:
    _HAS_FUSED_ATTN = False
    _HAS_FLASH_ATTN = False


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


def fused_linear_int4_available() -> bool:
    return _lib is not None and _HAS_FUSED_INT4


def fused_linear_int3_available() -> bool:
    return _lib is not None and _HAS_FUSED_INT3


def fused_linear_int4(input_array: np.ndarray, packed_weight: np.ndarray, scales: np.ndarray, n: int) -> np.ndarray:
    if _lib is None or not _HAS_FUSED_INT4:
        raise RuntimeError("vspec_cuda_bridge.dll not found")
    if input_array.dtype != np.float32:
        input_array = input_array.astype(np.float32)
    if packed_weight.dtype != np.uint8:
        packed_weight = packed_weight.astype(np.uint8)
    if scales.dtype != np.float32:
        scales = scales.astype(np.float32)

    if input_array.ndim == 1:
        input_array = input_array[None, :]

    input_array = np.ascontiguousarray(input_array)
    packed_weight = np.ascontiguousarray(packed_weight)
    scales = np.ascontiguousarray(scales)

    m, k = input_array.shape
    output = np.empty((m, n), dtype=np.float32)
    ok = _lib.vspec_cuda_fused_linear_int4_bridge(
        input_array.ctypes.data_as(POINTER(c_float)),
        packed_weight.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
        scales.ctypes.data_as(POINTER(c_float)),
        c_size_t(m),
        c_size_t(k),
        c_size_t(n),
        output.ctypes.data_as(POINTER(c_float)),
    )
    if ok != 1:
        raise RuntimeError("vspec_cuda_fused_linear_int4_bridge failed")
    return output


def fused_linear_int3(input_array: np.ndarray, packed_weight: np.ndarray, scales: np.ndarray, n: int) -> np.ndarray:
    if _lib is None or not _HAS_FUSED_INT3:
        raise RuntimeError("vspec_cuda_bridge.dll not found")
    if input_array.dtype != np.float32:
        input_array = input_array.astype(np.float32)
    if packed_weight.dtype != np.uint8:
        packed_weight = packed_weight.astype(np.uint8)
    if scales.dtype != np.float32:
        scales = scales.astype(np.float32)

    if input_array.ndim == 1:
        input_array = input_array[None, :]

    input_array = np.ascontiguousarray(input_array)
    packed_weight = np.ascontiguousarray(packed_weight)
    scales = np.ascontiguousarray(scales)

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
