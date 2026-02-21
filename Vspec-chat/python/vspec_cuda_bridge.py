import ctypes
from ctypes import c_float, c_int, c_size_t, POINTER
from pathlib import Path

import numpy as np


def _load_lib() -> ctypes.CDLL | None:
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
