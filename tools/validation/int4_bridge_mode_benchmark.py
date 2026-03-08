from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from statistics import mean

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CHAT_PY = ROOT / "Vspec-chat" / "python"
if str(CHAT_PY) not in sys.path:
    sys.path.insert(0, str(CHAT_PY))

from vspec_cuda_bridge import fused_linear_int4, fused_linear_int4_available


def _pack_signed_rowwise_int4(q: np.ndarray) -> np.ndarray:
    n, k = q.shape
    codes = (q.astype(np.int16, copy=False) & 0x0F).astype(np.uint8, copy=False)
    if (k & 1) != 0:
        pad = np.zeros((n, 1), dtype=np.uint8)
        codes = np.concatenate([codes, pad], axis=1)
    lo = codes[:, 0::2]
    hi = codes[:, 1::2] << 4
    return (lo | hi).astype(np.uint8, copy=False).reshape(-1)


def _quantize_rowwise_int4(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w = weight.astype(np.float32, copy=False)
    max_abs = np.max(np.abs(w), axis=1)
    scales = np.where(max_abs > 0.0, max_abs / 7.0, 1.0).astype(np.float32, copy=False)
    q = np.clip(np.round(w / scales[:, None]), -8.0, 7.0).astype(np.int8, copy=False)
    packed = _pack_signed_rowwise_int4(q)
    return packed, scales


def _bench(mode: str, vec: np.ndarray, packed: np.ndarray, scales: np.ndarray, n: int, warmup: int, repeat: int) -> tuple[float, float]:
    os.environ["VSPEC_INT4_COMPUTE_MODE"] = mode
    for _ in range(warmup):
        _ = fused_linear_int4(vec, packed, scales, n)

    times_ms: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        _ = fused_linear_int4(vec, packed, scales, n)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)
    return mean(times_ms), min(times_ms)


def main() -> int:
    if not fused_linear_int4_available():
        print("SKIP: fused_linear_int4 not available")
        return 0

    seed = int(os.getenv("VSPEC_INT4_BENCH_SEED", "20260308"))
    rng = np.random.default_rng(seed)

    m = int(os.getenv("VSPEC_INT4_BENCH_M", "1"))
    k = int(os.getenv("VSPEC_INT4_BENCH_K", "4096"))
    n = int(os.getenv("VSPEC_INT4_BENCH_N", "4096"))
    warmup = int(os.getenv("VSPEC_INT4_BENCH_WARMUP", "8"))
    repeat = int(os.getenv("VSPEC_INT4_BENCH_REPEAT", "24"))

    vec = rng.standard_normal((m, k), dtype=np.float32)
    w = (rng.standard_normal((n, k), dtype=np.float32) * 0.35).astype(np.float32, copy=False)
    packed, scales = _quantize_rowwise_int4(w)

    mean_native, min_native = _bench("native", vec, packed, scales, n, warmup, repeat)
    mean_deq, min_deq = _bench("dequant-cublas", vec, packed, scales, n, warmup, repeat)

    speedup = mean_native / max(mean_deq, 1e-6)

    print(f"bench_shape: m={m} k={k} n={n}")
    print(f"native_mean_ms={mean_native:.3f} native_min_ms={min_native:.3f}")
    print(f"dequant_cublas_mean_ms={mean_deq:.3f} dequant_cublas_min_ms={min_deq:.3f}")
    print(f"speedup_mean_x={speedup:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
