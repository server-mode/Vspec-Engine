from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CHAT_PY = ROOT / "Vspec-chat" / "python"
if str(CHAT_PY) not in sys.path:
    sys.path.insert(0, str(CHAT_PY))

from runtime_lowbit_module import (  # noqa: E402
    fused_linear_int4_available,
    _pack_signed_rowwise_int4,
    _unpack_int4_rowwise,
)
from vspec_cuda_bridge import fused_linear_int4  # noqa: E402


def quantize_rowwise_int4(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = np.asarray(weight, dtype=np.float32)
    max_abs = np.max(np.abs(w), axis=1)
    scales = np.where(max_abs > 0.0, max_abs / 7.0, 1.0).astype(np.float32)
    q = np.clip(np.round(w / scales[:, None]), -8.0, 7.0).astype(np.int8)
    packed = _pack_signed_rowwise_int4(q)
    dequant = q.astype(np.float32) * scales[:, None]
    return packed, scales, dequant


def run_case(out_n: int, k: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal((1, k), dtype=np.float32)
    w = (rng.standard_normal((out_n, k), dtype=np.float32) * 0.35).astype(np.float32)

    packed, scales, dequant_ref = quantize_rowwise_int4(w)
    dequant_from_unpack = _unpack_int4_rowwise(packed, scales, out_n, k)

    unpack_diff = float(np.max(np.abs(dequant_ref - dequant_from_unpack)))

    ref = vec @ dequant_from_unpack.T
    got = fused_linear_int4(vec, packed, scales, out_n)
    max_abs_err = float(np.max(np.abs(ref - got)))
    denom = float(np.max(np.abs(ref)) + 1e-6)
    max_rel_err = float(max_abs_err / denom)

    return unpack_diff, max_rel_err


def main() -> int:
    if not fused_linear_int4_available():
        print("INT4 compare skipped: fused_linear_int4 kernel not available")
        return 2

    cases = [
        (64, 96, 20260308),
        (128, 256, 20260309),
        (192, 384, 20260310),
    ]

    worst_unpack = 0.0
    worst_rel = 0.0
    for idx, (out_n, k, seed) in enumerate(cases, start=1):
        unpack_diff, rel_err = run_case(out_n, k, seed)
        worst_unpack = max(worst_unpack, unpack_diff)
        worst_rel = max(worst_rel, rel_err)
        print(f"case#{idx}: out_n={out_n} k={k} unpack_max_abs={unpack_diff:.6f} kernel_max_rel={rel_err:.6f}")

    print(f"summary: worst_unpack_max_abs={worst_unpack:.6f} worst_kernel_max_rel={worst_rel:.6f}")

    threshold = float(os.getenv("VSPEC_INT4_COMPARE_REL_THRESHOLD", "0.10"))
    if worst_rel > threshold:
        print(f"FAIL: worst kernel relative error {worst_rel:.6f} > threshold {threshold:.6f}")
        return 1

    print("PASS: INT4 runtime kernel compare within threshold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
