from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CHAT_PY = ROOT / "Vspec-chat" / "python"
if str(CHAT_PY) not in sys.path:
    sys.path.insert(0, str(CHAT_PY))

from vspec_cuda_bridge import (  # noqa: E402
    attention_single_f32,
    attention_fused_single_f32,
    attention_single_f32_available,
    attention_fused_single_f32_available,
)


def _ref_attention(query: np.ndarray, keys: np.ndarray, values: np.ndarray) -> np.ndarray:
    scale = np.float32(1.0 / np.sqrt(float(query.shape[0])))
    scores = (keys @ query) * scale
    scores = scores - np.max(scores)
    probs = np.exp(scores)
    probs = probs / np.maximum(np.sum(probs), 1e-12)
    return probs @ values


def _run_case(seq_len: int, head_dim: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((head_dim,), dtype=np.float32)
    k = rng.standard_normal((seq_len, head_dim), dtype=np.float32)
    v = rng.standard_normal((seq_len, head_dim), dtype=np.float32)

    ref = _ref_attention(q, k, v).astype(np.float32, copy=False)

    max_rel_single = 0.0
    if attention_single_f32_available():
        got_single = attention_single_f32(q, k, v).astype(np.float32, copy=False)
        err_single = float(np.max(np.abs(got_single - ref)))
        denom = float(np.max(np.abs(ref)) + 1e-6)
        max_rel_single = float(err_single / denom)

    max_rel_fused = 0.0
    if attention_fused_single_f32_available():
        got_fused = attention_fused_single_f32(q, k, v).astype(np.float32, copy=False)
        err_fused = float(np.max(np.abs(got_fused - ref)))
        denom = float(np.max(np.abs(ref)) + 1e-6)
        max_rel_fused = float(err_fused / denom)

    return max_rel_single, max_rel_fused


def main() -> int:
    if not attention_single_f32_available() and not attention_fused_single_f32_available():
        print("SKIP: no CUDA attention kernels available")
        return 0

    cases = [
        (32, 128, 20260311),
        (96, 128, 20260312),
        (192, 128, 20260313),
    ]

    worst_single = 0.0
    worst_fused = 0.0
    for idx, (seq_len, head_dim, seed) in enumerate(cases, start=1):
        rel_single, rel_fused = _run_case(seq_len, head_dim, seed)
        worst_single = max(worst_single, rel_single)
        worst_fused = max(worst_fused, rel_fused)
        print(
            f"case#{idx}: seq_len={seq_len} head_dim={head_dim} "
            f"single_max_rel={rel_single:.6f} fused_max_rel={rel_fused:.6f}"
        )

    print(f"summary: worst_single_max_rel={worst_single:.6f} worst_fused_max_rel={worst_fused:.6f}")

    threshold = 0.05
    if worst_single > threshold or worst_fused > threshold:
        print(f"FAIL: attention kernel relative error above threshold {threshold:.6f}")
        return 1

    print("PASS: attention kernels match numpy reference within threshold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
