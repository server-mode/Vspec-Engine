# Week 12 Optimization Notes

This document captures the Week 12 optimization and tuning work.

## 1) CPU int4 matmul micro-optimization

- Decode two int4 weights per byte.
- Hoist scale and row pointers to reduce repeated loads.
- Preserve correctness with odd-`k` fallback.

## 2) CUDA warp tuning hooks

- `VSPEC_CUDA_BLOCK_X` / `VSPEC_CUDA_BLOCK_Y` allow block tuning at compile time.
- Kernel uses `__launch_bounds__` to help occupancy decisions.

Example build overrides:

```
cmake -S . -B build -DVSPEC_ENABLE_CUDA=ON
cmake --build build --config Release
```

Set compile definitions for your target if needed (future work):

- `VSPEC_CUDA_BLOCK_X=32`
- `VSPEC_CUDA_BLOCK_Y=8`

## 3) Graph optimization pass

A simple dead-node elimination pass removes nodes whose outputs are not in a live set.
This is a placeholder for larger IR optimization pipelines.

## 4) Next steps

- SIMD / vectorized pack/unpack
- Kernel-level block tiling for int4 matmul
- Attention kernel fusion
