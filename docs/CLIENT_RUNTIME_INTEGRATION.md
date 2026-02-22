# Vspec Runtime Integration Guide for Client Applications

This document explains how an external client application (C/C++ or Python) can use `Vspec Engine` as its inference runtime backend.

## 1) Integration goals

- Use `vspec_engine` for runtime compute (linear/attention, quant metadata).
- Preserve the low-bit execution path `< 4-bit` (INT2/INT3/INT4), aligned with the project objective.
- Optionally use `vspec_engine_capi` for Python/FFI clients.
- Optionally use `vspec_cuda_bridge` for CUDA-native kernels in Python tooling.

---

## 2) Required build artifacts

From the repository root:

```powershell
cmake -S . -B build -DVSPEC_ENABLE_CUDA=ON
cmake --build build --config Release
```

Main artifacts:

- Runtime core library: `build/Release/vspec_engine.lib` (Windows) or equivalent on Linux/macOS.
- Python C API DLL: `build/Release/vspec_engine_capi.dll`.
- CUDA bridge DLL (when CUDA is enabled): `build/Release/vspec_cuda_bridge.dll`.

> If your client only needs the C/C++ runtime, the minimum requirement is `vspec_engine` + headers in `include/`.

---

## 3) Integrating into a C/C++ client

### 3.1. Minimum required headers

- `include/vspec/runtime/runtime.h`
- `include/vspec/kernel/context.h`
- `include/vspec/quant/quant.h`
- (for INT4/INT3 packing) `include/vspec/quant/int4.h`, `include/vspec/quant/int3.h`

### 3.2. Link libraries

- Link against `vspec_engine`.
- If running the CUDA backend, ensure a compatible CUDA runtime is available.

### 3.3. Runtime invocation flow

Standard client flow:

1. Prepare input/weight/output buffers.
2. Create `VspecKernelContext`.
3. Fill quant metadata (`VspecQuantMeta`) with `type`, `scales`, `scale_count`.
4. Set `config.m/n/k`.
5. Call:
   - `vspec_runtime_init_default()`
   - `vspec_linear_forward(&ctx)` or `vspec_attention_forward(&ctx)`

Reference examples:

- `examples/int4_demo.c`
- `examples/attention_demo.c`
- `examples/cuda_fused_demo.c`

---

## 4) Python integration (via C API)

`vspec-python` already provides ctypes bridge modules:

- `vspec-python/src/vspec/runtime_bridge.py`
- `vspec-python/src/vspec/torch_like_api.py`

Quick usage:

```python
from vspec.torch_like_api import load

model = load("sample_weights.vpt")  # or .safetensors
print(model.tensor_count)
print(model.generate("Hello from client app"))
```

Requirements:

- `vspec_engine_capi.dll` must be resolvable by `ctypes`.
- If using CUDA bridge, ensure `CUDA_PATH/bin` is in the DLL search path.

---

## 5) Runtime backend configuration

Backend APIs are defined in `include/vspec/kernel/backend.h`:

- `vspec_set_backend(...)`
- `vspec_make_cpu_backend()`
- `vspec_make_cuda_backend()` + `vspec_cuda_backend_available()`
- `vspec_make_rocm_backend()`, `vspec_make_sycl_backend()` (planned/stub based on project status)

Recommendations:

- Select backend once during startup.
- Avoid switching backend while multiple inference threads are running.

---

## 6) Checklist to preserve the < 4-bit objective

To keep client apps aligned with low-bit goals:

- Use `VspecQuantMeta.type` as `VSPEC_QUANT_INT2`, `VSPEC_QUANT_INT3`, or `VSPEC_QUANT_INT4`.
- Ensure `scales` and `scale_count` match the packed weight layout.
- For CUDA fused paths, prioritize benchmark + reference checks:
  - `vspec_cuda_fused_demo`
  - `vspec_cuda_fused_benchmark`
- Track drift/perplexity with validation tools under `tools/validation` before release.

---

## 7) Language Stability Guard for chat applications

The C/CUDA engine layer does not enforce language rules by itself.

If your client is a multilingual chat application, add a decode-layer guard in the application layer, for example:

- `Vspec-chat/python/language_stability_guard.py`

Benefits:

- Reduces script/language drift during decoding.
- Does not alter kernel compute/quant paths, so it does not conflict with the `<4-bit` objective.

---

## 8) Production rollout recommendations

- Pin ABI versioning according to `docs/ABI.md`.
- Package runtime + model converter + config as one consistent release bundle.
- Set up CI with three test groups:
  - correctness (reference comparison),
  - performance (tokens/sec, latency),
  - quality (perplexity drift / language drift).

---

## 9) Quick troubleshooting

- DLL load failure: verify `build/Release` location and PATH/DLL directories.
- High low-bit error: verify packing format + scales.
- Unexpected slowness: confirm active backend (CPU vs CUDA), then rerun benchmark targets.

---

If you want, the next step is to add a ready-to-run `client_template/` (CMake sample project) so another application can call `vspec_runtime` using `add_subdirectory()` in minutes.