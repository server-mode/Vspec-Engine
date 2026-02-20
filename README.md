# Vspec Engine

Vspec Engine is a kernel-first runtime for low-bit LLM and diffusion inference. It targets native under-4-bit execution, modular backend support, and memory-aware scheduling. This project inherits core ideas from PyLittle and evolves them into a standalone runtime. The current codebase includes CPU reference paths, CUDA kernels (when a toolkit is available), a compact IR, and a Python bridge for early integration tests.

## Key goals

- Native 2/3/4-bit packed execution with dynamic quant mapping.
- CUDA-first backend with ROCm and SYCL planned.
- PyTorch model compatibility via conversion to Vspec IR (no torch runtime).
- Memory-first design: KV cache controls, pool/arena allocators, streaming attention.

## Capabilities

- native mixed-bit
- runtime IR-centric
- multi-backend abstraction (CUDA + ROCm + SYCL)
- KV aware memory system
- Python API independent of PyTorch
- dynamic scheduling & graph rewrite
- kernel level extensibility

## Requirements

- CMake 3.20+
- MSVC toolchain on Windows (or clang/gcc on Linux)
- CUDA toolkit (optional, for CUDA targets)
- Python 3.9+ (optional, for the Python bridge/tools)

## Build (Windows)

- Configure: cmake -S . -B build
- Build: cmake --build build --config Release

CUDA detection is automatic when the CUDA toolkit is installed. If CMake does not pick it up, configure with explicit paths:

- cmake -S . -B build -DCMAKE_CUDA_COMPILER="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe" -DCUDAToolkit_ROOT="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8" -DCMAKE_VS_GLOBALS="CudaToolkitDir=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"

## Build (Linux or macOS)

- Configure: cmake -S . -B build
- Build: cmake --build build -j

## Demos and tools

- int4 demo: ./build/Release/vspec_int4_demo.exe
- benchmark: ./build/Release/vspec_benchmark.exe
- stress test: ./build/Release/vspec_stress_test.exe
- attention v2 demo: ./build/Release/vspec_attention_v2_demo.exe
- CUDA fused demo (CUDA only): ./build/Release/vspec_cuda_fused_demo.exe
- phase 3 demo: ./build/Release/vspec_phase3_demo.exe
- custom LLM bench (report builder): python tools/benchmark/custom_bench.py --model-id Qwen/Qwen3-8B --ir qwen3_8b_ir_full.json --baseline-precision fp16 --vspec-bits 4 --kv-tokens 2048 --kv-heads 64 --kv-head-dim 128 --output tools/benchmark/qwen3_report.json
- custom LLM bench from logs: python tools/benchmark/custom_bench.py --model-id Qwen/Qwen3-8B --ir qwen3_8b_ir_full.json --baseline-precision fp16 --vspec-bits 4 --force-vspec-bits --kv-tokens 2048 --kv-heads 64 --kv-head-dim 128 --vspec-log logs/vspec_run.txt --baseline-log logs/baseline_run.txt --output tools/benchmark/qwen3_report.json
- sample log generator: python tools/benchmark/make_sample_logs.py --out-dir logs

## Benchmarking and comparisons

Use the custom bench report to compare Vspec against traditional runs (FP16/FP32) and external runtimes like llama.cpp. The report supports:

- Memory estimate: baseline vs Vspec quantized weights, plus KV cache.
- Throughput: tokens/sec for Vspec and baseline, plus speedup.
- Extra metrics: perplexity drift, SM occupancy, memory bandwidth, warp stall reason, sequence scaling.

Workflow:

1) Run a baseline inference (traditional or llama.cpp) and save a log with tokens/seconds and optional extra metrics.
2) Run a Vspec inference and save a log with the same fields.
3) Generate the report using the custom bench tool.

## Python bridge (optional)

1) Build the shared C API library:
   - ./build/Release/vspec_engine_capi.dll
2) Run the Python demo:
   - PYTHONPATH=vspec-python/src python examples/python_torch_like_demo.py

The Python package source lives in ./vspec-python.

## Converter (CLI)

- Convert manifest or safetensors header to IR JSON:
  - python tools/converter/vspec_converter.py --input sample_weights.vpt --output out_ir.json

## Notes

- CUDA is optional. If CUDA is not detected, CPU reference kernels are used.
- The roadmap lives in ROADMAP.md (ignored by default in git).
- See docs/OPTIMIZATION.md and docs/ABI.md for technical details.
