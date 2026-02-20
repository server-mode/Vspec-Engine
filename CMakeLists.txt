cmake_minimum_required(VERSION 3.20)
project(vspec_engine C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(VSPEC_ENABLE_CUDA "Enable CUDA backend" ON)

set(VSPEC_HAS_CUDA 0)
if(VSPEC_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(VSPEC_HAS_CUDA 1)
    endif()
endif()

add_library(vspec_engine
    src/core/tensor.c
    src/core/tensor_meta.c
    src/compat/types.c
    src/compat/pytorch_loader.c
    src/compat/safetensors_parser.c
    src/compat/graph_rewrite.c
    src/compat/weight_mapper.c
    src/kernel/backend_registry.c
    src/kernel/cpu/cpu_backend.c
    src/kernel/cuda/cuda_backend_stub.c
    src/kernel/rocm/rocm_backend_stub.c
    src/kernel/sycl/sycl_backend_stub.c
    src/attention/kv_cache.c
    src/attention/attention_ref.c
    src/attention/rotary.c
    src/attention/sparse.c
    src/attention/flash_stub.c
    src/attention/flash_block.c
    src/attention/kv_ring.c
    src/attention/streaming.c
    src/attention/fused_graph.c
    src/graph/ir.c
    src/graph/optimizer.c
    src/quant/int4.c
    src/quant/pack.c
    src/quant/int3.c
    src/quant/dynamic_map.c
    src/quant/quant.c
    src/validation/perplexity.c
    src/validation/drift_analyzer.c
    src/validation/quant_sensitivity.c
    src/validation/baseline_compare.c
    src/memory/pool.c
    src/memory/arena_allocator.c
    src/memory/block_pool.c
    src/memory/kv_arena.c
    src/memory/activation_reuse.c
    src/memory/memory_metrics.c
    src/memory/vram_scheduler.c
    src/runtime/runtime.c
    src/runtime/mixed_bit.c
    src/runtime/mixed_bit_runtime.c
    src/runtime/mixed_bit_policy.c
    src/scheduler/dynamic.c
    src/parallel/multi_gpu.c
)

if(VSPEC_HAS_CUDA)
    target_sources(vspec_engine PRIVATE
        src/kernel/cuda/cuda_linear_int4.cu
        src/kernel/cuda/fused_linear_int4.cu
        src/kernel/cuda/fused_linear_int3.cu
        src/kernel/cuda/attention_fused.cu
        src/kernel/cuda/autotune.cu
    )
    set_target_properties(vspec_engine PROPERTIES CUDA_STANDARD 14)
endif()

target_compile_definitions(vspec_engine PUBLIC VSPEC_HAS_CUDA=${VSPEC_HAS_CUDA})

target_include_directories(vspec_engine
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_library(vspec_engine_capi SHARED
    src/python/capi.c
    src/compat/types.c
    src/compat/pytorch_loader.c
    src/compat/safetensors_parser.c
    src/compat/graph_rewrite.c
    src/graph/ir.c
)

target_include_directories(vspec_engine_capi
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_definitions(vspec_engine_capi PRIVATE VSPEC_PYTHON_CAPI_BUILD)

set_target_properties(vspec_engine_capi PROPERTIES
    OUTPUT_NAME "vspec_engine_capi"
)

add_executable(vspec_int4_demo
    examples/int4_demo.c
)

target_link_libraries(vspec_int4_demo PRIVATE vspec_engine)

add_executable(vspec_memory_pool_demo
    examples/memory_pool_demo.c
)

target_link_libraries(vspec_memory_pool_demo PRIVATE vspec_engine)

add_executable(vspec_quant_pack_demo
    examples/quant_pack_demo.c
)

target_link_libraries(vspec_quant_pack_demo PRIVATE vspec_engine)

add_executable(vspec_attention_demo
    examples/attention_demo.c
)

target_link_libraries(vspec_attention_demo PRIVATE vspec_engine)

add_executable(vspec_graph_ir_demo
    examples/graph_ir_demo.c
)

target_link_libraries(vspec_graph_ir_demo PRIVATE vspec_engine)

add_executable(vspec_compat_demo
    examples/compat_demo.c
)

target_link_libraries(vspec_compat_demo PRIVATE vspec_engine)

add_executable(vspec_benchmark
    tools/benchmark/benchmark_suite.c
)

target_link_libraries(vspec_benchmark PRIVATE vspec_engine)

add_executable(vspec_stress_test
    tools/benchmark/stress_test.c
)

target_link_libraries(vspec_stress_test PRIVATE vspec_engine)

add_executable(vspec_memory_profiler
    tools/profiler/memory_profiler.c
)

target_link_libraries(vspec_memory_profiler PRIVATE vspec_engine)

add_executable(vspec_missing_features_demo
    examples/missing_features_demo.c
)

target_link_libraries(vspec_missing_features_demo PRIVATE vspec_engine)

add_executable(vspec_optimizer_demo
    examples/optimizer_demo.c
)

target_link_libraries(vspec_optimizer_demo PRIVATE vspec_engine)

add_executable(vspec_phase2_demo
    examples/phase2_demo.c
)

target_link_libraries(vspec_phase2_demo PRIVATE vspec_engine)

add_executable(vspec_phase3_demo
    examples/phase3_demo.c
)

target_link_libraries(vspec_phase3_demo PRIVATE vspec_engine)

if(VSPEC_HAS_CUDA)
    add_executable(vspec_cuda_fused_demo
        examples/cuda_fused_demo.c
    )

    target_link_libraries(vspec_cuda_fused_demo PRIVATE vspec_engine)
endif()

add_executable(vspec_validation_demo
    tools/validation/quant_validation_demo.c
)

target_link_libraries(vspec_validation_demo PRIVATE vspec_engine)

add_executable(vspec_attention_v2_demo
    examples/attention_v2_demo.c
)

target_link_libraries(vspec_attention_v2_demo PRIVATE vspec_engine)

add_executable(vspec_memory_v2_demo
    tools/profiler/memory_v2_demo.c
)

target_link_libraries(vspec_memory_v2_demo PRIVATE vspec_engine)
