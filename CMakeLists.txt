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
    src/attention/kv_paged_cache.c
    src/model/qwen_ops.c
    src/graph/ir.c
    src/graph/optimizer.c
    src/quant/int4.c
    src/quant/pack.c
    src/quant/int3.c
    src/quant/dynamic_map.c
    src/quant/quant.c
    src/quant/formats/q4_k.c
    src/quant/formats/special_quant.c
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
    src/runtime/continuous_batch.c
    src/runtime/torch_compat_module.c
    src/runtime/three_bit_runtime_modules.c
    src/runtime/decode_session.c
    src/runtime/continuous_batch.c
    src/runtime/sampling_core.c
    src/runtime/hw_performance_manager.c
    src/runtime/language_structure_guard.c
    src/runtime/output_guard.c
    src/runtime/runtime_behavior_monitor.c
    src/runtime/ultimate_optimizer.c
    src/runtime/qlora_adapter.c
    src/runtime/mixed_bit.c
    src/runtime/mixed_bit_runtime.c
    src/runtime/mixed_bit_policy.c
    src/runtime/adaptive_precision_engine.c
    src/runtime/adaptive/runtime_controller.c
    src/runtime/adaptive/token_scheduler.c
    src/runtime/adaptive/precision_router.c
    src/runtime/adaptive/memory_policy.c
    src/runtime/plugin/plugin_api.c
    src/scheduler/dynamic.c
    src/scheduler/request_scheduler.c
    src/parallel/multi_gpu.c
)

if(VSPEC_HAS_CUDA)
    find_package(CUDAToolkit)

    target_sources(vspec_engine PRIVATE
        src/kernel/cuda/cuda_linear_int4.cu
        src/kernel/cuda/fused_linear_int4.cu
        src/kernel/cuda/fused_linear_int3.cu
        src/kernel/cuda/cuda_ultimate_branch.cu
        src/kernel/cuda/attention_fused.cu
        src/kernel/cuda/autotune.cu
        src/kernel/cuda/cuda_rmsnorm_f32.cu
        src/kernel/cuda/cuda_linear_f32.cu
        src/kernel/cuda/cuda_gemm_f32.cu
        src/kernel/cuda/cuda_attention_f32.cu
        src/kernel/cuda/cuda_elemwise_f32.cu
    )
    set_target_properties(vspec_engine PROPERTIES CUDA_STANDARD 14)

    if (CUDAToolkit_FOUND)
        target_link_libraries(vspec_engine PRIVATE CUDA::cublas)
        target_compile_definitions(vspec_engine PRIVATE VSPEC_USE_CUBLAS=1)
    else()
        target_compile_definitions(vspec_engine PRIVATE VSPEC_USE_CUBLAS=0)
    endif()
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
    src/compat/weight_mapper.c
    src/attention/kv_paged_cache.c
    src/memory/vram_scheduler.c
    src/scheduler/request_scheduler.c
    src/runtime/continuous_batch.c
    src/runtime/decode_session.c
    src/runtime/sampling_core.c
    src/runtime/plugin/plugin_api.c
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

if(VSPEC_HAS_CUDA)
    add_library(vspec_cuda_bridge SHARED
        src/python/cuda_bridge.c
        src/attention/flash_block.c
        src/kernel/cuda/cuda_rmsnorm_f32.cu
        src/kernel/cuda/cuda_linear_f32.cu
        src/kernel/cuda/cuda_gemm_f32.cu
        src/kernel/cuda/cuda_attention_f32.cu
        src/kernel/cuda/attention_fused.cu
        src/kernel/cuda/cuda_elemwise_f32.cu
        src/kernel/cuda/fused_linear_int4.cu
        src/kernel/cuda/fused_linear_int3.cu
    )

    if (CUDAToolkit_FOUND)
        target_link_libraries(vspec_cuda_bridge PRIVATE CUDA::cublas)
        target_compile_definitions(vspec_cuda_bridge PRIVATE VSPEC_USE_CUBLAS=1)
    else()
        target_compile_definitions(vspec_cuda_bridge PRIVATE VSPEC_USE_CUBLAS=0)
    endif()

    target_include_directories(vspec_cuda_bridge
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    set_target_properties(vspec_cuda_bridge PROPERTIES
        OUTPUT_NAME "vspec_cuda_bridge"
    )
endif()

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

add_executable(vspec_special_quant_demo
    examples/special_quant_demo.c
)

target_link_libraries(vspec_special_quant_demo PRIVATE vspec_engine)

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

add_executable(vspec_ultimate_quality_benchmark
    tools/benchmark/ultimate_quality_benchmark.c
)

target_link_libraries(vspec_ultimate_quality_benchmark PRIVATE vspec_engine)

add_executable(vspec_qlora_manifest_smoke
    tools/benchmark/qlora_manifest_smoke.c
)

target_link_libraries(vspec_qlora_manifest_smoke PRIVATE vspec_engine)

add_executable(vspec_week25_scheduler_stress
    tools/benchmark/week25_memory_scheduler_stress.c
)

target_link_libraries(vspec_week25_scheduler_stress PRIVATE vspec_engine)

if(VSPEC_HAS_CUDA)
    add_executable(vspec_job3_storage_compute_benchmark
        tools/benchmark/job3_int3_storage_compute_benchmark.c
    )

    target_link_libraries(vspec_job3_storage_compute_benchmark PRIVATE vspec_engine)
endif()

add_executable(vspec_stress_test
    tools/benchmark/stress_test.c
)

target_link_libraries(vspec_stress_test PRIVATE vspec_engine)

add_executable(vspec_request_scheduler_benchmark
    tools/benchmark/runtime_request_scheduler_benchmark.c
)

target_link_libraries(vspec_request_scheduler_benchmark PRIVATE vspec_engine)

add_executable(vspec_continuous_batch_benchmark
    tools/benchmark/continuous_batch_benchmark.c
)

target_link_libraries(vspec_continuous_batch_benchmark PRIVATE vspec_engine)

add_executable(vspec_memory_profiler
    tools/profiler/memory_profiler.c
)

target_link_libraries(vspec_memory_profiler PRIVATE vspec_engine)

add_executable(vspec_kv_paged_cache_profiler
    tools/profiler/kv_paged_cache_profiler.c
)

target_link_libraries(vspec_kv_paged_cache_profiler PRIVATE vspec_engine)

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

add_executable(vspec_adaptive_runtime_demo
    examples/adaptive_runtime_demo.c
)

target_link_libraries(vspec_adaptive_runtime_demo PRIVATE vspec_engine)

add_executable(vspec_adaptive_plugin_demo
    examples/adaptive_plugin_demo.c
)

target_link_libraries(vspec_adaptive_plugin_demo PRIVATE vspec_engine)

add_executable(vspec_torch_compat_demo
    examples/torch_compat_demo.c
)

target_link_libraries(vspec_torch_compat_demo PRIVATE vspec_engine)

if(VSPEC_HAS_CUDA)
    add_executable(vspec_cuda_fused_demo
        examples/cuda_fused_demo.c
    )

    target_link_libraries(vspec_cuda_fused_demo PRIVATE vspec_engine)

    add_executable(vspec_cuda_fused_benchmark
        tools/benchmark/cuda_fused_linear_benchmark.c
    )

    target_link_libraries(vspec_cuda_fused_benchmark PRIVATE vspec_engine)

    add_executable(vspec_unified_runtime_benchmark
        tools/benchmark/unified_runtime_benchmark.c
    )

    target_link_libraries(vspec_unified_runtime_benchmark PRIVATE vspec_engine)
endif()

add_executable(vspec_validation_demo
    tools/validation/quant_validation_demo.c
)

target_link_libraries(vspec_validation_demo PRIVATE vspec_engine)

add_executable(vspec_job29_now_validation
    tools/validation/job29_now_validation_suite.c
)

target_link_libraries(vspec_job29_now_validation PRIVATE vspec_engine)

add_executable(vspec_attention_v2_demo
    examples/attention_v2_demo.c
)

target_link_libraries(vspec_attention_v2_demo PRIVATE vspec_engine)

add_executable(vspec_memory_v2_demo
    tools/profiler/memory_v2_demo.c
)

target_link_libraries(vspec_memory_v2_demo PRIVATE vspec_engine)

add_executable(vspec_adaptive_profile_benchmark
    tools/benchmark/adaptive_profile_benchmark.c
)

target_link_libraries(vspec_adaptive_profile_benchmark PRIVATE vspec_engine)
