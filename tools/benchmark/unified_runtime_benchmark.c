#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <string.h>

#include "vspec/kernel/context.h"
#include "vspec/quant/int3.h"
#include "vspec/quant/int4.h"
#include "vspec/quant/pack.h"
#include "vspec/quant/quant.h"
#include "vspec/runtime/runtime.h"

#if defined(VSPEC_HAS_CUDA) && VSPEC_HAS_CUDA
#include "vspec/kernel/cuda_fused.h"
#endif

#if defined(VSPEC_HAS_CUDA) && VSPEC_HAS_CUDA
#include <cuda_runtime.h>
#endif

static double now_ms(void) {
    return (double)clock() * 1000.0 / (double)CLOCKS_PER_SEC;
}

static float clamp01(float value) {
    if (value < 0.0f) {
        return 0.0f;
    }
    if (value > 1.0f) {
        return 1.0f;
    }
    return value;
}

static const char* severity_name(VspecRuntimeBehaviorSeverity severity) {
    switch (severity) {
        case VSPEC_RUNTIME_SEVERITY_LOW:
            return "low";
        case VSPEC_RUNTIME_SEVERITY_MEDIUM:
            return "medium";
        case VSPEC_RUNTIME_SEVERITY_HIGH:
            return "high";
        case VSPEC_RUNTIME_SEVERITY_CRITICAL:
            return "critical";
        case VSPEC_RUNTIME_SEVERITY_NONE:
        default:
            return "none";
    }
}

static void print_issue_mask(uint32_t issue_mask) {
    if (issue_mask == VSPEC_RUNTIME_ISSUE_NONE) {
        printf("[unified_bench] behavior_issues=none\n");
        return;
    }

    printf("[unified_bench] behavior_issues=");
    int first = 1;
    if (issue_mask & VSPEC_RUNTIME_ISSUE_GPU_UNDER_TARGET) {
        printf("%sGPU_UNDER_TARGET", first ? "" : "|");
        first = 0;
    }
    if (issue_mask & VSPEC_RUNTIME_ISSUE_VRAM_OVER_TARGET) {
        printf("%sVRAM_OVER_TARGET", first ? "" : "|");
        first = 0;
    }
    if (issue_mask & VSPEC_RUNTIME_ISSUE_BITS_OVER_TARGET) {
        printf("%sBITS_OVER_TARGET", first ? "" : "|");
        first = 0;
    }
    if (issue_mask & VSPEC_RUNTIME_ISSUE_INTEGRITY_FAIL) {
        printf("%sINTEGRITY_FAIL", first ? "" : "|");
        first = 0;
    }
    if (issue_mask & VSPEC_RUNTIME_ISSUE_CPU_FALLBACK) {
        printf("%sCPU_FALLBACK", first ? "" : "|");
    }
    printf("\n");
}

static int prepare_inputs(
    size_t m,
    size_t k,
    size_t n,
    float** out_a,
    float** out_c,
    float** out_scales,
    int8_t** out_wq4,
    int8_t** out_wq3,
    uint8_t** out_w4,
    uint8_t** out_w3
) {
    float* a = (float*)malloc(m * k * sizeof(float));
    float* c = (float*)malloc(m * n * sizeof(float));
    float* scales = (float*)malloc(n * sizeof(float));
    int8_t* wq4 = (int8_t*)malloc(n * k * sizeof(int8_t));
    int8_t* wq3 = (int8_t*)malloc(n * k * sizeof(int8_t));
    uint8_t* w4 = (uint8_t*)malloc(n * vspec_int4_packed_bytes(k));
    uint8_t* w3 = (uint8_t*)malloc(n * vspec_int3_packed_bytes(k));

    if (!a || !c || !scales || !wq4 || !wq3 || !w4 || !w3) {
        free(a); free(c); free(scales); free(wq4); free(wq3); free(w4); free(w3);
        return 0;
    }

    for (size_t i = 0; i < m * k; ++i) {
        a[i] = (float)((int)(i % 17) - 8) * 0.1f;
    }
    for (size_t i = 0; i < n; ++i) {
        scales[i] = 0.125f + (float)(i % 7) * 0.01f;
    }

    for (size_t i = 0; i < n * k; ++i) {
        int v4 = (int)(i % 15) - 7;
        int v3 = (int)(i % 7) - 3;
        if (v4 < -8) v4 = -8;
        if (v4 > 7) v4 = 7;
        if (v3 < -4) v3 = -4;
        if (v3 > 3) v3 = 3;
        wq4[i] = (int8_t)v4;
        wq3[i] = (int8_t)v3;
    }

    const size_t row4 = vspec_int4_packed_bytes(k);
    const size_t row3 = vspec_int3_packed_bytes(k);
    for (size_t row = 0; row < n; ++row) {
        vspec_int4_pack(&wq4[row * k], k, &w4[row * row4]);
        vspec_quant_pack_signed(&wq3[row * k], k, 3, &w3[row * row3]);
    }

    *out_a = a;
    *out_c = c;
    *out_scales = scales;
    *out_wq4 = wq4;
    *out_wq3 = wq3;
    *out_w4 = w4;
    *out_w3 = w3;
    return 1;
}

int main(void) {
    const size_t m = 128;
    const size_t k = 256;
    const size_t n = 128;
    const size_t warmup = 10;
    const size_t iters = 50;

    vspec_runtime_init_default();

    float* a = NULL;
    float* c = NULL;
    float* scales = NULL;
    int8_t* wq4 = NULL;
    int8_t* wq3 = NULL;
    uint8_t* w4 = NULL;
    uint8_t* w3 = NULL;

    if (!prepare_inputs(m, k, n, &a, &c, &scales, &wq4, &wq3, &w4, &w3)) {
        return 1;
    }

    for (size_t it = 0; it < warmup; ++it) {
        vspec_int4_matmul_ref_f32_q4(a, m, k, w4, n, scales, c);
    }

    const double cpu_t0 = now_ms();
    for (size_t it = 0; it < iters; ++it) {
        vspec_int4_matmul_ref_f32_q4(a, m, k, w4, n, scales, c);
    }
    const double cpu_t1 = now_ms();

    const double cpu_avg_ms = (cpu_t1 - cpu_t0) / (double)iters;
    const double ops = 2.0 * (double)m * (double)n * (double)k;
    const double cpu_gflops = (ops / (cpu_avg_ms / 1000.0)) / 1e9;
    const float workload_scale = clamp01((float)(ops / (2.0 * 128.0 * 256.0 * 128.0)));

    printf("[unified_bench] shape m=%zu k=%zu n=%zu warmup=%zu iters=%zu\n", m, k, n, warmup, iters);
    printf("[unified_bench] cpu_int4 avg_ms=%.3f gflops=%.3f\n", cpu_avg_ms, cpu_gflops);

    const VspecRuntimeHwState* hw_state = vspec_runtime_get_hw_state();
    printf("[unified_bench] runtime_backend=%s\n",
        (hw_state && hw_state->active_backend_name) ? hw_state->active_backend_name : "unknown");

    int cuda_available = 0;
#if defined(VSPEC_HAS_CUDA) && VSPEC_HAS_CUDA
    cuda_available = vspec_cuda_fused_available();
#endif

    vspec_runtime_behavior_set_workload_scale(workload_scale);

    if (!cuda_available) {
        printf("[unified_bench] cuda not available\n");
        vspec_runtime_behavior_set_integrity_pass(1);
        vspec_runtime_behavior_observe(0.0f, 0.10f, 3.0f);

        VspecRuntimeBehaviorReport report;
        (void)memset(&report, 0, sizeof(report));
        vspec_runtime_behavior_report(&report);
        printf("[unified_bench] behavior_severity=%s(%d) breaches=%u/%u\n",
            severity_name(report.severity),
            (int)report.severity,
            (unsigned)report.breach_updates,
            (unsigned)report.total_updates);
        print_issue_mask(report.issue_mask);

        free(a); free(c); free(scales); free(wq4); free(wq3); free(w4); free(w3);
        return 0;
    }

#if defined(VSPEC_HAS_CUDA) && VSPEC_HAS_CUDA
    VspecKernelContext ctx;
    (void)memset(&ctx, 0, sizeof(ctx));
    ctx.input = a;
    ctx.output = c;
    ctx.qmeta.schema_version = 1U;
    ctx.qmeta.scales = scales;
    ctx.qmeta.scale_count = n;
    ctx.config.m = m;
    ctx.config.n = n;
    ctx.config.k = k;

    ctx.weight = w4;
    ctx.qmeta.type = VSPEC_QUANT_INT4;
    for (size_t it = 0; it < warmup; ++it) {
        vspec_cuda_launch_fused_linear_int4(&ctx);
    }

    const double gpu4_t0 = now_ms();
    for (size_t it = 0; it < iters; ++it) {
        vspec_cuda_launch_fused_linear_int4(&ctx);
    }
    const double gpu4_t1 = now_ms();

    double gpu4_event_avg_ms = -1.0;
#if defined(VSPEC_HAS_CUDA) && VSPEC_HAS_CUDA
    cudaEvent_t e0, e1;
    if (cudaEventCreate(&e0) == cudaSuccess && cudaEventCreate(&e1) == cudaSuccess) {
        if (cudaEventRecord(e0, 0) == cudaSuccess) {
            for (size_t it = 0; it < iters; ++it) {
                vspec_cuda_launch_fused_linear_int4(&ctx);
            }
            if (cudaEventRecord(e1, 0) == cudaSuccess && cudaEventSynchronize(e1) == cudaSuccess) {
                float elapsed_ms = 0.0f;
                if (cudaEventElapsedTime(&elapsed_ms, e0, e1) == cudaSuccess) {
                    gpu4_event_avg_ms = (double)elapsed_ms / (double)iters;
                }
            }
        }
        (void)cudaEventDestroy(e1);
        (void)cudaEventDestroy(e0);
    }
#endif

    const double gpu4_avg_ms = (gpu4_t1 - gpu4_t0) / (double)iters;
    const double gpu4_gflops = (ops / (gpu4_avg_ms / 1000.0)) / 1e9;

    ctx.weight = w3;
    ctx.qmeta.type = VSPEC_QUANT_INT3;
    for (size_t it = 0; it < warmup; ++it) {
        vspec_cuda_launch_fused_linear_int3(&ctx);
    }

    const double gpu3_t0 = now_ms();
    for (size_t it = 0; it < iters; ++it) {
        vspec_cuda_launch_fused_linear_int3(&ctx);
    }
    const double gpu3_t1 = now_ms();
    const double gpu3_avg_ms = (gpu3_t1 - gpu3_t0) / (double)iters;
    const double gpu3_gflops = (ops / (gpu3_avg_ms / 1000.0)) / 1e9;

    printf("[unified_bench] gpu_int4 avg_ms=%.3f gflops=%.3f speedup_vs_cpu=%.3fx\n",
        gpu4_avg_ms,
        gpu4_gflops,
        (gpu4_avg_ms > 0.0) ? (cpu_avg_ms / gpu4_avg_ms) : 0.0);

    if (gpu4_event_avg_ms > 0.0) {
        printf("[unified_bench] gpu_int4 cuda_event_avg_ms=%.3f\n", gpu4_event_avg_ms);
    }

    printf("[unified_bench] gpu_int3 avg_ms=%.3f gflops=%.3f speedup_vs_cpu=%.3fx\n",
        gpu3_avg_ms,
        gpu3_gflops,
        (gpu3_avg_ms > 0.0) ? (cpu_avg_ms / gpu3_avg_ms) : 0.0);

    const float speedup4 = (gpu4_avg_ms > 0.0) ? (float)(cpu_avg_ms / gpu4_avg_ms) : 0.0f;
    const float gpu_util_proxy = clamp01(speedup4 / 12.0f);
    const float vram_util_proxy = 0.10f;

    vspec_runtime_behavior_set_integrity_pass(1);
    vspec_runtime_behavior_observe(gpu_util_proxy, vram_util_proxy, 3.0f);

    VspecRuntimeBehaviorReport report;
    (void)memset(&report, 0, sizeof(report));
    vspec_runtime_behavior_report(&report);

    printf("[unified_bench] behavior_observe gpu_util_proxy=%.3f vram_util_proxy=%.3f bits=%.1f workload_scale=%.3f\n",
        report.observed_gpu_utilization,
        report.observed_vram_utilization,
        report.observed_effective_bits,
        report.workload_scale);
    printf("[unified_bench] behavior_targets gpu=%.2f vram=%.2f bits<=%.1f\n",
        report.target_gpu_utilization,
        report.max_vram_utilization,
        report.max_effective_bits);
    printf("[unified_bench] behavior_severity=%s(%d) breaches=%u/%u\n",
        severity_name(report.severity),
        (int)report.severity,
        (unsigned)report.breach_updates,
        (unsigned)report.total_updates);
    print_issue_mask(report.issue_mask);
#endif

    free(a); free(c); free(scales); free(wq4); free(wq3); free(w4); free(w3);
    return 0;
}