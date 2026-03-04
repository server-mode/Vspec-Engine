#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "vspec/kernel/context.h"
#include "vspec/kernel/cuda_fused.h"
#include "vspec/quant/int3.h"
#include "vspec/quant/int4.h"
#include "vspec/quant/pack.h"

static double now_ms(void) {
    return (double)clock() * 1000.0 / (double)CLOCKS_PER_SEC;
}

static int prepare_inputs(
    size_t m,
    size_t k,
    size_t n,
    float** out_a,
    float** out_c4,
    float** out_c3,
    float** out_scales,
    int8_t** out_wq4,
    int8_t** out_wq3,
    uint8_t** out_w4,
    uint8_t** out_w3
) {
    float* a = (float*)malloc(m * k * sizeof(float));
    float* c4 = (float*)malloc(m * n * sizeof(float));
    float* c3 = (float*)malloc(m * n * sizeof(float));
    float* scales = (float*)malloc(n * sizeof(float));
    int8_t* wq4 = (int8_t*)malloc(n * k * sizeof(int8_t));
    int8_t* wq3 = (int8_t*)malloc(n * k * sizeof(int8_t));
    uint8_t* w4 = (uint8_t*)malloc(n * vspec_int4_packed_bytes(k));
    uint8_t* w3 = (uint8_t*)malloc(n * vspec_int3_packed_bytes(k));

    if (!a || !c4 || !c3 || !scales || !wq4 || !wq3 || !w4 || !w3) {
        free(a);
        free(c4);
        free(c3);
        free(scales);
        free(wq4);
        free(wq3);
        free(w4);
        free(w3);
        return 0;
    }

    for (size_t i = 0; i < m * k; ++i) {
        a[i] = (float)((int)(i % 31) - 15) * 0.045f;
    }
    for (size_t i = 0; i < n; ++i) {
        scales[i] = 0.08f + (float)(i % 11) * 0.01f;
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
    *out_c4 = c4;
    *out_c3 = c3;
    *out_scales = scales;
    *out_wq4 = wq4;
    *out_wq3 = wq3;
    *out_w4 = w4;
    *out_w3 = w3;
    return 1;
}

static float relative_l2(const float* a, const float* b, size_t n) {
    double num = 0.0;
    double den = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double d = (double)a[i] - (double)b[i];
        num += d * d;
        den += (double)a[i] * (double)a[i];
    }
    if (den <= 1e-18) {
        return 0.0f;
    }
    return (float)sqrt(num / den);
}

int main(void) {
    const size_t m = 128;
    const size_t k = 256;
    const size_t n = 128;
    const size_t warmup = 8;
    const size_t iters = 40;

    if (!vspec_cuda_fused_available()) {
        printf("[job3-storage] cuda fused backend unavailable\n");
        return 0;
    }

    float* a = NULL;
    float* c4 = NULL;
    float* c3 = NULL;
    float* scales = NULL;
    int8_t* wq4 = NULL;
    int8_t* wq3 = NULL;
    uint8_t* w4 = NULL;
    uint8_t* w3 = NULL;

    if (!prepare_inputs(m, k, n, &a, &c4, &c3, &scales, &wq4, &wq3, &w4, &w3)) {
        return 1;
    }

    VspecKernelContext ctx4;
    memset(&ctx4, 0, sizeof(ctx4));
    ctx4.input = a;
    ctx4.output = c4;
    ctx4.weight = w4;
    ctx4.qmeta.schema_version = 1U;
    ctx4.qmeta.type = VSPEC_QUANT_INT4;
    ctx4.qmeta.scales = scales;
    ctx4.qmeta.scale_count = n;
    ctx4.config.m = m;
    ctx4.config.n = n;
    ctx4.config.k = k;

    VspecKernelContext ctx3 = ctx4;
    ctx3.output = c3;
    ctx3.weight = w3;
    ctx3.qmeta.type = VSPEC_QUANT_INT3;

    for (size_t i = 0; i < warmup; ++i) {
        vspec_cuda_launch_fused_linear_int4(&ctx4);
        vspec_cuda_launch_fused_linear_int3_storage(&ctx3);
    }

    const double t4_0 = now_ms();
    for (size_t i = 0; i < iters; ++i) {
        vspec_cuda_launch_fused_linear_int4(&ctx4);
    }
    const double t4_1 = now_ms();

    const double t3_0 = now_ms();
    for (size_t i = 0; i < iters; ++i) {
        vspec_cuda_launch_fused_linear_int3_storage(&ctx3);
    }
    const double t3_1 = now_ms();

    const double ms4 = (t4_1 - t4_0) / (double)iters;
    const double ms3_storage = (t3_1 - t3_0) / (double)iters;
    const double ops = 2.0 * (double)m * (double)n * (double)k;

    const double gflops4 = (ops / (ms4 / 1000.0)) / 1e9;
    const double gflops3 = (ops / (ms3_storage / 1000.0)) / 1e9;

    const float drift = relative_l2(c4, c3, m * n);

    printf("[job3-storage] pipeline=int3_weight_storage_to_int4_compute\n");
    printf("[job3-storage] shape m=%zu k=%zu n=%zu iters=%zu\n", m, k, n, iters);
    printf("[job3-storage] int4_compute avg_ms=%.4f gflops=%.3f\n", ms4, gflops4);
    printf("[job3-storage] int3_storage_int4_compute avg_ms=%.4f gflops=%.3f\n", ms3_storage, gflops3);
    printf("[job3-storage] tps_impact_ratio=%.4f\n", (ms3_storage > 0.0) ? (ms4 / ms3_storage) : 0.0);
    printf("[job3-storage] output_rel_l2=%.6f\n", drift);

    free(a);
    free(c4);
    free(c3);
    free(scales);
    free(wq4);
    free(wq3);
    free(w4);
    free(w3);

    return 0;
}
