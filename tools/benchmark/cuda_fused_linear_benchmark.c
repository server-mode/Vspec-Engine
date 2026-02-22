#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <string.h>

#include "vspec/kernel/cuda_fused.h"
#include "vspec/kernel/context.h"
#include "vspec/quant/int3.h"
#include "vspec/quant/int4.h"
#include "vspec/quant/pack.h"
#include "vspec/quant/quant.h"

static double now_ms(void) {
    return (double)clock() * 1000.0 / (double)CLOCKS_PER_SEC;
}

int main(void) {
    if (!vspec_cuda_fused_available()) {
        printf("[cuda_fused_bench] cuda not available\n");
        return 0;
    }

    const size_t m = 64;
    const size_t k = 256;
    const size_t n = 128;
    const size_t iters = 20;

    float* a = (float*)malloc(m * k * sizeof(float));
    float* c = (float*)malloc(m * n * sizeof(float));
    float* scales = (float*)malloc(n * sizeof(float));
    int8_t* wq4 = (int8_t*)malloc(n * k * sizeof(int8_t));
    int8_t* wq3 = (int8_t*)malloc(n * k * sizeof(int8_t));
    uint8_t* w4 = (uint8_t*)malloc(n * vspec_int4_packed_bytes(k));
    uint8_t* w3 = (uint8_t*)malloc(n * vspec_int3_packed_bytes(k));

    if (!a || !c || !scales || !wq4 || !wq3 || !w4 || !w3) {
        free(a); free(c); free(scales); free(wq4); free(wq3); free(w4); free(w3);
        return 1;
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

    VspecKernelContext ctx;
    memset(&ctx, 0, sizeof(ctx));
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
    double t0 = now_ms();
    for (size_t it = 0; it < iters; ++it) {
        vspec_cuda_launch_fused_linear_int4(&ctx);
    }
    double t1 = now_ms();

    ctx.weight = w3;
    ctx.qmeta.type = VSPEC_QUANT_INT3;
    double t2 = now_ms();
    for (size_t it = 0; it < iters; ++it) {
        vspec_cuda_launch_fused_linear_int3(&ctx);
    }
    double t3 = now_ms();

    const double avg4 = (t1 - t0) / (double)iters;
    const double avg3 = (t3 - t2) / (double)iters;
    const double ops = 2.0 * (double)m * (double)n * (double)k;
    const double gflops4 = (ops / (avg4 / 1000.0)) / 1e9;
    const double gflops3 = (ops / (avg3 / 1000.0)) / 1e9;

    printf("[cuda_fused_bench] m=%zu k=%zu n=%zu iters=%zu\n", m, k, n, iters);
    printf("[cuda_fused_bench] int4 avg_ms=%.3f gflops=%.3f\n", avg4, gflops4);
    printf("[cuda_fused_bench] int3 avg_ms=%.3f gflops=%.3f\n", avg3, gflops3);

    free(a); free(c); free(scales); free(wq4); free(wq3); free(w4); free(w3);
    return 0;
}
