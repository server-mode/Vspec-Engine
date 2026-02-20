#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <string.h>

#include "vspec/quant/int4.h"

static double now_ms(void) {
    return (double)clock() * 1000.0 / (double)CLOCKS_PER_SEC;
}

int main(void) {
    const size_t m = 128;
    const size_t k = 256;
    const size_t n = 128;
    const size_t iters = 50;

    float* a = (float*)malloc(m * k * sizeof(float));
    float* c = (float*)malloc(m * n * sizeof(float));
    float* scales = (float*)malloc(n * sizeof(float));
    int8_t* wq = (int8_t*)malloc(n * k * sizeof(int8_t));
    uint8_t* wpacked = (uint8_t*)malloc(n * vspec_int4_packed_bytes(k));

    if (!a || !c || !scales || !wq || !wpacked) {
        free(a); free(c); free(scales); free(wq); free(wpacked);
        return 1;
    }

    for (size_t i = 0; i < m * k; ++i) {
        a[i] = (float)((int)(i % 17) - 8) * 0.1f;
    }
    for (size_t i = 0; i < n; ++i) {
        scales[i] = 0.125f + (float)(i % 7) * 0.01f;
    }
    for (size_t i = 0; i < n * k; ++i) {
        int v = (int)(i % 19) - 9;
        if (v < -8) v = -8;
        if (v > 7) v = 7;
        wq[i] = (int8_t)v;
    }

    for (size_t row = 0; row < n; ++row) {
        vspec_int4_pack(&wq[row * k], k, &wpacked[row * vspec_int4_packed_bytes(k)]);
    }

    double t0 = now_ms();
    for (size_t it = 0; it < iters; ++it) {
        vspec_int4_matmul_ref_f32_q4(a, m, k, wpacked, n, scales, c);
    }
    double t1 = now_ms();

    const double elapsed_ms = t1 - t0;
    const double avg_ms = elapsed_ms / (double)iters;
    const double ops = 2.0 * (double)m * (double)n * (double)k;
    const double gflops = (ops / (avg_ms / 1000.0)) / 1e9;

    double checksum = 0.0;
    for (size_t i = 0; i < m * n; ++i) {
        checksum += c[i];
    }

    printf("[benchmark] int4_ref_matmul m=%zu k=%zu n=%zu iters=%zu\n", m, k, n, iters);
    printf("[benchmark] total_ms=%.3f avg_ms=%.3f gflops=%.3f checksum=%.6f\n",
        elapsed_ms, avg_ms, gflops, checksum);

    free(a); free(c); free(scales); free(wq); free(wpacked);
    return 0;
}
