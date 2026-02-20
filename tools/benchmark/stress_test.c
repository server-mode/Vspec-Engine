#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "vspec/quant/int4.h"
#include "vspec/attention/streaming.h"
#include "vspec/memory/pool.h"

static double now_ms(void) {
    return (double)clock() * 1000.0 / (double)CLOCKS_PER_SEC;
}

int main(void) {
    const size_t m = 64;
    const size_t k = 128;
    const size_t n = 64;

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
        a[i] = (float)((int)(i % 23) - 11) * 0.05f;
    }
    for (size_t i = 0; i < n; ++i) {
        scales[i] = 0.1f + (float)(i % 5) * 0.01f;
    }
    for (size_t i = 0; i < n * k; ++i) {
        int v = (int)(i % 15) - 7;
        if (v < -8) v = -8;
        if (v > 7) v = 7;
        wq[i] = (int8_t)v;
    }
    for (size_t row = 0; row < n; ++row) {
        vspec_int4_pack(&wq[row * k], k, &wpacked[row * vspec_int4_packed_bytes(k)]);
    }

    uint8_t* arena = (uint8_t*)malloc(1024 * 1024);
    VspecMemoryPool pool;
    vspec_memory_pool_init(&pool, arena, 1024 * 1024);

    const size_t iters = 100;
    double t0 = now_ms();
    for (size_t it = 0; it < iters; ++it) {
        vspec_int4_matmul_ref_f32_q4(a, m, k, wpacked, n, scales, c);
        (void)vspec_memory_pool_alloc(&pool, 1024, 64);
        if ((it % 10) == 0) {
            vspec_memory_pool_reset(&pool);
        }
    }
    double t1 = now_ms();

    const size_t tokens = 32;
    const size_t head_dim = 16;
    float* q = (float*)malloc(head_dim * sizeof(float));
    float* kbuf = (float*)malloc(tokens * head_dim * sizeof(float));
    float* vbuf = (float*)malloc(tokens * head_dim * sizeof(float));
    float* out = (float*)malloc(head_dim * sizeof(float));

    for (size_t i = 0; i < head_dim; ++i) {
        q[i] = (float)i * 0.01f;
    }
    for (size_t i = 0; i < tokens * head_dim; ++i) {
        kbuf[i] = (float)((int)(i % 11) - 5) * 0.02f;
        vbuf[i] = (float)((int)(i % 9) - 4) * 0.03f;
    }

    double t2 = now_ms();
    for (size_t it = 0; it < iters; ++it) {
        vspec_attention_streaming_ref(q, kbuf, vbuf, tokens, head_dim, 8, out);
    }
    double t3 = now_ms();

    printf("stress matmul ms=%.2f attention ms=%.2f\n", (t1 - t0), (t3 - t2));
    printf("pool peak=%zu allocs=%zu fails=%zu\n",
        vspec_memory_pool_peak_used(&pool),
        vspec_memory_pool_alloc_count(&pool),
        vspec_memory_pool_failed_alloc_count(&pool));

    free(a); free(c); free(scales); free(wq); free(wpacked);
    free(arena);
    free(q); free(kbuf); free(vbuf); free(out);
    return 0;
}
