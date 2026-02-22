#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "vspec/kernel/cuda_fused.h"
#include "vspec/quant/int3.h"
#include "vspec/quant/int4.h"
#include "vspec/quant/quant.h"

static float max_abs_err(const float* a, const float* b, size_t count) {
    float maxv = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        float d = fabsf(a[i] - b[i]);
        if (d > maxv) {
            maxv = d;
        }
    }
    return maxv;
}

int main(void) {
    if (!vspec_cuda_fused_available()) {
        printf("cuda fused not available\n");
        return 0;
    }

    const size_t m = 2, k = 4, n = 2;
    float a[8] = {1, 2, 3, 4, 4, 3, 2, 1};
    int8_t wq[8] = {1, 0, -1, 2, 2, -2, 1, 0};
    float scales[2] = {0.5f, 0.25f};

    const size_t row_bytes = vspec_int4_packed_bytes(k);
    uint8_t* wpacked = (uint8_t*)malloc(n * row_bytes);
    memset(wpacked, 0, n * row_bytes);
    for (size_t row = 0; row < n; ++row) {
        vspec_int4_pack(&wq[row * k], k, &wpacked[row * row_bytes]);
    }

    float out4[4] = {0};
    float ref4[4] = {0};
    VspecKernelContext ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.input = a;
    ctx.weight = wpacked;
    ctx.output = out4;
    ctx.qmeta.type = VSPEC_QUANT_INT4;
    ctx.qmeta.scales = scales;
    ctx.config.m = m;
    ctx.config.n = n;
    ctx.config.k = k;

    vspec_cuda_launch_fused_linear_int4(&ctx);
    vspec_int4_matmul_ref_f32_q4(a, m, k, wpacked, n, scales, ref4);

    printf("[int4] fused out: %.3f %.3f %.3f %.3f\n", out4[0], out4[1], out4[2], out4[3]);
    printf("[int4] ref   out: %.3f %.3f %.3f %.3f\n", ref4[0], ref4[1], ref4[2], ref4[3]);
    printf("[int4] max_abs_err=%.6f\n", max_abs_err(out4, ref4, 4));

    int8_t wq3[8] = {1, -1, -2, 2, 2, -3, 1, 0};
    const size_t row_bytes3 = vspec_int3_packed_bytes(k);
    uint8_t* wpacked3 = (uint8_t*)malloc(n * row_bytes3);
    memset(wpacked3, 0, n * row_bytes3);
    for (size_t row = 0; row < n; ++row) {
        vspec_quant_pack_signed(&wq3[row * k], k, 3, &wpacked3[row * row_bytes3]);
    }

    float out3[4] = {0};
    float ref3[4] = {0};
    ctx.weight = wpacked3;
    ctx.output = out3;
    ctx.qmeta.type = VSPEC_QUANT_INT3;
    vspec_cuda_launch_fused_linear_int3(&ctx);
    vspec_int3_matmul_ref_f32_q3(a, m, k, wpacked3, n, scales, ref3);

    printf("[int3] fused out: %.3f %.3f %.3f %.3f\n", out3[0], out3[1], out3[2], out3[3]);
    printf("[int3] ref   out: %.3f %.3f %.3f %.3f\n", ref3[0], ref3[1], ref3[2], ref3[3]);
    printf("[int3] max_abs_err=%.6f\n", max_abs_err(out3, ref3, 4));

    free(wpacked);
    free(wpacked3);
    return 0;
}
