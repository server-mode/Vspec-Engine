#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vspec/kernel/cuda_fused.h"
#include "vspec/quant/int4.h"
#include "vspec/quant/quant.h"

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

    float out[4] = {0};
    VspecKernelContext ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.input = a;
    ctx.weight = wpacked;
    ctx.output = out;
    ctx.qmeta.type = VSPEC_QUANT_INT4;
    ctx.qmeta.scales = scales;
    ctx.config.m = m;
    ctx.config.n = n;
    ctx.config.k = k;

    vspec_cuda_launch_fused_linear_int4(&ctx);

    printf("fused out: %.3f %.3f %.3f %.3f\n", out[0], out[1], out[2], out[3]);

    free(wpacked);
    return 0;
}
