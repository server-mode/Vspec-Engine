#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#include "vspec/kernel/context.h"
#include "vspec/quant/int4.h"
#include "vspec/quant/quant.h"
#include "vspec/version.h"
#include "vspec/runtime/runtime.h"

int main(void) {
    const size_t m = 2;
    const size_t k = 4;
    const size_t n = 3;

    const float a[8] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        4.0f, 3.0f, 2.0f, 1.0f
    };

    const int8_t b_q[12] = {
         1,  0, -1,  2,
         2, -2,  1,  0,
        -1,  1,  1, -2
    };

    const float scales[3] = {0.5f, 0.25f, 0.75f};

    const size_t packed_bytes = vspec_int4_packed_bytes(12);
    uint8_t* b_packed = (uint8_t*)malloc(packed_bytes);
    if (!b_packed) {
        return 1;
    }
    memset(b_packed, 0, packed_bytes);
    vspec_int4_pack(b_q, 12, b_packed);

    float out[6] = {0};

    VspecKernelContext ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.input = (void*)a;
    ctx.weight = (void*)b_packed;
    ctx.output = (void*)out;
    ctx.qmeta.type = VSPEC_QUANT_INT4;
    ctx.qmeta.scales = scales;
    ctx.qmeta.scale_count = n;
    ctx.config.m = m;
    ctx.config.n = n;
    ctx.config.k = k;

    vspec_runtime_init_default();
    vspec_linear_forward(&ctx);

    printf("Output (%zux%zu):\n", m, n);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            printf("%8.3f ", out[i * n + j]);
        }
        printf("\n");
    }

    free(b_packed);

    return 0;
}
