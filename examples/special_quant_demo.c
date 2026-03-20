#include <stdio.h>
#include <string.h>

#include "vspec/quant/formats/special_quant.h"

static unsigned short float_to_half_simple(float x) {
    union {
        float f;
        unsigned int u;
    } v;
    v.f = x;

    unsigned int sign = (v.u >> 16U) & 0x8000U;
    int exp = (int)((v.u >> 23U) & 0xFFU) - 127 + 15;
    unsigned int frac = (v.u >> 13U) & 0x3FFU;

    if (exp <= 0) {
        return (unsigned short)sign;
    }
    if (exp >= 31) {
        return (unsigned short)(sign | 0x7C00U);
    }
    return (unsigned short)(sign | ((unsigned int)exp << 10U) | frac);
}

int main(void) {
    const VspecSpecialQuantOps* ops = vspec_special_quant_get_ops(VSPEC_SPECIAL_QUANT_Q4_K);
    if (!ops) {
        printf("special quant q4_k unavailable\n");
        return 1;
    }

    VspecQ4KBlock block;
    memset(&block, 0, sizeof(block));
    block.d = float_to_half_simple(0.2f);
    block.dmin = float_to_half_simple(0.05f);

    for (size_t i = 0U; i < sizeof(block.scales); ++i) {
        block.scales[i] = 0xFFU;
    }
    for (size_t i = 0U; i < sizeof(block.qs); ++i) {
        block.qs[i] = 0x88U;
    }

    float row[256];
    ops->dequantize_row(&block, 256U, row);

    float a[256];
    for (size_t i = 0U; i < 256U; ++i) {
        a[i] = 1.0f;
    }

    float c = 0.0f;
    ops->matmul_ref_f32(a, 1U, 256U, &block, 1U, &c);

    printf("special quant=%s block_bytes=%zu sample0=%.6f dot=%.6f\n", ops->name, ops->block_bytes, row[0], c);
    return 0;
}
