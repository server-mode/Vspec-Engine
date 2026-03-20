#include "vspec/quant/formats/special_quant.h"

#include <stddef.h>

static void dequant_q4k_adapter(const void* row_blocks, size_t elements, float* out) {
    vspec_q4k_dequantize_row((const VspecQ4KBlock*)row_blocks, elements, out);
}

static void matmul_q4k_adapter(const float* a, size_t m, size_t k, const void* b_rows, size_t n, float* c) {
    vspec_q4k_matmul_ref_f32(a, m, k, (const VspecQ4KBlock*)b_rows, n, c);
}

static const VspecSpecialQuantOps k_q4k_ops = {
    VSPEC_SPECIAL_QUANT_Q4_K,
    "q4_k",
    VSPEC_Q4_K_BLOCK_ELEMENTS,
    sizeof(VspecQ4KBlock),
    dequant_q4k_adapter,
    matmul_q4k_adapter,
};

int vspec_special_quant_supported(VspecSpecialQuantFormat format) {
    return (format == VSPEC_SPECIAL_QUANT_Q4_K) ? 1 : 0;
}

const VspecSpecialQuantOps* vspec_special_quant_get_ops(VspecSpecialQuantFormat format) {
    if (format == VSPEC_SPECIAL_QUANT_Q4_K) {
        return &k_q4k_ops;
    }
    return NULL;
}
