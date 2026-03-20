#ifndef VSPEC_QUANT_FORMATS_SPECIAL_QUANT_H
#define VSPEC_QUANT_FORMATS_SPECIAL_QUANT_H

#include <stddef.h>

#include "vspec/quant/formats/q4_k.h"

typedef enum VspecSpecialQuantFormat {
    VSPEC_SPECIAL_QUANT_NONE = 0,
    VSPEC_SPECIAL_QUANT_Q4_K = 1
} VspecSpecialQuantFormat;

typedef struct VspecSpecialQuantOps {
    VspecSpecialQuantFormat format;
    const char* name;
    size_t block_elements;
    size_t block_bytes;
    void (*dequantize_row)(const void* row_blocks, size_t elements, float* out);
    void (*matmul_ref_f32)(const float* a, size_t m, size_t k, const void* b_rows, size_t n, float* c);
} VspecSpecialQuantOps;

int vspec_special_quant_supported(VspecSpecialQuantFormat format);
const VspecSpecialQuantOps* vspec_special_quant_get_ops(VspecSpecialQuantFormat format);

#endif
