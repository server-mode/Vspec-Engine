#ifndef VSPEC_QUANT_FORMATS_Q4_K_H
#define VSPEC_QUANT_FORMATS_Q4_K_H

#include <stddef.h>
#include <stdint.h>

#define VSPEC_Q4_K_BLOCK_ELEMENTS 256U
#define VSPEC_Q4_K_SUBBLOCK_ELEMENTS 32U
#define VSPEC_Q4_K_SUBBLOCKS 8U

typedef struct VspecQ4KBlock {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[128];
} VspecQ4KBlock;

size_t vspec_q4k_blocks_for_elements(size_t elements);
size_t vspec_q4k_storage_bytes(size_t elements);

void vspec_q4k_dequantize_row(
    const VspecQ4KBlock* row_blocks,
    size_t elements,
    float* out
);

void vspec_q4k_matmul_ref_f32(
    const float* a,
    size_t m,
    size_t k,
    const VspecQ4KBlock* b_rows,
    size_t n,
    float* c
);

#endif
