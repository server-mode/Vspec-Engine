#ifndef VSPEC_QUANT_INT3_H
#define VSPEC_QUANT_INT3_H

#include <stddef.h>
#include <stdint.h>

size_t vspec_int3_packed_bytes(size_t elements);

void vspec_int3_matmul_ref_f32_q3(
    const float* a,
    size_t m,
    size_t k,
    const uint8_t* b_packed,
    size_t n,
    const float* scales,
    float* c
);

#endif
