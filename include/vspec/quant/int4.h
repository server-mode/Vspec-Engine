#ifndef VSPEC_QUANT_INT4_H
#define VSPEC_QUANT_INT4_H

#include <stddef.h>
#include <stdint.h>

size_t vspec_int4_packed_bytes(size_t elements);
void vspec_int4_pack(const int8_t* src, size_t elements, uint8_t* dst);
int8_t vspec_int4_get(const uint8_t* packed, size_t index);

void vspec_int4_matmul_ref_f32_q4(
    const float* a,
    size_t m,
    size_t k,
    const uint8_t* b_packed,
    size_t n,
    const float* scales,
    float* c
);

#endif
