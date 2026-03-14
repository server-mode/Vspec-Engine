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

void vspec_int3_compute_zero_points(
    const uint8_t* b_packed,
    size_t k,
    size_t n,
    float* zero_points
);

void vspec_int3_matmul_ref_f32_q3_with_zero_points(
    const float* a,
    size_t m,
    size_t k,
    const uint8_t* b_packed,
    size_t n,
    const float* scales,
    const float* zero_points,
    float* c
);

#endif
