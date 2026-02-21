#ifndef VSPEC_MODEL_QWEN_OPS_H
#define VSPEC_MODEL_QWEN_OPS_H

#include <stddef.h>

void vspec_rmsnorm_f32(const float* input, const float* weight, size_t dim, float eps, float* output);
void vspec_silu_inplace(float* data, size_t count);
void vspec_mul_inplace(float* data, const float* other, size_t count);

void vspec_linear_f32(
    const float* input,
    size_t m,
    size_t k,
    const float* weight,
    size_t n,
    const float* bias,
    float* output
);

void vspec_qwen_mlp_ref(
    const float* input,
    size_t m,
    size_t k,
    const float* w1,
    const float* w2,
    const float* w3,
    float* output
);

#endif
