#ifndef VSPEC_KERNEL_CUDA_OPS_H
#define VSPEC_KERNEL_CUDA_OPS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void vspec_cuda_rmsnorm_f32(const float* input, const float* weight, float eps, size_t rows, size_t dim, float* output);
void vspec_cuda_linear_f32(const float* input, const float* weight, size_t m, size_t k, size_t n, float* output);
void vspec_cuda_gemm_f32(const float* input, const float* weight, size_t m, size_t k, size_t n, float* output);
void vspec_cuda_attention_single_f32(const float* query, const float* keys, const float* values, size_t seq_len, size_t head_dim, float* output);
void vspec_cuda_silu_f32(float* data, size_t count);
void vspec_cuda_mul_f32(float* data, const float* other, size_t count);

#ifdef __cplusplus
}
#endif

#endif
