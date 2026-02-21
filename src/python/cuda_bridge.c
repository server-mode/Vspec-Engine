#include <stddef.h>
#include <cuda_runtime.h>

#include "vspec/kernel/cuda_ops.h"

#if defined(_WIN32)
  #define VSPEC_CUDA_API __declspec(dllexport)
#else
  #define VSPEC_CUDA_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

VSPEC_CUDA_API void vspec_cuda_rmsnorm_f32_bridge(
    const float* input,
    const float* weight,
    float eps,
    size_t rows,
    size_t dim,
    float* output
) {
    vspec_cuda_rmsnorm_f32(input, weight, eps, rows, dim, output);
}

VSPEC_CUDA_API void vspec_cuda_linear_f32_bridge(
  const float* input,
  const float* weight,
  size_t m,
  size_t k,
  size_t n,
  float* output
) {
  vspec_cuda_linear_f32(input, weight, m, k, n, output);
}

VSPEC_CUDA_API void vspec_cuda_gemm_f32_bridge(
  const float* input,
  const float* weight,
  size_t m,
  size_t k,
  size_t n,
  float* output
) {
  vspec_cuda_gemm_f32(input, weight, m, k, n, output);
}

VSPEC_CUDA_API void vspec_cuda_attention_single_f32_bridge(
  const float* query,
  const float* keys,
  const float* values,
  size_t seq_len,
  size_t head_dim,
  float* output
) {
  vspec_cuda_attention_single_f32(query, keys, values, seq_len, head_dim, output);
}

VSPEC_CUDA_API void vspec_cuda_silu_f32_bridge(float* data, size_t count) {
  vspec_cuda_silu_f32(data, count);
}

VSPEC_CUDA_API void vspec_cuda_mul_f32_bridge(const float* a, const float* b, size_t count, float* output) {
  if (!output) {
    return;
  }
  for (size_t i = 0; i < count; ++i) {
    output[i] = a[i];
  }
  vspec_cuda_mul_f32(output, b, count);
}

VSPEC_CUDA_API int vspec_cuda_mem_info_bridge(size_t* free_bytes, size_t* total_bytes) {
  if (!free_bytes || !total_bytes) {
    return 0;
  }
  size_t free_mem = 0U;
  size_t total_mem = 0U;
  cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
  if (err != cudaSuccess) {
    return 0;
  }
  *free_bytes = free_mem;
  *total_bytes = total_mem;
  return 1;
}

#ifdef __cplusplus
}
#endif
