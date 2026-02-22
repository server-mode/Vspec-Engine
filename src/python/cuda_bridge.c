#include <stddef.h>
#include <cuda_runtime.h>
#include <string.h>

#include "vspec/kernel/cuda_ops.h"
#include "vspec/kernel/cuda_fused.h"
#include "vspec/kernel/context.h"

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

VSPEC_CUDA_API void vspec_cuda_attention_fused_single_f32_bridge(
  const float* query,
  const float* keys,
  const float* values,
  size_t seq_len,
  size_t head_dim,
  float* output
) {
  vspec_cuda_attention_fused_single_f32(query, keys, values, seq_len, head_dim, output);
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

VSPEC_CUDA_API int vspec_cuda_fused_linear_int4_bridge(
  const float* input,
  const unsigned char* packed_weight,
  const float* scales,
  size_t m,
  size_t k,
  size_t n,
  float* output
) {
  if (!input || !packed_weight || !scales || !output || m == 0 || k == 0 || n == 0) {
    return 0;
  }
  typedef struct Cache4 {
    const unsigned char* host_w;
    const float* host_s;
    unsigned char* d_w;
    float* d_s;
    size_t bytes_w;
    size_t bytes_s;
  } Cache4;
  static Cache4 cache[64];
  static size_t cache_count = 0U;
  static float* d_in = NULL;
  static float* d_out = NULL;
  static size_t cap_in = 0U;
  static size_t cap_out = 0U;

  const size_t packed_k = (k + 1U) / 2U;
  const size_t bytes_w = n * packed_k * sizeof(unsigned char);
  const size_t bytes_s = n * sizeof(float);
  const size_t bytes_in = m * k * sizeof(float);
  const size_t bytes_out = m * n * sizeof(float);

  Cache4* entry = NULL;
  for (size_t i = 0; i < cache_count; ++i) {
    if (cache[i].host_w == packed_weight && cache[i].host_s == scales && cache[i].bytes_w == bytes_w && cache[i].bytes_s == bytes_s) {
      entry = &cache[i];
      break;
    }
  }

  if (!entry) {
    if (cache_count < 64U) {
      entry = &cache[cache_count++];
    } else {
      entry = &cache[cache_count - 1U];
      if (entry->d_w) cudaFree(entry->d_w);
      if (entry->d_s) cudaFree(entry->d_s);
    }
    memset(entry, 0, sizeof(*entry));
    entry->host_w = packed_weight;
    entry->host_s = scales;
    entry->bytes_w = bytes_w;
    entry->bytes_s = bytes_s;
    if (cudaMalloc((void**)&entry->d_w, bytes_w) != cudaSuccess) return 0;
    if (cudaMalloc((void**)&entry->d_s, bytes_s) != cudaSuccess) return 0;
    if (cudaMemcpy(entry->d_w, packed_weight, bytes_w, cudaMemcpyHostToDevice) != cudaSuccess) return 0;
    if (cudaMemcpy(entry->d_s, scales, bytes_s, cudaMemcpyHostToDevice) != cudaSuccess) return 0;
  }

  if (cap_in < bytes_in) {
    if (d_in) cudaFree(d_in);
    d_in = NULL;
    if (cudaMalloc((void**)&d_in, bytes_in) != cudaSuccess) return 0;
    cap_in = bytes_in;
  }
  if (cap_out < bytes_out) {
    if (d_out) cudaFree(d_out);
    d_out = NULL;
    if (cudaMalloc((void**)&d_out, bytes_out) != cudaSuccess) return 0;
    cap_out = bytes_out;
  }

  if (cudaMemcpy(d_in, input, bytes_in, cudaMemcpyHostToDevice) != cudaSuccess) return 0;
  vspec_cuda_fused_linear_int4_device(d_in, entry->d_w, entry->d_s, d_out, m, n, k);
  if (cudaDeviceSynchronize() != cudaSuccess) return 0;
  if (cudaMemcpy(output, d_out, bytes_out, cudaMemcpyDeviceToHost) != cudaSuccess) return 0;
  return 1;
}

VSPEC_CUDA_API int vspec_cuda_fused_linear_int3_bridge(
  const float* input,
  const unsigned char* packed_weight,
  const float* scales,
  size_t m,
  size_t k,
  size_t n,
  float* output
) {
  if (!input || !packed_weight || !scales || !output || m == 0 || k == 0 || n == 0) {
    return 0;
  }
  typedef struct Cache3 {
    const unsigned char* host_w;
    const float* host_s;
    unsigned char* d_w;
    float* d_s;
    size_t bytes_w;
    size_t bytes_s;
  } Cache3;
  static Cache3 cache[64];
  static size_t cache_count = 0U;
  static float* d_in = NULL;
  static float* d_out = NULL;
  static size_t cap_in = 0U;
  static size_t cap_out = 0U;

  const size_t packed_k = (k * 3U + 7U) / 8U;
  const size_t bytes_w = n * packed_k * sizeof(unsigned char);
  const size_t bytes_s = n * sizeof(float);
  const size_t bytes_in = m * k * sizeof(float);
  const size_t bytes_out = m * n * sizeof(float);

  Cache3* entry = NULL;
  for (size_t i = 0; i < cache_count; ++i) {
    if (cache[i].host_w == packed_weight && cache[i].host_s == scales && cache[i].bytes_w == bytes_w && cache[i].bytes_s == bytes_s) {
      entry = &cache[i];
      break;
    }
  }

  if (!entry) {
    if (cache_count < 64U) {
      entry = &cache[cache_count++];
    } else {
      entry = &cache[cache_count - 1U];
      if (entry->d_w) cudaFree(entry->d_w);
      if (entry->d_s) cudaFree(entry->d_s);
    }
    memset(entry, 0, sizeof(*entry));
    entry->host_w = packed_weight;
    entry->host_s = scales;
    entry->bytes_w = bytes_w;
    entry->bytes_s = bytes_s;
    if (cudaMalloc((void**)&entry->d_w, bytes_w) != cudaSuccess) return 0;
    if (cudaMalloc((void**)&entry->d_s, bytes_s) != cudaSuccess) return 0;
    if (cudaMemcpy(entry->d_w, packed_weight, bytes_w, cudaMemcpyHostToDevice) != cudaSuccess) return 0;
    if (cudaMemcpy(entry->d_s, scales, bytes_s, cudaMemcpyHostToDevice) != cudaSuccess) return 0;
  }

  if (cap_in < bytes_in) {
    if (d_in) cudaFree(d_in);
    d_in = NULL;
    if (cudaMalloc((void**)&d_in, bytes_in) != cudaSuccess) return 0;
    cap_in = bytes_in;
  }
  if (cap_out < bytes_out) {
    if (d_out) cudaFree(d_out);
    d_out = NULL;
    if (cudaMalloc((void**)&d_out, bytes_out) != cudaSuccess) return 0;
    cap_out = bytes_out;
  }

  if (cudaMemcpy(d_in, input, bytes_in, cudaMemcpyHostToDevice) != cudaSuccess) return 0;
  vspec_cuda_fused_linear_int3_device(d_in, entry->d_w, entry->d_s, d_out, m, n, k);
  if (cudaDeviceSynchronize() != cudaSuccess) return 0;
  if (cudaMemcpy(output, d_out, bytes_out, cudaMemcpyDeviceToHost) != cudaSuccess) return 0;
  return 1;
}

#ifdef __cplusplus
}
#endif
