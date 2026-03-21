#include <stddef.h>
#include <cuda_runtime.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#if defined(VSPEC_USE_CUBLAS) && VSPEC_USE_CUBLAS
#include <cublas_v2.h>
#endif

#include "vspec/attention/flash_block.h"
#include "vspec/kernel/cuda_ops.h"
#include "vspec/kernel/cuda_fused.h"
#include "vspec/kernel/context.h"

#if defined(_WIN32)
  #define VSPEC_CUDA_API __declspec(dllexport)
  #define VSPEC_THREAD_LOCAL __declspec(thread)
#else
  #define VSPEC_CUDA_API
  #define VSPEC_THREAD_LOCAL __thread
#endif

#ifdef __cplusplus
extern "C" {
#endif

static uint64_t vspec_hash_sample_bytes(const void* ptr, size_t bytes, size_t sample_count) {
  if (!ptr || bytes == 0U) {
    return 0U;
  }
  const uint8_t* p = (const uint8_t*)ptr;
  const size_t samples = (sample_count == 0U) ? 64U : sample_count;
  const size_t step = (bytes / samples) + 1U;
  uint64_t h = 1469598103934665603ULL;
  for (size_t i = 0U; i < bytes; i += step) {
    h ^= (uint64_t)p[i];
    h *= 1099511628211ULL;
  }
  h ^= (uint64_t)bytes;
  h *= 1099511628211ULL;
  return h;
}

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

VSPEC_CUDA_API void vspec_attention_flash_single_f32_bridge(
  const float* query,
  const float* keys,
  const float* values,
  size_t seq_len,
  size_t head_dim,
  size_t block_tokens,
  float* output
) {
  vspec_cuda_attention_flash_single_f32(query, keys, values, seq_len, head_dim, block_tokens, output);
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
  const float* zero_points,
  size_t m,
  size_t k,
  size_t n,
  size_t n_blocks,
  float* output
) {
  if (!input || !packed_weight || !scales || !output || m == 0 || k == 0 || n == 0) {
    return 0;
  }
  if (n_blocks == 0U) {
    n_blocks = 1U;
  }
  typedef struct Cache4 {
    const unsigned char* host_w;
    const float* host_s;
    const float* host_zp;
    unsigned char* d_w;
    float* d_s;
    float* d_zp;
    float* d_w_f32;
    size_t bytes_w;
    size_t bytes_s;
    size_t bytes_zp;
    size_t bytes_w_f32;
    size_t n;
    size_t k;
    size_t n_blocks;
    uint64_t fp_w;
    uint64_t fp_s;
    uint64_t fp_zp;
    uint64_t last_used;
  } Cache4;
  static const size_t kCacheCap4 = 128U;
  static VSPEC_THREAD_LOCAL Cache4 cache[128];
  static VSPEC_THREAD_LOCAL size_t cache_count = 0U;
  static VSPEC_THREAD_LOCAL uint64_t use_clock = 0U;
  static VSPEC_THREAD_LOCAL float* d_in = NULL;
  static VSPEC_THREAD_LOCAL float* d_out = NULL;
  static VSPEC_THREAD_LOCAL size_t cap_in = 0U;
  static VSPEC_THREAD_LOCAL size_t cap_out = 0U;
#if defined(VSPEC_USE_CUBLAS) && VSPEC_USE_CUBLAS
  static VSPEC_THREAD_LOCAL cublasHandle_t handle = NULL;
#endif

  const size_t packed_k = (k + 1U) / 2U;
  const size_t bytes_w = n * packed_k * sizeof(unsigned char);
  const size_t bytes_s = n * n_blocks * sizeof(float);
  const size_t bytes_zp = n * n_blocks * sizeof(float);
  const size_t bytes_in = m * k * sizeof(float);
  const size_t bytes_out = m * n * sizeof(float);
  const uint64_t fp_w = vspec_hash_sample_bytes(packed_weight, bytes_w, 128U);
  const uint64_t fp_s = vspec_hash_sample_bytes(scales, bytes_s, 64U);
  const uint64_t fp_zp = vspec_hash_sample_bytes(zero_points, bytes_zp, 64U);

  Cache4* entry = NULL;
  Cache4* stale_entry = NULL;

  int use_dequant_cublas = 0;
  const char* mode_env = getenv("VSPEC_INT4_COMPUTE_MODE");
  if (mode_env && mode_env[0] != '\0') {
    if (strcmp(mode_env, "dequant-cublas") == 0 || strcmp(mode_env, "cublas") == 0) {
#if defined(VSPEC_USE_CUBLAS) && VSPEC_USE_CUBLAS
      use_dequant_cublas = 1;
#endif
    }
  }

#if defined(VSPEC_USE_CUBLAS) && VSPEC_USE_CUBLAS
  if (use_dequant_cublas && !handle) {
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
      use_dequant_cublas = 0;
    }
  }
#endif

  for (size_t i = 0; i < cache_count; ++i) {
    if (cache[i].host_w == packed_weight && cache[i].host_s == scales && cache[i].host_zp == zero_points &&
        cache[i].bytes_w == bytes_w && cache[i].bytes_s == bytes_s &&
        cache[i].bytes_zp == bytes_zp &&
        cache[i].fp_w == fp_w && cache[i].fp_s == fp_s && cache[i].fp_zp == fp_zp) {
      entry = &cache[i];
      break;
    }
    if (cache[i].host_w == packed_weight && cache[i].host_s == scales && cache[i].host_zp == zero_points &&
        cache[i].bytes_w == bytes_w && cache[i].bytes_s == bytes_s && cache[i].bytes_zp == bytes_zp) {
      stale_entry = &cache[i];
    }
  }

  if (!entry) {
    if (stale_entry) {
      entry = stale_entry;
    } else {
      if (cache_count < kCacheCap4) {
        entry = &cache[cache_count++];
      } else {
        size_t lru_idx = 0U;
        for (size_t i = 1U; i < kCacheCap4; ++i) {
          if (cache[i].last_used < cache[lru_idx].last_used) {
            lru_idx = i;
          }
        }
        entry = &cache[lru_idx];
      }
    }
    if (entry->d_w) cudaFree(entry->d_w);
    if (entry->d_s) cudaFree(entry->d_s);
    if (entry->d_zp) cudaFree(entry->d_zp);
    if (entry->d_w_f32) cudaFree(entry->d_w_f32);
    memset(entry, 0, sizeof(*entry));
    entry->host_w = packed_weight;
    entry->host_s = scales;
    entry->host_zp = zero_points;
    entry->bytes_w = bytes_w;
    entry->bytes_s = bytes_s;
    entry->bytes_zp = bytes_zp;
    entry->bytes_w_f32 = n * k * sizeof(float);
    entry->n = n;
    entry->k = k;
    entry->n_blocks = n_blocks;
    entry->fp_w = fp_w;
    entry->fp_s = fp_s;
    entry->fp_zp = fp_zp;
    if (cudaMalloc((void**)&entry->d_w, bytes_w) != cudaSuccess) return 0;
    if (cudaMalloc((void**)&entry->d_s, bytes_s) != cudaSuccess) return 0;
    if (cudaMalloc((void**)&entry->d_zp, bytes_zp) != cudaSuccess) return 0;
    if (cudaMemcpy(entry->d_w, packed_weight, bytes_w, cudaMemcpyHostToDevice) != cudaSuccess) return 0;
    if (cudaMemcpy(entry->d_s, scales, bytes_s, cudaMemcpyHostToDevice) != cudaSuccess) return 0;
    if (zero_points) {
      if (cudaMemcpy(entry->d_zp, zero_points, bytes_zp, cudaMemcpyHostToDevice) != cudaSuccess) return 0;
    } else {
      if (cudaMemset(entry->d_zp, 0, bytes_zp) != cudaSuccess) return 0;
    }

    if (use_dequant_cublas) {
      if (cudaMalloc((void**)&entry->d_w_f32, entry->bytes_w_f32) != cudaSuccess) return 0;
      vspec_cuda_dequant_int4_to_f32_device(entry->d_w, entry->d_s, entry->d_zp, entry->d_w_f32, n, k, n_blocks);
      if (cudaDeviceSynchronize() != cudaSuccess) return 0;
    }
  }
  entry->last_used = ++use_clock;

  if (use_dequant_cublas && !entry->d_w_f32) {
    entry->bytes_w_f32 = n * k * sizeof(float);
    entry->n = n;
    entry->k = k;
    if (cudaMalloc((void**)&entry->d_w_f32, entry->bytes_w_f32) != cudaSuccess) return 0;
    vspec_cuda_dequant_int4_to_f32_device(entry->d_w, entry->d_s, entry->d_zp, entry->d_w_f32, n, k, n_blocks);
    if (cudaDeviceSynchronize() != cudaSuccess) return 0;
  }

  if (!use_dequant_cublas && entry->d_w_f32) {
    cudaFree(entry->d_w_f32);
    entry->d_w_f32 = NULL;
    entry->bytes_w_f32 = 0U;
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

#if defined(VSPEC_USE_CUBLAS) && VSPEC_USE_CUBLAS
  if (use_dequant_cublas && entry->d_w_f32 && handle) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t st = cublasSgemm(
      handle,
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      (int)n,
      (int)m,
      (int)k,
      &alpha,
      entry->d_w_f32,
      (int)k,
      d_in,
      (int)k,
      &beta,
      d_out,
      (int)n
    );
    if (st != CUBLAS_STATUS_SUCCESS) return 0;
  } else {
    vspec_cuda_fused_linear_int4_device(d_in, entry->d_w, entry->d_s, entry->d_zp, d_out, m, n, k, n_blocks);
  }
#else
  (void)use_dequant_cublas;
  vspec_cuda_fused_linear_int4_device(d_in, entry->d_w, entry->d_s, entry->d_zp, d_out, m, n, k, n_blocks);
#endif

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
    uint64_t fp_w;
    uint64_t fp_s;
    uint64_t last_used;
  } Cache3;
  static const size_t kCacheCap3 = 128U;
  static VSPEC_THREAD_LOCAL Cache3 cache[128];
  static VSPEC_THREAD_LOCAL size_t cache_count = 0U;
  static VSPEC_THREAD_LOCAL uint64_t use_clock = 0U;
  static VSPEC_THREAD_LOCAL float* d_in = NULL;
  static VSPEC_THREAD_LOCAL float* d_out = NULL;
  static VSPEC_THREAD_LOCAL size_t cap_in = 0U;
  static VSPEC_THREAD_LOCAL size_t cap_out = 0U;

  const size_t packed_k = (k * 3U + 7U) / 8U;
  const size_t bytes_w = n * packed_k * sizeof(unsigned char);
  const size_t bytes_s = n * sizeof(float);
  const size_t bytes_in = m * k * sizeof(float);
  const size_t bytes_out = m * n * sizeof(float);
  const uint64_t fp_w = vspec_hash_sample_bytes(packed_weight, bytes_w, 128U);
  const uint64_t fp_s = vspec_hash_sample_bytes(scales, bytes_s, 64U);

  Cache3* entry = NULL;
  Cache3* stale_entry = NULL;
  for (size_t i = 0; i < cache_count; ++i) {
    if (cache[i].host_w == packed_weight && cache[i].host_s == scales &&
        cache[i].bytes_w == bytes_w && cache[i].bytes_s == bytes_s &&
        cache[i].fp_w == fp_w && cache[i].fp_s == fp_s) {
      entry = &cache[i];
      break;
    }
    if (cache[i].host_w == packed_weight && cache[i].host_s == scales &&
        cache[i].bytes_w == bytes_w && cache[i].bytes_s == bytes_s) {
      stale_entry = &cache[i];
    }
  }

  if (!entry) {
    if (stale_entry) {
      entry = stale_entry;
    } else {
      if (cache_count < kCacheCap3) {
        entry = &cache[cache_count++];
      } else {
        size_t lru_idx = 0U;
        for (size_t i = 1U; i < kCacheCap3; ++i) {
          if (cache[i].last_used < cache[lru_idx].last_used) {
            lru_idx = i;
          }
        }
        entry = &cache[lru_idx];
      }
    }
    if (entry->d_w) cudaFree(entry->d_w);
    if (entry->d_s) cudaFree(entry->d_s);
    memset(entry, 0, sizeof(*entry));
    entry->host_w = packed_weight;
    entry->host_s = scales;
    entry->bytes_w = bytes_w;
    entry->bytes_s = bytes_s;
    entry->fp_w = fp_w;
    entry->fp_s = fp_s;
    if (cudaMalloc((void**)&entry->d_w, bytes_w) != cudaSuccess) return 0;
    if (cudaMalloc((void**)&entry->d_s, bytes_s) != cudaSuccess) return 0;
    if (cudaMemcpy(entry->d_w, packed_weight, bytes_w, cudaMemcpyHostToDevice) != cudaSuccess) return 0;
    if (cudaMemcpy(entry->d_s, scales, bytes_s, cudaMemcpyHostToDevice) != cudaSuccess) return 0;
  }
  entry->last_used = ++use_clock;

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
  vspec_cuda_fused_linear_int3_device(d_in, entry->d_w, entry->d_s, d_out, m, n, k, 1);
  if (cudaMemcpy(output, d_out, bytes_out, cudaMemcpyDeviceToHost) != cudaSuccess) return 0;
  return 1;
}

#ifdef __cplusplus
}
#endif
