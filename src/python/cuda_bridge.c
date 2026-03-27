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
  #if defined(_MSC_VER)
  #pragma comment(linker, "/NODEFAULTLIB:LIBCMT")
  #endif
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

typedef struct VspecInt4RegisteredWeight {
  int active;
  int handle;
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
} VspecInt4RegisteredWeight;

static const size_t kInt4RegisteredCapMax = 512U;
static const size_t kInt4RegisteredCapDefault = 256U;
static VSPEC_THREAD_LOCAL VspecInt4RegisteredWeight g_int4_registered[512];
static VSPEC_THREAD_LOCAL size_t g_int4_registered_count = 0U;
static VSPEC_THREAD_LOCAL uint64_t g_int4_registered_clock = 0U;
static VSPEC_THREAD_LOCAL float* g_int4_cached_d_in = NULL;
static VSPEC_THREAD_LOCAL float* g_int4_cached_d_out = NULL;
static VSPEC_THREAD_LOCAL size_t g_int4_cached_cap_in = 0U;
static VSPEC_THREAD_LOCAL size_t g_int4_cached_cap_out = 0U;
static VSPEC_THREAD_LOCAL uint64_t g_int4_cached_dispatch_calls = 0U;
static VSPEC_THREAD_LOCAL uint64_t g_int4_cached_dispatch_hits = 0U;
static VSPEC_THREAD_LOCAL uint64_t g_int4_cached_dispatch_misses = 0U;
static VSPEC_THREAD_LOCAL uint64_t g_int4_cached_register_calls = 0U;
static VSPEC_THREAD_LOCAL uint64_t g_int4_cached_register_reuse = 0U;
static VSPEC_THREAD_LOCAL uint64_t g_int4_cached_register_evictions = 0U;
#if defined(VSPEC_USE_CUBLAS) && VSPEC_USE_CUBLAS
static VSPEC_THREAD_LOCAL cublasHandle_t g_int4_cached_cublas = NULL;
#endif

static size_t vspec_int4_registered_cap(void) {
  size_t cap = kInt4RegisteredCapDefault;
  const char* cap_env = getenv("VSPEC_INT4_BRIDGE_CACHE_CAP");
  if (cap_env && cap_env[0] != '\0') {
    long parsed = atol(cap_env);
    if (parsed > 0L) {
      cap = (size_t)parsed;
    }
  }
  if (cap == 0U) {
    cap = 1U;
  }
  if (cap > kInt4RegisteredCapMax) {
    cap = kInt4RegisteredCapMax;
  }
  return cap;
}

static void vspec_int4_registered_release_entry(VspecInt4RegisteredWeight* entry) {
  if (!entry) {
    return;
  }
  if (entry->d_w) {
    cudaFree(entry->d_w);
  }
  if (entry->d_s) {
    cudaFree(entry->d_s);
  }
  if (entry->d_zp) {
    cudaFree(entry->d_zp);
  }
  if (entry->d_w_f32) {
    cudaFree(entry->d_w_f32);
  }
  memset(entry, 0, sizeof(*entry));
}

static VspecInt4RegisteredWeight* vspec_int4_registered_find_by_handle(int handle) {
  if (handle <= 0) {
    return NULL;
  }
  const size_t idx = (size_t)(handle - 1);
  if (idx >= kInt4RegisteredCapMax) {
    return NULL;
  }
  if (!g_int4_registered[idx].active || g_int4_registered[idx].handle != handle) {
    return NULL;
  }
  return &g_int4_registered[idx];
}

static VspecInt4RegisteredWeight* vspec_int4_registered_find_existing(
  const unsigned char* packed_weight,
  const float* scales,
  const float* zero_points,
  size_t bytes_w,
  size_t bytes_s,
  size_t bytes_zp,
  uint64_t fp_w,
  uint64_t fp_s,
  uint64_t fp_zp,
  size_t k,
  size_t n,
  size_t n_blocks
) {
  const size_t cap = vspec_int4_registered_cap();
  for (size_t i = 0U; i < cap; ++i) {
    VspecInt4RegisteredWeight* e = &g_int4_registered[i];
    if (!e->active) {
      continue;
    }
    if (e->host_w == packed_weight && e->host_s == scales && e->host_zp == zero_points &&
        e->bytes_w == bytes_w && e->bytes_s == bytes_s && e->bytes_zp == bytes_zp &&
        e->fp_w == fp_w && e->fp_s == fp_s && e->fp_zp == fp_zp &&
        e->k == k && e->n == n && e->n_blocks == n_blocks) {
      return e;
    }
  }
  return NULL;
}

static VspecInt4RegisteredWeight* vspec_int4_registered_acquire_slot(int* evicted) {
  const size_t cap = vspec_int4_registered_cap();
  if (evicted) {
    *evicted = 0;
  }
  for (size_t i = 0U; i < cap; ++i) {
    if (!g_int4_registered[i].active) {
      if (g_int4_registered_count < (i + 1U)) {
        g_int4_registered_count = i + 1U;
      }
      return &g_int4_registered[i];
    }
  }

  size_t lru_idx = 0U;
  uint64_t lru_used = g_int4_registered[0].last_used;
  for (size_t i = 1U; i < cap; ++i) {
    if (g_int4_registered[i].last_used < lru_used) {
      lru_used = g_int4_registered[i].last_used;
      lru_idx = i;
    }
  }
  if (evicted) {
    *evicted = 1;
  }
  return &g_int4_registered[lru_idx];
}

static int vspec_int4_registered_ensure_cublas(int use_dequant_cublas) {
#if defined(VSPEC_USE_CUBLAS) && VSPEC_USE_CUBLAS
  if (use_dequant_cublas && !g_int4_cached_cublas) {
    if (cublasCreate(&g_int4_cached_cublas) != CUBLAS_STATUS_SUCCESS) {
      return 0;
    }
    (void)cublasSetMathMode(g_int4_cached_cublas, CUBLAS_TF32_TENSOR_OP_MATH);
  }
  return 1;
#else
  (void)use_dequant_cublas;
  return 0;
#endif
}

static int vspec_int4_registered_compute_mode_dequant_cublas(void) {
  const char* mode_env = getenv("VSPEC_INT4_COMPUTE_MODE");
  if (mode_env && mode_env[0] != '\0') {
    if (strcmp(mode_env, "dequant-cublas") == 0 || strcmp(mode_env, "cublas") == 0 || strcmp(mode_env, "tensorcore") == 0) {
#if defined(VSPEC_USE_CUBLAS) && VSPEC_USE_CUBLAS
      return 1;
#else
      return 0;
#endif
    }
  }
  return 0;
}

VSPEC_CUDA_API int vspec_cuda_fused_linear_int4_register_weight_bridge(
  const unsigned char* packed_weight,
  const float* scales,
  const float* zero_points,
  size_t k,
  size_t n,
  size_t n_blocks
) {
  g_int4_cached_register_calls += 1U;
  if (!packed_weight || !scales || k == 0U || n == 0U) {
    return 0;
  }
  if (n_blocks == 0U) {
    n_blocks = 1U;
  }

  const size_t packed_k = (k + 1U) / 2U;
  const size_t bytes_w = n * packed_k * sizeof(unsigned char);
  const size_t bytes_s = n * n_blocks * sizeof(float);
  const size_t bytes_zp = n * n_blocks * sizeof(float);
  const uint64_t fp_w = vspec_hash_sample_bytes(packed_weight, bytes_w, 128U);
  const uint64_t fp_s = vspec_hash_sample_bytes(scales, bytes_s, 64U);
  const uint64_t fp_zp = zero_points ? vspec_hash_sample_bytes(zero_points, bytes_zp, 64U) : 0U;

  VspecInt4RegisteredWeight* existing = vspec_int4_registered_find_existing(
    packed_weight,
    scales,
    zero_points,
    bytes_w,
    bytes_s,
    bytes_zp,
    fp_w,
    fp_s,
    fp_zp,
    k,
    n,
    n_blocks
  );
  if (existing) {
    existing->last_used = ++g_int4_registered_clock;
    g_int4_cached_register_reuse += 1U;
    return existing->handle;
  }

  int evicted = 0;
  VspecInt4RegisteredWeight* slot = vspec_int4_registered_acquire_slot(&evicted);
  if (!slot) {
    return 0;
  }
  if (evicted) {
    g_int4_cached_register_evictions += 1U;
  }
  if (slot->active) {
    vspec_int4_registered_release_entry(slot);
  }

  const size_t idx = (size_t)(slot - &g_int4_registered[0]);
  memset(slot, 0, sizeof(*slot));
  slot->active = 1;
  slot->handle = (int)(idx + 1U);
  slot->host_w = packed_weight;
  slot->host_s = scales;
  slot->host_zp = zero_points;
  slot->bytes_w = bytes_w;
  slot->bytes_s = bytes_s;
  slot->bytes_zp = bytes_zp;
  slot->bytes_w_f32 = n * k * sizeof(float);
  slot->n = n;
  slot->k = k;
  slot->n_blocks = n_blocks;
  slot->fp_w = fp_w;
  slot->fp_s = fp_s;
  slot->fp_zp = fp_zp;
  slot->last_used = ++g_int4_registered_clock;

  if (cudaMalloc((void**)&slot->d_w, bytes_w) != cudaSuccess) {
    vspec_int4_registered_release_entry(slot);
    return 0;
  }
  if (cudaMalloc((void**)&slot->d_s, bytes_s) != cudaSuccess) {
    vspec_int4_registered_release_entry(slot);
    return 0;
  }
  if (cudaMalloc((void**)&slot->d_zp, bytes_zp) != cudaSuccess) {
    vspec_int4_registered_release_entry(slot);
    return 0;
  }

  if (cudaMemcpy(slot->d_w, packed_weight, bytes_w, cudaMemcpyHostToDevice) != cudaSuccess) {
    vspec_int4_registered_release_entry(slot);
    return 0;
  }
  if (cudaMemcpy(slot->d_s, scales, bytes_s, cudaMemcpyHostToDevice) != cudaSuccess) {
    vspec_int4_registered_release_entry(slot);
    return 0;
  }
  if (zero_points) {
    if (cudaMemcpy(slot->d_zp, zero_points, bytes_zp, cudaMemcpyHostToDevice) != cudaSuccess) {
      vspec_int4_registered_release_entry(slot);
      return 0;
    }
  } else {
    if (cudaMemset(slot->d_zp, 0, bytes_zp) != cudaSuccess) {
      vspec_int4_registered_release_entry(slot);
      return 0;
    }
  }

  return slot->handle;
}

VSPEC_CUDA_API int vspec_cuda_fused_linear_int4_cached_bridge(
  const float* input,
  size_t m,
  size_t k,
  int weight_handle,
  size_t expected_n,
  float* output
) {
  g_int4_cached_dispatch_calls += 1U;
  if (!input || !output || m == 0U || k == 0U || expected_n == 0U || weight_handle <= 0) {
    g_int4_cached_dispatch_misses += 1U;
    return 0;
  }

  VspecInt4RegisteredWeight* entry = vspec_int4_registered_find_by_handle(weight_handle);
  if (!entry || !entry->active || !entry->d_w || !entry->d_s || !entry->d_zp) {
    g_int4_cached_dispatch_misses += 1U;
    return 0;
  }
  if (entry->k != k || entry->n != expected_n) {
    g_int4_cached_dispatch_misses += 1U;
    return 0;
  }
  entry->last_used = ++g_int4_registered_clock;

  const size_t bytes_in = m * k * sizeof(float);
  const size_t bytes_out = m * expected_n * sizeof(float);
  if (g_int4_cached_cap_in < bytes_in) {
    if (g_int4_cached_d_in) {
      cudaFree(g_int4_cached_d_in);
    }
    g_int4_cached_d_in = NULL;
    if (cudaMalloc((void**)&g_int4_cached_d_in, bytes_in) != cudaSuccess) {
      g_int4_cached_dispatch_misses += 1U;
      return 0;
    }
    g_int4_cached_cap_in = bytes_in;
  }
  if (g_int4_cached_cap_out < bytes_out) {
    if (g_int4_cached_d_out) {
      cudaFree(g_int4_cached_d_out);
    }
    g_int4_cached_d_out = NULL;
    if (cudaMalloc((void**)&g_int4_cached_d_out, bytes_out) != cudaSuccess) {
      g_int4_cached_dispatch_misses += 1U;
      return 0;
    }
    g_int4_cached_cap_out = bytes_out;
  }

  if (cudaMemcpy(g_int4_cached_d_in, input, bytes_in, cudaMemcpyHostToDevice) != cudaSuccess) {
    g_int4_cached_dispatch_misses += 1U;
    return 0;
  }

  int use_dequant_cublas = vspec_int4_registered_compute_mode_dequant_cublas();
  if (use_dequant_cublas) {
    if (!vspec_int4_registered_ensure_cublas(use_dequant_cublas)) {
      use_dequant_cublas = 0;
    }
  }

#if defined(VSPEC_USE_CUBLAS) && VSPEC_USE_CUBLAS
  if (use_dequant_cublas) {
    if (!entry->d_w_f32) {
      if (cudaMalloc((void**)&entry->d_w_f32, entry->bytes_w_f32) != cudaSuccess) {
        g_int4_cached_dispatch_misses += 1U;
        return 0;
      }
      vspec_cuda_dequant_int4_to_f32_device(entry->d_w, entry->d_s, entry->d_zp, entry->d_w_f32, expected_n, entry->k, entry->n_blocks);
      if (cudaDeviceSynchronize() != cudaSuccess) {
        g_int4_cached_dispatch_misses += 1U;
        return 0;
      }
    }
    {
      const float alpha = 1.0f;
      const float beta = 0.0f;
      cublasStatus_t st = cublasSgemm(
        g_int4_cached_cublas,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        (int)expected_n,
        (int)m,
        (int)entry->k,
        &alpha,
        entry->d_w_f32,
        (int)entry->k,
        g_int4_cached_d_in,
        (int)entry->k,
        &beta,
        g_int4_cached_d_out,
        (int)expected_n
      );
      if (st != CUBLAS_STATUS_SUCCESS) {
        g_int4_cached_dispatch_misses += 1U;
        return 0;
      }
    }
  } else
#endif
  {
    if (entry->d_w_f32) {
      cudaFree(entry->d_w_f32);
      entry->d_w_f32 = NULL;
      entry->bytes_w_f32 = 0U;
    }
    vspec_cuda_fused_linear_int4_device(g_int4_cached_d_in, entry->d_w, entry->d_s, entry->d_zp, g_int4_cached_d_out, m, expected_n, k, entry->n_blocks);
  }

  if (cudaMemcpy(output, g_int4_cached_d_out, bytes_out, cudaMemcpyDeviceToHost) != cudaSuccess) {
    g_int4_cached_dispatch_misses += 1U;
    return 0;
  }

  g_int4_cached_dispatch_hits += 1U;
  return 1;
}

VSPEC_CUDA_API int vspec_cuda_fused_linear_int4_cached_many_bridge(
  const float* input,
  size_t m,
  size_t k,
  const int* weight_handles,
  const size_t* output_dims,
  size_t handle_count,
  float* output
) {
  if (!input || !weight_handles || !output_dims || !output || m == 0U || k == 0U || handle_count == 0U) {
    return 0;
  }

  size_t out_offset = 0U;
  for (size_t i = 0U; i < handle_count; ++i) {
    const int handle = weight_handles[i];
    const size_t out_n = output_dims[i];
    if (handle <= 0 || out_n == 0U) {
      return 0;
    }

    VspecInt4RegisteredWeight* entry = vspec_int4_registered_find_by_handle(handle);
    if (!entry || !entry->active || entry->n != out_n || entry->k != k) {
      return 0;
    }

    if (!vspec_cuda_fused_linear_int4_cached_bridge(input, m, k, handle, out_n, output + out_offset)) {
      return 0;
    }
    out_offset += (m * out_n);
  }

  return 1;
}

VSPEC_CUDA_API int vspec_cuda_int4_cached_stats_bridge(
  uint64_t* out_dispatch_calls,
  uint64_t* out_dispatch_hits,
  uint64_t* out_dispatch_misses,
  uint64_t* out_register_calls,
  uint64_t* out_register_reuse,
  uint64_t* out_register_evictions
) {
  if (!out_dispatch_calls || !out_dispatch_hits || !out_dispatch_misses ||
      !out_register_calls || !out_register_reuse || !out_register_evictions) {
    return 0;
  }
  *out_dispatch_calls = g_int4_cached_dispatch_calls;
  *out_dispatch_hits = g_int4_cached_dispatch_hits;
  *out_dispatch_misses = g_int4_cached_dispatch_misses;
  *out_register_calls = g_int4_cached_register_calls;
  *out_register_reuse = g_int4_cached_register_reuse;
  *out_register_evictions = g_int4_cached_register_evictions;
  return 1;
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

VSPEC_CUDA_API int vspec_cuda_device_capability_bridge(int* out_major, int* out_minor, int* out_multiprocessors) {
  if (!out_major || !out_minor || !out_multiprocessors) {
    return 0;
  }
  int dev = 0;
  struct cudaDeviceProp prop;
  if (cudaGetDevice(&dev) != cudaSuccess) {
    if (cudaSetDevice(0) != cudaSuccess) {
      return 0;
    }
    if (cudaGetDevice(&dev) != cudaSuccess) {
      return 0;
    }
  }
  if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) {
    return 0;
  }
  *out_major = (int)prop.major;
  *out_minor = (int)prop.minor;
  *out_multiprocessors = (int)prop.multiProcessorCount;
  return 1;
}

VSPEC_CUDA_API int vspec_cuda_int4_tensorcore_available_bridge(void) {
  int major = 0;
  int minor = 0;
  int sms = 0;
  if (!vspec_cuda_device_capability_bridge(&major, &minor, &sms)) {
    return 0;
  }
#if defined(VSPEC_USE_CUBLAS) && VSPEC_USE_CUBLAS
  return (major >= 8) ? 1 : 0;
#else
  (void)minor;
  (void)sms;
  return 0;
#endif
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
  static const size_t kCacheCap4Max = 512U;
  static const size_t kCacheCap4Default = 256U;
  static VSPEC_THREAD_LOCAL Cache4 cache[512];
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
  size_t cache_cap4 = kCacheCap4Default;
  const char* cache_cap_env4 = getenv("VSPEC_INT4_BRIDGE_CACHE_CAP");
  if (cache_cap_env4 && cache_cap_env4[0] != '\0') {
    long parsed = atol(cache_cap_env4);
    if (parsed > 0L) {
      cache_cap4 = (size_t)parsed;
    }
  }
  if (cache_cap4 == 0U) {
    cache_cap4 = 1U;
  }
  if (cache_cap4 > kCacheCap4Max) {
    cache_cap4 = kCacheCap4Max;
  }
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
    if (strcmp(mode_env, "dequant-cublas") == 0 || strcmp(mode_env, "cublas") == 0 || strcmp(mode_env, "tensorcore") == 0) {
#if defined(VSPEC_USE_CUBLAS) && VSPEC_USE_CUBLAS
      use_dequant_cublas = 1;
#endif
    }
  }

#if defined(VSPEC_USE_CUBLAS) && VSPEC_USE_CUBLAS
  if (use_dequant_cublas && !handle) {
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
      use_dequant_cublas = 0;
    } else {
      (void)cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
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
      if (cache_count < cache_cap4) {
        entry = &cache[cache_count++];
      } else {
        size_t lru_idx = 0U;
        for (size_t i = 1U; i < cache_cap4; ++i) {
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
  static const size_t kCacheCap3Max = 512U;
  static const size_t kCacheCap3Default = 256U;
  static VSPEC_THREAD_LOCAL Cache3 cache[512];
  static VSPEC_THREAD_LOCAL size_t cache_count = 0U;
  static VSPEC_THREAD_LOCAL uint64_t use_clock = 0U;
  static VSPEC_THREAD_LOCAL float* d_in = NULL;
  static VSPEC_THREAD_LOCAL float* d_out = NULL;
  static VSPEC_THREAD_LOCAL size_t cap_in = 0U;
  static VSPEC_THREAD_LOCAL size_t cap_out = 0U;

  const size_t packed_k = (k * 3U + 7U) / 8U;
  size_t cache_cap3 = kCacheCap3Default;
  const char* cache_cap_env3 = getenv("VSPEC_INT3_BRIDGE_CACHE_CAP");
  if (cache_cap_env3 && cache_cap_env3[0] != '\0') {
    long parsed = atol(cache_cap_env3);
    if (parsed > 0L) {
      cache_cap3 = (size_t)parsed;
    }
  }
  if (cache_cap3 == 0U) {
    cache_cap3 = 1U;
  }
  if (cache_cap3 > kCacheCap3Max) {
    cache_cap3 = kCacheCap3Max;
  }
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
      if (cache_count < cache_cap3) {
        entry = &cache[cache_count++];
      } else {
        size_t lru_idx = 0U;
        for (size_t i = 1U; i < cache_cap3; ++i) {
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
