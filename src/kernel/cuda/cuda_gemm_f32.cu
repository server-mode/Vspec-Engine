#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "vspec/kernel/cuda_ops.h"

static const size_t VSPEC_GEMM_WEIGHT_CACHE_CAPACITY = 256U;

extern "C" void vspec_cuda_gemm_f32(
    const float* input,
    const float* weight,
    size_t m,
    size_t k,
    size_t n,
    float* output
) {
    if (!input || !weight || !output || m == 0 || k == 0 || n == 0) {
        return;
    }

    const size_t bytes_in = m * k * sizeof(float);
    const size_t bytes_w = n * k * sizeof(float);
    const size_t bytes_out = m * n * sizeof(float);

    typedef struct GemmWeightCache {
        const float* host_w;
        float* d_w;
        size_t bytes_w;
        uint64_t fingerprint;
        uint64_t last_used;
    } GemmWeightCache;

    static thread_local GemmWeightCache cache[VSPEC_GEMM_WEIGHT_CACHE_CAPACITY];
    static thread_local size_t cache_count = 0U;
    static thread_local uint64_t use_clock = 0U;
    static thread_local float* d_in = NULL;
    static thread_local float* d_out = NULL;
    static thread_local size_t cap_in = 0U;
    static thread_local size_t cap_out = 0U;
    static thread_local cublasHandle_t handle = NULL;
    static thread_local int tensorcore_mode = -1;

    auto hash_weight_fingerprint = [](const float* w, size_t bytes) -> uint64_t {
        if (!w || bytes < sizeof(float)) {
            return (uint64_t)bytes;
        }
        const size_t elems = bytes / sizeof(float);
        const size_t samples = 64U;
        const size_t step = (elems / samples) + 1U;
        uint64_t h = 1469598103934665603ULL;
        for (size_t i = 0U; i < elems; i += step) {
            uint32_t bits = 0U;
            memcpy(&bits, w + i, sizeof(uint32_t));
            h ^= (uint64_t)bits;
            h *= 1099511628211ULL;
        }
        h ^= (uint64_t)elems;
        h *= 1099511628211ULL;
        return h;
    };

    if (!handle) {
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
            return;
        }
    }
    if (tensorcore_mode < 0) {
        const char* env = getenv("VSPEC_USE_TENSORCORE_GEMM");
        tensorcore_mode = (!env || env[0] == '\0' || strcmp(env, "0") != 0) ? 1 : 0;
    }

    const uint64_t weight_fp = hash_weight_fingerprint(weight, bytes_w);

    GemmWeightCache* entry = NULL;
    GemmWeightCache* stale_entry = NULL;
    for (size_t i = 0; i < cache_count; ++i) {
        if (cache[i].host_w == weight && cache[i].bytes_w == bytes_w && cache[i].fingerprint == weight_fp) {
            entry = &cache[i];
            break;
        }
        if (cache[i].host_w == weight && cache[i].bytes_w == bytes_w) {
            stale_entry = &cache[i];
        }
    }

    if (!entry) {
        if (stale_entry) {
            entry = stale_entry;
        } else {
            if (cache_count < VSPEC_GEMM_WEIGHT_CACHE_CAPACITY) {
                entry = &cache[cache_count++];
            } else {
                size_t lru_idx = 0U;
                for (size_t i = 1U; i < VSPEC_GEMM_WEIGHT_CACHE_CAPACITY; ++i) {
                    if (cache[i].last_used < cache[lru_idx].last_used) {
                        lru_idx = i;
                    }
                }
                entry = &cache[lru_idx];
            }
        }
        if (entry->d_w) cudaFree(entry->d_w);
        memset(entry, 0, sizeof(*entry));
        entry->host_w = weight;
        entry->bytes_w = bytes_w;
        entry->fingerprint = weight_fp;
        if (cudaMalloc((void**)&entry->d_w, bytes_w) != cudaSuccess) {
            return;
        }
        if (cudaMemcpy(entry->d_w, weight, bytes_w, cudaMemcpyHostToDevice) != cudaSuccess) {
            return;
        }
    }
    entry->last_used = ++use_clock;

    if (cap_in < bytes_in) {
        if (d_in) cudaFree(d_in);
        d_in = NULL;
        if (cudaMalloc((void**)&d_in, bytes_in) != cudaSuccess) return;
        cap_in = bytes_in;
    }
    if (cap_out < bytes_out) {
        if (d_out) cudaFree(d_out);
        d_out = NULL;
        if (cudaMalloc((void**)&d_out, bytes_out) != cudaSuccess) return;
        cap_out = bytes_out;
    }

    if (cudaMemcpy(d_in, input, bytes_in, cudaMemcpyHostToDevice) != cudaSuccess) return;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasStatus_t st;
    if (tensorcore_mode) {
        (void)cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
        st = cublasGemmEx(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            (int)n,
            (int)m,
            (int)k,
            &alpha,
            entry->d_w,
            CUDA_R_32F,
            (int)k,
            d_in,
            CUDA_R_32F,
            (int)k,
            &beta,
            d_out,
            CUDA_R_32F,
            (int)n,
            CUBLAS_COMPUTE_32F_FAST_TF32,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        );
        if (st != CUBLAS_STATUS_SUCCESS) {
            (void)cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
        }
    } else {
        st = CUBLAS_STATUS_NOT_SUPPORTED;
    }
    if (st != CUBLAS_STATUS_SUCCESS) {
        st = cublasSgemm(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            (int)n,
            (int)m,
            (int)k,
            &alpha,
            entry->d_w,
            (int)k,
            d_in,
            (int)k,
            &beta,
            d_out,
            (int)n
        );
    }
    if (st != CUBLAS_STATUS_SUCCESS) return;

    if (cudaDeviceSynchronize() != cudaSuccess) return;
    (void)cudaMemcpy(output, d_out, bytes_out, cudaMemcpyDeviceToHost);
}
