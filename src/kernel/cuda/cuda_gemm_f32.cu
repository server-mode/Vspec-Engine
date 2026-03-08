#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdlib.h>
#include <string.h>

#include "vspec/kernel/cuda_ops.h"

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
    } GemmWeightCache;

    static GemmWeightCache cache[64];
    static size_t cache_count = 0U;
    static float* d_in = NULL;
    static float* d_out = NULL;
    static size_t cap_in = 0U;
    static size_t cap_out = 0U;
    static cublasHandle_t handle = NULL;
    static int tensorcore_mode = -1;

    if (!handle) {
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
            return;
        }
    }
    if (tensorcore_mode < 0) {
        const char* env = getenv("VSPEC_USE_TENSORCORE_GEMM");
        tensorcore_mode = (!env || env[0] == '\0' || strcmp(env, "0") != 0) ? 1 : 0;
    }

    GemmWeightCache* entry = NULL;
    for (size_t i = 0; i < cache_count; ++i) {
        if (cache[i].host_w == weight && cache[i].bytes_w == bytes_w) {
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
        }
        memset(entry, 0, sizeof(*entry));
        entry->host_w = weight;
        entry->bytes_w = bytes_w;
        if (cudaMalloc((void**)&entry->d_w, bytes_w) != cudaSuccess) {
            return;
        }
        if (cudaMemcpy(entry->d_w, weight, bytes_w, cudaMemcpyHostToDevice) != cudaSuccess) {
            return;
        }
    }

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
