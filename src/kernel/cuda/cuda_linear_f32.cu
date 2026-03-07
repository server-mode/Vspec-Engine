#include <cuda_runtime.h>
#include <string.h>

#include "vspec/kernel/cuda_ops.h"

#ifndef VSPEC_CUDA_BLOCK_LIN
#define VSPEC_CUDA_BLOCK_LIN 16
#endif

__global__ static void linear_f32_kernel(
    const float* input,
    const float* weight,
    size_t m,
    size_t k,
    size_t n,
    float* output
) {
    const size_t col = (size_t)(blockIdx.x * blockDim.x + threadIdx.x);
    const size_t row = (size_t)(blockIdx.y * blockDim.y + threadIdx.y);

    if (row >= m || col >= n) {
        return;
    }

    const float* in_row = input + row * k;
    const float* w_row = weight + col * k;
    float acc = 0.0f;

    for (size_t i = 0; i < k; ++i) {
        acc += in_row[i] * w_row[i];
    }

    output[row * n + col] = acc;
}

extern "C" void vspec_cuda_linear_f32(
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

    typedef struct LinearWeightCache {
        const float* host_w;
        float* d_w;
        size_t bytes_w;
    } LinearWeightCache;

    static LinearWeightCache cache[64];
    static size_t cache_count = 0U;
    static float* d_in = NULL;
    static float* d_out = NULL;
    static size_t cap_in = 0U;
    static size_t cap_out = 0U;

    LinearWeightCache* entry = NULL;
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
        if (cudaMalloc((void**)&entry->d_w, bytes_w) != cudaSuccess) return;
        if (cudaMemcpy(entry->d_w, weight, bytes_w, cudaMemcpyHostToDevice) != cudaSuccess) return;
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

    dim3 block(VSPEC_CUDA_BLOCK_LIN, VSPEC_CUDA_BLOCK_LIN);
    dim3 grid(
        (unsigned)((n + block.x - 1U) / block.x),
        (unsigned)((m + block.y - 1U) / block.y)
    );

    linear_f32_kernel<<<grid, block>>>(d_in, entry->d_w, m, k, n, d_out);

    if (cudaDeviceSynchronize() != cudaSuccess) return;
    (void)cudaMemcpy(output, d_out, bytes_out, cudaMemcpyDeviceToHost);
}
