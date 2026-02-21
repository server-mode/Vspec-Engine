#include <cuda_runtime.h>

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

    float* d_in = NULL;
    float* d_w = NULL;
    float* d_out = NULL;

    if (cudaMalloc((void**)&d_in, bytes_in) != cudaSuccess) goto cleanup;
    if (cudaMalloc((void**)&d_w, bytes_w) != cudaSuccess) goto cleanup;
    if (cudaMalloc((void**)&d_out, bytes_out) != cudaSuccess) goto cleanup;

    if (cudaMemcpy(d_in, input, bytes_in, cudaMemcpyHostToDevice) != cudaSuccess) goto cleanup;
    if (cudaMemcpy(d_w, weight, bytes_w, cudaMemcpyHostToDevice) != cudaSuccess) goto cleanup;

    dim3 block(VSPEC_CUDA_BLOCK_LIN, VSPEC_CUDA_BLOCK_LIN);
    dim3 grid(
        (unsigned)((n + block.x - 1U) / block.x),
        (unsigned)((m + block.y - 1U) / block.y)
    );

    linear_f32_kernel<<<grid, block>>>(d_in, d_w, m, k, n, d_out);

    if (cudaDeviceSynchronize() != cudaSuccess) goto cleanup;
    if (cudaMemcpy(output, d_out, bytes_out, cudaMemcpyDeviceToHost) != cudaSuccess) goto cleanup;

+cleanup:
+    if (d_out) cudaFree(d_out);
+    if (d_w) cudaFree(d_w);
+    if (d_in) cudaFree(d_in);
+}
