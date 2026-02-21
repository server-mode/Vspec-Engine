#include <cuda_runtime.h>

#include "vspec/kernel/cuda_ops.h"

#ifndef VSPEC_CUDA_BLOCK_RMS
#define VSPEC_CUDA_BLOCK_RMS 256
#endif

__global__ static void rmsnorm_kernel(const float* input, const float* weight, float eps, size_t dim, float* output) {
    const size_t row = blockIdx.x;
    const float* in = input + row * dim;
    float* out = output + row * dim;

    __shared__ float shared_sum;
    float local = 0.0f;

    for (size_t i = threadIdx.x; i < dim; i += blockDim.x) {
        const float v = in[i];
        local += v * v;
    }

    if (threadIdx.x == 0) {
        shared_sum = 0.0f;
    }
    __syncthreads();

    atomicAdd(&shared_sum, local);
    __syncthreads();

    const float mean = shared_sum / (float)dim;
    const float scale = rsqrtf(mean + eps);

    for (size_t i = threadIdx.x; i < dim; i += blockDim.x) {
        out[i] = in[i] * scale * weight[i];
    }
}

extern "C" void vspec_cuda_rmsnorm_f32(const float* input, const float* weight, float eps, size_t rows, size_t dim, float* output) {
    if (!input || !weight || !output || rows == 0 || dim == 0) {
        return;
    }

    const size_t bytes = rows * dim * sizeof(float);
    const size_t weight_bytes = dim * sizeof(float);

    float* d_in = NULL;
    float* d_w = NULL;
    float* d_out = NULL;

    if (cudaMalloc((void**)&d_in, bytes) != cudaSuccess) goto cleanup;
    if (cudaMalloc((void**)&d_w, weight_bytes) != cudaSuccess) goto cleanup;
    if (cudaMalloc((void**)&d_out, bytes) != cudaSuccess) goto cleanup;

    if (cudaMemcpy(d_in, input, bytes, cudaMemcpyHostToDevice) != cudaSuccess) goto cleanup;
    if (cudaMemcpy(d_w, weight, weight_bytes, cudaMemcpyHostToDevice) != cudaSuccess) goto cleanup;

    dim3 grid((unsigned)rows);
    dim3 block(VSPEC_CUDA_BLOCK_RMS);
    rmsnorm_kernel<<<grid, block>>>(d_in, d_w, eps, dim, d_out);

    if (cudaDeviceSynchronize() != cudaSuccess) goto cleanup;
    if (cudaMemcpy(output, d_out, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) goto cleanup;

cleanup:
    if (d_out) cudaFree(d_out);
    if (d_w) cudaFree(d_w);
    if (d_in) cudaFree(d_in);
}