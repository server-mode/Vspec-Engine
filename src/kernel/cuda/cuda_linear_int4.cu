#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#include "vspec/kernel/context.h"
#include "vspec/quant/quant.h"

#ifndef VSPEC_CUDA_BLOCK_X
#define VSPEC_CUDA_BLOCK_X 16
#endif

#ifndef VSPEC_CUDA_BLOCK_Y
#define VSPEC_CUDA_BLOCK_Y 16
#endif

__device__ static int8_t decode_int4(uint8_t nibble) {
    nibble &= 0x0F;
    if (nibble & 0x08) {
        return (int8_t)(nibble - 16);
    }
    return (int8_t)nibble;
}

__global__ __launch_bounds__(VSPEC_CUDA_BLOCK_X * VSPEC_CUDA_BLOCK_Y) static void int4_linear_kernel(
    const float* a,
    const uint8_t* b_packed,
    const float* scales,
    float* c,
    size_t m,
    size_t n,
    size_t k,
    size_t packed_k
) {
    const size_t j = (size_t)(blockIdx.x * blockDim.x + threadIdx.x);
    const size_t i = (size_t)(blockIdx.y * blockDim.y + threadIdx.y);

    if (i >= m || j >= n) {
        return;
    }

    const uint8_t* b_row = b_packed + (j * packed_k);
    float acc = 0.0f;

    for (size_t t = 0; t < k; ++t) {
        const float av = a[i * k + t];
        const uint8_t byte = b_row[t >> 1U];
        const uint8_t nibble = (t & 1U) ? ((byte >> 4) & 0x0F) : (byte & 0x0F);
        const float w = (float)decode_int4(nibble) * scales[j];
        acc += av * w;
    }

    c[i * n + j] = acc;
}

extern "C" int vspec_cuda_runtime_available(void) {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        return 0;
    }
    return count > 0 ? 1 : 0;
}

extern "C" void vspec_cuda_launch_linear_impl(VspecKernelContext* ctx) {
    if (!ctx || !ctx->input || !ctx->weight || !ctx->output) {
        return;
    }

    if (ctx->qmeta.type != VSPEC_QUANT_INT4 || !ctx->qmeta.scales) {
        return;
    }

    const size_t m = ctx->config.m;
    const size_t n = ctx->config.n;
    const size_t k = ctx->config.k;
    const size_t packed_k = (k + 1U) / 2U;

    dim3 block(VSPEC_CUDA_BLOCK_X, VSPEC_CUDA_BLOCK_Y);
    dim3 grid((unsigned)((n + block.x - 1U) / block.x), (unsigned)((m + block.y - 1U) / block.y));

    const size_t bytes_a = m * k * sizeof(float);
    const size_t bytes_b = n * packed_k * sizeof(uint8_t);
    const size_t bytes_s = n * sizeof(float);
    const size_t bytes_c = m * n * sizeof(float);

    float* d_a = NULL;
    uint8_t* d_b = NULL;
    float* d_s = NULL;
    float* d_c = NULL;

    if (cudaMalloc((void**)&d_a, bytes_a) != cudaSuccess) goto cleanup;
    if (cudaMalloc((void**)&d_b, bytes_b) != cudaSuccess) goto cleanup;
    if (cudaMalloc((void**)&d_s, bytes_s) != cudaSuccess) goto cleanup;
    if (cudaMalloc((void**)&d_c, bytes_c) != cudaSuccess) goto cleanup;

    if (cudaMemcpy(d_a, ctx->input, bytes_a, cudaMemcpyHostToDevice) != cudaSuccess) goto cleanup;
    if (cudaMemcpy(d_b, ctx->weight, bytes_b, cudaMemcpyHostToDevice) != cudaSuccess) goto cleanup;
    if (cudaMemcpy(d_s, ctx->qmeta.scales, bytes_s, cudaMemcpyHostToDevice) != cudaSuccess) goto cleanup;

    int4_linear_kernel<<<grid, block>>>(d_a, d_b, d_s, d_c, m, n, k, packed_k);

    if (cudaDeviceSynchronize() != cudaSuccess) goto cleanup;
    if (cudaMemcpy(ctx->output, d_c, bytes_c, cudaMemcpyDeviceToHost) != cudaSuccess) goto cleanup;

cleanup:
    if (d_c) cudaFree(d_c);
    if (d_s) cudaFree(d_s);
    if (d_b) cudaFree(d_b);
    if (d_a) cudaFree(d_a);
}

extern "C" void vspec_cuda_launch_attention_impl(VspecKernelContext* ctx) {
    (void)ctx;
}
