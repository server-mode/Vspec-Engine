#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#include "vspec/kernel/cuda_fused.h"
#include "vspec/quant/quant.h"

__device__ static int8_t decode_int3_at(const uint8_t* packed, size_t index) {
    const size_t bit_pos = index * 3U;
    const size_t byte_idx = bit_pos >> 3U;
    const uint8_t shift = (uint8_t)(bit_pos & 7U);

    uint16_t code = (uint16_t)(packed[byte_idx] >> shift);
    if ((uint8_t)(8U - shift) < 3U) {
        code |= (uint16_t)(packed[byte_idx + 1U] << (8U - shift));
    }
    code &= 0x7U;
    return (code & 0x4U) ? (int8_t)(code - 8U) : (int8_t)code;
}

__global__ static void fused_int3_kernel(
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
    const float scale = scales[j];
    float acc = 0.0f;

    for (size_t t = 0; t < k; ++t) {
        const float av = a[i * k + t];
        const int8_t wq = decode_int3_at(b_row, t);
        acc += av * ((float)wq * scale);
    }

    c[i * n + j] = acc;
}

extern "C" void vspec_cuda_fused_linear_int3_device(
    const float* d_a,
    const unsigned char* d_b_packed,
    const float* d_scales,
    float* d_c,
    size_t m,
    size_t n,
    size_t k
) {
    if (!d_a || !d_b_packed || !d_scales || !d_c || m == 0U || n == 0U || k == 0U) {
        return;
    }
    const size_t packed_k = (k * 3U + 7U) / 8U;
    dim3 block(16, 16);
    dim3 grid((unsigned)((n + block.x - 1U) / block.x), (unsigned)((m + block.y - 1U) / block.y));
    fused_int3_kernel<<<grid, block>>>(d_a, d_b_packed, d_scales, d_c, m, n, k, packed_k);
}

extern "C" void vspec_cuda_launch_fused_linear_int3(VspecKernelContext* ctx) {
    if (!ctx || !ctx->input || !ctx->weight || !ctx->output) {
        return;
    }

    if (ctx->qmeta.type != VSPEC_QUANT_INT3 || !ctx->qmeta.scales) {
        return;
    }

    const size_t m = ctx->config.m;
    const size_t n = ctx->config.n;
    const size_t k = ctx->config.k;
    const size_t packed_k = (k * 3U + 7U) / 8U;

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

    vspec_cuda_fused_linear_int3_device(d_a, d_b, d_s, d_c, m, n, k);

    if (cudaDeviceSynchronize() != cudaSuccess) goto cleanup;
    (void)cudaMemcpy(ctx->output, d_c, bytes_c, cudaMemcpyDeviceToHost);

cleanup:
    if (d_c) cudaFree(d_c);
    if (d_s) cudaFree(d_s);
    if (d_b) cudaFree(d_b);
    if (d_a) cudaFree(d_a);
}
