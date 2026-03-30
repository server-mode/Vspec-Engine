#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#include "vspec/kernel/cuda_fused.h"
#include "vspec/quant/quant.h"

__device__ static float decode_int2_value(uint8_t packed, uint8_t lane, float scale) {
    const uint8_t q = (uint8_t)((packed >> (lane * 2U)) & 0x3U);
    const float centered = (float)q - 1.5f;
    return centered * scale;
}

__global__ static void fused_int2_kernel_tiled(
    const float* a,
    const uint8_t* b_packed,
    const float* scales,
    float* c,
    size_t m,
    size_t n,
    size_t k,
    size_t packed_k
) {
    const int TILE_M = 16;
    const int TILE_N = 16;
    const int TILE_K = 32;

    __shared__ float sA[TILE_M][TILE_K];
    __shared__ float sB[TILE_N][TILE_K];

    const size_t local_x = (size_t)threadIdx.x;
    const size_t local_y = (size_t)threadIdx.y;
    const size_t col = (size_t)(blockIdx.x * TILE_N + local_x);
    const size_t row = (size_t)(blockIdx.y * TILE_M + local_y);
    const size_t tid = local_y * TILE_N + local_x;
    const int valid = (row < m && col < n) ? 1 : 0;

    float acc = 0.0f;
    for (size_t base = 0U; base < k; base += (size_t)TILE_K) {
        for (size_t idx = tid; idx < (size_t)(TILE_M * TILE_K); idx += (size_t)(TILE_M * TILE_N)) {
            const size_t r = idx / (size_t)TILE_K;
            const size_t c_local = idx % (size_t)TILE_K;
            const size_t g_i = (size_t)blockIdx.y * (size_t)TILE_M + r;
            const size_t g_k = base + c_local;
            sA[r][c_local] = (g_i < m && g_k < k) ? a[g_i * k + g_k] : 0.0f;
        }

        for (size_t idx = tid; idx < (size_t)(TILE_N * TILE_K); idx += (size_t)(TILE_M * TILE_N)) {
            const size_t r = idx / (size_t)TILE_K;
            const size_t c_local = idx % (size_t)TILE_K;
            const size_t g_j = (size_t)blockIdx.x * (size_t)TILE_N + r;
            const size_t g_k = base + c_local;
            float w = 0.0f;
            if (g_j < n && g_k < k) {
                const uint8_t* wr = b_packed + (g_j * packed_k);
                const uint8_t pack = wr[g_k >> 2U];
                const uint8_t lane = (uint8_t)(g_k & 0x3U);
                const float s = scales ? scales[g_j] : 1.0f;
                w = decode_int2_value(pack, lane, s);
            }
            sB[r][c_local] = w;
        }

        __syncthreads();

#pragma unroll
        for (int t = 0; t < TILE_K; ++t) {
            acc += sA[local_y][(size_t)t] * sB[local_x][(size_t)t];
        }
        __syncthreads();
    }

    if (valid) {
        c[row * n + col] = acc;
    }
}

extern "C" void vspec_cuda_fused_linear_int2_device(
    const float* d_a,
    const unsigned char* d_b_packed,
    const float* d_scales,
    float* d_c,
    size_t m,
    size_t n,
    size_t k
) {
    if (!d_a || !d_b_packed || !d_c || m == 0U || n == 0U || k == 0U) {
        return;
    }

    const size_t packed_k = (k + 3U) / 4U;
    dim3 block(16, 16);
    dim3 grid((unsigned)((n + block.x - 1U) / block.x), (unsigned)((m + block.y - 1U) / block.y));
    fused_int2_kernel_tiled<<<grid, block>>>(d_a, d_b_packed, d_scales, d_c, m, n, k, packed_k);
}

__global__ static void dequant_int2_rowwise_kernel(
    const uint8_t* b_packed,
    const float* scales,
    float* w_f32,
    size_t n,
    size_t k,
    size_t packed_k
) {
    const size_t row = (size_t)(blockIdx.y * blockDim.y + threadIdx.y);
    const size_t packed_col = (size_t)(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= n || packed_col >= packed_k) {
        return;
    }

    const uint8_t byte = b_packed[row * packed_k + packed_col];
    const float s = scales ? scales[row] : 1.0f;

    const size_t t0 = packed_col * 4U;
    const size_t t1 = t0 + 1U;
    const size_t t2 = t0 + 2U;
    const size_t t3 = t0 + 3U;

    if (t0 < k) {
        const uint8_t q = (uint8_t)(byte & 0x3U);
        w_f32[row * k + t0] = ((float)q - 1.5f) * s;
    }
    if (t1 < k) {
        const uint8_t q = (uint8_t)((byte >> 2U) & 0x3U);
        w_f32[row * k + t1] = ((float)q - 1.5f) * s;
    }
    if (t2 < k) {
        const uint8_t q = (uint8_t)((byte >> 4U) & 0x3U);
        w_f32[row * k + t2] = ((float)q - 1.5f) * s;
    }
    if (t3 < k) {
        const uint8_t q = (uint8_t)((byte >> 6U) & 0x3U);
        w_f32[row * k + t3] = ((float)q - 1.5f) * s;
    }
}

extern "C" void vspec_cuda_dequant_int2_to_f32_device(
    const unsigned char* d_b_packed,
    const float* d_scales,
    float* d_w_f32,
    size_t n,
    size_t k
) {
    if (!d_b_packed || !d_w_f32 || n == 0U || k == 0U) {
        return;
    }
    const size_t packed_k = (k + 3U) / 4U;
    dim3 block(128, 1);
    dim3 grid((unsigned)((packed_k + block.x - 1U) / block.x), (unsigned)n);
    dequant_int2_rowwise_kernel<<<grid, block>>>(d_b_packed, d_scales, d_w_f32, n, k, packed_k);
}

extern "C" void vspec_cuda_launch_fused_linear_int2(VspecKernelContext* ctx) {
    if (!ctx || !ctx->input || !ctx->weight || !ctx->output) {
        return;
    }
    if (ctx->qmeta.type != VSPEC_QUANT_INT2 || !ctx->qmeta.scales) {
        return;
    }

    const size_t m = ctx->config.m;
    const size_t n = ctx->config.n;
    const size_t k = ctx->config.k;
    const size_t packed_k = (k + 3U) / 4U;

    const size_t bytes_a = m * k * sizeof(float);
    const size_t bytes_b = n * packed_k * sizeof(uint8_t);
    const size_t bytes_s = n * sizeof(float);
    const size_t bytes_c = m * n * sizeof(float);

    float* d_a = NULL;
    uint8_t* d_b = NULL;
    float* d_s = NULL;
    float* d_c = NULL;

    if (cudaMalloc((void**)&d_a, bytes_a) != cudaSuccess) {
        return;
    }
    if (cudaMalloc((void**)&d_b, bytes_b) != cudaSuccess) {
        cudaFree(d_a);
        return;
    }
    if (cudaMalloc((void**)&d_s, bytes_s) != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        return;
    }
    if (cudaMalloc((void**)&d_c, bytes_c) != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_s);
        return;
    }

    if (cudaMemcpy(d_a, ctx->input, bytes_a, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_b, ctx->weight, bytes_b, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_s, ctx->qmeta.scales, bytes_s, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_s);
        cudaFree(d_c);
        return;
    }

    vspec_cuda_fused_linear_int2_device(d_a, d_b, d_s, d_c, m, n, k);
    (void)cudaMemcpy(ctx->output, d_c, bytes_c, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_s);
    cudaFree(d_c);
}
