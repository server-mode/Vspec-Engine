#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <stdlib.h>

#include "vspec/kernel/cuda_fused.h"
#include "vspec/quant/quant.h"

__device__ static float clampf_device(float x, float lo, float hi) {
    if (x < lo) {
        return lo;
    }
    if (x > hi) {
        return hi;
    }
    return x;
}

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

static size_t vspec_env_size_or_default(const char* name, size_t fallback) {
    const char* value = getenv(name);
    if (!value || value[0] == '\0') {
        return fallback;
    }
    const long parsed = atol(value);
    if (parsed <= 0L) {
        return fallback;
    }
    return (size_t)parsed;
}

static float vspec_env_float_or_default(const char* name, float fallback) {
    const char* value = getenv(name);
    if (!value || value[0] == '\0') {
        return fallback;
    }
    const float parsed = (float)atof(value);
    if (!(parsed == parsed)) {
        return fallback;
    }
    return parsed;
}

static int vspec_env_flag_or_default(const char* name, int fallback) {
    const char* value = getenv(name);
    if (!value || value[0] == '\0') {
        return fallback;
    }
    const char c = value[0];
    if (c == '1' || c == 'y' || c == 'Y' || c == 't' || c == 'T') {
        return 1;
    }
    if (c == '0' || c == 'n' || c == 'N' || c == 'f' || c == 'F') {
        return 0;
    }
    return fallback;
}

__global__ static void int3_to_int4_expand_kernel(
    const uint8_t* b_int3,
    uint8_t* b_int4,
    size_t n,
    size_t k,
    size_t packed_k3,
    size_t packed_k4
) {
    const size_t row = (size_t)(blockIdx.y * blockDim.y + threadIdx.y);
    const size_t out_byte = (size_t)(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= n || out_byte >= packed_k4) {
        return;
    }

    const uint8_t* row3 = b_int3 + row * packed_k3;
    uint8_t* row4 = b_int4 + row * packed_k4;

    const size_t t0 = out_byte * 2U;
    const size_t t1 = t0 + 1U;

    int8_t q0 = 0;
    int8_t q1 = 0;
    if (t0 < k) {
        q0 = decode_int3_at(row3, t0);
    }
    if (t1 < k) {
        q1 = decode_int3_at(row3, t1);
    }

    uint8_t n0 = (q0 < 0) ? (uint8_t)(q0 + 16) : (uint8_t)q0;
    uint8_t n1 = (q1 < 0) ? (uint8_t)(q1 + 16) : (uint8_t)q1;
    row4[out_byte] = (uint8_t)((n0 & 0x0F) | ((n1 & 0x0F) << 4));
}

__global__ static void clamp_rows_std_kernel(
    const float* input,
    float* output,
    size_t rows,
    size_t cols,
    float alpha
) {
    const size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    const float* in_row = input + row * cols;
    float* out_row = output + row * cols;

    float mean = 0.0f;
    for (size_t i = 0; i < cols; ++i) {
        mean += in_row[i];
    }
    mean /= (float)cols;

    float var = 0.0f;
    for (size_t i = 0; i < cols; ++i) {
        const float d = in_row[i] - mean;
        var += d * d;
    }
    var /= (float)cols;
    const float std = sqrtf(fmaxf(var, 0.0f));
    const float th = fmaxf(1e-6f, fabsf(alpha) * std);

    for (size_t i = 0; i < cols; ++i) {
        out_row[i] = clampf_device(in_row[i], -th, th);
    }
}

__global__ __launch_bounds__(256) static void fused_int4_kernel_tiled(
    const float* a,
    const uint8_t* b_packed,
    const float* scales,
    const float* zero_points,
    float* c,
    size_t m,
    size_t n,
    size_t k,
    size_t packed_k,
    size_t n_blocks
) {
    const int TILE_M = 16;
    const int TILE_N = 16;
    const int TILE_K = 32;

    __shared__ float sA[TILE_M][TILE_K];
    __shared__ float sB[TILE_N][TILE_K];

    const size_t local_x = (size_t)threadIdx.x;
    const size_t local_y = (size_t)threadIdx.y;
    const size_t j = (size_t)(blockIdx.x * TILE_N + local_x);
    const size_t i = (size_t)(blockIdx.y * TILE_M + local_y);
    const size_t tid = local_y * TILE_N + local_x;
    const int valid = (i < m && j < n) ? 1 : 0;

    float acc = 0.0f;

    for (size_t base = 0; base < k; base += (size_t)TILE_K) {
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
                const uint8_t* b_row = b_packed + (g_j * packed_k);
                const uint8_t byte = b_row[g_k >> 1U];
                const uint8_t nibble = (g_k & 1U) ? ((byte >> 4) & 0x0F) : (byte & 0x0F);
                const int8_t wq = (nibble & 0x08) ? (int8_t)(nibble - 16) : (int8_t)nibble;
                const size_t block_id = (n_blocks > 1U) ? ((g_k * n_blocks) / k) : 0U;
                const size_t sb = (block_id < n_blocks) ? block_id : (n_blocks - 1U);
                const size_t idx = g_j * n_blocks + sb;
                const float scale = scales[idx];
                const float zp = (zero_points != NULL) ? zero_points[idx] : 0.0f;
                w = ((float)wq - zp) * scale;
            }
            sB[r][c_local] = w;
        }

        __syncthreads();

        float tile_acc = 0.0f;
#pragma unroll
        for (int t = 0; t < TILE_K; ++t) {
            tile_acc += sA[local_y][(size_t)t] * sB[local_x][(size_t)t];
        }

        acc += tile_acc;
        __syncthreads();
    }

    if (valid) {
        c[i * n + j] = acc;
    }
}

__global__ static void dequant_int4_rowwise_kernel(
    const uint8_t* b_packed,
    const float* scales,
    const float* zero_points,
    float* w_f32,
    size_t n,
    size_t k,
    size_t packed_k,
    size_t n_blocks
) {
    const size_t row = (size_t)(blockIdx.y * blockDim.y + threadIdx.y);
    const size_t packed_col = (size_t)(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= n || packed_col >= packed_k) {
        return;
    }

    const size_t base_packed = row * packed_k;
    const uint8_t byte = b_packed[base_packed + packed_col];
    const size_t t0 = packed_col * 2U;
    const size_t t1 = t0 + 1U;

    size_t block_id0 = (n_blocks > 1U) ? ((t0 * n_blocks) / k) : 0U;
    if (block_id0 >= n_blocks) {
        block_id0 = n_blocks - 1U;
    }
    const size_t idx0 = row * n_blocks + block_id0;
    const float scale0 = scales[idx0];
    const float zp0 = (zero_points != NULL) ? zero_points[idx0] : 0.0f;

    int8_t q0 = (int8_t)(byte & 0x0F);
    if (q0 >= 8) {
        q0 = (int8_t)(q0 - 16);
    }
    if (t0 < k) {
        w_f32[row * k + t0] = ((float)q0 - zp0) * scale0;
    }

    size_t block_id1 = (n_blocks > 1U) ? ((t1 * n_blocks) / k) : 0U;
    if (block_id1 >= n_blocks) {
        block_id1 = n_blocks - 1U;
    }
    const size_t idx1 = row * n_blocks + block_id1;
    const float scale1 = scales[idx1];
    const float zp1 = (zero_points != NULL) ? zero_points[idx1] : 0.0f;

    int8_t q1 = (int8_t)((byte >> 4) & 0x0F);
    if (q1 >= 8) {
        q1 = (int8_t)(q1 - 16);
    }
    if (t1 < k) {
        w_f32[row * k + t1] = ((float)q1 - zp1) * scale1;
    }
}

extern "C" int vspec_cuda_fused_available(void) {
    int count = 0;
    return (cudaGetDeviceCount(&count) == cudaSuccess && count > 0) ? 1 : 0;
}

extern "C" void vspec_cuda_fused_linear_int4_device(
    const float* d_a,
    const unsigned char* d_b_packed,
    const float* d_scales,
    const float* d_zero_points,
    float* d_c,
    size_t m,
    size_t n,
    size_t k,
    size_t n_blocks
) {
    if (!d_a || !d_b_packed || !d_scales || !d_c || m == 0U || n == 0U || k == 0U) {
        return;
    }
    if (n_blocks == 0U) {
        n_blocks = 1U;
    }
    const size_t packed_k = (k + 1U) / 2U;
    size_t block_x = vspec_env_size_or_default("VSPEC_INT4_BLOCK_X", 16U);
    size_t block_y = vspec_env_size_or_default("VSPEC_INT4_BLOCK_Y", 16U);
    if (block_x != 16U || block_y != 16U) {
        block_x = 16U;
        block_y = 16U;
    }

    dim3 block((unsigned)block_x, (unsigned)block_y);
    dim3 grid((unsigned)((n + 15U) / 16U), (unsigned)((m + 15U) / 16U));
    fused_int4_kernel_tiled<<<grid, block>>>(d_a, d_b_packed, d_scales, d_zero_points, d_c, m, n, k, packed_k, n_blocks);
}

extern "C" void vspec_cuda_dequant_int4_to_f32_device(
    const unsigned char* d_b_packed,
    const float* d_scales,
    const float* d_zero_points,
    float* d_w_f32,
    size_t n,
    size_t k,
    size_t n_blocks
) {
    if (!d_b_packed || !d_scales || !d_w_f32 || n == 0U || k == 0U) {
        return;
    }
    if (n_blocks == 0U) {
        n_blocks = 1U;
    }
    const size_t packed_k = (k + 1U) / 2U;
    dim3 block(128, 1);
    dim3 grid((unsigned)((packed_k + block.x - 1U) / block.x), (unsigned)n);
    dequant_int4_rowwise_kernel<<<grid, block>>>(d_b_packed, d_scales, d_zero_points, d_w_f32, n, k, packed_k, n_blocks);
}

extern "C" void vspec_cuda_expand_int3_to_int4_device(
    const unsigned char* d_b_int3,
    unsigned char* d_b_int4,
    size_t n,
    size_t k
) {
    if (!d_b_int3 || !d_b_int4 || n == 0U || k == 0U) {
        return;
    }

    const size_t packed_k3 = (k * 3U + 7U) / 8U;
    const size_t packed_k4 = (k + 1U) / 2U;
    dim3 block(128, 1);
    dim3 grid((unsigned)((packed_k4 + block.x - 1U) / block.x), (unsigned)n);
    int3_to_int4_expand_kernel<<<grid, block>>>(d_b_int3, d_b_int4, n, k, packed_k3, packed_k4);
}

extern "C" void vspec_cuda_launch_fused_linear_int4(VspecKernelContext* ctx) {
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

    const size_t bytes_a = m * k * sizeof(float);
    const size_t bytes_b = n * packed_k * sizeof(uint8_t);
    const size_t bytes_s = n * sizeof(float);
    const size_t bytes_c = m * n * sizeof(float);
    const int clamp_enabled = vspec_env_flag_or_default("VSPEC_FUSED_INPUT_CLAMP_ENABLE", 0);
    const float clamp_alpha = vspec_env_float_or_default("VSPEC_FUSED_INPUT_CLAMP_ALPHA", 6.0f);

    static thread_local float* d_a = NULL;
    static thread_local float* d_a_clamped = NULL;
    static thread_local uint8_t* d_b = NULL;
    static thread_local float* d_s = NULL;
    static thread_local float* d_c = NULL;
    static thread_local size_t cap_a = 0U;
    static thread_local size_t cap_a_clamped = 0U;
    static thread_local size_t cap_b = 0U;
    static thread_local size_t cap_s = 0U;
    static thread_local size_t cap_c = 0U;

    if (cap_a < bytes_a) {
        if (d_a) cudaFree(d_a);
        d_a = NULL;
        if (cudaMalloc((void**)&d_a, bytes_a) != cudaSuccess) return;
        cap_a = bytes_a;
    }
    if (clamp_enabled && clamp_alpha > 0.0f) {
        if (cap_a_clamped < bytes_a) {
            if (d_a_clamped) cudaFree(d_a_clamped);
            d_a_clamped = NULL;
            if (cudaMalloc((void**)&d_a_clamped, bytes_a) != cudaSuccess) return;
            cap_a_clamped = bytes_a;
        }
    } else if (d_a_clamped) {
        cudaFree(d_a_clamped);
        d_a_clamped = NULL;
        cap_a_clamped = 0U;
    }
    if (cap_b < bytes_b) {
        if (d_b) cudaFree(d_b);
        d_b = NULL;
        if (cudaMalloc((void**)&d_b, bytes_b) != cudaSuccess) return;
        cap_b = bytes_b;
    }
    if (cap_s < bytes_s) {
        if (d_s) cudaFree(d_s);
        d_s = NULL;
        if (cudaMalloc((void**)&d_s, bytes_s) != cudaSuccess) return;
        cap_s = bytes_s;
    }
    if (cap_c < bytes_c) {
        if (d_c) cudaFree(d_c);
        d_c = NULL;
        if (cudaMalloc((void**)&d_c, bytes_c) != cudaSuccess) return;
        cap_c = bytes_c;
    }

    if (cudaMemcpy(d_a, ctx->input, bytes_a, cudaMemcpyHostToDevice) != cudaSuccess) return;
    if (cudaMemcpy(d_b, ctx->weight, bytes_b, cudaMemcpyHostToDevice) != cudaSuccess) return;
    if (cudaMemcpy(d_s, ctx->qmeta.scales, bytes_s, cudaMemcpyHostToDevice) != cudaSuccess) return;

    const float* d_input = d_a;
    if (clamp_enabled && clamp_alpha > 0.0f) {
        dim3 clamp_block(128);
        dim3 clamp_grid((unsigned)((m + clamp_block.x - 1U) / clamp_block.x));
        clamp_rows_std_kernel<<<clamp_grid, clamp_block>>>(d_a, d_a_clamped, m, k, clamp_alpha);
        d_input = d_a_clamped;
    }

    vspec_cuda_fused_linear_int4_device(d_input, d_b, d_s, NULL, d_c, m, n, k, 1U);

    if (cudaDeviceSynchronize() != cudaSuccess) return;
    (void)cudaMemcpy(ctx->output, d_c, bytes_c, cudaMemcpyDeviceToHost);
}

extern "C" void vspec_cuda_launch_fused_linear_int3_storage(VspecKernelContext* ctx) {
    if (!ctx || !ctx->input || !ctx->weight || !ctx->output) {
        return;
    }

    if (ctx->qmeta.type != VSPEC_QUANT_INT3 || !ctx->qmeta.scales) {
        return;
    }

    const size_t m = ctx->config.m;
    const size_t n = ctx->config.n;
    const size_t k = ctx->config.k;
    const size_t packed_k3 = (k * 3U + 7U) / 8U;
    const size_t packed_k4 = (k + 1U) / 2U;

    const size_t bytes_a = m * k * sizeof(float);
    const size_t bytes_b3 = n * packed_k3 * sizeof(uint8_t);
    const size_t bytes_b4 = n * packed_k4 * sizeof(uint8_t);
    const size_t bytes_s = n * sizeof(float);
    const size_t bytes_c = m * n * sizeof(float);
    const int clamp_enabled = vspec_env_flag_or_default("VSPEC_FUSED_INPUT_CLAMP_ENABLE", 0);
    const float clamp_alpha = vspec_env_float_or_default("VSPEC_FUSED_INPUT_CLAMP_ALPHA", 6.0f);

    static thread_local float* d_a = NULL;
    static thread_local float* d_a_clamped = NULL;
    static thread_local uint8_t* d_b3 = NULL;
    static thread_local uint8_t* d_b4 = NULL;
    static thread_local float* d_s = NULL;
    static thread_local float* d_c = NULL;
    static thread_local size_t cap_a = 0U;
    static thread_local size_t cap_a_clamped = 0U;
    static thread_local size_t cap_b3 = 0U;
    static thread_local size_t cap_b4 = 0U;
    static thread_local size_t cap_s = 0U;
    static thread_local size_t cap_c = 0U;

    if (cap_a < bytes_a) {
        if (d_a) cudaFree(d_a);
        d_a = NULL;
        if (cudaMalloc((void**)&d_a, bytes_a) != cudaSuccess) return;
        cap_a = bytes_a;
    }
    if (clamp_enabled && clamp_alpha > 0.0f) {
        if (cap_a_clamped < bytes_a) {
            if (d_a_clamped) cudaFree(d_a_clamped);
            d_a_clamped = NULL;
            if (cudaMalloc((void**)&d_a_clamped, bytes_a) != cudaSuccess) return;
            cap_a_clamped = bytes_a;
        }
    } else if (d_a_clamped) {
        cudaFree(d_a_clamped);
        d_a_clamped = NULL;
        cap_a_clamped = 0U;
    }
    if (cap_b3 < bytes_b3) {
        if (d_b3) cudaFree(d_b3);
        d_b3 = NULL;
        if (cudaMalloc((void**)&d_b3, bytes_b3) != cudaSuccess) return;
        cap_b3 = bytes_b3;
    }
    if (cap_b4 < bytes_b4) {
        if (d_b4) cudaFree(d_b4);
        d_b4 = NULL;
        if (cudaMalloc((void**)&d_b4, bytes_b4) != cudaSuccess) return;
        cap_b4 = bytes_b4;
    }
    if (cap_s < bytes_s) {
        if (d_s) cudaFree(d_s);
        d_s = NULL;
        if (cudaMalloc((void**)&d_s, bytes_s) != cudaSuccess) return;
        cap_s = bytes_s;
    }
    if (cap_c < bytes_c) {
        if (d_c) cudaFree(d_c);
        d_c = NULL;
        if (cudaMalloc((void**)&d_c, bytes_c) != cudaSuccess) return;
        cap_c = bytes_c;
    }

    if (cudaMemcpy(d_a, ctx->input, bytes_a, cudaMemcpyHostToDevice) != cudaSuccess) return;
    if (cudaMemcpy(d_b3, ctx->weight, bytes_b3, cudaMemcpyHostToDevice) != cudaSuccess) return;
    if (cudaMemcpy(d_s, ctx->qmeta.scales, bytes_s, cudaMemcpyHostToDevice) != cudaSuccess) return;

    vspec_cuda_expand_int3_to_int4_device(d_b3, d_b4, n, k);

    const float* d_input = d_a;
    if (clamp_enabled && clamp_alpha > 0.0f) {
        dim3 clamp_block(128);
        dim3 clamp_grid((unsigned)((m + clamp_block.x - 1U) / clamp_block.x));
        clamp_rows_std_kernel<<<clamp_grid, clamp_block>>>(d_a, d_a_clamped, m, k, clamp_alpha);
        d_input = d_a_clamped;
    }

    vspec_cuda_fused_linear_int4_device(d_input, d_b4, d_s, NULL, d_c, m, n, k, 1U);

    if (cudaDeviceSynchronize() != cudaSuccess) return;
    (void)cudaMemcpy(ctx->output, d_c, bytes_c, cudaMemcpyDeviceToHost);
}

