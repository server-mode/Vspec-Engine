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

__device__ static float prng_uniform_01(uint32_t state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return (float)(state & 0x00FFFFFFU) / 16777216.0f;
}

__device__ static float stochastic_roundf_device(float x, uint32_t seed) {
    float lo = floorf(x);
    float frac = x - lo;
    float u = prng_uniform_01(seed);
    return (u < frac) ? (lo + 1.0f) : lo;
}

__global__ static void fused_int3_kernel(
    const float* a,
    const uint8_t* b_packed,
    const float* scales,
    float* c,
    size_t m,
    size_t n,
    size_t k,
    size_t packed_k,
    int stochastic_rounding
) {
    const size_t j = (size_t)(blockIdx.x * blockDim.x + threadIdx.x);
    const size_t i = (size_t)(blockIdx.y * blockDim.y + threadIdx.y);
    if (i >= m || j >= n) {
        return;
    }

    const uint8_t* b_row = b_packed + (j * packed_k);
    const float scale = scales[j];
    float acc = 0.0f;

    const size_t block_k = 32U;
    for (size_t base = 0; base < k; base += block_k) {
        size_t end = base + block_k;
        if (end > k) {
            end = k;
        }

        float block_abs = 0.0f;
        for (size_t t = base; t < end; ++t) {
            const float av = a[i * k + t];
            const int8_t wq = decode_int3_at(b_row, t);
            const float wv = (float)wq * scale;
            const float aabs = fabsf(av);
            const float wabs = fabsf(wv);
            if (aabs > block_abs) {
                block_abs = aabs;
            }
            if (wabs > block_abs) {
                block_abs = wabs;
            }
        }

        const float block_scale = fmaxf(1e-6f, block_abs);
        float block_acc = 0.0f;
        for (size_t t = base; t < end; ++t) {
            const float av = a[i * k + t] / block_scale;
            const int8_t wq = decode_int3_at(b_row, t);
            float wv = ((float)wq * scale) / block_scale;
            if (stochastic_rounding) {
                uint32_t seed = (uint32_t)(
                    (uint32_t)(i * 2654435761U) ^
                    (uint32_t)(j * 2246822519U) ^
                    (uint32_t)(t * 3266489917U) ^
                    0x9E3779B9U
                );
                float quant_steps = wv * 16.0f;
                wv = stochastic_roundf_device(quant_steps, seed) / 16.0f;
            }
            block_acc += av * wv;
        }
        acc += block_acc * block_scale * block_scale;
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
    size_t k,
    int stochastic_rounding
) {
    if (!d_a || !d_b_packed || !d_scales || !d_c || m == 0U || n == 0U || k == 0U) {
        return;
    }
    const size_t packed_k = (k * 3U + 7U) / 8U;
    dim3 block(16, 16);
    dim3 grid((unsigned)((n + block.x - 1U) / block.x), (unsigned)((m + block.y - 1U) / block.y));
    fused_int3_kernel<<<grid, block>>>(d_a, d_b_packed, d_scales, d_c, m, n, k, packed_k, stochastic_rounding);
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
    const float clamp_alpha = 2.8f;
    int stochastic_rounding = 1;
    const char* sr_env = getenv("VSPEC_3BIT_STOCHASTIC_ROUNDING");
    if (sr_env && sr_env[0] != '\0') {
        if (sr_env[0] == '0' || sr_env[0] == 'n' || sr_env[0] == 'N' || sr_env[0] == 'f' || sr_env[0] == 'F') {
            stochastic_rounding = 0;
        }
    }

    static float* d_a = NULL;
    static float* d_a_clamped = NULL;
    static uint8_t* d_b = NULL;
    static float* d_s = NULL;
    static float* d_c = NULL;
    static size_t cap_a = 0U;
    static size_t cap_a_clamped = 0U;
    static size_t cap_b = 0U;
    static size_t cap_s = 0U;
    static size_t cap_c = 0U;

    if (cap_a < bytes_a) {
        if (d_a) cudaFree(d_a);
        d_a = NULL;
        if (cudaMalloc((void**)&d_a, bytes_a) != cudaSuccess) return;
        cap_a = bytes_a;
    }
    if (cap_a_clamped < bytes_a) {
        if (d_a_clamped) cudaFree(d_a_clamped);
        d_a_clamped = NULL;
        if (cudaMalloc((void**)&d_a_clamped, bytes_a) != cudaSuccess) return;
        cap_a_clamped = bytes_a;
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

    dim3 clamp_block(128);
    dim3 clamp_grid((unsigned)((m + clamp_block.x - 1U) / clamp_block.x));
    clamp_rows_std_kernel<<<clamp_grid, clamp_block>>>(d_a, d_a_clamped, m, k, clamp_alpha);

    vspec_cuda_fused_linear_int3_device(d_a_clamped, d_b, d_s, d_c, m, n, k, stochastic_rounding);

    if (cudaDeviceSynchronize() != cudaSuccess) return;
    (void)cudaMemcpy(ctx->output, d_c, bytes_c, cudaMemcpyDeviceToHost);
}
