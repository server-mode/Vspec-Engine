#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>

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

__global__ static void fused_int4_kernel(
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
    const float scale = scales[j];
    const size_t block_k = 64U;

    for (size_t base = 0; base < k; base += block_k) {
        size_t end = base + block_k;
        if (end > k) {
            end = k;
        }

        float block_abs = 0.0f;
        for (size_t t = base; t < end; ++t) {
            const float av = a[i * k + t];
            const uint8_t byte = b_row[t >> 1U];
            const uint8_t nibble = (t & 1U) ? ((byte >> 4) & 0x0F) : (byte & 0x0F);
            const int8_t wq = (nibble & 0x08) ? (int8_t)(nibble - 16) : (int8_t)nibble;
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
            const uint8_t byte = b_row[t >> 1U];
            const uint8_t nibble = (t & 1U) ? ((byte >> 4) & 0x0F) : (byte & 0x0F);
            const int8_t wq = (nibble & 0x08) ? (int8_t)(nibble - 16) : (int8_t)nibble;
            const float wv = ((float)wq * scale) / block_scale;
            block_acc += av * wv;
        }
        acc += block_acc * block_scale * block_scale;
    }

    c[i * n + j] = acc;
}

extern "C" int vspec_cuda_fused_available(void) {
    int count = 0;
    return (cudaGetDeviceCount(&count) == cudaSuccess && count > 0) ? 1 : 0;
}

extern "C" void vspec_cuda_fused_linear_int4_device(
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
    const size_t packed_k = (k + 1U) / 2U;
    dim3 block(16, 16);
    dim3 grid((unsigned)((n + block.x - 1U) / block.x), (unsigned)((m + block.y - 1U) / block.y));
    fused_int4_kernel<<<grid, block>>>(d_a, d_b_packed, d_scales, d_c, m, n, k, packed_k);
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
    const float clamp_alpha = 2.8f;

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

    vspec_cuda_fused_linear_int4_device(d_a_clamped, d_b, d_s, d_c, m, n, k);

    if (cudaDeviceSynchronize() != cudaSuccess) return;
    (void)cudaMemcpy(ctx->output, d_c, bytes_c, cudaMemcpyDeviceToHost);
}

