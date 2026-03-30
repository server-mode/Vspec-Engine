#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#include "vspec/kernel/cuda_fused.h"

__device__ static float decode_int4_value(
    const unsigned char* w_row,
    size_t idx,
    const float* scales,
    const float* zero_points,
    size_t n_blocks,
    size_t k
) {
    const unsigned char packed = w_row[idx >> 1U];
    const unsigned char q = (idx & 1U) ? (unsigned char)((packed >> 4U) & 0x0FU) : (unsigned char)(packed & 0x0FU);
    const size_t block_size = (n_blocks > 1U) ? ((k + n_blocks - 1U) / n_blocks) : k;
    size_t b = 0U;
    if (block_size > 0U && n_blocks > 1U) {
        b = idx / block_size;
        if (b >= n_blocks) {
            b = n_blocks - 1U;
        }
    }
    const float s = scales ? scales[b] : 1.0f;
    const float zp = zero_points ? zero_points[b] : 0.0f;
    return (((float)q) - 8.0f) * s + zp;
}

__global__ static void refine_hot_int4_kernel(
    const float* a,
    const unsigned char* b_int4,
    const float* s_int4,
    const float* zp_int4,
    float* c,
    const uint32_t* hot_indices,
    size_t hot_count,
    size_t m,
    size_t n,
    size_t k,
    size_t n_blocks
) {
    const size_t hot_pos = (size_t)(blockIdx.x * blockDim.x + threadIdx.x);
    const size_t row = (size_t)(blockIdx.y * blockDim.y + threadIdx.y);
    if (hot_pos >= hot_count || row >= m) {
        return;
    }

    const size_t col = (size_t)hot_indices[hot_pos];
    if (col >= n) {
        return;
    }

    const size_t packed_k4 = (k + 1U) / 2U;
    const unsigned char* w_row = b_int4 + (col * packed_k4);
    const float* s_row = s_int4 ? (s_int4 + (col * n_blocks)) : NULL;
    const float* zp_row = zp_int4 ? (zp_int4 + (col * n_blocks)) : NULL;

    float acc = 0.0f;
    for (size_t t = 0U; t < k; ++t) {
        const float w = decode_int4_value(w_row, t, s_row, zp_row, n_blocks, k);
        acc += a[row * k + t] * w;
    }

    c[row * n + col] = acc;
}

extern "C" void vspec_cuda_fused_linear_hybrid_device(
    const float* d_a,
    const unsigned char* d_b_int2_packed,
    const float* d_s_int2,
    const unsigned char* d_b_int4_packed,
    const float* d_s_int4,
    const float* d_zp_int4,
    float* d_c,
    const uint32_t* d_hot_indices,
    size_t hot_count,
    size_t m,
    size_t n,
    size_t k,
    size_t n_blocks_int4
) {
    if (!d_a || !d_b_int2_packed || !d_s_int2 || !d_c || m == 0U || n == 0U || k == 0U) {
        return;
    }

    // Draft pass for all neurons.
    vspec_cuda_fused_linear_int2_device(
        d_a,
        d_b_int2_packed,
        d_s_int2,
        d_c,
        m,
        n,
        k
    );

    vspec_cuda_refine_hot_int4_device(
        d_a,
        d_b_int4_packed,
        d_s_int4,
        d_zp_int4,
        d_c,
        d_hot_indices,
        hot_count,
        m,
        n,
        k,
        n_blocks_int4
    );
}

extern "C" void vspec_cuda_refine_hot_int4_device(
    const float* d_a,
    const unsigned char* d_b_int4_packed,
    const float* d_s_int4,
    const float* d_zp_int4,
    float* d_c,
    const uint32_t* d_hot_indices,
    size_t hot_count,
    size_t m,
    size_t n,
    size_t k,
    size_t n_blocks_int4
) {
    if (!d_a || !d_b_int4_packed || !d_s_int4 || !d_c || !d_hot_indices || hot_count == 0U ||
        m == 0U || n == 0U || k == 0U) {
        return;
    }

    if (n_blocks_int4 == 0U) {
        n_blocks_int4 = 1U;
    }

    dim3 block(16, 16);
    dim3 grid(
        (unsigned)((hot_count + block.x - 1U) / block.x),
        (unsigned)((m + block.y - 1U) / block.y)
    );
    refine_hot_int4_kernel<<<grid, block>>>(
        d_a,
        d_b_int4_packed,
        d_s_int4,
        d_zp_int4,
        d_c,
        d_hot_indices,
        hot_count,
        m,
        n,
        k,
        n_blocks_int4
    );
}
