#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "vspec/attention/kv_cache.h"
#include "vspec/kernel/cuda_fused.h"

__device__ static float clampf_device(float x, float lo, float hi) {
    if (x < lo) {
        return lo;
    }
    if (x > hi) {
        return hi;
    }
    return x;
}

__device__ static int8_t decode_int3_at_device(const uint8_t* packed, size_t index) {
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

__global__ static void dequant_int3_kv_rows_kernel(
    const uint8_t* packed_rows,
    const float* scales,
    float* out,
    size_t seq_len,
    size_t head_dim,
    size_t packed_head_bytes,
    size_t blocks_per_head,
    size_t block_size
) {
    const size_t col = (size_t)(blockIdx.x * blockDim.x + threadIdx.x);
    const size_t row = (size_t)(blockIdx.y * blockDim.y + threadIdx.y);
    if (row >= seq_len || col >= head_dim) {
        return;
    }

    const uint8_t* packed = packed_rows + row * packed_head_bytes;
    const float* row_scales = scales + row * blocks_per_head;
    const size_t b = col / block_size;
    const float s = row_scales[b];
    const int8_t q = decode_int3_at_device(packed, col);
    out[row * head_dim + col] = (float)q * s;
}

__global__ static void clamp_vector_std_kernel(
    const float* input,
    float* output,
    size_t n,
    float alpha
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    if (!input || !output || n == 0U) {
        return;
    }

    float mean = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        mean += input[i];
    }
    mean /= (float)n;

    float var = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float d = input[i] - mean;
        var += d * d;
    }
    var /= (float)n;
    const float std = sqrtf(fmaxf(var, 0.0f));
    const float th = fmaxf(1e-6f, fabsf(alpha) * std);

    for (size_t i = 0; i < n; ++i) {
        output[i] = clampf_device(input[i], -th, th);
    }
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
        float d = in_row[i] - mean;
        var += d * d;
    }
    var /= (float)cols;
    const float std = sqrtf(fmaxf(var, 0.0f));
    const float th = fmaxf(1e-6f, fabsf(alpha) * std);

    for (size_t i = 0; i < cols; ++i) {
        out_row[i] = clampf_device(in_row[i], -th, th);
    }
}

__global__ static void fused_attention_scores_kernel(
    const float* query,
    const float* keys,
    float* scores,
    size_t seq_len,
    size_t head_dim,
    float inv_sqrt_dim
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len) {
        return;
    }

    const float* k = keys + idx * head_dim;
    float acc = 0.0f;
    for (size_t i = 0; i < head_dim; ++i) {
        acc += query[i] * k[i];
    }
    scores[idx] = acc * inv_sqrt_dim;
}

__global__ static void fused_attention_softmax_kernel(float* scores, size_t seq_len, float denom_floor) {
    __shared__ float red[256];
    __shared__ float max_val;
    __shared__ float sum_val;

    float local_max = -FLT_MAX;
    for (size_t i = threadIdx.x; i < seq_len; i += blockDim.x) {
        float v = scores[i];
        if (v > local_max) {
            local_max = v;
        }
    }
    red[threadIdx.x] = local_max;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2U; stride > 0U; stride >>= 1U) {
        if (threadIdx.x < stride) {
            float rhs = red[threadIdx.x + stride];
            if (rhs > red[threadIdx.x]) {
                red[threadIdx.x] = rhs;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0U) {
        max_val = red[0];
    }
    __syncthreads();

    float local_sum = 0.0f;
    for (size_t i = threadIdx.x; i < seq_len; i += blockDim.x) {
        float e = expf(scores[i] - max_val);
        scores[i] = e;
        local_sum += e;
    }
    red[threadIdx.x] = local_sum;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2U; stride > 0U; stride >>= 1U) {
        if (threadIdx.x < stride) {
            red[threadIdx.x] += red[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0U) {
        sum_val = fmaxf(red[0], denom_floor);
    }
    __syncthreads();

    for (size_t i = threadIdx.x; i < seq_len; i += blockDim.x) {
        scores[i] = scores[i] / sum_val;
    }
}

__global__ static void fused_attention_output_kernel(
    const float* scores,
    const float* values,
    float* output,
    size_t seq_len,
    size_t head_dim
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= head_dim) {
        return;
    }

    float acc = 0.0f;
    for (size_t i = 0; i < seq_len; ++i) {
        acc += scores[i] * values[i * head_dim + idx];
    }
    output[idx] = acc;
}

extern "C" void vspec_cuda_attention_fused_single_f32(
    const float* query,
    const float* keys,
    const float* values,
    size_t seq_len,
    size_t head_dim,
    float* output
) {
    if (!query || !keys || !values || !output || seq_len == 0U || head_dim == 0U) {
        return;
    }

    const size_t bytes_q = head_dim * sizeof(float);
    const size_t bytes_kv = seq_len * head_dim * sizeof(float);
    const size_t bytes_out = head_dim * sizeof(float);
    const float softmax_denom_floor = 1e-12f;

    static float* d_q = NULL;
    static float* d_k = NULL;
    static float* d_v = NULL;
    static float* d_scores = NULL;
    static float* d_out = NULL;
    static size_t cap_q = 0U;
    static size_t cap_kv = 0U;
    static size_t cap_scores = 0U;
    static size_t cap_out = 0U;

    if (cap_q < bytes_q) {
        if (d_q) cudaFree(d_q);
        d_q = NULL;
        if (cudaMalloc((void**)&d_q, bytes_q) != cudaSuccess) return;
        cap_q = bytes_q;
    }
    if (cap_kv < bytes_kv) {
        if (d_k) cudaFree(d_k);
        if (d_v) cudaFree(d_v);
        d_k = NULL;
        d_v = NULL;
        if (cudaMalloc((void**)&d_k, bytes_kv) != cudaSuccess) return;
        if (cudaMalloc((void**)&d_v, bytes_kv) != cudaSuccess) return;
        cap_kv = bytes_kv;
    }
    if (cap_out < bytes_out) {
        if (d_out) cudaFree(d_out);
        d_out = NULL;
        if (cudaMalloc((void**)&d_out, bytes_out) != cudaSuccess) return;
        cap_out = bytes_out;
    }
    if (cap_scores < seq_len * sizeof(float)) {
        if (d_scores) cudaFree(d_scores);
        d_scores = NULL;
        if (cudaMalloc((void**)&d_scores, seq_len * sizeof(float)) != cudaSuccess) return;
        cap_scores = seq_len * sizeof(float);
    }

    if (cudaMemcpy(d_q, query, bytes_q, cudaMemcpyHostToDevice) != cudaSuccess) return;
    if (cudaMemcpy(d_k, keys, bytes_kv, cudaMemcpyHostToDevice) != cudaSuccess) return;
    if (cudaMemcpy(d_v, values, bytes_kv, cudaMemcpyHostToDevice) != cudaSuccess) return;

    const float inv_sqrt_dim = 1.0f / sqrtf((float)head_dim);
    unsigned int block_size = 256U;
    if (head_dim <= 128U && seq_len <= 512U) {
        block_size = 128U;
    }
    dim3 block(block_size);
    dim3 grid_scores((unsigned)((seq_len + block.x - 1U) / block.x));
    dim3 grid_out((unsigned)((head_dim + block.x - 1U) / block.x));

    fused_attention_scores_kernel<<<grid_scores, block>>>(d_q, d_k, d_scores, seq_len, head_dim, inv_sqrt_dim);
    fused_attention_softmax_kernel<<<1, block>>>(d_scores, seq_len, softmax_denom_floor);
    fused_attention_output_kernel<<<grid_out, block>>>(d_scores, d_v, d_out, seq_len, head_dim);
    if (cudaGetLastError() != cudaSuccess) return;
    if (cudaMemcpy(output, d_out, bytes_out, cudaMemcpyDeviceToHost) != cudaSuccess) return;
}

extern "C" void vspec_cuda_attention_flash_single_f32(
    const float* query,
    const float* keys,
    const float* values,
    size_t seq_len,
    size_t head_dim,
    size_t block_tokens,
    float* output
) {
    (void)block_tokens;
    vspec_cuda_attention_fused_single_f32(query, keys, values, seq_len, head_dim, output);
}

static void vspec_cuda_attention_fused_single_int3kv_f32(
    const float* query,
    const uint8_t* key_packed,
    const uint8_t* value_packed,
    const float* key_scales,
    const float* value_scales,
    size_t seq_len,
    size_t head_dim,
    size_t packed_head_bytes,
    size_t blocks_per_head,
    size_t block_size,
    float* output
) {
    if (!query || !key_packed || !value_packed || !key_scales || !value_scales || !output ||
        seq_len == 0U || head_dim == 0U || packed_head_bytes == 0U || blocks_per_head == 0U || block_size == 0U) {
        return;
    }

    const size_t bytes_q = head_dim * sizeof(float);
    const size_t bytes_qpacked = seq_len * packed_head_bytes * sizeof(uint8_t);
    const size_t bytes_scales = seq_len * blocks_per_head * sizeof(float);
    const size_t bytes_kv = seq_len * head_dim * sizeof(float);
    const size_t bytes_out = head_dim * sizeof(float);
    const float softmax_denom_floor = 1e-12f;

    static float* d_q = NULL;
    static uint8_t* d_kq = NULL;
    static uint8_t* d_vq = NULL;
    static float* d_ks = NULL;
    static float* d_vs = NULL;
    static float* d_k = NULL;
    static float* d_k_clamped = NULL;
    static float* d_v = NULL;
    static float* d_scores = NULL;
    static float* d_out = NULL;
    static size_t cap_q = 0U;
    static size_t cap_qpacked = 0U;
    static size_t cap_scales = 0U;
    static size_t cap_kv = 0U;
    static size_t cap_scores = 0U;
    static size_t cap_out = 0U;

    if (cap_q < bytes_q) {
        if (d_q) cudaFree(d_q);
        d_q = NULL;
        if (cudaMalloc((void**)&d_q, bytes_q) != cudaSuccess) return;
        cap_q = bytes_q;
    }
    if (cap_qpacked < bytes_qpacked) {
        if (d_kq) cudaFree(d_kq);
        if (d_vq) cudaFree(d_vq);
        d_kq = NULL;
        d_vq = NULL;
        if (cudaMalloc((void**)&d_kq, bytes_qpacked) != cudaSuccess) return;
        if (cudaMalloc((void**)&d_vq, bytes_qpacked) != cudaSuccess) return;
        cap_qpacked = bytes_qpacked;
    }
    if (cap_scales < bytes_scales) {
        if (d_ks) cudaFree(d_ks);
        if (d_vs) cudaFree(d_vs);
        d_ks = NULL;
        d_vs = NULL;
        if (cudaMalloc((void**)&d_ks, bytes_scales) != cudaSuccess) return;
        if (cudaMalloc((void**)&d_vs, bytes_scales) != cudaSuccess) return;
        cap_scales = bytes_scales;
    }
    if (cap_kv < bytes_kv) {
        if (d_k) cudaFree(d_k);
        if (d_v) cudaFree(d_v);
        d_k = NULL;
        d_v = NULL;
        if (cudaMalloc((void**)&d_k, bytes_kv) != cudaSuccess) return;
        if (cudaMalloc((void**)&d_v, bytes_kv) != cudaSuccess) return;
        cap_kv = bytes_kv;
    }
    if (cap_out < bytes_out) {
        if (d_out) cudaFree(d_out);
        d_out = NULL;
        if (cudaMalloc((void**)&d_out, bytes_out) != cudaSuccess) return;
        cap_out = bytes_out;
    }
    if (cap_scores < seq_len * sizeof(float)) {
        if (d_scores) cudaFree(d_scores);
        d_scores = NULL;
        if (cudaMalloc((void**)&d_scores, seq_len * sizeof(float)) != cudaSuccess) return;
        cap_scores = seq_len * sizeof(float);
    }

    if (cudaMemcpy(d_q, query, bytes_q, cudaMemcpyHostToDevice) != cudaSuccess) return;
    if (cudaMemcpy(d_kq, key_packed, bytes_qpacked, cudaMemcpyHostToDevice) != cudaSuccess) return;
    if (cudaMemcpy(d_vq, value_packed, bytes_qpacked, cudaMemcpyHostToDevice) != cudaSuccess) return;
    if (cudaMemcpy(d_ks, key_scales, bytes_scales, cudaMemcpyHostToDevice) != cudaSuccess) return;
    if (cudaMemcpy(d_vs, value_scales, bytes_scales, cudaMemcpyHostToDevice) != cudaSuccess) return;

    dim3 deq_block(32, 4);
    dim3 deq_grid(
        (unsigned)((head_dim + deq_block.x - 1U) / deq_block.x),
        (unsigned)((seq_len + deq_block.y - 1U) / deq_block.y)
    );
    dequant_int3_kv_rows_kernel<<<deq_grid, deq_block>>>(
        d_kq, d_ks, d_k, seq_len, head_dim, packed_head_bytes, blocks_per_head, block_size
    );
    dequant_int3_kv_rows_kernel<<<deq_grid, deq_block>>>(
        d_vq, d_vs, d_v, seq_len, head_dim, packed_head_bytes, blocks_per_head, block_size
    );

    const float inv_sqrt_dim = 1.0f / sqrtf((float)head_dim);
    unsigned int block_size_softmax = 256U;
    if (head_dim <= 128U && seq_len <= 512U) {
        block_size_softmax = 128U;
    }
    dim3 block(block_size_softmax);
    dim3 grid_scores((unsigned)((seq_len + block.x - 1U) / block.x));
    dim3 grid_out((unsigned)((head_dim + block.x - 1U) / block.x));

    fused_attention_scores_kernel<<<grid_scores, block>>>(d_q, d_k, d_scores, seq_len, head_dim, inv_sqrt_dim);
    fused_attention_softmax_kernel<<<1, block>>>(d_scores, seq_len, softmax_denom_floor);
    fused_attention_output_kernel<<<grid_out, block>>>(d_scores, d_v, d_out, seq_len, head_dim);
    if (cudaGetLastError() != cudaSuccess) return;
    if (cudaMemcpy(output, d_out, bytes_out, cudaMemcpyDeviceToHost) != cudaSuccess) return;
}

extern "C" void vspec_cuda_launch_attention_fused(VspecKernelContext* ctx) {
    if (!ctx || !ctx->input || !ctx->weight || !ctx->output) {
        return;
    }

    const VspecKVCache* cache = (const VspecKVCache*)ctx->weight;
    if (!cache || cache->current_tokens == 0U || cache->num_heads == 0U || cache->head_dim == 0U) {
        return;
    }

    const float* q = (const float*)ctx->input;
    float* out = (float*)ctx->output;
    const size_t heads = cache->num_heads;
    const size_t d = cache->head_dim;
    const size_t tmax = cache->current_tokens;

    for (size_t h = 0; h < heads; ++h) {
        const float* qh = q + h * d;
        if (cache->int3_compressed) {
            const size_t packed_head_bytes = cache->packed_head_bytes;
            const size_t blocks_per_head = cache->blocks_per_head;
            const size_t block_size = cache->block_size;
            if (packed_head_bytes == 0U || blocks_per_head == 0U || block_size == 0U) {
                return;
            }

            uint8_t* key_head_packed = (uint8_t*)malloc(tmax * packed_head_bytes * sizeof(uint8_t));
            uint8_t* value_head_packed = (uint8_t*)malloc(tmax * packed_head_bytes * sizeof(uint8_t));
            float* key_head_scales = (float*)malloc(tmax * blocks_per_head * sizeof(float));
            float* value_head_scales = (float*)malloc(tmax * blocks_per_head * sizeof(float));
            if (!key_head_packed || !value_head_packed || !key_head_scales || !value_head_scales) {
                free(key_head_packed);
                free(value_head_packed);
                free(key_head_scales);
                free(value_head_scales);
                return;
            }

            for (size_t t = 0; t < tmax; ++t) {
                const size_t qoff_src = (t * heads + h) * packed_head_bytes;
                const size_t soff_src = (t * heads + h) * blocks_per_head;
                memcpy(key_head_packed + t * packed_head_bytes, cache->key_int3 + qoff_src, packed_head_bytes * sizeof(uint8_t));
                memcpy(value_head_packed + t * packed_head_bytes, cache->value_int3 + qoff_src, packed_head_bytes * sizeof(uint8_t));
                memcpy(key_head_scales + t * blocks_per_head, cache->key_scales + soff_src, blocks_per_head * sizeof(float));
                memcpy(value_head_scales + t * blocks_per_head, cache->value_scales + soff_src, blocks_per_head * sizeof(float));
            }

            float* oh = out + h * d;
            vspec_cuda_attention_fused_single_int3kv_f32(
                qh,
                key_head_packed,
                value_head_packed,
                key_head_scales,
                value_head_scales,
                tmax,
                d,
                packed_head_bytes,
                blocks_per_head,
                block_size,
                oh
            );

            free(key_head_packed);
            free(value_head_packed);
            free(key_head_scales);
            free(value_head_scales);
        } else {
            const float* kh = cache->key + h * cache->max_tokens * d;
            const float* vh = cache->value + h * cache->max_tokens * d;
            float* oh = out + h * d;
            vspec_cuda_attention_fused_single_f32(qh, kh, vh, tmax, d, oh);
        }
    }
}
