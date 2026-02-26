#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

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

__global__ static void fused_attention_softmax_kernel(float* scores, size_t seq_len, float score_clip, float denom_floor, float temp_min) {
    __shared__ float red[256];
    __shared__ float max_val;
    __shared__ float sum_val;
    __shared__ float adaptive_temp;

    float local_max = -FLT_MAX;
    for (size_t i = threadIdx.x; i < seq_len; i += blockDim.x) {
        float v = clampf_device(scores[i], -score_clip, score_clip);
        scores[i] = v;
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

    float local_mean = 0.0f;
    for (size_t i = threadIdx.x; i < seq_len; i += blockDim.x) {
        local_mean += scores[i];
    }
    red[threadIdx.x] = local_mean;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2U; stride > 0U; stride >>= 1U) {
        if (threadIdx.x < stride) {
            red[threadIdx.x] += red[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float mean = red[0] / fmaxf((float)seq_len, 1.0f);

    float local_var = 0.0f;
    for (size_t i = threadIdx.x; i < seq_len; i += blockDim.x) {
        float d = scores[i] - mean;
        local_var += d * d;
    }
    red[threadIdx.x] = local_var;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2U; stride > 0U; stride >>= 1U) {
        if (threadIdx.x < stride) {
            red[threadIdx.x] += red[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0U) {
        float var = red[0] / fmaxf((float)seq_len, 1.0f);
        float std = sqrtf(fmaxf(var, 0.0f));
        adaptive_temp = fmaxf(temp_min, fminf(1.0f, 0.60f + 0.20f * std));
    }
    __syncthreads();

    float local_sum = 0.0f;
    for (size_t i = threadIdx.x; i < seq_len; i += blockDim.x) {
        float e = expf((scores[i] - max_val) / adaptive_temp);
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
    const float clamp_alpha = 2.8f;
    const float softmax_score_clip = 24.0f;
    const float softmax_denom_floor = 1e-12f;
    const float softmax_temp_min = 0.70f;

    static float* d_q = NULL;
    static float* d_q_clamped = NULL;
    static float* d_k = NULL;
    static float* d_k_clamped = NULL;
    static float* d_v = NULL;
    static float* d_scores = NULL;
    static float* d_out = NULL;
    static size_t cap_q = 0U;
    static size_t cap_kv = 0U;
    static size_t cap_scores = 0U;
    static size_t cap_out = 0U;

    if (cap_q < bytes_q) {
        if (d_q) cudaFree(d_q);
        if (d_q_clamped) cudaFree(d_q_clamped);
        d_q = NULL;
        d_q_clamped = NULL;
        if (cudaMalloc((void**)&d_q, bytes_q) != cudaSuccess) return;
        if (cudaMalloc((void**)&d_q_clamped, bytes_q) != cudaSuccess) return;
        cap_q = bytes_q;
    }
    if (cap_kv < bytes_kv) {
        if (d_k) cudaFree(d_k);
        if (d_k_clamped) cudaFree(d_k_clamped);
        if (d_v) cudaFree(d_v);
        d_k = NULL;
        d_k_clamped = NULL;
        d_v = NULL;
        if (cudaMalloc((void**)&d_k, bytes_kv) != cudaSuccess) return;
        if (cudaMalloc((void**)&d_k_clamped, bytes_kv) != cudaSuccess) return;
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

    clamp_vector_std_kernel<<<1, 1>>>(d_q, d_q_clamped, head_dim, clamp_alpha);
    dim3 clamp_block(128);
    dim3 clamp_grid((unsigned)((seq_len + clamp_block.x - 1U) / clamp_block.x));
    clamp_rows_std_kernel<<<clamp_grid, clamp_block>>>(d_k, d_k_clamped, seq_len, head_dim, clamp_alpha);

    const float inv_sqrt_dim = 1.0f / sqrtf((float)head_dim);
    unsigned int block_size = 256U;
    if (head_dim <= 128U && seq_len <= 512U) {
        block_size = 128U;
    }
    dim3 block(block_size);
    dim3 grid_scores((unsigned)((seq_len + block.x - 1U) / block.x));
    dim3 grid_out((unsigned)((head_dim + block.x - 1U) / block.x));

    fused_attention_scores_kernel<<<grid_scores, block>>>(d_q_clamped, d_k_clamped, d_scores, seq_len, head_dim, inv_sqrt_dim);
    fused_attention_softmax_kernel<<<1, block>>>(d_scores, seq_len, softmax_score_clip, softmax_denom_floor, softmax_temp_min);
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
        const float* kh = cache->key + h * cache->max_tokens * d;
        const float* vh = cache->value + h * cache->max_tokens * d;
        float* oh = out + h * d;
        vspec_cuda_attention_fused_single_f32(qh, kh, vh, tmax, d, oh);
    }
}
