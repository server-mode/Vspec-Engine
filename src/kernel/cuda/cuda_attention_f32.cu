#include <cuda_runtime.h>
#include <math.h>

#include "vspec/kernel/cuda_ops.h"

__global__ static void attention_scores_kernel(
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

__global__ static void attention_softmax_kernel(float* scores, size_t seq_len) {
    __shared__ float max_val;
    __shared__ float sum_val;

    if (threadIdx.x == 0) {
        max_val = scores[0];
        for (size_t i = 1; i < seq_len; ++i) {
            if (scores[i] > max_val) {
                max_val = scores[i];
            }
        }
        sum_val = 0.0f;
        for (size_t i = 0; i < seq_len; ++i) {
            sum_val += expf(scores[i] - max_val);
        }
    }
    __syncthreads();

    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len) {
        return;
    }
    scores[idx] = expf(scores[idx] - max_val) / sum_val;
}

__global__ static void attention_output_kernel(
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

extern "C" void vspec_cuda_attention_single_f32(
    const float* query,
    const float* keys,
    const float* values,
    size_t seq_len,
    size_t head_dim,
    float* output
) {
    if (!query || !keys || !values || !output || seq_len == 0 || head_dim == 0) {
        return;
    }

    const size_t bytes_q = head_dim * sizeof(float);
    const size_t bytes_k = seq_len * head_dim * sizeof(float);
    const size_t bytes_scores = seq_len * sizeof(float);
    const size_t bytes_out = head_dim * sizeof(float);

    float* d_q = NULL;
    float* d_k = NULL;
    float* d_v = NULL;
    float* d_scores = NULL;
    float* d_out = NULL;

    if (cudaMalloc((void**)&d_q, bytes_q) != cudaSuccess) goto cleanup;
    if (cudaMalloc((void**)&d_k, bytes_k) != cudaSuccess) goto cleanup;
    if (cudaMalloc((void**)&d_v, bytes_k) != cudaSuccess) goto cleanup;
    if (cudaMalloc((void**)&d_scores, bytes_scores) != cudaSuccess) goto cleanup;
    if (cudaMalloc((void**)&d_out, bytes_out) != cudaSuccess) goto cleanup;

    if (cudaMemcpy(d_q, query, bytes_q, cudaMemcpyHostToDevice) != cudaSuccess) goto cleanup;
    if (cudaMemcpy(d_k, keys, bytes_k, cudaMemcpyHostToDevice) != cudaSuccess) goto cleanup;
    if (cudaMemcpy(d_v, values, bytes_k, cudaMemcpyHostToDevice) != cudaSuccess) goto cleanup;

    const float inv_sqrt_dim = 1.0f / sqrtf((float)head_dim);
    dim3 block(256);
    dim3 grid_scores((unsigned)((seq_len + block.x - 1U) / block.x));
    attention_scores_kernel<<<grid_scores, block>>>(d_q, d_k, d_scores, seq_len, head_dim, inv_sqrt_dim);

    dim3 grid_softmax(1);
    attention_softmax_kernel<<<grid_softmax, block>>>(d_scores, seq_len);

    dim3 grid_out((unsigned)((head_dim + block.x - 1U) / block.x));
    attention_output_kernel<<<grid_out, block>>>(d_scores, d_v, d_out, seq_len, head_dim);

    if (cudaDeviceSynchronize() != cudaSuccess) goto cleanup;
    if (cudaMemcpy(output, d_out, bytes_out, cudaMemcpyDeviceToHost) != cudaSuccess) goto cleanup;

+cleanup:
+    if (d_out) cudaFree(d_out);
+    if (d_scores) cudaFree(d_scores);
+    if (d_v) cudaFree(d_v);
+    if (d_k) cudaFree(d_k);
+    if (d_q) cudaFree(d_q);
+}
