#include <cuda_runtime.h>
#include <math.h>

#include "vspec/kernel/cuda_ops.h"

__global__ static void silu_kernel(float* data, size_t count) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    const float x = data[idx];
    data[idx] = x / (1.0f + expf(-x));
}

__global__ static void mul_kernel(float* data, const float* other, size_t count) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    data[idx] *= other[idx];
}

extern "C" void vspec_cuda_silu_f32(float* data, size_t count) {
    if (!data || count == 0) {
        return;
    }
    const size_t bytes = count * sizeof(float);
    float* d_data = NULL;
    if (cudaMalloc((void**)&d_data, bytes) != cudaSuccess) return;
    if (cudaMemcpy(d_data, data, bytes, cudaMemcpyHostToDevice) != cudaSuccess) goto cleanup;

    dim3 block(256);
    dim3 grid((unsigned)((count + block.x - 1U) / block.x));
    silu_kernel<<<grid, block>>>(d_data, count);

    if (cudaDeviceSynchronize() != cudaSuccess) goto cleanup;
    (void)cudaMemcpy(data, d_data, bytes, cudaMemcpyDeviceToHost);

+cleanup:
+    if (d_data) cudaFree(d_data);
+}
+
+extern "C" void vspec_cuda_mul_f32(float* data, const float* other, size_t count) {
+    if (!data || !other || count == 0) {
+        return;
+    }
+    const size_t bytes = count * sizeof(float);
+    float* d_data = NULL;
+    float* d_other = NULL;
+    if (cudaMalloc((void**)&d_data, bytes) != cudaSuccess) return;
+    if (cudaMalloc((void**)&d_other, bytes) != cudaSuccess) goto cleanup;
+    if (cudaMemcpy(d_data, data, bytes, cudaMemcpyHostToDevice) != cudaSuccess) goto cleanup;
+    if (cudaMemcpy(d_other, other, bytes, cudaMemcpyHostToDevice) != cudaSuccess) goto cleanup;
+
+    dim3 block(256);
+    dim3 grid((unsigned)((count + block.x - 1U) / block.x));
+    mul_kernel<<<grid, block>>>(d_data, d_other, count);
+
+    if (cudaDeviceSynchronize() != cudaSuccess) goto cleanup;
+    (void)cudaMemcpy(data, d_data, bytes, cudaMemcpyDeviceToHost);
+
+cleanup:
+    if (d_other) cudaFree(d_other);
+    if (d_data) cudaFree(d_data);
+}
