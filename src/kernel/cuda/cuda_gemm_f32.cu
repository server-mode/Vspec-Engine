#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "vspec/kernel/cuda_ops.h"

extern "C" void vspec_cuda_gemm_f32(
    const float* input,
    const float* weight,
    size_t m,
    size_t k,
    size_t n,
    float* output
) {
    if (!input || !weight || !output || m == 0 || k == 0 || n == 0) {
        return;
    }

    const size_t bytes_in = m * k * sizeof(float);
    const size_t bytes_w = n * k * sizeof(float);
    const size_t bytes_out = m * n * sizeof(float);

    float* d_in = NULL;
    float* d_w = NULL;
    float* d_out = NULL;
    cublasHandle_t handle = NULL;

    if (cudaMalloc((void**)&d_in, bytes_in) != cudaSuccess) goto cleanup;
    if (cudaMalloc((void**)&d_w, bytes_w) != cudaSuccess) goto cleanup;
    if (cudaMalloc((void**)&d_out, bytes_out) != cudaSuccess) goto cleanup;

    if (cudaMemcpy(d_in, input, bytes_in, cudaMemcpyHostToDevice) != cudaSuccess) goto cleanup;
    if (cudaMemcpy(d_w, weight, bytes_w, cudaMemcpyHostToDevice) != cudaSuccess) goto cleanup;

    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) goto cleanup;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasStatus_t st = cublasSgemm(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        (int)n,
        (int)m,
        (int)k,
        &alpha,
        d_w,
        (int)k,
        d_in,
        (int)k,
        &beta,
        d_out,
        (int)n
    );
    if (st != CUBLAS_STATUS_SUCCESS) goto cleanup;

    if (cudaDeviceSynchronize() != cudaSuccess) goto cleanup;
    if (cudaMemcpy(output, d_out, bytes_out, cudaMemcpyDeviceToHost) != cudaSuccess) goto cleanup;

+cleanup:
+    if (handle) cublasDestroy(handle);
+    if (d_out) cudaFree(d_out);
+    if (d_w) cudaFree(d_w);
+    if (d_in) cudaFree(d_in);
+}
