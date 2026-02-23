#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#if defined(VSPEC_USE_CUBLAS) && VSPEC_USE_CUBLAS
#include <cublas_v2.h>
#endif

#include "vspec/kernel/context.h"
#include "vspec/kernel/cuda_fused.h"
#include "vspec/kernel/cuda_ops.h"
#include "vspec/quant/int4.h"
#include "vspec/quant/pack.h"
#include "vspec/quant/quant.h"
#include "vspec/runtime/qlora_adapter.h"

static int vspec_env_enabled(const char* key, int default_value) {
    const char* v = getenv(key);
    if (!v || v[0] == '\0') {
        return default_value;
    }
    if (strcmp(v, "1") == 0 || strcmp(v, "true") == 0 || strcmp(v, "TRUE") == 0 || strcmp(v, "yes") == 0) {
        return 1;
    }
    return 0;
}

static float vspec_env_float(const char* key, float default_value) {
    const char* v = getenv(key);
    if (!v || v[0] == '\0') {
        return default_value;
    }
    return (float)atof(v);
}

static void vspec_dequant_row_to_float(
    VspecQuantType type,
    const uint8_t* row_packed,
    size_t k,
    float scale,
    float* out_row
) {
    if (!row_packed || !out_row || k == 0U) {
        return;
    }

    if (type == VSPEC_QUANT_INT4) {
        for (size_t t = 0; t < k; ++t) {
            const int8_t q = vspec_int4_get(row_packed, t);
            out_row[t] = ((float)q) * scale;
        }
        return;
    }

    if (type == VSPEC_QUANT_INT3) {
        for (size_t t = 0; t < k; ++t) {
            const int8_t q = vspec_quant_get_signed(row_packed, t, 3);
            out_row[t] = ((float)q) * scale;
        }
        return;
    }
}

static void vspec_unpack_quant_to_float_kxn(
    VspecKernelContext* ctx,
    float* w_kxn
) {
    const size_t n = ctx->config.n;
    const size_t k = ctx->config.k;
    const uint8_t* packed = (const uint8_t*)ctx->weight;
    const float* scales = ctx->qmeta.scales;

    size_t row_bytes = 0U;
    if (ctx->qmeta.type == VSPEC_QUANT_INT4) {
        row_bytes = vspec_int4_packed_bytes(k);
    } else if (ctx->qmeta.type == VSPEC_QUANT_INT3) {
        row_bytes = vspec_quant_packed_bytes(k, 3);
    }

    for (size_t j = 0; j < n; ++j) {
        const uint8_t* row = packed + (j * row_bytes);
        float* temp = (float*)malloc(k * sizeof(float));
        if (!temp) {
            continue;
        }
        vspec_dequant_row_to_float(ctx->qmeta.type, row, k, scales[j], temp);
        for (size_t t = 0; t < k; ++t) {
            w_kxn[t * n + j] = temp[t];
        }
        free(temp);
    }
}

static float vspec_outlier_ratio_from_input(const float* input, size_t count, float threshold) {
    if (!input || count == 0U || threshold <= 0.0f) {
        return 0.0f;
    }
    size_t outliers = 0U;
    for (size_t i = 0; i < count; ++i) {
        if (fabsf(input[i]) > threshold) {
            outliers += 1U;
        }
    }
    return (float)outliers / (float)count;
}

#if defined(VSPEC_USE_CUBLAS) && VSPEC_USE_CUBLAS
static int vspec_cuda_linear_tensorcore_tf32(
    const float* a_row_major,
    const float* b_row_major_kxn,
    size_t m,
    size_t k,
    size_t n,
    float* c_row_major
) {
    if (!a_row_major || !b_row_major_kxn || !c_row_major || m == 0U || n == 0U || k == 0U) {
        return 0;
    }

    static cublasHandle_t handle = NULL;
    static int handle_ready = 0;
    if (!handle_ready) {
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
            return 0;
        }
        (void)cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
        handle_ready = 1;
    }

    const size_t bytes_a = m * k * sizeof(float);
    const size_t bytes_b = k * n * sizeof(float);
    const size_t bytes_c = m * n * sizeof(float);

    float* d_a = NULL;
    float* d_b = NULL;
    float* d_c = NULL;

    if (cudaMalloc((void**)&d_a, bytes_a) != cudaSuccess) goto fail;
    if (cudaMalloc((void**)&d_b, bytes_b) != cudaSuccess) goto fail;
    if (cudaMalloc((void**)&d_c, bytes_c) != cudaSuccess) goto fail;

    if (cudaMemcpy(d_a, a_row_major, bytes_a, cudaMemcpyHostToDevice) != cudaSuccess) goto fail;
    if (cudaMemcpy(d_b, b_row_major_kxn, bytes_b, cudaMemcpyHostToDevice) != cudaSuccess) goto fail;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasStatus_t status = cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_T,
        (int)n,
        (int)m,
        (int)k,
        &alpha,
        d_b,
        CUDA_R_32F,
        (int)k,
        d_a,
        CUDA_R_32F,
        (int)m,
        &beta,
        d_c,
        CUDA_R_32F,
        (int)n,
        CUBLAS_COMPUTE_32F_FAST_TF32,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    if (status != CUBLAS_STATUS_SUCCESS) {
        goto fail;
    }

    if (cudaMemcpy(c_row_major, d_c, bytes_c, cudaMemcpyDeviceToHost) != cudaSuccess) goto fail;

    cudaFree(d_c);
    cudaFree(d_b);
    cudaFree(d_a);
    return 1;

fail:
    if (d_c) cudaFree(d_c);
    if (d_b) cudaFree(d_b);
    if (d_a) cudaFree(d_a);
    return 0;
}
#endif

int vspec_cuda_ultimate_tensorcore_available(void) {
#if defined(VSPEC_USE_CUBLAS) && VSPEC_USE_CUBLAS
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) {
        return 0;
    }
    return 1;
#else
    return 0;
#endif
}

void vspec_cuda_launch_linear_ultimate(VspecKernelContext* ctx) {
    if (!ctx || !ctx->input || !ctx->weight || !ctx->output || !ctx->qmeta.scales) {
        return;
    }

    const int enable_outlier = vspec_env_enabled("VSPEC_ULTIMATE_OUTLIER_AWARE", 1);
    const int enable_tensor_core = vspec_env_enabled("VSPEC_ULTIMATE_TENSORCORE", 1);
    const int enable_qlora = vspec_env_enabled("VSPEC_ULTIMATE_QLORA", 1);
    const float outlier_threshold = vspec_env_float("VSPEC_ULTIMATE_OUTLIER_TH", 6.0f);
    const float quality_bias = vspec_env_float("VSPEC_ULTIMATE_QUALITY_BIAS", 0.8f);
    const float qlora_alpha = vspec_env_float("VSPEC_ULTIMATE_QLORA_ALPHA", 0.02f);
    const uint32_t layer_id = (uint32_t)(ctx->config.flags & 0xFFFFFFFFu);

    const size_t m = ctx->config.m;
    const size_t n = ctx->config.n;
    const size_t k = ctx->config.k;

    const float* input = (const float*)ctx->input;
    float* output = (float*)ctx->output;

    float outlier_ratio = 0.0f;
    if (enable_outlier) {
        outlier_ratio = vspec_outlier_ratio_from_input(input, m * k, outlier_threshold);
    }

    int prefer_high_precision = 0;
    if (outlier_ratio >= 0.02f || quality_bias >= 0.85f || ctx->qmeta.type == VSPEC_QUANT_NONE) {
        prefer_high_precision = 1;
    }

    if (prefer_high_precision) {
        float* w_kxn = (float*)malloc(k * n * sizeof(float));
        if (!w_kxn) {
            return;
        }

        if (ctx->qmeta.type == VSPEC_QUANT_NONE) {
            (void)memcpy(w_kxn, ctx->weight, k * n * sizeof(float));
        } else {
            vspec_unpack_quant_to_float_kxn(ctx, w_kxn);
        }

        int handled = 0;
#if defined(VSPEC_USE_CUBLAS) && VSPEC_USE_CUBLAS
        if (enable_tensor_core && vspec_cuda_ultimate_tensorcore_available()) {
            handled = vspec_cuda_linear_tensorcore_tf32(input, w_kxn, m, k, n, output);
        }
#endif
        if (!handled) {
            vspec_cuda_linear_f32(input, w_kxn, m, k, n, output);
        }

        free(w_kxn);
    } else {
        if (ctx->qmeta.type == VSPEC_QUANT_INT3) {
            vspec_cuda_launch_fused_linear_int3(ctx);
        } else {
            vspec_cuda_launch_fused_linear_int4(ctx);
        }
    }

    if (enable_qlora && layer_id != 0U && vspec_qlora_adapter_has_layer(layer_id)) {
        vspec_qlora_adapter_apply_layer_f32(layer_id, input, m, k, n, output);
    } else if (enable_qlora && qlora_alpha > 0.0f) {
        for (size_t i = 0; i < m * n; ++i) {
            output[i] += qlora_alpha * output[i];
        }
    }
}