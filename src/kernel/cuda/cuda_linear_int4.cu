#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "vspec/kernel/context.h"
#include "vspec/kernel/cuda_fused.h"
#include "vspec/kernel/cuda_ultimate.h"
#include "vspec/quant/quant.h"

extern "C" void vspec_cuda_launch_attention_fused(VspecKernelContext* ctx);

#ifndef VSPEC_CUDA_BLOCK_X
#define VSPEC_CUDA_BLOCK_X 16
#endif

#ifndef VSPEC_CUDA_BLOCK_Y
#define VSPEC_CUDA_BLOCK_Y 16
#endif

__device__ static int8_t decode_int4(uint8_t nibble) {
    nibble &= 0x0F;
    if (nibble & 0x08) {
        return (int8_t)(nibble - 16);
    }
    return (int8_t)nibble;
}

static int vspec_env_enabled_host(const char* key, int default_value) {
    const char* v = getenv(key);
    if (!v || v[0] == '\0') {
        return default_value;
    }
    if (strcmp(v, "1") == 0 || strcmp(v, "true") == 0 || strcmp(v, "TRUE") == 0 || strcmp(v, "yes") == 0) {
        return 1;
    }
    return 0;
}

__global__ __launch_bounds__(VSPEC_CUDA_BLOCK_X * VSPEC_CUDA_BLOCK_Y) static void int4_linear_kernel(
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

    for (size_t t = 0; t < k; ++t) {
        const float av = a[i * k + t];
        const uint8_t byte = b_row[t >> 1U];
        const uint8_t nibble = (t & 1U) ? ((byte >> 4) & 0x0F) : (byte & 0x0F);
        const float w = (float)decode_int4(nibble) * scales[j];
        acc += av * w;
    }

    c[i * n + j] = acc;
}

extern "C" int vspec_cuda_runtime_available(void) {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        return 0;
    }
    return count > 0 ? 1 : 0;
}

extern "C" void vspec_cuda_launch_linear_impl(VspecKernelContext* ctx) {
    if (!ctx || !ctx->input || !ctx->weight || !ctx->output) {
        return;
    }

    const char* ultimate_enabled = getenv("VSPEC_ULTIMATE_ENABLE");
    if (ultimate_enabled && (strcmp(ultimate_enabled, "1") == 0 || strcmp(ultimate_enabled, "true") == 0)) {
        vspec_cuda_launch_linear_ultimate(ctx);
        return;
    }

    if (!ctx->qmeta.scales) {
        return;
    }

    if (ctx->qmeta.type == VSPEC_QUANT_INT4 && vspec_env_enabled_host("VSPEC_FORCE_TENSORCORE_4BIT", 0)) {
        vspec_cuda_launch_linear_ultimate(ctx);
        return;
    }

    if (ctx->qmeta.type == VSPEC_QUANT_INT4) {
        vspec_cuda_launch_fused_linear_int4(ctx);
        return;
    }

    if (ctx->qmeta.type == VSPEC_QUANT_INT3) {
        vspec_cuda_launch_fused_linear_int3_storage(ctx);
        return;
    }
}

extern "C" void vspec_cuda_launch_attention_impl(VspecKernelContext* ctx) {
    vspec_cuda_launch_attention_fused(ctx);
}
