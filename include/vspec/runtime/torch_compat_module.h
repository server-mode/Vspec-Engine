#ifndef VSPEC_RUNTIME_TORCH_COMPAT_MODULE_H
#define VSPEC_RUNTIME_TORCH_COMPAT_MODULE_H

#include <stddef.h>

#include "vspec/attention/kv_cache.h"
#include "vspec/core/tensor.h"
#include "vspec/kernel/context.h"
#include "vspec/quant/quant.h"

typedef enum VspecTorchCompatDevice {
    VSPEC_TORCH_COMPAT_DEVICE_AUTO = 0,
    VSPEC_TORCH_COMPAT_DEVICE_CPU = 1,
    VSPEC_TORCH_COMPAT_DEVICE_CUDA = 2,
} VspecTorchCompatDevice;

typedef struct VspecTorchCompatRuntime {
    int initialized;
    VspecTorchCompatDevice device;
    const char* active_backend;
} VspecTorchCompatRuntime;

typedef struct VspecTorchCompatCapabilities {
    int linear_q4;
    int attention;
    int kv_cache;
    int rmsnorm;
    int silu;
    int softmax;
    int matmul_f32;
    int add_f32;
    int mul_f32;
    int gelu_f32;
    int layernorm_f32;
    int argmax_f32;
} VspecTorchCompatCapabilities;

void vspec_torch_compat_runtime_init(
    VspecTorchCompatRuntime* runtime,
    VspecTorchCompatDevice device
);

void vspec_torch_compat_query_capabilities(
    const VspecTorchCompatRuntime* runtime,
    VspecTorchCompatCapabilities* out_caps
);

void vspec_torch_compat_tensor_view(
    VspecTensor* tensor,
    void* data,
    VspecDType dtype,
    size_t ndim,
    const size_t* shape
);

void vspec_torch_compat_linear_forward_q4(
    const VspecTorchCompatRuntime* runtime,
    const float* input,
    size_t m,
    size_t k,
    const unsigned char* q4_weight,
    size_t n,
    const float* scales,
    float* output
);

void vspec_torch_compat_attention_forward(
    const VspecTorchCompatRuntime* runtime,
    const float* query,
    const VspecKVCache* cache,
    float* out
);

void vspec_torch_compat_output_projection_3bit(
    const float* input,
    const float* weight,
    const float* bias,
    size_t in_dim,
    size_t out_dim,
    float* output
);

int vspec_torch_compat_kv_cache_init(
    VspecKVCache* cache,
    float* key_buffer,
    float* value_buffer,
    size_t max_tokens,
    size_t num_heads,
    size_t head_dim
);

int vspec_torch_compat_kv_cache_append(
    VspecKVCache* cache,
    const float* key_token,
    const float* value_token
);

int vspec_torch_compat_kv_cache_enable_int3_compression(
    VspecKVCache* cache,
    size_t block_size
);

void vspec_torch_compat_kv_cache_disable_int3_compression(VspecKVCache* cache);

void vspec_torch_compat_kv_cache_reset(VspecKVCache* cache);

void vspec_torch_compat_rmsnorm_f32(
    const float* input,
    const float* weight,
    const float* bias,
    size_t n,
    float eps,
    float* output
);

void vspec_torch_compat_silu_f32(
    const float* input,
    size_t n,
    float* output
);

void vspec_torch_compat_softmax_f32(
    const float* input,
    size_t n,
    float* output
);

void vspec_torch_compat_matmul_f32(
    const float* a,
    const float* b,
    size_t m,
    size_t k,
    size_t n,
    float* out
);

void vspec_torch_compat_add_f32(
    const float* a,
    const float* b,
    size_t n,
    float* out
);

void vspec_torch_compat_mul_f32(
    const float* a,
    const float* b,
    size_t n,
    float* out
);

void vspec_torch_compat_gelu_f32(
    const float* input,
    size_t n,
    float* out
);

void vspec_torch_compat_layernorm_f32(
    const float* input,
    const float* gamma,
    const float* beta,
    size_t n,
    float eps,
    float* out
);

size_t vspec_torch_compat_argmax_f32(
    const float* input,
    size_t n
);

#endif
