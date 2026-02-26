#include "vspec/runtime/torch_compat_module.h"

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <string.h>

#include "vspec/kernel/backend.h"
#include "vspec/runtime/runtime.h"
#include "vspec/runtime/three_bit_runtime_modules.h"

static void _fill_default_stride(VspecTensor* tensor) {
    if (!tensor || tensor->ndim == 0) {
        return;
    }

    size_t running = 1;
    for (size_t i = tensor->ndim; i > 0; --i) {
        size_t idx = i - 1;
        tensor->stride[idx] = running;
        running *= tensor->shape[idx];
    }
}

void vspec_torch_compat_runtime_init(
    VspecTorchCompatRuntime* runtime,
    VspecTorchCompatDevice device
) {
    if (!runtime) {
        return;
    }

    vspec_runtime_init_default();
    runtime->initialized = 1;
    runtime->device = device;

    const VspecRuntimeHwState* hw = vspec_runtime_get_hw_state();
    runtime->active_backend = (hw && hw->active_backend_name) ? hw->active_backend_name : "cpu-ref";
}

void vspec_torch_compat_query_capabilities(
    const VspecTorchCompatRuntime* runtime,
    VspecTorchCompatCapabilities* out_caps
) {
    if (!out_caps) {
        return;
    }

    (void)memset(out_caps, 0, sizeof(*out_caps));
    if (!runtime || !runtime->initialized) {
        return;
    }

    out_caps->linear_q4 = 1;
    out_caps->attention = 1;
    out_caps->kv_cache = 1;
    out_caps->rmsnorm = 1;
    out_caps->silu = 1;
    out_caps->softmax = 1;
    out_caps->matmul_f32 = 1;
    out_caps->add_f32 = 1;
    out_caps->mul_f32 = 1;
    out_caps->gelu_f32 = 1;
    out_caps->layernorm_f32 = 1;
    out_caps->argmax_f32 = 1;
}

void vspec_torch_compat_tensor_view(
    VspecTensor* tensor,
    void* data,
    VspecDType dtype,
    size_t ndim,
    const size_t* shape
) {
    if (!tensor || !shape || ndim == 0 || ndim > 4) {
        return;
    }

    (void)memset(tensor, 0, sizeof(*tensor));
    tensor->data = data;
    tensor->dtype = dtype;
    tensor->ndim = ndim;
    for (size_t i = 0; i < ndim; ++i) {
        tensor->shape[i] = shape[i];
    }
    _fill_default_stride(tensor);
}

void vspec_torch_compat_linear_forward_q4(
    const VspecTorchCompatRuntime* runtime,
    const float* input,
    size_t m,
    size_t k,
    const unsigned char* q4_weight,
    size_t n,
    const float* scales,
    float* output
) {
    (void)runtime;
    if (!input || !q4_weight || !scales || !output || m == 0 || k == 0 || n == 0) {
        return;
    }

    VspecKernelContext ctx;
    (void)memset(&ctx, 0, sizeof(ctx));
    ctx.input = (void*)input;
    ctx.weight = (void*)q4_weight;
    ctx.output = (void*)output;
    ctx.config.m = m;
    ctx.config.k = k;
    ctx.config.n = n;
    vspec_quant_meta_init(&ctx.qmeta);
    ctx.qmeta.type = VSPEC_QUANT_INT4;
    ctx.qmeta.scales = scales;

    vspec_linear_forward(&ctx);
}

void vspec_torch_compat_attention_forward(
    const VspecTorchCompatRuntime* runtime,
    const float* query,
    const VspecKVCache* cache,
    float* out
) {
    (void)runtime;
    if (!query || !cache || !out) {
        return;
    }

    VspecKernelContext ctx;
    (void)memset(&ctx, 0, sizeof(ctx));
    ctx.input = (void*)query;
    ctx.weight = (void*)cache;
    ctx.output = (void*)out;

    vspec_attention_forward(&ctx);
}

int vspec_torch_compat_kv_cache_init(
    VspecKVCache* cache,
    float* key_buffer,
    float* value_buffer,
    size_t max_tokens,
    size_t num_heads,
    size_t head_dim
) {
    return vspec_kv_cache_init(cache, key_buffer, value_buffer, max_tokens, num_heads, head_dim);
}

int vspec_torch_compat_kv_cache_append(
    VspecKVCache* cache,
    const float* key_token,
    const float* value_token
) {
    return vspec_kv_cache_append(cache, key_token, value_token);
}

void vspec_torch_compat_kv_cache_reset(VspecKVCache* cache) {
    vspec_kv_cache_reset(cache);
}

void vspec_torch_compat_rmsnorm_f32(
    const float* input,
    const float* weight,
    const float* bias,
    size_t n,
    float eps,
    float* output
) {
    if (!input || !weight || !output || n == 0) {
        return;
    }

    float mean_sq = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        mean_sq += input[i] * input[i];
    }
    mean_sq /= (float)n;
    const float inv_rms = 1.0f / sqrtf(mean_sq + eps);

    for (size_t i = 0; i < n; ++i) {
        float v = input[i] * inv_rms * weight[i];
        if (bias) {
            v += bias[i];
        }
        output[i] = v;
    }
}

void vspec_torch_compat_silu_f32(
    const float* input,
    size_t n,
    float* output
) {
    if (!input || !output || n == 0) {
        return;
    }

    for (size_t i = 0; i < n; ++i) {
        float x = input[i];
        output[i] = x / (1.0f + expf(-x));
    }
}

void vspec_torch_compat_softmax_f32(
    const float* input,
    size_t n,
    float* output
) {
    if (!input || !output || n == 0) {
        return;
    }

    Vspec3BitSoftmaxManager manager;
    vspec_3bit_softmax_manager_default(&manager);
    if (manager.enabled) {
        vspec_3bit_softmax_apply(&manager, input, n, output);
        return;
    }

    float max_v = -FLT_MAX;
    for (size_t i = 0; i < n; ++i) {
        if (input[i] > max_v) {
            max_v = input[i];
        }
    }

    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        output[i] = expf(input[i] - max_v);
        sum += output[i];
    }
    if (sum <= 0.0f) {
        return;
    }
    for (size_t i = 0; i < n; ++i) {
        output[i] /= sum;
    }
}

void vspec_torch_compat_output_projection_3bit(
    const float* input,
    const float* weight,
    const float* bias,
    size_t in_dim,
    size_t out_dim,
    float* output
) {
    if (!input || !weight || !output || in_dim == 0 || out_dim == 0) {
        return;
    }

    Vspec3BitAttentionManager manager;
    vspec_3bit_attention_manager_default(&manager);
    vspec_3bit_attention_output_projection(
        &manager,
        input,
        weight,
        bias,
        in_dim,
        out_dim,
        output
    );
}

void vspec_torch_compat_matmul_f32(
    const float* a,
    const float* b,
    size_t m,
    size_t k,
    size_t n,
    float* out
) {
    if (!a || !b || !out || m == 0 || k == 0 || n == 0) {
        return;
    }

    for (size_t row = 0; row < m; ++row) {
        for (size_t col = 0; col < n; ++col) {
            float acc = 0.0f;
            for (size_t t = 0; t < k; ++t) {
                acc += a[row * k + t] * b[t * n + col];
            }
            out[row * n + col] = acc;
        }
    }
}

void vspec_torch_compat_add_f32(
    const float* a,
    const float* b,
    size_t n,
    float* out
) {
    if (!a || !b || !out || n == 0) {
        return;
    }
    for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}

void vspec_torch_compat_mul_f32(
    const float* a,
    const float* b,
    size_t n,
    float* out
) {
    if (!a || !b || !out || n == 0) {
        return;
    }
    for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] * b[i];
    }
}

void vspec_torch_compat_gelu_f32(
    const float* input,
    size_t n,
    float* out
) {
    if (!input || !out || n == 0) {
        return;
    }

    const float sqrt_2_over_pi = 0.7978845608028654f;
    for (size_t i = 0; i < n; ++i) {
        float x = input[i];
        float x3 = x * x * x;
        float inner = sqrt_2_over_pi * (x + 0.044715f * x3);
        out[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

void vspec_torch_compat_layernorm_f32(
    const float* input,
    const float* gamma,
    const float* beta,
    size_t n,
    float eps,
    float* out
) {
    if (!input || !gamma || !out || n == 0) {
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
    float inv = 1.0f / sqrtf(var + eps);

    for (size_t i = 0; i < n; ++i) {
        float v = (input[i] - mean) * inv;
        v *= gamma[i];
        if (beta) {
            v += beta[i];
        }
        out[i] = v;
    }
}

size_t vspec_torch_compat_argmax_f32(
    const float* input,
    size_t n
) {
    if (!input || n == 0) {
        return 0;
    }

    size_t best_idx = 0;
    float best_val = input[0];
    for (size_t i = 1; i < n; ++i) {
        if (input[i] > best_val) {
            best_val = input[i];
            best_idx = i;
        }
    }
    return best_idx;
}
