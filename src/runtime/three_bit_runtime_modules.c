#include "vspec/runtime/three_bit_runtime_modules.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static float _clipf(float x, float lo, float hi) {
    if (x < lo) {
        return lo;
    }
    if (x > hi) {
        return hi;
    }
    return x;
}

static int _get_env_flag(const char* name) {
    const char* value = getenv(name);
    if (!value || !value[0]) {
        return 0;
    }
    return (value[0] == '1' || value[0] == 'y' || value[0] == 'Y' || value[0] == 't' || value[0] == 'T') ? 1 : 0;
}

static uint8_t _parse_env_bits_or_default(const char* name, uint8_t fallback) {
    const char* value = getenv(name);
    if (!value || !value[0]) {
        return fallback;
    }
    int parsed = atoi(value);
    if (parsed < 2) {
        parsed = 2;
    }
    if (parsed > 4) {
        parsed = 4;
    }
    return (uint8_t)parsed;
}

size_t vspec_3bit_resolve_block_size(size_t requested) {
    if (requested == 32U || requested == 64U) {
        return requested;
    }
    return 64U;
}

static size_t _parse_env_block_size_or_default(const char* name, size_t fallback) {
    const char* value = getenv(name);
    if (!value || !value[0]) {
        return vspec_3bit_resolve_block_size(fallback);
    }
    int parsed = atoi(value);
    if (parsed != 32 && parsed != 64) {
        return vspec_3bit_resolve_block_size(fallback);
    }
    return (size_t)parsed;
}

int vspec_runtime_3bit_enabled(void) {
    const char* fused_bits = getenv("VSPEC_FUSED_BITS");
    if (fused_bits && fused_bits[0] == '3' && fused_bits[1] == '\0') {
        return 1;
    }
    return _get_env_flag("VSPEC_3BIT_RUNTIME_MODULE");
}

uint8_t vspec_runtime_3bit_bits_for_component(const char* component_name) {
    if (!component_name || !component_name[0]) {
        return _parse_env_bits_or_default("VSPEC_3BIT_DEFAULT_BITS", 3U);
    }

    if (strcmp(component_name, "attention.projection") == 0) {
        return _parse_env_bits_or_default("VSPEC_3BIT_ATTN_PROJ_BITS", 4U);
    }
    if (strcmp(component_name, "attention.qk") == 0) {
        return _parse_env_bits_or_default("VSPEC_3BIT_ATTN_QK_BITS", 3U);
    }
    if (strcmp(component_name, "mlp") == 0) {
        return _parse_env_bits_or_default("VSPEC_3BIT_MLP_BITS", 3U);
    }
    if (strcmp(component_name, "lm_head") == 0) {
        return _parse_env_bits_or_default("VSPEC_3BIT_LM_HEAD_BITS", 4U);
    }

    return _parse_env_bits_or_default("VSPEC_3BIT_DEFAULT_BITS", 3U);
}

void vspec_3bit_softmax_manager_default(Vspec3BitSoftmaxManager* manager) {
    if (!manager) {
        return;
    }
    manager->enabled = vspec_runtime_3bit_enabled();
    manager->logit_clip = 10.0f;
    manager->temperature_floor = 0.70f;
    manager->min_denom = 1e-12f;
}

void vspec_3bit_accum_manager_default(Vspec3BitAccumManager* manager) {
    if (!manager) {
        return;
    }
    manager->enabled = vspec_runtime_3bit_enabled();
    manager->compensation_limit = 1e7f;
    manager->pairwise_block = _parse_env_block_size_or_default("VSPEC_3BIT_BLOCK_SIZE", 64U);
}

void vspec_3bit_noise_reducer_default(Vspec3BitNoiseReducer* reducer) {
    if (!reducer) {
        return;
    }
    reducer->enabled = vspec_runtime_3bit_enabled();
    reducer->input_clip = 8.0f;
    reducer->smooth_alpha = 0.20f;
    reducer->outlier_threshold = 4.5f;
    reducer->activation_clamp_alpha = 2.8f;
}

void vspec_3bit_attention_manager_default(Vspec3BitAttentionManager* manager) {
    if (!manager) {
        return;
    }
    manager->enabled = vspec_runtime_3bit_enabled();
    manager->qk_compute_bits = vspec_runtime_3bit_bits_for_component("attention.qk");
    manager->output_projection_bits = vspec_runtime_3bit_bits_for_component("attention.projection");
    manager->mlp_compute_bits = vspec_runtime_3bit_bits_for_component("mlp");
    manager->qk_scale_min = 0.05f;
    manager->qk_scale_max = 4.00f;
    manager->output_clip = 16.0f;
    vspec_3bit_softmax_manager_default(&manager->softmax);
    vspec_3bit_accum_manager_default(&manager->accum);
    vspec_3bit_noise_reducer_default(&manager->noise);
    if (manager->output_projection_bits >= 4U) {
        manager->output_clip = 32.0f;
    }
}

void vspec_3bit_noise_reduce_vector(
    const Vspec3BitNoiseReducer* reducer,
    const float* input,
    size_t n,
    float* output
) {
    if (!input || !output || n == 0) {
        return;
    }
    if (!reducer || !reducer->enabled) {
        for (size_t i = 0; i < n; ++i) {
            output[i] = input[i];
        }
        return;
    }

    const float clip = fmaxf(1.0f, reducer->input_clip);
    const float alpha = _clipf(reducer->smooth_alpha, 0.0f, 0.95f);
    const float outlier_th = fmaxf(1.0f, reducer->outlier_threshold);

    float prev = _clipf(input[0], -clip, clip);
    output[0] = prev;
    for (size_t i = 1; i < n; ++i) {
        float v = _clipf(input[i], -clip, clip);
        float delta = v - prev;
        if (fabsf(delta) > outlier_th) {
            v = prev + ((delta > 0.0f) ? outlier_th : -outlier_th);
        }
        output[i] = prev * alpha + v * (1.0f - alpha);
        prev = output[i];
    }
}

void vspec_3bit_dynamic_clamp_std(
    const float* input,
    size_t n,
    float alpha,
    float* output
) {
    if (!input || !output || n == 0) {
        return;
    }

    double mean = 0.0;
    for (size_t i = 0; i < n; ++i) {
        mean += (double)input[i];
    }
    mean /= (double)n;

    double var = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double d = (double)input[i] - mean;
        var += d * d;
    }
    var /= (double)n;
    const float std = (float)sqrt(var);
    const float th = fmaxf(1e-6f, fabsf(alpha) * std);

    for (size_t i = 0; i < n; ++i) {
        output[i] = _clipf(input[i], -th, th);
    }
}

void vspec_3bit_softmax_apply(
    const Vspec3BitSoftmaxManager* manager,
    const float* logits,
    size_t n,
    float* probs
) {
    if (!manager || !logits || !probs || n == 0) {
        return;
    }

    const float effective_temp = (manager->enabled) ? _clipf(1.0f, manager->temperature_floor, 1.0f) : 1.0f;
    const float inv_temp = 1.0f / fmaxf(1e-6f, effective_temp);

    float max_v = -FLT_MAX;
    for (size_t i = 0; i < n; ++i) {
        float v = logits[i] * inv_temp;
        if (manager->enabled) {
            v = _clipf(v, -manager->logit_clip, manager->logit_clip);
        }
        probs[i] = v;
        if (v > max_v) {
            max_v = v;
        }
    }

    double denom = 0.0;
    for (size_t i = 0; i < n; ++i) {
        float shifted = probs[i] - max_v;
        float ex = expf(shifted);
        probs[i] = ex;
        denom += (double)ex;
    }

    if (denom < (double)fmaxf(manager->min_denom, 1e-30f)) {
        const float uniform = 1.0f / (float)n;
        for (size_t i = 0; i < n; ++i) {
            probs[i] = uniform;
        }
        return;
    }

    const float inv_denom = 1.0f / (float)denom;
    for (size_t i = 0; i < n; ++i) {
        probs[i] *= inv_denom;
    }
}

void vspec_3bit_accum_reset(Vspec3BitAccumState* state) {
    if (!state) {
        return;
    }
    state->sum = 0.0;
    state->compensation = 0.0;
}

void vspec_3bit_accum_add(
    const Vspec3BitAccumManager* manager,
    Vspec3BitAccumState* state,
    float value
) {
    if (!state) {
        return;
    }
    if (!manager || !manager->enabled) {
        state->sum += (double)value;
        return;
    }

    const double y = (double)value - state->compensation;
    const double t = state->sum + y;
    state->compensation = (t - state->sum) - y;
    state->sum = t;

    if (fabs(state->sum) > (double)manager->compensation_limit) {
        state->sum = _clipf((float)state->sum, -manager->compensation_limit, manager->compensation_limit);
        state->compensation = 0.0;
    }
}

float vspec_3bit_accum_value(const Vspec3BitAccumState* state) {
    if (!state) {
        return 0.0f;
    }
    return (float)state->sum;
}

float vspec_3bit_accum_dot_f32(
    const Vspec3BitAccumManager* manager,
    const float* a,
    const float* b,
    size_t n
) {
    if (!a || !b || n == 0) {
        return 0.0f;
    }

    if (!manager || !manager->enabled) {
        float out = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            out += a[i] * b[i];
        }
        return out;
    }

    const size_t block = vspec_3bit_resolve_block_size((manager->pairwise_block == 0U) ? 64U : manager->pairwise_block);
    Vspec3BitAccumState total;
    vspec_3bit_accum_reset(&total);

    for (size_t base = 0; base < n; base += block) {
        size_t end = base + block;
        if (end > n) {
            end = n;
        }
        float block_abs = 0.0f;
        for (size_t i = base; i < end; ++i) {
            float av = fabsf(a[i]);
            float bv = fabsf(b[i]);
            if (av > block_abs) {
                block_abs = av;
            }
            if (bv > block_abs) {
                block_abs = bv;
            }
        }
        const float block_scale = fmaxf(block_abs, 1e-6f);
        Vspec3BitAccumState local;
        vspec_3bit_accum_reset(&local);
        for (size_t i = base; i < end; ++i) {
            const float an = a[i] / block_scale;
            const float bn = b[i] / block_scale;
            vspec_3bit_accum_add(manager, &local, an * bn);
        }
        vspec_3bit_accum_add(manager, &total, vspec_3bit_accum_value(&local) * block_scale * block_scale);
    }

    return vspec_3bit_accum_value(&total);

}

float vspec_3bit_attention_qk_score(
    const Vspec3BitAttentionManager* manager,
    const float* query,
    const float* key,
    size_t head_dim,
    float inv_sqrt_d
) {
    if (!query || !key || head_dim == 0) {
        return 0.0f;
    }

    float score = vspec_3bit_accum_dot_f32(manager ? &manager->accum : NULL, query, key, head_dim);
    float scale = inv_sqrt_d;
    if (manager && manager->enabled) {
        scale = _clipf(scale, manager->qk_scale_min, manager->qk_scale_max);
    }
    return score * scale;
}

void vspec_3bit_attention_output_projection(
    const Vspec3BitAttentionManager* manager,
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

    float* input_clamped = (float*)malloc(in_dim * sizeof(float));
    float* row_clamped = (float*)malloc(in_dim * sizeof(float));
    if (!input_clamped || !row_clamped) {
        free(input_clamped);
        free(row_clamped);
        return;
    }

    const float alpha = (manager && manager->noise.enabled) ? manager->noise.activation_clamp_alpha : 2.8f;
    vspec_3bit_dynamic_clamp_std(input, in_dim, alpha, input_clamped);

    for (size_t out_idx = 0; out_idx < out_dim; ++out_idx) {
        const float* row = weight + out_idx * in_dim;
        vspec_3bit_dynamic_clamp_std(row, in_dim, alpha, row_clamped);
        float v = vspec_3bit_accum_dot_f32(manager ? &manager->accum : NULL, input_clamped, row_clamped, in_dim);
        if (bias) {
            v += bias[out_idx];
        }
        if (manager && manager->enabled) {
            v = _clipf(v, -manager->output_clip, manager->output_clip);
        }
        output[out_idx] = v;
    }

    free(input_clamped);
    free(row_clamped);
}
