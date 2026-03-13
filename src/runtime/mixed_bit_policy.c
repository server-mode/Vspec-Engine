#include "vspec/runtime/mixed_bit_policy.h"
#include "vspec/runtime/adaptive_precision_engine.h"

#include <math.h>
#include <stdlib.h>

typedef struct VspecLayerSignalState {
    uint32_t layer_id;
    VspecLayerType type;
    float last_rms;
    int used;
} VspecLayerSignalState;

static VspecLayerSignalState g_layer_signal_state[256];

static float vspec_env_float_or_default(const char* name, float fallback) {
    const char* value = getenv(name);
    if (!value || value[0] == '\0') {
        return fallback;
    }
    return (float)atof(value);
}

static uint8_t vspec_env_u8_or_default(const char* name, uint8_t fallback) {
    const char* value = getenv(name);
    if (!value || value[0] == '\0') {
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

static float vspec_layer_rms(const float* data, size_t count) {
    if (!data || count == 0U) {
        return 0.0f;
    }
    double acc = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const double v = (double)data[i];
        acc += v * v;
    }
    return (float)sqrt(acc / (double)count);
}

static float vspec_attention_entropy_collapse_score(const float* data, size_t count) {
    if (!data || count < 2U) {
        return 0.0f;
    }

    const size_t n = (count > 128U) ? 128U : count;
    float max_v = data[0];
    for (size_t i = 1U; i < n; ++i) {
        if (data[i] > max_v) {
            max_v = data[i];
        }
    }

    double sum_exp = 0.0;
    for (size_t i = 0U; i < n; ++i) {
        const double x = (double)data[i] - (double)max_v;
        sum_exp += exp(x);
    }
    if (sum_exp <= 1e-18) {
        return 0.0f;
    }

    double entropy = 0.0;
    for (size_t i = 0U; i < n; ++i) {
        const double x = (double)data[i] - (double)max_v;
        const double p = exp(x) / sum_exp;
        if (p > 1e-18) {
            entropy += -p * log(p);
        }
    }

    const double max_entropy = log((double)n);
    if (max_entropy <= 1e-18) {
        return 0.0f;
    }
    const double normalized = entropy / max_entropy;
    double collapse = 1.0 - normalized;
    if (collapse < 0.0) collapse = 0.0;
    if (collapse > 1.0) collapse = 1.0;
    return (float)collapse;
}

static float vspec_activation_norm_drift(uint32_t layer_id, VspecLayerType type, float current_rms) {
    int free_idx = -1;
    for (size_t i = 0U; i < 256U; ++i) {
        if (g_layer_signal_state[i].used) {
            if (g_layer_signal_state[i].layer_id == layer_id && g_layer_signal_state[i].type == type) {
                const float prev = g_layer_signal_state[i].last_rms;
                g_layer_signal_state[i].last_rms = current_rms;
                if (prev <= 1e-6f) {
                    return 0.0f;
                }
                return fabsf(current_rms - prev) / prev;
            }
        } else if (free_idx < 0) {
            free_idx = (int)i;
        }
    }

    if (free_idx >= 0) {
        g_layer_signal_state[free_idx].used = 1;
        g_layer_signal_state[free_idx].layer_id = layer_id;
        g_layer_signal_state[free_idx].type = type;
        g_layer_signal_state[free_idx].last_rms = current_rms;
    }
    return 0.0f;
}

static uint16_t vspec_model_id_from_layer_id(uint32_t layer_id) {
    return (uint16_t)((layer_id >> 16U) & 0xFFFFU);
}

static float vspec_pressure_from_metrics(
    const VspecMemoryMetrics* metrics,
    const VspecVramBudget* budget,
    size_t target_bytes
) {
    if (budget && budget->total_bytes > 0U) {
        return (float)budget->used_bytes / (float)budget->total_bytes;
    }
    if (metrics && target_bytes > 0U) {
        size_t used = metrics->weight_bytes + metrics->activation_bytes + metrics->kv_bytes + metrics->scratch_bytes;
        return (float)used / (float)target_bytes;
    }
    return 0.0f;
}

static uint8_t vspec_base_bits_from_policy(const VspecMixedBitPolicy* policy, VspecLayerType type) {
    if (!policy) {
        switch (type) {
            case VSPEC_LAYER_ATTENTION:
                return 3;
            case VSPEC_LAYER_MLP:
                return 3;
            case VSPEC_LAYER_EMBED:
                return 3;
            default:
                return 3;
        }
    }

    switch (type) {
        case VSPEC_LAYER_ATTENTION:
            return policy->attention_bits;
        case VSPEC_LAYER_ATTENTION_QK:
            return policy->attention_qk_bits;
        case VSPEC_LAYER_ATTENTION_PROJ:
            return policy->attention_projection_bits;
        case VSPEC_LAYER_MLP:
            return policy->mlp_bits;
        case VSPEC_LAYER_EMBED:
            return policy->embed_bits;
        case VSPEC_LAYER_LM_HEAD:
            return policy->lm_head_bits;
        default:
            return policy->attention_bits;
    }
}

static uint8_t vspec_clamp_bits(uint8_t bits, uint8_t min_bits, uint8_t max_bits) {
    if (bits < min_bits) {
        return min_bits;
    }
    if (bits > max_bits) {
        return max_bits;
    }
    return bits;
}

static int vspec_layer_requires_4bit(VspecLayerType type) {
    return (type == VSPEC_LAYER_ATTENTION_QK ||
            type == VSPEC_LAYER_ATTENTION_PROJ ||
            type == VSPEC_LAYER_LM_HEAD ||
            type == VSPEC_LAYER_EMBED);
}

static uint8_t vspec_apply_realtime_pressure_downshift(
    uint8_t bits,
    const VspecMixedBitPolicy* policy,
    VspecLayerType type,
    const VspecMixedBitPressureProfile* pressure,
    uint8_t min_bits,
    uint8_t max_bits
) {
    if (!policy || !pressure) {
        return vspec_clamp_bits(bits, min_bits, max_bits);
    }

    if (vspec_layer_requires_4bit(type) && bits >= 4U) {
        return 4U;
    }

    float peak_pressure = pressure->vram_pressure;
    if (pressure->kv_pressure > peak_pressure) {
        peak_pressure = pressure->kv_pressure;
    }

    uint8_t out = bits;
    if (peak_pressure >= policy->pressure_critical) {
        const uint8_t step = (uint8_t)(policy->downshift_step * 2U);
        if (out > step) {
            out -= step;
        } else {
            out = min_bits;
        }
    } else if (peak_pressure >= policy->pressure_high) {
        if (out > policy->downshift_step) {
            out -= policy->downshift_step;
        } else {
            out = min_bits;
        }
    }

    if (pressure->kv_fragmentation >= 0.35f && out > min_bits) {
        out = (out > 1U) ? (uint8_t)(out - 1U) : min_bits;
    }

    if (pressure->kv_max_tokens > 0U) {
        float token_pressure = (float)pressure->kv_active_tokens / (float)pressure->kv_max_tokens;
        if (token_pressure >= 0.92f && out > min_bits) {
            out = (out > 1U) ? (uint8_t)(out - 1U) : min_bits;
        }
    }

    return vspec_clamp_bits(out, min_bits, max_bits);
}

void vspec_mixed_bit_policy_default(VspecMixedBitPolicy* policy) {
    if (!policy) {
        return;
    }
    policy->attention_bits = 4;
    policy->attention_qk_bits = 4;
    policy->attention_projection_bits = 4;
    policy->mlp_bits = 4;
    policy->embed_bits = 4;
    policy->lm_head_bits = 4;
    policy->min_bits = 3;
    policy->max_bits = 4;
    policy->downshift_step = 1;
    policy->pressure_high = 0.80f;
    policy->pressure_critical = 0.92f;
    policy->enable_bit_escalation = 1;
    policy->residual_rms_escalate_threshold = 1.35f;
    policy->attention_entropy_escalate_threshold = 0.65f;
    policy->activation_norm_drift_threshold = 0.30f;
    policy->memory_target_bytes = 0U;
    vspec_dynamic_quant_default(&policy->dyn_cfg);
}

uint8_t vspec_mixed_bit_select_bits(
    const VspecMixedBitRuntime* runtime,
    const VspecMixedBitPolicy* policy,
    uint32_t layer_id,
    VspecLayerType type,
    const float* data,
    size_t count,
    const VspecMemoryMetrics* metrics,
    const VspecVramBudget* budget
) {
    uint8_t override_bits = 0U;
    if (vspec_mixed_bit_runtime_get(runtime, layer_id, type, &override_bits)) {
        if (vspec_layer_requires_4bit(type) && override_bits < 4U) {
            return 4U;
        }
        return override_bits;
    }

    uint8_t bits = vspec_base_bits_from_policy(policy, type);

    if (vspec_layer_requires_4bit(type)) {
        return vspec_clamp_bits(4U, 2U, 4U);
    }

    uint8_t min_bits = policy ? policy->min_bits : 3;
    uint8_t max_bits = policy ? policy->max_bits : 4;
    if (min_bits < 2U) {
        min_bits = 2U;
    }
    if (max_bits > 4U) {
        max_bits = 4U;
    }
    if (min_bits > max_bits) {
        min_bits = max_bits;
    }

    const float pressure = vspec_pressure_from_metrics(metrics, budget, policy ? policy->memory_target_bytes : 0U);
    const uint16_t model_id = vspec_model_id_from_layer_id(layer_id);
    const float precision_downgrade_trigger_fallback = vspec_env_float_or_default(
        "VSPEC_PRECISION_DOWNGRADE_TRIGGER",
        policy ? policy->pressure_high : 0.80f
    );
    const float precision_downgrade_trigger = vspec_adaptive_precision_resolve_precision_downgrade_trigger(
        model_id,
        precision_downgrade_trigger_fallback
    );
    const float cache_compression_trigger = vspec_env_float_or_default(
        "VSPEC_CACHE_COMPRESSION_TRIGGER",
        policy ? policy->pressure_critical : 0.92f
    );
    const uint8_t per_model_env_cap = vspec_env_u8_or_default("VSPEC_PER_MODEL_ADAPTIVE_BIT_CAP", 4U);
    const uint8_t per_model_bit_cap = vspec_adaptive_precision_resolve_bit_cap(model_id, per_model_env_cap);

    if (per_model_bit_cap < max_bits) {
        max_bits = per_model_bit_cap;
        if (min_bits > max_bits) {
            min_bits = max_bits;
        }
    }

    const int kv_compress = vspec_adaptive_precision_should_compress_kv(model_id, pressure, cache_compression_trigger);
#if defined(_WIN32)
    _putenv_s("VSPEC_KV_CACHE_COMPRESS_INT3", kv_compress ? "1" : "0");
#else
    setenv("VSPEC_KV_CACHE_COMPRESS_INT3", kv_compress ? "1" : "0", 1);
#endif

    if (data && count > 0U && pressure >= precision_downgrade_trigger) {
        VspecDynamicQuantConfig cfg = policy ? policy->dyn_cfg : (VspecDynamicQuantConfig){3, 4, 32, 99.5f};
        if (type == VSPEC_LAYER_MLP || type == VSPEC_LAYER_EMBED || type == VSPEC_LAYER_ATTENTION) {
            cfg.group_size = 32U;
        }
        if (cfg.min_bits < 3U) {
            cfg.min_bits = 3U;
        }
        if (cfg.max_bits > 4U) {
            cfg.max_bits = 4U;
        }
        VspecDynamicQuantDecision decision = vspec_dynamic_quant_decide(data, count, &cfg);
        bits = vspec_clamp_bits(decision.bits, 3U, 4U);
    } else {
        bits = 4U;
    }

    bits = vspec_clamp_bits(bits, min_bits, max_bits);

    int quality_escalated = 0;
    if (policy && policy->enable_bit_escalation && data && count > 0U) {
        const float rms = vspec_layer_rms(data, count);
        const float drift = vspec_activation_norm_drift(layer_id, type, rms);
        float entropy_collapse = 0.0f;
        if (type == VSPEC_LAYER_ATTENTION || type == VSPEC_LAYER_ATTENTION_QK) {
            entropy_collapse = vspec_attention_entropy_collapse_score(data, count);
        }

        if (rms >= policy->residual_rms_escalate_threshold ||
            drift >= policy->activation_norm_drift_threshold ||
            entropy_collapse >= policy->attention_entropy_escalate_threshold) {
            bits = 4U;
            quality_escalated = 1;
        }
    }

    if (policy && !quality_escalated) {
        if (pressure >= policy->pressure_critical) {
            if (bits > policy->downshift_step * 2U) {
                bits -= (uint8_t)(policy->downshift_step * 2U);
            } else {
                bits = min_bits;
            }
        } else if (pressure >= policy->pressure_high) {
            if (bits > policy->downshift_step) {
                bits -= policy->downshift_step;
            } else {
                bits = min_bits;
            }
        }
    }

    if (vspec_layer_requires_4bit(type)) {
        bits = 4U;
    }

    return vspec_clamp_bits(bits, min_bits, max_bits);
}

uint8_t vspec_mixed_bit_select_bits_realtime(
    const VspecMixedBitRuntime* runtime,
    const VspecMixedBitPolicy* policy,
    uint32_t layer_id,
    VspecLayerType type,
    const float* data,
    size_t count,
    const VspecMemoryMetrics* metrics,
    const VspecVramBudget* budget,
    const VspecMixedBitPressureProfile* pressure
) {
    uint8_t bits = vspec_mixed_bit_select_bits(runtime, policy, layer_id, type, data, count, metrics, budget);
    uint8_t min_bits = policy ? policy->min_bits : 3U;
    uint8_t max_bits = policy ? policy->max_bits : 4U;
    if (min_bits < 2U) {
        min_bits = 2U;
    }
    if (max_bits > 4U) {
        max_bits = 4U;
    }
    if (min_bits > max_bits) {
        min_bits = max_bits;
    }
    bits = vspec_apply_realtime_pressure_downshift(bits, policy, type, pressure, min_bits, max_bits);
    if (vspec_layer_requires_4bit(type)) {
        bits = 4U;
    }
    return bits;
}
