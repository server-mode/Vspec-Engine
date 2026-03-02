#include "vspec/runtime/mixed_bit_policy.h"

#include <math.h>

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

static uint8_t vspec_apply_realtime_pressure_downshift(
    uint8_t bits,
    const VspecMixedBitPolicy* policy,
    const VspecMixedBitPressureProfile* pressure,
    uint8_t min_bits,
    uint8_t max_bits
) {
    if (!policy || !pressure) {
        return vspec_clamp_bits(bits, min_bits, max_bits);
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
    policy->attention_bits = 3;
    policy->attention_qk_bits = 3;
    policy->attention_projection_bits = 4;
    policy->mlp_bits = 3;
    policy->embed_bits = 3;
    policy->lm_head_bits = 4;
    policy->min_bits = 2;
    policy->max_bits = 4;
    policy->downshift_step = 1;
    policy->pressure_high = 0.80f;
    policy->pressure_critical = 0.92f;
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
        return override_bits;
    }

    uint8_t bits = vspec_base_bits_from_policy(policy, type);

    if (type == VSPEC_LAYER_ATTENTION_PROJ || type == VSPEC_LAYER_LM_HEAD) {
        return vspec_clamp_bits(4U, 2U, 4U);
    }

    uint8_t min_bits = policy ? policy->min_bits : 2;
    uint8_t max_bits = policy ? policy->max_bits : 3;
    if (min_bits < 2U) {
        min_bits = 2U;
    }
    if (max_bits > 4U) {
        max_bits = 4U;
    }
    if (min_bits > max_bits) {
        min_bits = max_bits;
    }

    if (data && count > 0U) {
        VspecDynamicQuantConfig cfg = policy ? policy->dyn_cfg : (VspecDynamicQuantConfig){2, 4, 32, 99.5f};
        if (type == VSPEC_LAYER_MLP || type == VSPEC_LAYER_ATTENTION_QK) {
            cfg.group_size = 32U;
        }
        if (cfg.min_bits < min_bits) {
            cfg.min_bits = min_bits;
        }
        if (cfg.max_bits > max_bits) {
            cfg.max_bits = max_bits;
        }
        VspecDynamicQuantDecision decision = vspec_dynamic_quant_decide(data, count, &cfg);
        bits = decision.bits;

        if (type == VSPEC_LAYER_ATTENTION_QK && cfg.max_bits >= 4U) {
            float max_abs = 0.0f;
            float rms_acc = 0.0f;
            for (size_t i = 0; i < count; ++i) {
                float v = fabsf(data[i]);
                if (v > max_abs) {
                    max_abs = v;
                }
                rms_acc += data[i] * data[i];
            }
            float rms = sqrtf(rms_acc / (float)count);
            float outlier_ratio = (rms > 1e-6f) ? (max_abs / rms) : max_abs;
            if (outlier_ratio >= 5.5f) {
                bits = 4U;
            }
        }
    }

    bits = vspec_clamp_bits(bits, min_bits, max_bits);

    if (policy) {
        float pressure = vspec_pressure_from_metrics(metrics, budget, policy->memory_target_bytes);
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
    uint8_t min_bits = policy ? policy->min_bits : 2U;
    uint8_t max_bits = policy ? policy->max_bits : 3U;
    if (min_bits < 2U) {
        min_bits = 2U;
    }
    if (max_bits > 4U) {
        max_bits = 4U;
    }
    if (min_bits > max_bits) {
        min_bits = max_bits;
    }
    return vspec_apply_realtime_pressure_downshift(bits, policy, pressure, min_bits, max_bits);
}
